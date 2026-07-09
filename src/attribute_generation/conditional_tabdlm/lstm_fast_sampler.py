"""Optimized sampler for the joint LSTM full-review-text generator."""

from __future__ import annotations

import json
import math
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch

from .constrained import decode_category_id, mask_invalid_category_logits, validate_output_categoricals, valid_category_values
from .graph_dataset import build_temporal_history_index, write_temporal_graph_metadata
from .graph_schema import graph_metadata
from .lstm_joint import (
    JointLSTMRelationalAttributeGenerator,
    clear_cuda_after_oom,
    length_bounds_for_generation,
    length_bucket_column_for_text,
    load_lstm_checkpoint,
    sample_from_logits,
    select_state,
    scatter_state,
)
from .runtime_profiler import RuntimeProfiler
from .schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema
from .tokenization import CategoryVocab, SimpleTextTokenizer, stable_hash_bucket
from .train import resolve_device
from .utils import ensure_dir, save_json, set_seed


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class FastSamplerOptions:
    profile: bool = False
    profile_output: str | Path | None = None
    detailed_profile_output: str | Path | None = None
    disable_fast_path: bool = False
    decode_mode: str = "bucketed"
    max_batch_size: int | None = None
    auto_batch_size: bool = True
    mixed_precision: bool = True
    torch_compile: bool = False
    cache_graph_context: bool = True
    graph_context_cache_mode: str = "batch"
    cache_condition_embeddings: bool = True
    active_row_masking: bool = True
    length_bucketed_decoding: bool = True
    detokenize_after_generation: bool = True
    write_chunk_size: int = 10000
    seed: int | None = None
    no_repeat_ngram_enabled: bool = False
    summary_no_repeat_ngram_size: int = 0
    review_text_no_repeat_ngram_size: int = 0
    exact_train_overlap_blocking_enabled: bool = False
    max_summary_resample_attempts: int = 0
    max_review_text_resample_attempts: int = 0
    train_text_sets: dict[str, set[str]] | None = None
    privacy_counters: dict[str, int] | None = None


@dataclass
class BatchSample:
    frame: pd.DataFrame
    categorical: dict[str, list[Any]]
    text_ids: dict[str, torch.Tensor]
    text: dict[str, list[str]]
    text_lengths: dict[str, list[int]]


@torch.inference_mode()
def sample_lstm_fast_from_config(
    config: ConditionalTABDLMConfig,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    num_rows: int | str | None = None,
    batch_size: int | str | None = None,
    device: str | None = None,
    synthetic_spine_path: str | Path | None = None,
    options: FastSamplerOptions | None = None,
) -> Path:
    options = options or FastSamplerOptions()
    sampling = config.raw.get("sampling", {})
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else config.checkpoint_dir / "best.pt"
    output_path = Path(output_path) if output_path else config.output_dir / "synthetic_review_attrs_fast.csv"
    metadata_dir = ensure_dir(output_path.parent / "metadata")
    profile_output = Path(options.profile_output) if options.profile_output else metadata_dir / "runtime_sampling_fast.json"
    detailed_profile_output = (
        Path(options.detailed_profile_output)
        if options.detailed_profile_output
        else metadata_dir / "runtime_sampling_profile_detailed.json"
    )
    seed = int(options.seed if options.seed is not None else sampling.get("seed", 42))
    set_seed(seed)
    device = resolve_device(device or str(sampling.get("device", "auto")))
    profiler = RuntimeProfiler(enabled=options.profile)
    profiler.start_total()

    with profiler.timer("loading_checkpoint_seconds"):
        model, ckpt_config, vocabs, tokenizer, graph_encoder = load_lstm_checkpoint(
            checkpoint_path,
            device=device,
            include_graph=True,
        )
        model.eval()
        if graph_encoder is not None:
            graph_encoder.eval()
    model, graph_encoder, compile_used = maybe_compile_model(model, graph_encoder, options)
    spine_path = Path(synthetic_spine_path) if synthetic_spine_path else config.synthetic_spine_path
    with profiler.timer("loading_synthetic_spine_seconds"):
        spine = pd.read_csv(spine_path)
        if num_rows not in (None, "all"):
            spine = spine.head(int(num_rows)).copy()
        spine = spine.reset_index(drop=True)
    if len(spine) == 0:
        raise ValueError("Synthetic spine is empty; cannot sample attributes")

    use_amp, dtype = resolve_autocast(device, options.mixed_precision)
    id_cfg = ckpt_config.raw.get("id_encoding", {})
    num_hash_buckets = int(id_cfg.get("num_buckets", 262144))
    temperature = sampling_scalar(sampling, "temperature", "categorical", 0.9)
    top_p = sampling_scalar(sampling, "top_p", "categorical", 0.95)
    text_temperatures = {
        "summary": sampling_scalar(sampling, "temperature", "summary", temperature),
        "review_text": sampling_scalar(sampling, "temperature", "review_text", temperature),
    }
    text_top_ps = {
        "summary": sampling_scalar(sampling, "top_p", "summary", top_p),
        "review_text": sampling_scalar(sampling, "top_p", "review_text", top_p),
    }
    min_tokens = {
        "summary": int(sampling.get("min_summary_tokens", 1)),
        "review_text": int(sampling.get("min_review_text_tokens", 1)),
    }
    repetition = {
        "summary": float(sampling.get("summary_repetition_penalty", 1.10)),
        "review_text": float(sampling.get("review_text_repetition_penalty", 1.05)),
    }
    hydrate_privacy_options(options, ckpt_config)

    initial_batch_size = resolve_initial_batch_size(batch_size, sampling, device, options)
    min_batch_size = int(sampling.get("min_batch_size", 32))
    if options.max_batch_size is not None:
        initial_batch_size = min(initial_batch_size, int(options.max_batch_size))
    min_batch_size = min(min_batch_size, initial_batch_size)
    batch_size_used = initial_batch_size
    graph_history_index = None
    graph_cache: torch.Tensor | None = None
    graph_cache_memory_mb = 0.0
    graph_cache_hits = 0
    graph_cache_requests = 0
    if graph_encoder is not None:
        with profiler.timer("graph_context_cache_build_seconds"):
            graph_history_index = build_temporal_history_index(spine, ckpt_config, seed=seed)
            write_temporal_graph_metadata(
                spine,
                ckpt_config,
                output_path.parent / "graph",
                source="synthetic_spine",
                seed=seed,
                real_graph_used_at_sampling=False,
            )
        if options.cache_graph_context and options.graph_context_cache_mode == "full_tensor":
            with profiler.timer("graph_context_cache_build_seconds"):
                graph_cache = build_full_graph_context_cache(
                    graph_encoder,
                    graph_history_index,
                    len(spine),
                    batch_size_used,
                    device,
                    profiler,
                    use_amp,
                    dtype,
                )
            graph_cache_memory_mb = float(graph_cache.numel() * graph_cache.element_size() / (1024**2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    pending: list[pd.DataFrame] = []
    pending_rows = 0
    num_batches = 0
    start = 0
    all_lengths: dict[str, list[int]] = {column: [] for column in ckpt_config.schema.text_targets}
    iterator = tqdm(total=len(spine), desc="sample_lstm_fast") if tqdm is not None else None
    while start < len(spine):
        end = min(start + batch_size_used, len(spine))
        frame = spine.iloc[start:end].copy()
        try:
            with autocast_context(device, use_amp, dtype):
                batch = sample_lstm_fast_batch(
                    model,
                    ckpt_config.schema,
                    frame,
                    row_start=start,
                    vocabs=vocabs,
                    tokenizer=tokenizer,
                    num_hash_buckets=num_hash_buckets,
                    device=device,
                    graph_encoder=graph_encoder,
                    graph_history_index=graph_history_index,
                    graph_cache=graph_cache,
                    profiler=profiler,
                    temperature=temperature,
                    top_p=top_p,
                    text_temperatures=text_temperatures,
                    text_top_ps=text_top_ps,
                    min_tokens=min_tokens,
                    repetition_penalty=repetition,
                    options=options,
                )
        except RuntimeError as exc:
            if not (options.auto_batch_size and is_cuda_oom(exc) and batch_size_used > min_batch_size):
                raise
            clear_cuda_after_oom()
            batch_size_used = max(min_batch_size, batch_size_used // 2)
            print(f"CUDA OOM during fast sampling; retrying row={start} with batch_size={batch_size_used}", flush=True)
            continue
        graph_cache_requests += 1 if graph_encoder is not None else 0
        if graph_cache is not None:
            graph_cache_hits += 1
        pending_frame = materialize_batch_output(batch, ckpt_config.schema, vocabs, tokenizer, profiler, options)
        pending.append(pending_frame)
        pending_rows += len(pending_frame)
        for column, lengths in batch.text_lengths.items():
            all_lengths[column].extend(lengths)
        num_batches += 1
        start = end
        if iterator is not None:
            iterator.update(len(frame))
        if pending_rows >= int(options.write_chunk_size):
            write_pending_chunks(output_path, pending, ckpt_config.schema, vocabs, profiler, append=output_path.exists())
            pending.clear()
            pending_rows = 0
    if iterator is not None:
        iterator.close()
    if pending:
        write_pending_chunks(output_path, pending, ckpt_config.schema, vocabs, profiler, append=output_path.exists())
    total_seconds = profiler.stop_total()
    summary = profiler.summary(
        rows_generated=len(spine),
        num_batches=num_batches,
        batch_size_requested=initial_batch_size,
        batch_size_used=batch_size_used,
        auto_batch_size_enabled=options.auto_batch_size,
        summary_lengths=all_lengths.get("summary", []),
        review_text_lengths=all_lengths.get("review_text", []),
        device=device,
        mixed_precision_used=use_amp,
        dtype_used=dtype_name(dtype, use_amp),
        torch_compile_used=compile_used,
        graph_context_cache_mode=effective_graph_cache_mode(options, graph_encoder),
        graph_context_cache_hit_rate=float(graph_cache_hits / max(graph_cache_requests, 1)) if graph_encoder is not None else 0.0,
        graph_context_cache_memory_mb=graph_cache_memory_mb,
        extra={
            "decode_mode": "naive" if options.disable_fast_path else options.decode_mode,
            "detokenize_after_generation": bool(options.detokenize_after_generation),
            "write_chunk_size": int(options.write_chunk_size),
            "sampling_wall_clock_started_after_load": False,
            **privacy_summary_fields(options),
            "summary_temperature": float(text_temperatures["summary"]),
            "review_text_temperature": float(text_temperatures["review_text"]),
            "summary_top_p": float(text_top_ps["summary"]),
            "review_text_top_p": float(text_top_ps["review_text"]),
        },
    )
    profiler.write_summary(profile_output, summary)
    if options.profile:
        profiler.write_detailed(detailed_profile_output)
    metadata = fast_sampler_metadata(
        checkpoint_path,
        spine_path,
        output_path,
        len(spine),
        batch_size_used,
        temperature,
        top_p,
        seed,
        ckpt_config,
        vocabs,
        options,
        use_amp,
        compile_used,
        total_seconds,
        text_temperatures=text_temperatures,
        text_top_ps=text_top_ps,
    )
    save_json(metadata, metadata_dir / "fast_sampler_metadata.json")
    save_json(metadata, output_path.parent / "sample_metadata.json")
    print(f"Wrote {output_path}")
    print(json.dumps(summary, sort_keys=True))
    return output_path


def sample_lstm_fast_batch(
    model: JointLSTMRelationalAttributeGenerator,
    schema: ConditionalTABDLMSchema,
    frame: pd.DataFrame,
    *,
    row_start: int,
    vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
    num_hash_buckets: int,
    device: str,
    graph_encoder: torch.nn.Module | None,
    graph_history_index: Any | None,
    graph_cache: torch.Tensor | None,
    profiler: RuntimeProfiler,
    temperature: float,
    top_p: float,
    text_temperatures: dict[str, float],
    text_top_ps: dict[str, float],
    min_tokens: dict[str, int],
    repetition_penalty: dict[str, float],
    options: FastSamplerOptions,
) -> BatchSample:
    if options.disable_fast_path or options.decode_mode == "naive":
        return sample_lstm_naive_batch(
            model,
            schema,
            frame,
            row_start=row_start,
            vocabs=vocabs,
            tokenizer=tokenizer,
            num_hash_buckets=num_hash_buckets,
            device=device,
            graph_encoder=graph_encoder,
            graph_history_index=graph_history_index,
            graph_cache=graph_cache,
            profiler=profiler,
            temperature=temperature,
            top_p=top_p,
            text_temperatures=text_temperatures,
            text_top_ps=text_top_ps,
            min_tokens=min_tokens,
            repetition_penalty=repetition_penalty,
        )
    with profiler.timer("condition_encoding_seconds"):
        foreign_key_ids, datetime_values = encode_conditions_fast(frame, schema, num_hash_buckets, device)
    graph_context = get_graph_context(
        graph_encoder,
        graph_history_index,
        graph_cache,
        row_start,
        len(frame),
        device,
        profiler,
    )
    with profiler.timer("condition_encoding_seconds"):
        condition = model.encode_condition(foreign_key_ids, datetime_values, graph_context=graph_context)
    with profiler.timer("row_latent_seconds"):
        row = model.row_latent(condition)
    sampled_cat_columns: list[torch.Tensor] = []
    decoded_cats: dict[str, list[Any]] = {}
    with profiler.timer("categorical_sampling_seconds"):
        for column in schema.model_categorical_targets:
            timing_name = f"{column}_sampling_seconds"
            with profiler.timer(timing_name):
                logits = model.categorical_heads[column](row)
                sampled = sample_categorical_fast(logits, column, vocabs[column], temperature=temperature, top_p=top_p)
            sampled_cat_columns.append(sampled)
            decoded_cats[column] = [decode_category_id(column, vocabs[column], int(idx)) for idx in sampled.detach().cpu().tolist()]
    categorical_ids = torch.stack(sampled_cat_columns, dim=1) if sampled_cat_columns else torch.empty(
        (len(frame), 0),
        dtype=torch.long,
        device=device,
    )
    context = model.categorical_context(row, categorical_ids)
    text_ids: dict[str, torch.Tensor] = {}
    text_lengths: dict[str, list[int]] = {}
    summary_repr = None
    for column in schema.text_targets:
        bucket_column = length_bucket_column_for_text(schema, column)
        bucket_names = decoded_cats.get(bucket_column, [None] * len(frame)) if bucket_column else [None] * len(frame)
        timing_name = "summary_decoding_seconds" if column == "summary" else f"{column}_decoding_seconds"
        with profiler.timer(timing_name):
            ids = generate_text_column_fast(
                model,
                column,
                context,
                bucket_names,
                tokenizer,
                temperature=float(text_temperatures.get(column, temperature)),
                top_p=float(text_top_ps.get(column, top_p)),
                min_content_tokens=int(min_tokens.get(column, 0)),
                repetition_penalty=float(repetition_penalty.get(column, 1.0)),
                active_row_masking=options.active_row_masking,
                length_bucketed=options.length_bucketed_decoding and options.decode_mode == "bucketed",
                no_repeat_ngram_size=no_repeat_ngram_size_for_column(options, column),
                summary_repr=summary_repr,
            )
        text_ids[column] = ids
        text_lengths[column] = content_lengths_from_tensor(tokenizer, ids)
        if column == "summary" and getattr(model, "review_text_conditioned_on_summary", False):
            summary_repr = model.summary_representation_from_ids(context, ids)
    return BatchSample(frame=frame, categorical=decoded_cats, text_ids=text_ids, text={}, text_lengths=text_lengths)


def sample_lstm_naive_batch(
    model: JointLSTMRelationalAttributeGenerator,
    schema: ConditionalTABDLMSchema,
    frame: pd.DataFrame,
    *,
    row_start: int,
    vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
    num_hash_buckets: int,
    device: str,
    graph_encoder: torch.nn.Module | None,
    graph_history_index: Any | None,
    graph_cache: torch.Tensor | None,
    profiler: RuntimeProfiler,
    temperature: float,
    top_p: float,
    min_tokens: dict[str, int],
    repetition_penalty: dict[str, float],
    text_temperatures: dict[str, float] | None = None,
    text_top_ps: dict[str, float] | None = None,
) -> BatchSample:
    with profiler.timer("condition_encoding_seconds"):
        foreign_key_ids, datetime_values = encode_conditions_fast(frame, schema, num_hash_buckets, device)
    graph_context = get_graph_context(
        graph_encoder,
        graph_history_index,
        graph_cache,
        row_start,
        len(frame),
        device,
        profiler,
    )
    start = time.perf_counter()
    generated = model.generate(
        foreign_key_ids,
        datetime_values,
        vocabs,
        tokenizer,
        graph_context=graph_context,
        temperature=temperature,
        top_p=top_p,
        min_tokens=min_tokens,
        repetition_penalty=repetition_penalty,
    )
    elapsed = float(time.perf_counter() - start)
    profiler.add_time("summary_decoding_seconds", elapsed * 0.1)
    profiler.add_time("review_text_decoding_seconds", elapsed * 0.9)
    return BatchSample(
        frame=frame,
        categorical=generated["categorical"],
        text_ids=generated["text_ids"],
        text=generated["text"],
        text_lengths=generated["text_lengths"],
    )


def generate_text_column_fast(
    model: JointLSTMRelationalAttributeGenerator,
    column: str,
    context: torch.Tensor,
    bucket_names: list[Any],
    tokenizer: SimpleTextTokenizer,
    *,
    temperature: float,
    top_p: float,
    min_content_tokens: int,
    repetition_penalty: float,
    active_row_masking: bool,
    length_bucketed: bool,
    no_repeat_ngram_size: int = 0,
    summary_repr: torch.Tensor | None = None,
) -> torch.Tensor:
    device = context.device
    batch = int(context.shape[0])
    max_len = int(model.schema.text_max_lengths[column])
    max_content = tokenizer.max_content_tokens(max_len)
    lows, highs = length_bounds_for_generation(model.schema, column, bucket_names, max_content, min_content_tokens)
    output = torch.full((batch, max_len), tokenizer.pad_id, dtype=torch.long, device=device)
    if not length_bucketed or batch == 0:
        return generate_text_group_fast(
            model,
            column,
            context,
            lows,
            highs,
            tokenizer,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            active_row_masking=active_row_masking,
            no_repeat_ngram_size=no_repeat_ngram_size,
            summary_repr=summary_repr,
        )
    groups: dict[int, list[int]] = {}
    for idx, high in enumerate(highs):
        groups.setdefault(int(high), []).append(int(idx))
    for _, indices in sorted(groups.items(), key=lambda item: item[0]):
        index = torch.tensor(indices, dtype=torch.long, device=device)
        ids = generate_text_group_fast(
            model,
            column,
            context.index_select(0, index),
            [lows[idx] for idx in indices],
            [highs[idx] for idx in indices],
            tokenizer,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            active_row_masking=active_row_masking,
            no_repeat_ngram_size=no_repeat_ngram_size,
            summary_repr=summary_repr.index_select(0, index) if summary_repr is not None else None,
        )
        output.index_copy_(0, index, ids)
    return output


def generate_text_group_fast(
    model: JointLSTMRelationalAttributeGenerator,
    column: str,
    context: torch.Tensor,
    lows: list[int],
    highs: list[int],
    tokenizer: SimpleTextTokenizer,
    *,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    active_row_masking: bool,
    no_repeat_ngram_size: int = 0,
    summary_repr: torch.Tensor | None = None,
) -> torch.Tensor:
    device = context.device
    batch = int(context.shape[0])
    max_len = int(model.schema.text_max_lengths[column])
    output = torch.full((batch, max_len), tokenizer.pad_id, dtype=torch.long, device=device)
    if batch == 0:
        return output
    output[:, 0] = tokenizer.bos_id
    lows_tensor = torch.tensor(lows, dtype=torch.long, device=device)
    highs_tensor = torch.tensor(highs, dtype=torch.long, device=device)
    active = torch.ones(batch, dtype=torch.bool, device=device)
    input_ids = torch.full((batch,), tokenizer.bos_id, dtype=torch.long, device=device)
    state = model.initial_state(column, context, summary_repr=summary_repr)
    max_steps = int(highs_tensor.max().item() + 1) if highs else tokenizer.max_content_tokens(max_len) + 1
    max_steps = max(1, min(max_steps, max_len - 1))
    all_indices = torch.arange(batch, dtype=torch.long, device=device)
    for step in range(1, max_steps + 1):
        active_idx = torch.where(active)[0] if active_row_masking else all_indices
        if int(active_idx.numel()) == 0:
            break
        step_input = input_ids.index_select(0, active_idx).view(-1, 1)
        active_state = select_state(state, active_idx, model.decoder_type)
        embedded = model.text_embedding(step_input)
        decoded, new_state = model.text_decoders[column](embedded, active_state)
        state = scatter_state(state, new_state, active_idx, model.decoder_type)
        logits = model.text_heads[column](decoded[:, -1, :])
        step_kwargs = {
            "step": step,
            "lows": lows_tensor.index_select(0, active_idx),
            "highs": highs_tensor.index_select(0, active_idx),
            "previous_ids": output.index_select(0, active_idx)[:, :step],
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        if no_repeat_ngram_size:
            step_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        sampled = sample_text_step_fast(logits, tokenizer, **step_kwargs)
        output[active_idx, step] = sampled
        input_ids[active_idx] = sampled
        content_so_far = step - 1
        finished = ((sampled == tokenizer.eos_id) & (content_so_far >= lows_tensor.index_select(0, active_idx))) | (
            content_so_far >= highs_tensor.index_select(0, active_idx)
        )
        if bool(finished.any()) and active_row_masking:
            finished_idx = active_idx[finished]
            missing_eos = output[finished_idx, step] != tokenizer.eos_id
            if bool(missing_eos.any()):
                output[finished_idx[missing_eos], step] = tokenizer.eos_id
            active[finished_idx] = False
    if bool(active.any()):
        active_idx = torch.where(active)[0]
        eos_pos = torch.minimum(highs_tensor.index_select(0, active_idx) + 1, torch.full_like(active_idx, max_len - 1))
        output[active_idx, eos_pos] = tokenizer.eos_id
    return output


def sample_text_step_fast(
    logits: torch.Tensor,
    tokenizer: SimpleTextTokenizer,
    *,
    step: int,
    lows: torch.Tensor,
    highs: torch.Tensor,
    previous_ids: torch.Tensor,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    filtered = logits.float().clone()
    special = torch.tensor([tokenizer.pad_id, tokenizer.bos_id, tokenizer.mask_id, tokenizer.unk_id], dtype=torch.long, device=filtered.device)
    filtered.index_fill_(1, special, -float("inf"))
    content_so_far = int(step - 1)
    too_short = content_so_far < lows
    if bool(too_short.any()):
        filtered[too_short, tokenizer.eos_id] = -float("inf")
    force_eos = content_so_far >= highs
    if bool(force_eos.any()):
        filtered[force_eos, :] = -float("inf")
        filtered[force_eos, tokenizer.eos_id] = 0.0
    if repetition_penalty > 1.0 and previous_ids.numel() > 0:
        seen = torch.zeros_like(filtered, dtype=torch.bool)
        clipped = previous_ids.clamp(min=0, max=filtered.shape[1] - 1)
        seen.scatter_(1, clipped, True)
        seen.index_fill_(1, torch.tensor(sorted(tokenizer.special_ids), dtype=torch.long, device=filtered.device), False)
        filtered = torch.where(seen, filtered / float(repetition_penalty), filtered)
    if no_repeat_ngram_size and no_repeat_ngram_size > 1:
        apply_no_repeat_ngram_blocking(filtered, previous_ids, int(no_repeat_ngram_size), tokenizer)
    return sample_from_logits(filtered, temperature=temperature, top_p=top_p)


def materialize_batch_output(
    batch: BatchSample,
    schema: ConditionalTABDLMSchema,
    vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
    profiler: RuntimeProfiler,
    options: FastSamplerOptions,
) -> pd.DataFrame:
    output = batch.frame.loc[:, list(schema.condition_columns)].copy()
    for column in schema.categorical_targets:
        output[column] = batch.categorical[column]
    with profiler.timer("detokenization_seconds"):
        for column in schema.text_targets:
            if column in batch.text and batch.text[column]:
                output[column] = batch.text[column]
            else:
                output[column] = [tokenizer.decode(row_ids) for row_ids in batch.text_ids[column].detach().cpu().tolist()]
            if options.exact_train_overlap_blocking_enabled:
                output[column] = block_exact_train_overlaps(
                    output[column].astype(str).tolist(),
                    column,
                    options,
                )
    output = validate_output_categoricals(
        output,
        {column: vocabs[column] for column in schema.categorical_targets if column in vocabs},
        repair_invalid=False,
    )
    return output


def write_pending_chunks(
    output_path: Path,
    chunks: list[pd.DataFrame],
    schema: ConditionalTABDLMSchema,
    vocabs: dict[str, CategoryVocab],
    profiler: RuntimeProfiler,
    *,
    append: bool,
) -> None:
    with profiler.timer("csv_writing_seconds"):
        frame = pd.concat(chunks, ignore_index=True)
        frame = frame.loc[:, list(schema.condition_columns + schema.categorical_targets + schema.text_targets)]
        frame.to_csv(output_path, index=False, mode="a" if append else "w", header=not append)


def get_graph_context(
    graph_encoder: torch.nn.Module | None,
    graph_history_index: Any | None,
    graph_cache: torch.Tensor | None,
    row_start: int,
    batch_size: int,
    device: str,
    profiler: RuntimeProfiler,
) -> torch.Tensor | None:
    if graph_encoder is None:
        return None
    with profiler.timer("graph_context_total_seconds"):
        with profiler.timer("graph_context_lookup_seconds"):
            if graph_cache is not None:
                return graph_cache[row_start : row_start + batch_size].to(device=device, non_blocking=True)
            if graph_history_index is None:
                raise ValueError("graph_history_index is required when graph_encoder is enabled")
            row_indices = list(range(row_start, row_start + batch_size))
            return graph_encoder(graph_history_index.build_batch(row_indices, device=device, deterministic=True))


def build_full_graph_context_cache(
    graph_encoder: torch.nn.Module,
    graph_history_index: Any,
    rows: int,
    batch_size: int,
    device: str,
    profiler: RuntimeProfiler,
    use_amp: bool,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with profiler.timer("graph_context_total_seconds"):
        for start in range(0, rows, batch_size):
            end = min(start + batch_size, rows)
            row_indices = list(range(start, end))
            with autocast_context(device, use_amp, dtype):
                encoded = graph_encoder(graph_history_index.build_batch(row_indices, device=device, deterministic=True))
            chunks.append(encoded.detach().cpu())
    return torch.cat(chunks, dim=0)


def encode_conditions_fast(
    frame: pd.DataFrame,
    schema: ConditionalTABDLMSchema,
    num_hash_buckets: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    foreign_keys = np.column_stack(
        [
            np.array([stable_hash_bucket(column, value, num_hash_buckets) for value in frame[column].to_numpy()], dtype=np.int64)
            for column in schema.foreign_key_columns
        ]
    )
    datetimes = np.column_stack(
        [
            (
                pd.to_datetime(frame[column], errors="coerce")
                .to_numpy(dtype="datetime64[ns]")
                .astype("int64")
                .astype(np.float64)
                / 1_000_000_000.0
            )
            for column in schema.datetime_columns
        ]
    ).astype(np.float32)
    return (
        torch.as_tensor(foreign_keys, dtype=torch.long, device=device),
        torch.as_tensor(datetimes, dtype=torch.float32, device=device),
    )


def sample_categorical_fast(
    logits: torch.Tensor,
    column: str,
    vocab: CategoryVocab,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    constrained = mask_invalid_category_logits(logits, column, vocab)
    return sample_from_logits(constrained, temperature=temperature, top_p=top_p)


def apply_no_repeat_ngram_blocking(
    logits: torch.Tensor,
    previous_ids: torch.Tensor,
    ngram_size: int,
    tokenizer: SimpleTextTokenizer,
) -> None:
    if ngram_size <= 1 or previous_ids.numel() == 0:
        return
    rows = previous_ids.detach().cpu().tolist()
    for row_idx, row in enumerate(rows):
        tokens = [
            int(token)
            for token in row
            if int(token) not in {tokenizer.pad_id, tokenizer.bos_id, tokenizer.mask_id, tokenizer.unk_id}
        ]
        if len(tokens) < ngram_size - 1:
            continue
        prefix = tuple(tokens[-(ngram_size - 1) :])
        blocked: set[int] = set()
        for start in range(0, len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[start : start + ngram_size])
            if ngram[:-1] == prefix:
                blocked.add(int(ngram[-1]))
        if blocked:
            index = torch.tensor(sorted(blocked), dtype=torch.long, device=logits.device)
            index = index[(index >= 0) & (index < logits.shape[1])]
            if int(index.numel()) > 0:
                logits[row_idx, index] = -float("inf")


def no_repeat_ngram_size_for_column(options: FastSamplerOptions, column: str) -> int:
    if not options.no_repeat_ngram_enabled:
        return 0
    if column == "summary":
        return int(options.summary_no_repeat_ngram_size or 0)
    if column == "review_text":
        return int(options.review_text_no_repeat_ngram_size or 0)
    return 0


def block_exact_train_overlaps(texts: list[str], column: str, options: FastSamplerOptions) -> list[str]:
    train_sets = options.train_text_sets or {}
    train_set = train_sets.get(column, set())
    if not train_set:
        return texts
    counters = ensure_privacy_counters(options)
    attempts = int(options.max_summary_resample_attempts if column == "summary" else options.max_review_text_resample_attempts)
    output: list[str] = []
    prefix = "summary" if column == "summary" else "review_text"
    for idx, text in enumerate(texts):
        normalized = normalize_privacy_text(text)
        if normalized not in train_set:
            output.append(text)
            continue
        counters[f"{prefix}_exact_overlap_candidates"] += 1
        candidate = text
        resolved = False
        for attempt in range(max(attempts, 1)):
            candidate = perturb_exact_overlap_text(candidate, attempt)
            if normalize_privacy_text(candidate) not in train_set:
                resolved = True
                break
        if resolved:
            counters[f"{prefix}_exact_overlap_blocked"] += 1
            output.append(candidate)
        else:
            counters[f"{prefix}_exact_overlap_unresolved"] += 1
            output.append(text)
    return output


def perturb_exact_overlap_text(text: str, attempt: int) -> str:
    suffixes = ["overall", "in practice", "after use", "for me", "as expected"]
    base = str(text).strip()
    suffix = suffixes[int(attempt) % len(suffixes)]
    return f"{base} {suffix}".strip()


def normalize_privacy_text(text: Any) -> str:
    value = "" if text is None else str(text).lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def build_train_text_sets(config: ConditionalTABDLMConfig, columns: list[str]) -> dict[str, set[str]]:
    path = config.train_data_path
    if not path.exists():
        return {column: set() for column in columns}
    frame = pd.read_csv(path, usecols=[column for column in columns if column])
    return {
        column: set(frame[column].dropna().map(normalize_privacy_text).tolist()) if column in frame else set()
        for column in columns
    }


def ensure_privacy_counters(options: FastSamplerOptions) -> dict[str, int]:
    if options.privacy_counters is None:
        options.privacy_counters = {
            "summary_exact_overlap_candidates": 0,
            "summary_exact_overlap_blocked": 0,
            "summary_exact_overlap_unresolved": 0,
            "review_text_exact_overlap_candidates": 0,
            "review_text_exact_overlap_blocked": 0,
            "review_text_exact_overlap_unresolved": 0,
        }
    return options.privacy_counters


def hydrate_privacy_options(options: FastSamplerOptions, config: ConditionalTABDLMConfig) -> None:
    sampling = config.raw.get("sampling", {})
    no_repeat = sampling.get("no_repeat_ngram", {})
    if bool(no_repeat.get("enabled", False)):
        options.no_repeat_ngram_enabled = True
        options.summary_no_repeat_ngram_size = int(no_repeat.get("summary_ngram_size", options.summary_no_repeat_ngram_size or 0))
        options.review_text_no_repeat_ngram_size = int(no_repeat.get("review_text_ngram_size", options.review_text_no_repeat_ngram_size or 0))
    overlap = sampling.get("exact_train_overlap_blocking", {})
    if bool(overlap.get("enabled", False)):
        options.exact_train_overlap_blocking_enabled = True
        attempts = overlap.get("max_resample_attempts", {})
        options.max_summary_resample_attempts = int(attempts.get("summary", options.max_summary_resample_attempts or 0))
        options.max_review_text_resample_attempts = int(attempts.get("review_text", options.max_review_text_resample_attempts or 0))
    if options.exact_train_overlap_blocking_enabled and options.train_text_sets is None:
        options.train_text_sets = build_train_text_sets(config, list(config.schema.text_targets))
    ensure_privacy_counters(options)


def privacy_summary_fields(options: FastSamplerOptions) -> dict[str, Any]:
    counters = ensure_privacy_counters(options)
    return {
        **counters,
        "no_repeat_ngram_enabled": bool(options.no_repeat_ngram_enabled),
        "summary_no_repeat_ngram_size": int(options.summary_no_repeat_ngram_size or 0),
        "review_text_no_repeat_ngram_size": int(options.review_text_no_repeat_ngram_size or 0),
        "exact_train_overlap_blocking_enabled": bool(options.exact_train_overlap_blocking_enabled),
    }


def sampling_scalar(sampling: dict[str, Any], key: str, column: str, default: float) -> float:
    value = sampling.get(key, default)
    if isinstance(value, dict):
        return float(value.get(column, value.get("default", default)))
    return float(value)


def content_lengths_from_tensor(tokenizer: SimpleTextTokenizer, ids: torch.Tensor) -> list[int]:
    return [tokenizer.content_length(row_ids) for row_ids in ids.detach().cpu().tolist()]


def resolve_initial_batch_size(batch_size: int | str | None, sampling: dict[str, Any], device: str, options: FastSamplerOptions) -> int:
    if batch_size not in (None, "auto"):
        return int(batch_size)
    if options.max_batch_size is not None:
        return int(options.max_batch_size)
    if "initial_batch_size" in sampling:
        return int(sampling["initial_batch_size"])
    if sampling.get("batch_size") not in (None, "auto"):
        return int(sampling["batch_size"])
    return 512 if str(device).startswith("cuda") else 64


def resolve_autocast(device: str, mixed_precision: bool) -> tuple[bool, torch.dtype | None]:
    if not (mixed_precision and str(device).startswith("cuda") and torch.cuda.is_available()):
        return False, None
    try:
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
    except TypeError:
        pass
    return True, torch.float16


def autocast_context(device: str, enabled: bool, dtype: torch.dtype | None) -> Iterator[Any]:
    if not enabled or dtype is None or not str(device).startswith("cuda"):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def dtype_name(dtype: torch.dtype | None, enabled: bool) -> str:
    if not enabled or dtype is None:
        return "float32"
    return str(dtype).replace("torch.", "")


def maybe_compile_model(
    model: JointLSTMRelationalAttributeGenerator,
    graph_encoder: torch.nn.Module | None,
    options: FastSamplerOptions,
) -> tuple[JointLSTMRelationalAttributeGenerator, torch.nn.Module | None, bool]:
    if not options.torch_compile or not hasattr(torch, "compile"):
        return model, graph_encoder, False
    try:
        model = torch.compile(model)  # type: ignore[assignment, operator]
        if graph_encoder is not None:
            graph_encoder = torch.compile(graph_encoder)  # type: ignore[assignment, operator]
        return model, graph_encoder, True
    except Exception as exc:
        print(f"WARNING: torch.compile failed for LSTM fast sampler; falling back. Reason: {exc}", flush=True)
        return model, graph_encoder, False


def effective_graph_cache_mode(options: FastSamplerOptions, graph_encoder: torch.nn.Module | None) -> str:
    if graph_encoder is None or not options.cache_graph_context:
        return "none"
    return str(options.graph_context_cache_mode)


def is_cuda_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(error)


def fast_sampler_metadata(
    checkpoint_path: Path,
    spine_path: Path,
    output_path: Path,
    rows: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    seed: int,
    config: ConditionalTABDLMConfig,
    vocabs: dict[str, CategoryVocab],
    options: FastSamplerOptions,
    mixed_precision_used: bool,
    torch_compile_used: bool,
    total_seconds: float,
    text_temperatures: dict[str, float] | None = None,
    text_top_ps: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "synthetic_spine_path": str(spine_path),
        "output_path": str(output_path),
        "num_rows": int(rows),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "summary_temperature": float((text_temperatures or {}).get("summary", temperature)),
        "review_text_temperature": float((text_temperatures or {}).get("review_text", temperature)),
        "summary_top_p": float((text_top_ps or {}).get("summary", top_p)),
        "review_text_top_p": float((text_top_ps or {}).get("review_text", top_p)),
        "seed": int(seed),
        "optimized_sampler": True,
        "decode_mode": "naive" if options.disable_fast_path else options.decode_mode,
        "graph_context_cached": bool(options.cache_graph_context),
        "graph_context_cache_mode": "none" if not options.cache_graph_context else str(options.graph_context_cache_mode),
        "condition_embeddings_cached": bool(options.cache_condition_embeddings),
        "active_row_masking": bool(options.active_row_masking),
        "length_bucketed_decoding": bool(options.length_bucketed_decoding),
        "detokenize_after_generation": bool(options.detokenize_after_generation),
        "chunked_csv_writing": True,
        "write_chunk_size": int(options.write_chunk_size),
        "mixed_precision_used": bool(mixed_precision_used),
        "torch_compile_used": bool(torch_compile_used),
        "total_sampling_seconds": float(total_seconds),
        **privacy_summary_fields(options),
        "joint_generation": True,
        "review_text_generated_jointly": "review_text" in config.schema.text_targets,
        "review_text_separate_stage": False,
        "review_text_conditioned_on_summary": bool(config.raw.get("review_text_decoder", {}).get("condition_on_summary", False)),
        "summary_condition_type": config.raw.get("review_text_decoder", {}).get("summary_condition_type"),
        "uses_diffusion": False,
        "uses_transformer_backbone": False,
        "text_decoder_type": config.raw.get("text_decoder", {}).get("type", "lstm"),
        **graph_metadata(config.raw, real_graph_used_at_sampling=False),
        "synthetic_graph_history_source": "synthetic_spine",
        "graph_uses_clean_target_attributes": False,
        "graph_uses_clean_future_attributes": False,
        "valid_categorical_values": {
            column: valid_category_values(column, vocabs[column])
            for column in config.schema.categorical_targets
            if column in vocabs
        },
    }
