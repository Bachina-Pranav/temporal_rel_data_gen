"""Optimized sampler for the joint LSTM full-review-text generator."""

from __future__ import annotations

import json
import math
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
    temperature = float(sampling.get("temperature", 0.9))
    top_p = float(sampling.get("top_p", 0.95))
    min_tokens = {
        "summary": int(sampling.get("min_summary_tokens", 1)),
        "review_text": int(sampling.get("min_review_text_tokens", 1)),
    }
    repetition = {
        "summary": float(sampling.get("summary_repetition_penalty", 1.10)),
        "review_text": float(sampling.get("review_text_repetition_penalty", 1.05)),
    }

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
                temperature=temperature,
                top_p=top_p,
                min_content_tokens=int(min_tokens.get(column, 0)),
                repetition_penalty=float(repetition_penalty.get(column, 1.0)),
                active_row_masking=options.active_row_masking,
                length_bucketed=options.length_bucketed_decoding and options.decode_mode == "bucketed",
            )
        text_ids[column] = ids
        text_lengths[column] = content_lengths_from_tensor(tokenizer, ids)
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
    state = model.initial_state(column, context)
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
        sampled = sample_text_step_fast(
            logits,
            tokenizer,
            step=step,
            lows=lows_tensor.index_select(0, active_idx),
            highs=highs_tensor.index_select(0, active_idx),
            previous_ids=output.index_select(0, active_idx)[:, :step],
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
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
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "synthetic_spine_path": str(spine_path),
        "output_path": str(output_path),
        "num_rows": int(rows),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "top_p": float(top_p),
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
        "joint_generation": True,
        "review_text_generated_jointly": "review_text" in config.schema.text_targets,
        "review_text_separate_stage": False,
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
