"""Hierarchical structured-then-text diffusion sampler for Conditional TABDLM."""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch

from .graph_dataset import build_temporal_history_index, write_temporal_graph_metadata
from .graph_encoder import TemporalStructureOnlyGraphEncoder
from .graph_schema import graph_conditioning_enabled, graph_metadata
from .hierarchical_schema import GenerationPlan, generation_plan_from_config, length_field_for_text
from .runtime_profiler import RuntimeProfiler
from .sample import (
    decode_category_id,
    encode_conditions,
    load_model_checkpoint,
    masked_denoising_schedule,
    profile_timer,
    resolve_inference_dtype,
    resolve_sampling_steps,
    reveal_probability_for_schedule,
    sample_categorical_logits,
    sample_length_bucket_logits,
    sample_logits,
    text_lengths,
    validate_spine,
)
from .schema import ConditionalTABDLMConfig
from .tokenization import CategoryVocab, SimpleTextTokenizer, sample_length_from_bucket
from .train import build_graph_encoder, resolve_device
from .utils import ensure_dir, jsonable, set_seed


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


GRAPH_MODES = {"correct", "zero", "shuffled", "no_graph"}


def hierarchical_sample_from_config(
    config: ConditionalTABDLMConfig,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    num_rows: int | str | None = None,
    sample_batch_size: int | None = None,
    structured_steps: int | str | None = None,
    text_steps: int | str | None = None,
    timestep_spacing: str | None = None,
    inference_dtype: str | None = None,
    text_top_k: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    graph_mode_override: str = "correct",
    device: str | None = None,
    seed: int | None = None,
    synthetic_spine_path: str | Path | None = None,
    profile: bool | None = None,
    profile_output: str | Path | None = None,
    debug_write_aux_targets: bool = False,
    oracle_structured_table_path: str | Path | None = None,
) -> Path:
    sampling = config.raw.get("sampling", {})
    diffusion = config.raw.get("diffusion", {})
    profiler = RuntimeProfiler(enabled=bool((profile if profile is not None else sampling.get("profile", False)) or profile_output is not None))
    profiler.start_total()
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else config.checkpoint_dir / "best.pt"
    output_path = Path(output_path) if output_path else config.output_dir / "synthetic_review_attrs.csv"
    graph_mode_override = str(graph_mode_override or sampling.get("graph_mode", "correct"))
    if graph_mode_override not in GRAPH_MODES:
        raise ValueError(f"graph_mode must be one of {sorted(GRAPH_MODES)}")
    num_rows = num_rows if num_rows is not None else sampling.get("num_rows", 100000)
    batch_size = int(sample_batch_size or sampling.get("sample_batch_size", sampling.get("batch_size", 128)))
    temperature = float(temperature if temperature is not None else sampling.get("temperature", 1.0))
    top_p = float(top_p if top_p is not None else sampling.get("top_p", 0.95))
    text_top_k = parse_optional_int(text_top_k if text_top_k is not None else sampling.get("text_top_k"))
    seed = int(seed if seed is not None else sampling.get("seed", 42))
    device = resolve_device(device or str(sampling.get("device", "auto")))
    inference_dtype = str(inference_dtype or sampling.get("inference_dtype", "float32"))
    train_timesteps = int(diffusion.get("timesteps", diffusion.get("train_timesteps", 50)))
    structured_steps = resolve_sampling_steps(
        structured_steps if structured_steps is not None else sampling.get("structured_steps", diffusion.get("sampling_steps", train_timesteps)),
        train_timesteps,
    )
    text_steps = resolve_sampling_steps(
        text_steps if text_steps is not None else sampling.get("text_steps", diffusion.get("sampling_steps", train_timesteps)),
        train_timesteps,
    )
    timestep_spacing = str(timestep_spacing or diffusion.get("timestep_spacing", sampling.get("timestep_spacing", "uniform")))
    set_seed(seed)

    with profile_timer(profiler, "loading_model_seconds", device=device):
        model, ckpt_config, vocabs, tokenizer, graph_encoder = load_model_checkpoint(checkpoint_path, device, include_graph=True)
    ckpt_config.raw.setdefault("generation", config.raw.get("generation", {}))
    plan = generation_plan_from_config(ckpt_config.raw, ckpt_config.schema)

    spine_path = Path(synthetic_spine_path) if synthetic_spine_path else config.synthetic_spine_path
    with profile_timer(profiler, "loading_spine_seconds"):
        spine = pd.read_csv(spine_path)
    validate_spine(spine, ckpt_config.schema)
    if num_rows not in (None, "all"):
        spine = spine.head(int(num_rows)).copy()
    spine = spine.reset_index(drop=True)

    if oracle_structured_table_path is not None:
        oracle_structured = load_oracle_structured(
            oracle_structured_table_path,
            plan,
            ckpt_config.schema,
            vocabs,
            tokenizer,
            num_rows=len(spine),
            device=device,
        )
    else:
        oracle_structured = None

    graph_history_index = None
    use_graph = graph_mode_override != "no_graph" and (graph_encoder is not None or graph_conditioning_enabled(ckpt_config.raw))
    if use_graph:
        graph_encoder = graph_encoder or build_graph_encoder(ckpt_config, vocabs, tokenizer).to(device)
        graph_encoder.eval()
        with profile_timer(profiler, "graph_history_build_seconds"):
            graph_history_index = build_temporal_history_index(spine, ckpt_config, seed=seed)
        write_temporal_graph_metadata(
            spine,
            ckpt_config,
            output_path.parent / "graph",
            source="synthetic_spine",
            seed=seed,
            real_graph_used_at_sampling=False,
        )
    else:
        graph_encoder = None

    attrs = hierarchical_sample_attributes(
        model=model,
        config=ckpt_config,
        plan=plan,
        categorical_vocabs=vocabs,
        tokenizer=tokenizer,
        spine=spine,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        text_top_k=text_top_k,
        device=device,
        seed=seed,
        structured_steps=structured_steps,
        text_steps=text_steps,
        timestep_spacing=timestep_spacing,
        inference_dtype=inference_dtype,
        graph_encoder=graph_encoder,
        graph_history_index=graph_history_index,
        graph_mode_override=graph_mode_override,
        oracle_structured=oracle_structured,
        profiler=profiler,
    )

    output = spine.loc[:, list(ckpt_config.schema.condition_columns)].copy()
    for column in ckpt_config.schema.categorical_targets:
        output[column] = attrs[column]
    if debug_write_aux_targets:
        for column in ckpt_config.schema.auxiliary_categorical_targets:
            output[column] = attrs[column]
    for column in ckpt_config.schema.text_targets:
        output[column] = attrs[column]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_timer(profiler, "csv_writing_seconds"):
        output.to_csv(output_path, index=False)

    profiler.stop_total()
    metadata_dir = ensure_dir(output_path.parent / "metadata")
    runtime_output = Path(profile_output) if profile_output else metadata_dir / "runtime_hierarchical_sampling.json"
    summary_lengths = text_lengths(output.get("summary")) if "summary" in output else []
    review_lengths = text_lengths(output.get("review_text")) if "review_text" in output else []
    runtime_summary = profiler.summary(
        rows_generated=int(len(output)),
        num_batches=int(math.ceil(len(spine) / max(batch_size, 1))),
        batch_size_requested=batch_size,
        batch_size_used=batch_size,
        auto_batch_size_enabled=False,
        summary_lengths=summary_lengths,
        review_text_lengths=review_lengths,
        device=device,
        mixed_precision_used=str(inference_dtype).lower() != "float32",
        dtype_used=inference_dtype,
        torch_compile_used=False,
        extra=attrs["_sampling_diagnostics"],
    )
    if profiler.enabled or profile_output is not None:
        profiler.write_summary(runtime_output, runtime_summary)
        profiler.write_detailed(Path(runtime_output).with_name("runtime_hierarchical_sampling_events.json"))

    metadata = {
        "experiment_name": ckpt_config.raw.get("experiment_name", Path(ckpt_config.output_dir).name),
        "model_family": "graph_conditioned_hierarchical_multimodal_diffusion",
        "checkpoint_path": str(checkpoint_path),
        "synthetic_spine_path": str(spine_path),
        "output_path": str(output_path),
        "num_rows": int(len(output)),
        "generation_plan": plan.to_dict(),
        "valid_generative_baseline": oracle_structured is None,
        "oracle_structured_conditioning": oracle_structured is not None,
        "oracle_structured_warning": "NOT A VALID GENERATIVE BASELINE" if oracle_structured is not None else None,
        "graph_mode": graph_mode_override,
        "structured_steps": int(structured_steps),
        "text_steps": int(text_steps),
        "timestep_spacing": timestep_spacing,
        "text_top_k": text_top_k,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "batch_size": batch_size,
        "inference_dtype": inference_dtype,
        "runtime_profile_path": str(runtime_output) if profiler.enabled or profile_output is not None else None,
        "uses_generated_structured_attributes_for_text": oracle_structured is None,
        "text_conditioning_tokens_maskable": False,
        "summary_review_text_jointly_modeled": len(ckpt_config.schema.text_targets) > 1,
    }
    if graph_encoder is not None:
        metadata.update(graph_metadata(ckpt_config.raw, real_graph_used_at_sampling=False))
    with (metadata_dir / "hierarchical_sample_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (metadata_dir / "timestep_schedule.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(attrs["_sampling_diagnostics"]), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {output_path}")
    return output_path


@torch.no_grad()
def hierarchical_sample_attributes(
    *,
    model: torch.nn.Module,
    config: ConditionalTABDLMConfig,
    plan: GenerationPlan,
    categorical_vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
    spine: pd.DataFrame,
    batch_size: int,
    temperature: float,
    top_p: float,
    text_top_k: int | None,
    device: str,
    seed: int,
    structured_steps: int,
    text_steps: int,
    timestep_spacing: str,
    inference_dtype: str,
    graph_encoder: TemporalStructureOnlyGraphEncoder | None,
    graph_history_index: Any | None,
    graph_mode_override: str,
    oracle_structured: torch.Tensor | None = None,
    profiler: RuntimeProfiler | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    schema = config.schema
    rng = random.Random(int(seed))
    train_timesteps = int(config.raw.get("diffusion", {}).get("timesteps", 50))
    structured_schedule = masked_denoising_schedule(train_timesteps, structured_steps, timestep_spacing)
    text_schedule = masked_denoising_schedule(train_timesteps, text_steps, timestep_spacing)
    autocast_dtype, autocast_enabled = resolve_inference_dtype(inference_dtype, device)
    result: dict[str, Any] = {column: [] for column in schema.model_categorical_targets + schema.text_targets}
    model.eval()
    if graph_encoder is not None:
        graph_encoder.eval()

    iterator = range(0, len(spine), int(batch_size))
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(spine) + int(batch_size) - 1) // int(batch_size), desc="sample_hier")
    for start in iterator:
        batch_frame = spine.iloc[start : start + int(batch_size)]
        foreign_key_ids, datetime_values = encode_conditions(batch_frame, schema, int(config.raw.get("id_encoding", {}).get("num_buckets", 262144)), device)
        graph_context = build_graph_context(
            graph_encoder,
            graph_history_index,
            row_indices=list(range(start, start + len(batch_frame))),
            device=device,
            mode=graph_mode_override,
        )

        if oracle_structured is not None:
            cat_input = oracle_structured[start : start + len(batch_frame)].clone().to(device)
        else:
            cat_input = sample_structured_stage(
                model,
                schema,
                categorical_vocabs,
                tokenizer,
                foreign_key_ids,
                datetime_values,
                graph_context,
                structured_schedule,
                train_timesteps,
                temperature,
                device,
                autocast_dtype,
                autocast_enabled,
                profiler,
            )
        with profile_timer(profiler, "length_target_preparation_seconds"):
            exact_lengths = exact_lengths_from_length_buckets(
                schema,
                categorical_vocabs,
                tokenizer,
                cat_input,
                rng,
                device,
            )
            text_input, text_attention, text_remaining = initial_length_masked_text_inputs(schema, tokenizer, exact_lengths, device, len(batch_frame))
        sample_text_stage(
            model,
            schema,
            foreign_key_ids,
            datetime_values,
            cat_input,
            text_input,
            text_attention,
            text_remaining,
            graph_context,
            text_schedule,
            train_timesteps,
            temperature,
            top_p,
            text_top_k,
            device,
            autocast_dtype,
            autocast_enabled,
            profiler,
        )
        for idx, column in enumerate(schema.model_categorical_targets):
            decoded = [decode_category_id(column, categorical_vocabs[column], value) for value in cat_input[:, idx].detach().cpu().tolist()]
            result[column].extend(decoded)
        for column in schema.text_targets:
            decoded = [tokenizer.decode(row) for row in text_input[column].detach().cpu().tolist()]
            result[column].extend(decoded)

    num_batches = int(math.ceil(len(spine) / max(int(batch_size), 1)))
    result["_sampling_diagnostics"] = {
        "structured_steps": int(len(structured_schedule)),
        "text_steps": int(len(text_schedule)),
        "structured_timestep_sequence": [int(value) for value in structured_schedule],
        "text_timestep_sequence": [int(value) for value in text_schedule],
        "num_batches": num_batches,
        "structured_forward_passes_total": int(num_batches * len(structured_schedule)),
        "text_forward_passes_total": int(num_batches * len(text_schedule)),
        "model_forward_passes_total": int(num_batches * (len(structured_schedule) + len(text_schedule))),
        "structured_conditioning_tokens_maskable": False,
        "uses_generated_structured_attributes_for_text": oracle_structured is None,
        "graph_mode": graph_mode_override,
    }
    return result


def sample_structured_stage(
    model: torch.nn.Module,
    schema: Any,
    categorical_vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
    foreign_key_ids: torch.Tensor,
    datetime_values: torch.Tensor,
    graph_context: torch.Tensor | None,
    schedule: list[int],
    train_timesteps: int,
    temperature: float,
    device: str,
    autocast_dtype: torch.dtype,
    autocast_enabled: bool,
    profiler: RuntimeProfiler | None,
) -> torch.Tensor:
    cat_input = torch.empty((foreign_key_ids.shape[0], len(schema.model_categorical_targets)), dtype=torch.long, device=device)
    for idx, column in enumerate(schema.model_categorical_targets):
        cat_input[:, idx] = categorical_vocabs[column].mask_id
    remaining = torch.ones_like(cat_input, dtype=torch.bool)
    text_input, text_attention = inactive_text_inputs(schema, tokenizer, device, foreign_key_ids.shape[0])
    with profile_timer(profiler, "structured_diffusion_seconds", device=device, cuda=str(device).startswith("cuda")):
        for schedule_idx, step in enumerate(schedule):
            next_step = schedule[schedule_idx + 1] if schedule_idx + 1 < len(schedule) else 0
            reveal_prob = reveal_probability_for_schedule(step, next_step)
            t = torch.full((foreign_key_ids.shape[0],), step / max(train_timesteps, 1), dtype=torch.float32, device=device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                logits = model(
                    foreign_key_ids=foreign_key_ids,
                    datetime_values=datetime_values,
                    categorical_input_ids=cat_input,
                    text_input_ids=text_input,
                    text_attention=text_attention,
                    diffusion_t=t,
                    graph_context=graph_context,
                )
            for idx, column in enumerate(schema.model_categorical_targets):
                mask = remaining[:, idx]
                if not bool(mask.any()):
                    continue
                if column in schema.length_bucket_targets:
                    sampled = sample_length_bucket_logits(logits["categorical"][column], column, categorical_vocabs[column], None, schema, temperature)
                else:
                    sampled = sample_categorical_logits(logits["categorical"][column], column, categorical_vocabs[column], temperature=temperature)
                reveal = mask & (torch.rand_like(mask.float()) < reveal_prob)
                if schedule_idx + 1 == len(schedule):
                    reveal = mask
                cat_input[reveal, idx] = sampled[reveal]
                remaining[reveal, idx] = False
    return cat_input


def sample_text_stage(
    model: torch.nn.Module,
    schema: Any,
    foreign_key_ids: torch.Tensor,
    datetime_values: torch.Tensor,
    cat_input: torch.Tensor,
    text_input: dict[str, torch.Tensor],
    text_attention: dict[str, torch.Tensor],
    text_remaining: dict[str, torch.Tensor],
    graph_context: torch.Tensor | None,
    schedule: list[int],
    train_timesteps: int,
    temperature: float,
    top_p: float,
    text_top_k: int | None,
    device: str,
    autocast_dtype: torch.dtype,
    autocast_enabled: bool,
    profiler: RuntimeProfiler | None,
) -> None:
    with profile_timer(profiler, "text_diffusion_seconds", device=device, cuda=str(device).startswith("cuda")):
        for schedule_idx, step in enumerate(schedule):
            next_step = schedule[schedule_idx + 1] if schedule_idx + 1 < len(schedule) else 0
            reveal_prob = reveal_probability_for_schedule(step, next_step)
            t = torch.full((foreign_key_ids.shape[0],), step / max(train_timesteps, 1), dtype=torch.float32, device=device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                logits = model(
                    foreign_key_ids=foreign_key_ids,
                    datetime_values=datetime_values,
                    categorical_input_ids=cat_input,
                    text_input_ids=text_input,
                    text_attention=text_attention,
                    diffusion_t=t,
                    graph_context=graph_context,
                )
            for column in schema.text_targets:
                remaining = text_remaining[column]
                if not bool(remaining.any()):
                    continue
                flat_logits = logits["text"][column].reshape(-1, logits["text"][column].shape[-1])
                sampled = sample_logits(flat_logits, temperature=temperature, top_p=top_p, top_k=text_top_k).view_as(text_input[column])
                reveal = remaining & (torch.rand_like(remaining.float()) < reveal_prob)
                if schedule_idx + 1 == len(schedule):
                    reveal = remaining
                text_input[column][reveal] = sampled[reveal]
                text_remaining[column][reveal] = False


def inactive_text_inputs(schema: Any, tokenizer: SimpleTextTokenizer, device: str, batch_size: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    text_input: dict[str, torch.Tensor] = {}
    text_attention: dict[str, torch.Tensor] = {}
    for column in schema.text_targets:
        length = int(schema.text_max_lengths[column])
        text_input[column] = torch.full((batch_size, length), tokenizer.pad_id, dtype=torch.long, device=device)
        if length > 0:
            text_input[column][:, 0] = tokenizer.bos_id
        text_attention[column] = torch.zeros((batch_size, length), dtype=torch.long, device=device)
    return text_input, text_attention


def initial_length_masked_text_inputs(
    schema: Any,
    tokenizer: SimpleTextTokenizer,
    exact_lengths: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    text_input: dict[str, torch.Tensor] = {}
    text_attention: dict[str, torch.Tensor] = {}
    text_remaining: dict[str, torch.Tensor] = {}
    for column in schema.text_targets:
        max_len = int(schema.text_max_lengths[column])
        target = exact_lengths[column].to(device=device, dtype=torch.long).clamp(min=0, max=tokenizer.max_content_tokens(max_len))
        pos = torch.arange(max_len, dtype=torch.long, device=device).view(1, -1)
        content_mask = (pos >= 1) & (pos <= target.view(-1, 1))
        eos_pos = (target + 1).clamp(max=max_len - 1)
        values = torch.full((batch_size, max_len), tokenizer.pad_id, dtype=torch.long, device=device)
        values[:, 0] = tokenizer.bos_id
        values[content_mask] = tokenizer.mask_id
        values.scatter_(1, eos_pos.view(-1, 1), tokenizer.eos_id)
        attention = ((pos == 0) | content_mask | (pos == eos_pos.view(-1, 1))).long()
        text_input[column] = values
        text_attention[column] = attention
        text_remaining[column] = content_mask.clone()
    return text_input, text_attention, text_remaining


def exact_lengths_from_length_buckets(
    schema: Any,
    categorical_vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
    cat_input: torch.Tensor,
    rng: random.Random,
    device: str,
) -> dict[str, torch.Tensor]:
    lengths: dict[str, torch.Tensor] = {}
    for text_column in schema.text_targets:
        bucket_column = length_field_for_text(schema, text_column)
        if bucket_column is None:
            raise ValueError(f"Text field {text_column!r} has no length field")
        bucket_idx = schema.model_categorical_targets.index(bucket_column)
        vocab = categorical_vocabs[bucket_column]
        bucket_names = [vocab.decode(idx) for idx in cat_input[:, bucket_idx].detach().cpu().tolist()]
        buckets = schema.buckets_for_length_bucket(bucket_column)
        max_content = tokenizer.max_content_tokens(int(schema.text_max_lengths[text_column]))
        values = [sample_length_from_bucket(name, buckets, max_content, rng) for name in bucket_names]
        lengths[text_column] = torch.tensor(values, dtype=torch.long, device=device)
    return lengths


def build_graph_context(
    graph_encoder: TemporalStructureOnlyGraphEncoder | None,
    graph_history_index: Any | None,
    *,
    row_indices: list[int],
    device: str,
    mode: str,
) -> torch.Tensor | None:
    if graph_encoder is None or graph_history_index is None or mode == "no_graph":
        return None
    graph_batch = graph_history_index.build_batch(row_indices, device=device, deterministic=True)
    if mode == "zero":
        context = graph_encoder(graph_batch)
        return torch.zeros_like(context)
    if mode == "shuffled":
        context = graph_encoder(graph_batch)
        if context.shape[0] <= 1:
            return context
        return context[torch.randperm(context.shape[0], device=context.device)]
    return graph_encoder(graph_batch)


def load_oracle_structured(
    path: str | Path,
    plan: GenerationPlan,
    schema: Any,
    categorical_vocabs: dict[str, CategoryVocab],
    tokenizer: SimpleTextTokenizer,
    *,
    num_rows: int,
    device: str,
) -> torch.Tensor:
    columns = list(schema.model_categorical_targets)
    required = set(schema.categorical_targets)
    for column in schema.length_bucket_targets:
        required.add(schema.text_column_for_length_bucket(column))
    frame = pd.read_csv(path, usecols=[column for column in required if column], nrows=num_rows)
    if len(frame) < int(num_rows):
        raise ValueError(f"Oracle structured source has {len(frame)} rows, requested {num_rows}")
    values = torch.empty((num_rows, len(columns)), dtype=torch.long, device=device)
    for idx, column in enumerate(columns):
        if column in schema.length_bucket_targets:
            text_column = schema.text_column_for_length_bucket(column)
            bucket_values = oracle_length_bucket_values(frame[text_column], schema, column, tokenizer)
            values[:, idx] = torch.tensor([categorical_vocabs[column].encode(value) for value in bucket_values], dtype=torch.long, device=device)
            continue
        if column not in frame:
            raise ValueError(f"Oracle structured source is missing {column!r}")
        vocab = categorical_vocabs[column]
        values[:, idx] = torch.tensor([vocab.encode(value) for value in frame[column].tolist()], dtype=torch.long, device=device)
    return values


def oracle_length_bucket_values(series: pd.Series, schema: Any, bucket_column: str, tokenizer: SimpleTextTokenizer) -> list[str]:
    text_column = schema.text_column_for_length_bucket(bucket_column)
    buckets = schema.buckets_for_length_bucket(bucket_column)
    max_tokens = int(schema.text_max_lengths[text_column])
    values = []
    for text in series.fillna("").astype(str).tolist():
        ids, _ = tokenizer.encode(text, max_length=max_tokens)
        length = tokenizer.content_length(ids)
        for name, (low, high) in buckets.items():
            if int(low) <= int(length) <= int(high):
                values.append(str(name))
                break
        else:
            values.append(str(list(buckets)[-1]))
    return values


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None
