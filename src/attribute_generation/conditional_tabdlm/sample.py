"""Sampling utilities for Conditional TABDLM."""

from __future__ import annotations

import math
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch

from .attribute_corruption import GraphAttributeStore, build_attribute_graph_batch
from .constrained import (
    decode_category_id,
    mask_invalid_category_logits,
    valid_category_values,
    validate_output_categoricals,
)
from .graph_dataset import build_temporal_history_index, write_temporal_graph_metadata
from .graph_encoder import TemporalAttributeDenoisingGraphEncoder, TemporalStructureOnlyGraphEncoder
from .graph_schema import graph_conditioning_enabled, graph_metadata, graph_mode
from .model import ConditionalTABDLM
from .runtime_profiler import RuntimeProfiler
from .schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema
from .tokenization import CategoryVocab, SimpleTextTokenizer, sample_length_from_bucket, stable_hash_bucket
from .train import build_graph_encoder, build_model, resolve_device
from .utils import ensure_dir, jsonable, set_seed


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def sample_from_config(
    config: ConditionalTABDLMConfig,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    num_rows: int | str | None = None,
    batch_size: int | None = None,
    sample_batch_size: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    device: str | None = None,
    seed: int | None = None,
    synthetic_spine_path: str | Path | None = None,
    debug_write_aux_targets: bool = False,
    disable_length_calibration: bool = False,
    repair_invalid_categoricals: bool = False,
    sampling_steps: int | None = None,
    timestep_spacing: str | None = None,
    inference_dtype: str | None = None,
    compile_model: bool | None = None,
    text_top_k: int | None = None,
    profile: bool | None = None,
    profile_output: str | Path | None = None,
) -> Path:
    sampling = config.raw.get("sampling", {})
    diffusion = config.raw.get("diffusion", {})
    profiler = RuntimeProfiler(enabled=bool((profile if profile is not None else sampling.get("profile", False)) or profile_output is not None))
    profiler.start_total()
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else config.checkpoint_dir / "best.pt"
    output_path = Path(output_path) if output_path else config.output_dir / "synthetic_review_attrs.csv"
    num_rows = num_rows if num_rows is not None else sampling.get("num_rows", 100000)
    batch_size = int(sample_batch_size or batch_size or sampling.get("sample_batch_size", sampling.get("batch_size", 128)))
    temperature = float(temperature if temperature is not None else sampling.get("temperature", 1.0))
    top_p = float(top_p if top_p is not None else sampling.get("top_p", 0.95))
    seed = int(seed if seed is not None else sampling.get("seed", 42))
    device = resolve_device(device or str(sampling.get("device", "auto")))
    inference_dtype = str(inference_dtype or sampling.get("inference_dtype", "float32"))
    sampling_steps = int(sampling_steps or diffusion.get("sampling_steps", diffusion.get("timesteps", 50)))
    timestep_spacing = str(timestep_spacing or diffusion.get("timestep_spacing", sampling.get("timestep_spacing", "uniform")))
    compile_model = bool(compile_model if compile_model is not None else sampling.get("compile_model", False))
    text_top_k = parse_optional_int(text_top_k if text_top_k is not None else sampling.get("text_top_k"))

    set_seed(seed)
    with profile_timer(profiler, "loading_model_seconds", device=device):
        model, ckpt_config, vocabs, tokenizer, graph_encoder = load_model_checkpoint(
            checkpoint_path,
            device,
            include_graph=True,
        )
    model, compile_used = maybe_compile_model(model, compile_model)
    spine_path = Path(synthetic_spine_path) if synthetic_spine_path else config.synthetic_spine_path
    with profile_timer(profiler, "loading_spine_seconds"):
        spine = pd.read_csv(spine_path)
    validate_spine(spine, ckpt_config.schema)
    if num_rows not in (None, "all"):
        spine = spine.head(int(num_rows)).copy()
    spine = spine.reset_index(drop=True)
    graph_history_index = None
    if graph_encoder is not None or graph_conditioning_enabled(ckpt_config.raw):
        graph_encoder = graph_encoder or build_graph_encoder(ckpt_config, vocabs, tokenizer).to(device)
        graph_encoder.eval()
        if graph_mode(ckpt_config.raw) == "temporal_attribute_denoising":
            spine = sort_spine_chronologically(spine, ckpt_config.schema)
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
    attrs = sample_attributes(
        model,
        ckpt_config.schema,
        vocabs,
        tokenizer,
        spine,
        config=ckpt_config,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        device=device,
        seed=seed,
        disable_length_calibration=disable_length_calibration,
        graph_encoder=graph_encoder,
        graph_history_index=graph_history_index,
        sampling_steps=sampling_steps,
        timestep_spacing=timestep_spacing,
        inference_dtype=inference_dtype,
        text_top_k=text_top_k,
        profiler=profiler,
    )
    with profile_timer(profiler, "postprocessing_seconds"):
        output = spine.loc[:, list(ckpt_config.schema.condition_columns)].copy()
        for column in ckpt_config.schema.categorical_targets:
            output[column] = attrs[column]
        if debug_write_aux_targets:
            for column in ckpt_config.schema.auxiliary_categorical_targets:
                output[column] = attrs[column]
        for column in ckpt_config.schema.text_targets:
            output[column] = attrs[column]
        output = validate_output_categoricals(
            output,
            {column: vocabs[column] for column in ckpt_config.schema.categorical_targets if column in vocabs},
            repair_invalid=repair_invalid_categoricals,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_timer(profiler, "csv_writing_seconds"):
        output.to_csv(output_path, index=False)
    runtime_output = Path(profile_output) if profile_output else output_path.parent / "metadata" / "runtime_diffusion_sampling.json"
    summary_lengths = text_lengths(output.get("summary")) if "summary" in output else []
    review_lengths = text_lengths(output.get("review_text")) if "review_text" in output else []
    metadata = {
        "experiment_name": ckpt_config.raw.get("experiment_name", Path(ckpt_config.output_dir).name),
        "checkpoint_path": str(checkpoint_path),
        "synthetic_spine_path": str(spine_path),
        "output_path": str(output_path),
        "num_rows": int(len(output)),
        "batch_size": batch_size,
        "sample_batch_size": batch_size,
        "temperature": temperature,
        "top_p": top_p,
        "text_top_k": text_top_k,
        "seed": seed,
        "diffusion_train_timesteps": int(ckpt_config.raw.get("diffusion", {}).get("timesteps", 50)),
        "diffusion_sampling_steps": int(sampling_steps),
        "diffusion_timestep_spacing": str(timestep_spacing),
        "inference_dtype": inference_dtype,
        "torch_compile_used": compile_used,
        "runtime_profile_path": str(runtime_output) if profiler.enabled or profile_output is not None else None,
        "condition_columns": list(ckpt_config.schema.condition_columns),
        "target_columns": {
            "categorical": list(ckpt_config.schema.categorical_targets),
            "numerical": list(ckpt_config.schema.numerical_targets),
            "text": list(ckpt_config.schema.text_targets),
        },
        "auxiliary_categorical_targets": list(ckpt_config.schema.auxiliary_categorical_targets),
        "joint_generation": True,
        "review_text_generated_jointly": "review_text" in ckpt_config.schema.text_targets,
        "review_text_separate_stage": False,
        "uses_summary_length_bucket": bool(ckpt_config.schema.summary_length_enabled),
        "uses_review_text_length_bucket": "review_text_length_bucket" in ckpt_config.schema.auxiliary_categorical_targets,
        "force_eos_after_sampled_length": ckpt_config.schema.force_eos_after_sampled_length,
        "force_pad_after_eos": bool(ckpt_config.schema.force_pad_after_eos),
        "length_calibration_disabled": bool(disable_length_calibration),
        "repair_invalid_categoricals": bool(repair_invalid_categoricals),
        "summary_max_tokens": int(ckpt_config.schema.text_max_lengths.get("summary", 0)) if "summary" in ckpt_config.schema.text_targets else None,
        "review_text_max_tokens": int(ckpt_config.schema.text_max_lengths.get("review_text", 0)) if "review_text" in ckpt_config.schema.text_targets else None,
        "review_text_max_tokens_strategy": ckpt_config.raw.get("review_text", {}).get("max_tokens_strategy"),
        "review_text_length_cap_source": ckpt_config.raw.get("review_text", {}).get("length_cap_source"),
        "review_text_truncation_rate_train": ckpt_config.raw.get("review_text", {}).get("truncation_rate_train"),
        "review_text_coverage_rate_train": ckpt_config.raw.get("review_text", {}).get("coverage_rate_train"),
        "valid_categorical_values": {
            column: valid_category_values(column, vocabs[column])
            for column in ckpt_config.schema.categorical_targets
            if column in vocabs
        },
    }
    if "rating" in vocabs:
        metadata["valid_rating_values"] = valid_category_values("rating", vocabs["rating"])
    if graph_encoder is not None:
        metadata.update(graph_metadata(ckpt_config.raw, real_graph_used_at_sampling=False))
        metadata.update(
            {
                "synthetic_graph_history_source": "synthetic_spine",
                "graph_uses_clean_target_attributes": False,
                "graph_uses_clean_future_attributes": False,
                "graph_history_rows": int(len(spine)),
                "sampling_chronological": graph_mode(ckpt_config.raw) == "temporal_attribute_denoising",
                "history_source_sampling": "generated_past_synthetic_attributes"
                if graph_mode(ckpt_config.raw) == "temporal_attribute_denoising"
                else "synthetic_spine_structure_only",
                "future_synthetic_events_used_at_sampling": False,
                "current_batch_events_used_as_history": False,
            }
        )
        if isinstance(graph_encoder, TemporalAttributeDenoisingGraphEncoder):
            gate_value = graph_encoder.summary_attr_gate_value()
            if gate_value is not None:
                metadata["summary_attr_gate"] = float(gate_value)
    with (output_path.parent / "sample_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with profile_timer(profiler, "debug_example_seconds"):
        write_debug_outputs(attrs, output_path.parent / "debug")
    profiler.stop_total()
    if profiler.enabled or profile_output is not None:
        runtime_summary = profiler.summary(
            rows_generated=int(len(output)),
            num_batches=int(math.ceil(len(spine) / max(batch_size, 1))),
            batch_size_requested=int(batch_size),
            batch_size_used=int(batch_size),
            auto_batch_size_enabled=False,
            summary_lengths=summary_lengths,
            review_text_lengths=review_lengths,
            device=device,
            mixed_precision_used=inference_dtype.lower() != "float32",
            dtype_used=inference_dtype,
            torch_compile_used=compile_used,
            extra={
                "diffusion_train_timesteps": int(ckpt_config.raw.get("diffusion", {}).get("timesteps", 50)),
                "diffusion_sampling_steps": int(sampling_steps),
                "diffusion_timestep_spacing": str(timestep_spacing),
                "text_top_k": text_top_k,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
        profiler.write_summary(runtime_output, runtime_summary)
        profiler.write_detailed(Path(runtime_output).with_name("runtime_diffusion_sampling_events.json"))
    print(f"Wrote {output_path}")
    return output_path


@contextmanager
def profile_timer(
    profiler: RuntimeProfiler | None,
    name: str,
    *,
    device: str | None = None,
    cuda: bool = False,
) -> Iterator[None]:
    if profiler is None or not profiler.enabled:
        yield
        return
    if cuda:
        synchronize_cuda(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            synchronize_cuda(device)
        profiler.add_time(name, float(time.perf_counter() - start))


def synchronize_cuda(device: str | None = None) -> None:
    if device is None or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(torch.device(device))
    except Exception:
        torch.cuda.synchronize()


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def maybe_compile_model(model: torch.nn.Module, enabled: bool) -> tuple[torch.nn.Module, bool]:
    if not enabled:
        return model, False
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        print("WARNING: torch.compile is unavailable; continuing without compilation.", flush=True)
        return model, False
    try:
        return compile_fn(model, mode="reduce-overhead"), True
    except Exception as exc:
        print(f"WARNING: torch.compile failed; continuing without compilation. Reason: {exc}", flush=True)
        return model, False


def resolve_inference_dtype(dtype_name: str, device: str) -> tuple[torch.dtype, bool]:
    name = str(dtype_name or "float32").lower()
    if name in {"float32", "fp32", "none"} or not str(device).startswith("cuda"):
        return torch.float32, False
    if name in {"bfloat16", "bf16"}:
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16, True
        print("WARNING: bfloat16 requested but unsupported; falling back to float32.", flush=True)
        return torch.float32, False
    if name in {"float16", "fp16", "half"}:
        return torch.float16, True
    raise ValueError("inference_dtype must be one of float32, float16, or bfloat16")


def masked_denoising_schedule(total_timesteps: int, sampling_steps: int, spacing: str = "uniform") -> list[int]:
    total = max(1, int(total_timesteps))
    steps = max(1, min(int(sampling_steps), total))
    if steps >= total:
        return list(range(total, 0, -1))
    if steps == 1:
        return [total]
    spacing = str(spacing or "uniform").lower()
    if spacing == "quadratic":
        raw = np.square(np.linspace(np.sqrt(total), 1.0, num=steps))
    elif spacing == "leading":
        raw = np.linspace(total, 1.0, num=steps)
        raw = total - np.square(np.linspace(0.0, np.sqrt(total - 1), num=steps))
    elif spacing == "trailing":
        raw = np.square(np.linspace(np.sqrt(total), 1.0, num=steps))
    elif spacing == "uniform":
        raw = np.linspace(total, 1.0, num=steps)
    else:
        raise ValueError("timestep_spacing must be one of uniform, quadratic, leading, or trailing")
    schedule = sorted({int(round(value)) for value in raw}, reverse=True)
    schedule = [min(total, max(1, value)) for value in schedule]
    if total not in schedule:
        schedule.insert(0, total)
    if 1 not in schedule:
        schedule[-1] = 1
    return sorted(set(schedule), reverse=True)


def reveal_probability_for_schedule(current_step: int, next_step: int) -> float:
    current = max(1, int(current_step))
    next_value = max(0, min(int(next_step), current - 1))
    if next_value <= 0:
        return 1.0
    return float(np.clip(1.0 - (next_value / float(current)), 0.0, 1.0))


def text_lengths(series: pd.Series | None) -> list[int]:
    if series is None:
        return []
    return [len(str(value).split()) for value in series.fillna("").astype(str).tolist()]


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    include_graph: bool = False,
) -> (
    tuple[ConditionalTABDLM, ConditionalTABDLMConfig, dict[str, CategoryVocab], SimpleTextTokenizer]
    | tuple[ConditionalTABDLM, ConditionalTABDLMConfig, dict[str, CategoryVocab], SimpleTextTokenizer, TemporalStructureOnlyGraphEncoder | None]
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_config = checkpoint["raw_config"]
    if checkpoint.get("summary_length_calibration") is not None:
        raw_config = dict(raw_config)
        raw_config["_summary_length_calibration"] = checkpoint["summary_length_calibration"]
    schema = ConditionalTABDLMSchema.from_config_dict(raw_config)
    config = ConditionalTABDLMConfig(raw=raw_config, schema=schema, config_path=None)
    vocabs = {
        column: CategoryVocab.from_dict(data)
        for column, data in checkpoint["categorical_vocabs"].items()
    }
    tokenizer = SimpleTextTokenizer.from_dict(checkpoint["tokenizer_metadata"])
    model = build_model(config, vocabs, tokenizer).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    graph_encoder = None
    if include_graph and graph_conditioning_enabled(raw_config):
        graph_encoder = build_graph_encoder(config, vocabs, tokenizer).to(device)
        state = checkpoint.get("graph_encoder_state_dict")
        if state is not None:
            graph_encoder.load_state_dict(state)
        graph_encoder.eval()
    if include_graph:
        return model, config, vocabs, tokenizer, graph_encoder
    return model, config, vocabs, tokenizer


@torch.no_grad()
def sample_attributes(
    model: ConditionalTABDLM,
    schema: ConditionalTABDLMSchema,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    spine: pd.DataFrame,
    config: ConditionalTABDLMConfig,
    batch_size: int,
    temperature: float,
    top_p: float,
    device: str,
    seed: int = 42,
    disable_length_calibration: bool = False,
    graph_encoder: TemporalStructureOnlyGraphEncoder | None = None,
    graph_history_index: Any | None = None,
    sampling_steps: int | None = None,
    timestep_spacing: str = "uniform",
    inference_dtype: str = "float32",
    text_top_k: int | None = None,
    profiler: RuntimeProfiler | None = None,
) -> dict[str, list[str]]:
    id_cfg = config.raw.get("id_encoding", {})
    diffusion = config.raw.get("diffusion", {})
    sampling_cfg = config.raw.get("sampling", {})
    num_hash_buckets = int(id_cfg.get("num_buckets", 262144))
    timesteps = int(diffusion.get("timesteps", diffusion.get("train_timesteps", 50)))
    schedule = masked_denoising_schedule(
        total_timesteps=timesteps,
        sampling_steps=int(sampling_steps or timesteps),
        spacing=timestep_spacing,
    )
    autocast_dtype, autocast_enabled = resolve_inference_dtype(inference_dtype, device)
    repetition_penalties = {
        column: float(sampling_cfg.get(f"{column}_content_repetition_penalty", sampling_cfg.get("summary_content_repetition_penalty", 1.0)))
        for column in schema.text_targets
    }
    min_content_tokens = {
        column: int(sampling_cfg.get(f"min_{column}_content_tokens", sampling_cfg.get("min_summary_content_tokens", 0)))
        for column in schema.text_targets
    }
    calibration = None if disable_length_calibration else config.raw.get("_summary_length_calibration")
    rng = random.Random(int(seed))
    result: dict[str, list[str] | list[dict[str, Any]]] = {
        column: [] for column in schema.model_categorical_targets + schema.text_targets
    }
    debug_examples: list[dict[str, Any]] = []
    attr_sampling = isinstance(graph_encoder, TemporalAttributeDenoisingGraphEncoder)
    generated_attr_store = (
        GraphAttributeStore.empty_generated(len(spine), schema, categorical_vocabs, text_tokenizer)
        if attr_sampling
        else None
    )
    iterator = range(0, len(spine), int(batch_size))
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(spine) + int(batch_size) - 1) // int(batch_size), desc="sample")
    for start in iterator:
        batch_frame = spine.iloc[start : start + int(batch_size)]
        with profile_timer(profiler, "condition_encoding_seconds", device=device):
            foreign_key_ids, datetime_values = encode_conditions(batch_frame, schema, num_hash_buckets, device)
        graph_context = None
        if graph_encoder is not None and not attr_sampling:
            if graph_history_index is None:
                raise ValueError("graph_history_index is required when graph_encoder is enabled")
            row_indices = list(range(start, start + len(batch_frame)))
            with profile_timer(profiler, "graph_context_total_seconds", device=device, cuda=device.startswith("cuda")):
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    graph_context = graph_encoder(
                        graph_history_index.build_batch(row_indices, device=device, deterministic=True)
                    )
        with profile_timer(profiler, "initial_noise_seconds", device=device):
            cat_input = torch.empty((len(batch_frame), len(schema.model_categorical_targets)), dtype=torch.long, device=device)
            for idx, column in enumerate(schema.model_categorical_targets):
                cat_input[:, idx] = categorical_vocabs[column].mask_id
            cat_remaining = torch.ones_like(cat_input, dtype=torch.bool)

            text_input: dict[str, torch.Tensor] = {}
            text_attention: dict[str, torch.Tensor] = {}
            text_remaining: dict[str, torch.Tensor] = {}
            for column in schema.text_targets:
                length = int(schema.text_max_lengths[column])
                text_input[column] = torch.full((len(batch_frame), length), text_tokenizer.mask_id, dtype=torch.long, device=device)
                text_input[column][:, 0] = text_tokenizer.bos_id
                text_attention[column] = torch.ones((len(batch_frame), length), dtype=torch.long, device=device)
                text_remaining[column] = torch.ones((len(batch_frame), length), dtype=torch.bool, device=device)
                text_remaining[column][:, 0] = False

        logits: dict[str, Any] | None = None
        with profile_timer(profiler, "denoising_loop_seconds", device=device, cuda=device.startswith("cuda")):
            for schedule_idx, step in enumerate(schedule):
                next_step = schedule[schedule_idx + 1] if schedule_idx + 1 < len(schedule) else 0
                reveal_prob = reveal_probability_for_schedule(step, next_step)
                t = torch.full((len(batch_frame),), step / max(timesteps, 1), dtype=torch.float32, device=device)
                if attr_sampling:
                    graph_context = sampling_graph_context(
                        graph_encoder,
                        graph_history_index,
                        generated_attr_store,
                        config,
                        row_indices=list(range(start, start + len(batch_frame))),
                        cat_input=cat_input,
                        text_input=text_input,
                        device=device,
                    )
                with profile_timer(profiler, "denoising_step_seconds", device=device, cuda=device.startswith("cuda")):
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                        enabled=autocast_enabled,
                    ):
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
                    remaining = cat_remaining[:, idx]
                    if not bool(remaining.any()):
                        continue
                    if column in schema.length_bucket_targets:
                        sampled = sample_length_bucket_logits(
                            logits["categorical"][column],
                            column,
                            categorical_vocabs[column],
                            length_calibration_for_column(calibration, column),
                            schema,
                            temperature=temperature,
                        )
                    else:
                        sampled = sample_categorical_logits(
                            logits["categorical"][column],
                            column,
                            categorical_vocabs[column],
                            temperature=temperature,
                        )
                    reveal = remaining & (torch.rand_like(remaining.float()) < reveal_prob)
                    if schedule_idx + 1 == len(schedule):
                        reveal = remaining
                    cat_input[reveal, idx] = sampled[reveal]
                    cat_remaining[reveal, idx] = False
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

        final_t = torch.zeros((len(batch_frame),), dtype=torch.float32, device=device)
        if attr_sampling:
            graph_context = sampling_graph_context(
                graph_encoder,
                graph_history_index,
                generated_attr_store,
                config,
                row_indices=list(range(start, start + len(batch_frame))),
                cat_input=cat_input,
                text_input=text_input,
                device=device,
            )
        with profile_timer(profiler, "final_forward_seconds", device=device, cuda=device.startswith("cuda")):
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                logits = model(
                    foreign_key_ids=foreign_key_ids,
                    datetime_values=datetime_values,
                    categorical_input_ids=cat_input,
                    text_input_ids=text_input,
                    text_attention=text_attention,
                    diffusion_t=final_t,
                    graph_context=graph_context,
                )
        with profile_timer(profiler, "length_enforcement_seconds", device=device, cuda=device.startswith("cuda")):
            length_debug_rows = enforce_length_constraints(
                schema=schema,
                categorical_vocabs=categorical_vocabs,
                text_tokenizer=text_tokenizer,
                cat_input=cat_input,
                text_input=text_input,
                text_logits=logits["text"],
                rng=rng,
                temperature=temperature,
                top_p=top_p,
                repetition_penalties=repetition_penalties,
                min_content_tokens=min_content_tokens,
            )

        decoded_cats: dict[str, list[str]] = {}
        with profile_timer(profiler, "categorical_decoding_seconds"):
            for idx, column in enumerate(schema.model_categorical_targets):
                decoded = [
                    decode_category_id(column, categorical_vocabs[column], value)
                    for value in cat_input[:, idx].detach().cpu().tolist()
                ]
                decoded_cats[column] = decoded
                result[column].extend(decoded)  # type: ignore[arg-type]
        with profile_timer(profiler, "text_decoding_seconds"):
            text_rows_by_column = {
                column: text_input[column].detach().cpu().tolist()
                for column in schema.text_targets
            }
            decoded_text_by_column = {
                column: [text_tokenizer.decode(row) for row in rows]
                for column, rows in text_rows_by_column.items()
            }
            for column, decoded in decoded_text_by_column.items():
                result[column].extend(decoded)  # type: ignore[arg-type]
        with profile_timer(profiler, "debug_example_seconds"):
            for column in schema.text_targets:
                decoded = decoded_text_by_column[column]
                rows = text_rows_by_column[column]
                for local_idx, decoded_text in enumerate(decoded):
                    if len(debug_examples) >= 200:
                        break
                    ids = rows[local_idx]
                    raw_tokens = [text_tokenizer.inv_vocab.get(int(idx), text_tokenizer.unk_token) for idx in ids]
                    eos_position = next((idx for idx, token_id in enumerate(ids) if int(token_id) == text_tokenizer.eos_id), None)
                    length_bucket_col = length_bucket_column_for_text(schema, column)
                    example = {
                        "text_column": column,
                        "customer_id": batch_frame.iloc[local_idx].get(schema.foreign_key_columns[0]),
                        "product_id": batch_frame.iloc[local_idx].get(schema.foreign_key_columns[1]) if len(schema.foreign_key_columns) > 1 else None,
                        "review_time": str(batch_frame.iloc[local_idx].get(schema.datetime_columns[0])),
                        "rating": decoded_cats.get("rating", [None] * len(batch_frame))[local_idx],
                        "verified": decoded_cats.get("verified", [None] * len(batch_frame))[local_idx],
                        f"{column}_length_bucket": decoded_cats.get(length_bucket_col, [None] * len(batch_frame))[local_idx] if length_bucket_col else None,
                        "summary_length_bucket": decoded_cats.get("summary_length_bucket", [None] * len(batch_frame))[local_idx],
                        "review_text_length_bucket": decoded_cats.get("review_text_length_bucket", [None] * len(batch_frame))[local_idx],
                        f"decoded_{column}": decoded_text,
                        "decoded_summary": decoded_text if column == "summary" else None,
                        "raw_tokens": raw_tokens,
                        "raw_summary_tokens": raw_tokens if column == "summary" else None,
                        "raw_review_text_tokens": raw_tokens if column == "review_text" else None,
                        "eos_position": eos_position,
                        "content_length": text_tokenizer.content_length(ids),
                        **length_debug_rows[local_idx].get(column, {}),
                    }
                    debug_examples.append(example)
        if attr_sampling and generated_attr_store is not None:
            generated_attr_store.update_rows(
                list(range(start, start + len(batch_frame))),
                cat_input,
                text_input,
            )
    result["_debug_examples"] = debug_examples
    result["_length_calibration"] = calibration or {}
    return result


def enforce_summary_length_constraints(
    schema: ConditionalTABDLMSchema,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    cat_input: torch.Tensor,
    text_input: dict[str, torch.Tensor],
    text_logits: dict[str, torch.Tensor],
    rng: random.Random,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
    min_content_tokens: int = 0,
) -> list[dict[str, Any]]:
    rows = enforce_length_constraints(
        schema=schema,
        categorical_vocabs=categorical_vocabs,
        text_tokenizer=text_tokenizer,
        cat_input=cat_input,
        text_input=text_input,
        text_logits=text_logits,
        rng=rng,
        temperature=temperature,
        top_p=top_p,
        repetition_penalties={column: repetition_penalty for column in schema.text_targets},
        min_content_tokens={column: min_content_tokens for column in schema.text_targets},
    )
    if not schema.text_targets:
        return []
    first = schema.text_targets[0]
    return [row.get(first, {}) for row in rows]


def enforce_length_constraints(
    schema: ConditionalTABDLMSchema,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    cat_input: torch.Tensor,
    text_input: dict[str, torch.Tensor],
    text_logits: dict[str, torch.Tensor],
    rng: random.Random,
    temperature: float,
    top_p: float,
    repetition_penalties: dict[str, float] | None = None,
    min_content_tokens: dict[str, int] | None = None,
) -> list[dict[str, dict[str, Any]]]:
    if not schema.text_targets:
        return []
    repetition_penalties = repetition_penalties or {}
    min_content_tokens = min_content_tokens or {}
    debug_by_row: list[dict[str, dict[str, Any]]] = [{} for _ in range(cat_input.shape[0])]

    for column in schema.text_targets:
        sequences = text_input[column]
        logits = text_logits[column]
        max_len = sequences.shape[1]
        max_content = text_tokenizer.max_content_tokens(max_len)
        bucket_column = length_bucket_column_for_text(schema, column)
        length_targets: list[int] | None = None
        length_buckets: list[str] | None = None
        buckets: dict[str, tuple[int, int]] = {}
        if (
            schema.use_length_bucket_in_sampling
            and bucket_column is not None
            and bucket_column in schema.model_categorical_targets
            and bucket_column in categorical_vocabs
        ):
            length_idx = schema.model_categorical_targets.index(bucket_column)
            length_vocab = categorical_vocabs[bucket_column]
            bucket_names = [length_vocab.decode(idx) for idx in cat_input[:, length_idx].detach().cpu().tolist()]
            length_buckets = bucket_names
            buckets = schema.buckets_for_length_bucket(bucket_column)
            length_targets = [
                sample_length_from_bucket(name, buckets, max_content, rng)
                for name in bucket_names
            ]
        for row_idx in range(sequences.shape[0]):
            target_len = length_targets[row_idx] if length_targets is not None else None
            bucket_name = length_buckets[row_idx] if length_buckets else None
            bucket_bounds = buckets.get(str(bucket_name), (None, None)) if bucket_name is not None else (None, None)
            sequences[row_idx, 0] = text_tokenizer.bos_id
            if target_len is not None and schema.force_eos_after_sampled_length:
                target_len = int(max(0, min(target_len, max_content)))
                mode = str(schema.force_eos_after_sampled_length).lower()
                if mode == "soft":
                    low, high = buckets.get(str(bucket_name), (target_len, target_len))
                    enforce_soft_length(
                        sequences[row_idx],
                        logits[row_idx],
                        text_tokenizer,
                        low=max(int(low), int(min_content_tokens.get(column, 0))),
                        high=min(int(high), max_content),
                        rng=rng,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=float(repetition_penalties.get(column, 1.0)),
                    )
                else:
                    eos_pos = min(target_len + 1, max_len - 1)
                    for pos in range(1, eos_pos):
                        token_id = int(sequences[row_idx, pos].item())
                        if token_id in text_tokenizer.special_ids:
                            sequences[row_idx, pos] = sample_content_token(
                                logits[row_idx, pos],
                                text_tokenizer,
                                temperature=temperature,
                                top_p=top_p,
                                previous_ids=sequences[row_idx, 1:pos].detach().cpu().tolist(),
                                repetition_penalty=float(repetition_penalties.get(column, 1.0)),
                            )
                    sequences[row_idx, eos_pos] = text_tokenizer.eos_id
                    if schema.force_pad_after_eos and eos_pos + 1 < max_len:
                        sequences[row_idx, eos_pos + 1 :] = text_tokenizer.pad_id
            elif schema.force_pad_after_eos:
                ids = sequences[row_idx].detach().cpu().tolist()
                eos_pos = next((idx for idx, token_id in enumerate(ids) if int(token_id) == text_tokenizer.eos_id), None)
                if eos_pos is not None and eos_pos + 1 < max_len:
                    sequences[row_idx, eos_pos + 1 :] = text_tokenizer.pad_id
            decoded_length = text_tokenizer.content_length(sequences[row_idx].detach().cpu().tolist())
            low, high = bucket_bounds
            target_bucket_respected = None
            if low is not None and high is not None:
                target_bucket_respected = int(int(low) <= int(decoded_length) <= int(high))
            debug_by_row[row_idx][column] = {
                "target_content_length": target_len,
                "target_bucket_low": low,
                "target_bucket_high": high,
                "target_bucket_respected": target_bucket_respected,
                "decoded_length_bucket": decoded_length_bucket(decoded_length, schema, bucket_column),
            }
    return debug_by_row


def sample_length_bucket_logits(
    logits: torch.Tensor,
    column: str,
    vocab: CategoryVocab,
    calibration: dict[str, Any] | None,
    schema: ConditionalTABDLMSchema,
    temperature: float,
) -> torch.Tensor:
    logits = mask_invalid_category_logits(logits, column, vocab)
    probs = calibrated_length_probs(logits, vocab, calibration, schema, column=column)
    if temperature != 1.0:
        logits = torch.log(probs.clamp_min(1e-12)) / max(float(temperature), 1e-6)
        probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def sample_categorical_logits(
    logits: torch.Tensor,
    column: str,
    vocab: CategoryVocab,
    temperature: float = 1.0,
) -> torch.Tensor:
    constrained = mask_invalid_category_logits(logits, column, vocab)
    return sample_logits(constrained, temperature=temperature, top_p=1.0)


def calibrated_length_probs(
    logits: torch.Tensor,
    vocab: CategoryVocab,
    calibration: dict[str, Any] | None,
    schema: ConditionalTABDLMSchema,
    column: str = "summary_length_bucket",
) -> torch.Tensor:
    probs = torch.softmax(torch.nan_to_num(logits.float(), nan=0.0), dim=-1)
    if column in schema.length_bucket_targets and calibration:
        ratio = calibration.get("calibration_ratio", {})
        strength = float(calibration.get("calibration_strength", 1.0))
        factors = torch.ones(vocab.size, dtype=probs.dtype, device=probs.device)
        id_to_token = vocab.id_to_token
        for idx in range(vocab.size):
            factors[idx] = float(ratio.get(id_to_token[idx], 1.0)) ** strength
        probs = probs * factors.view(1, -1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


def enforce_soft_length(
    sequence: torch.Tensor,
    logits: torch.Tensor,
    tokenizer: SimpleTextTokenizer,
    low: int,
    high: int,
    rng: random.Random,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> None:
    max_len = int(sequence.shape[0])
    high = max(int(low), min(int(high), max_len - 2))
    low = max(0, min(int(low), high))
    ids = sequence.detach().cpu().tolist()
    eos_pos = next((idx for idx, token_id in enumerate(ids) if int(token_id) == tokenizer.eos_id), None)
    pad_pos = next((idx for idx, token_id in enumerate(ids) if int(token_id) == tokenizer.pad_id), None)
    stop_pos = eos_pos if eos_pos is not None else pad_pos
    decoded_len = max(0, (stop_pos - 1) if stop_pos is not None else tokenizer.content_length(ids))

    if decoded_len < low:
        eos_pos = min(low + 1, max_len - 1)
        for pos in range(1, eos_pos):
            token_id = int(sequence[pos].item())
            if token_id in tokenizer.special_ids:
                sequence[pos] = sample_content_token(
                    logits[pos],
                    tokenizer,
                    temperature=temperature,
                    top_p=top_p,
                    previous_ids=sequence[1:pos].detach().cpu().tolist(),
                    repetition_penalty=repetition_penalty,
                )
    elif decoded_len > high:
        eos_pos = min(high + 1, max_len - 1)
    else:
        eos_pos = stop_pos if stop_pos is not None else min(decoded_len + 1, max_len - 1)

    eos_pos = min(max(int(eos_pos or 1), 1), max_len - 1)
    sequence[eos_pos] = tokenizer.eos_id
    if eos_pos + 1 < max_len:
        sequence[eos_pos + 1 :] = tokenizer.pad_id


def decoded_length_bucket(content_length: int, schema: ConditionalTABDLMSchema, column: str | None = "summary_length_bucket") -> str | None:
    if column is None:
        return None
    for name, (low, high) in schema.buckets_for_length_bucket(column).items():
        if int(low) <= int(content_length) <= int(high):
            return str(name)
    return None


def length_bucket_for_calibration(calibration: dict[str, Any] | None, column: str) -> dict[str, Any] | None:
    return length_calibration_for_column(calibration, column)


def length_calibration_for_column(calibration: dict[str, Any] | None, column: str) -> dict[str, Any] | None:
    if not calibration:
        return None
    if "calibration_ratio" in calibration:
        return calibration
    value = calibration.get(column)
    return value if isinstance(value, dict) else None


def length_bucket_column_for_text(schema: ConditionalTABDLMSchema, text_column: str) -> str | None:
    for column in schema.length_bucket_targets:
        try:
            if schema.text_column_for_length_bucket(column) == text_column:
                return column
        except (KeyError, IndexError):
            continue
    return None


def sample_content_token(
    logits: torch.Tensor,
    tokenizer: SimpleTextTokenizer,
    temperature: float,
    top_p: float,
    previous_ids: list[int] | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    filtered = logits.clone()
    forbidden = list(tokenizer.special_ids)
    filtered[torch.tensor(forbidden, dtype=torch.long, device=filtered.device)] = -float("inf")
    if previous_ids and repetition_penalty > 1.0:
        for token_id in set(int(idx) for idx in previous_ids if int(idx) not in tokenizer.special_ids):
            filtered[token_id] = filtered[token_id] / float(repetition_penalty)
    if not torch.isfinite(filtered).any():
        return torch.tensor(tokenizer.content_token_ids[0], dtype=torch.long, device=filtered.device)
    return sample_logits(filtered.view(1, -1), temperature=temperature, top_p=top_p).squeeze(0)


def write_debug_outputs(attrs: dict[str, Any], debug_dir: str | Path) -> None:
    debug_dir = ensure_dir(debug_dir)
    examples = attrs.get("_debug_examples", [])
    examples_path = debug_dir / "generated_examples.jsonl"
    with examples_path.open("w") as handle:
        for row in examples:
            json.dump(jsonable(row), handle, sort_keys=True)
            handle.write("\n")
    for column in ["summary_length_bucket", "review_text_length_bucket"]:
        if column in attrs:
            counts = pd.Series(attrs[column]).value_counts().sort_index()
            counts.rename_axis(column).reset_index(name="count").to_csv(
                debug_dir / f"{column}_histogram.csv",
                index=False,
            )
    if "_length_calibration" in attrs:
        with (debug_dir / "length_calibration.json").open("w") as handle:
            json.dump(jsonable(attrs["_length_calibration"]), handle, indent=2, sort_keys=True)
            handle.write("\n")
    if "summary" in attrs:
        summaries = pd.Series(attrs["summary"]).fillna("").astype(str)
        top = summaries.value_counts().head(100).rename_axis("summary").reset_index(name="count")
        top["rate"] = top["count"] / max(len(summaries), 1)
        top.to_csv(debug_dir / "top_generated_summaries.csv", index=False)
    if "review_text" in attrs:
        reviews = pd.Series(attrs["review_text"]).fillna("").astype(str)
        top = reviews.value_counts().head(100).rename_axis("review_text").reset_index(name="count")
        top["rate"] = top["count"] / max(len(reviews), 1)
        top.to_csv(debug_dir / "top_generated_review_texts.csv", index=False)
    if examples:
        metrics = decoding_metrics_from_examples([row for row in examples if row.get("text_column") in (None, "summary")])
        with (debug_dir / "summary_length_decoding_metrics.json").open("w") as handle:
            json.dump(jsonable(metrics), handle, indent=2, sort_keys=True)
            handle.write("\n")
        review_examples = [row for row in examples if row.get("text_column") == "review_text"]
        if review_examples:
            review_metrics = decoding_metrics_from_examples(review_examples, bucket_column="review_text_length_bucket")
            with (debug_dir / "review_text_length_decoding_metrics.json").open("w") as handle:
                json.dump(jsonable(review_metrics), handle, indent=2, sort_keys=True)
                handle.write("\n")


def decoding_metrics_from_examples(examples: list[dict[str, Any]], bucket_column: str = "summary_length_bucket") -> dict[str, Any]:
    sampled = [row.get(bucket_column) for row in examples if row.get(bucket_column) is not None]
    decoded = [row.get("decoded_length_bucket") for row in examples if row.get("decoded_length_bucket") is not None]
    respected = [row.get("target_bucket_respected") for row in examples if row.get("target_bucket_respected") is not None]
    errors = []
    for row in examples:
        if row.get("target_content_length") is not None and row.get("content_length") is not None:
            errors.append(abs(float(row["target_content_length"]) - float(row["content_length"])))
    return {
        "sampled_length_bucket_distribution": normalized_counts(sampled),
        "decoded_length_bucket_distribution": normalized_counts(decoded),
        "target_bucket_respected_rate": float(sum(respected) / len(respected)) if respected else None,
        "target_vs_decoded_length_mae": float(sum(errors) / len(errors)) if errors else None,
    }


def normalized_counts(values: list[Any]) -> dict[str, float]:
    if not values:
        return {}
    counts = pd.Series(values).value_counts(normalize=True).sort_index()
    return {str(key): float(value) for key, value in counts.items()}


def sort_spine_chronologically(spine: pd.DataFrame, schema: ConditionalTABDLMSchema) -> pd.DataFrame:
    timestamp_col = schema.datetime_columns[0]
    frame = spine.copy()
    frame["_original_row_index"] = range(len(frame))
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    return frame.sort_values([timestamp_col, "_original_row_index"], kind="mergesort").drop(columns=["_original_row_index"]).reset_index(drop=True)


def sampling_graph_context(
    graph_encoder: TemporalStructureOnlyGraphEncoder | None,
    graph_history_index: Any | None,
    generated_attr_store: GraphAttributeStore | None,
    config: ConditionalTABDLMConfig,
    *,
    row_indices: list[int],
    cat_input: torch.Tensor,
    text_input: dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor | None:
    if graph_encoder is None:
        return None
    if graph_history_index is None or generated_attr_store is None:
        raise ValueError("v3 sampling requires graph_history_index and generated_attr_store")
    graph_batch = graph_history_index.build_batch(row_indices, device=device, deterministic=True)
    target_batch = {
        "categorical_input_ids": cat_input,
        "text_input_ids": text_input,
    }
    attr_batch, _ = build_attribute_graph_batch(
        graph_batch,
        target_batch,
        generated_attr_store,
        config,
        device=device,
        training=False,
    )
    graph_batch.update(attr_batch)
    return graph_encoder(graph_batch)


def encode_conditions(
    frame: pd.DataFrame,
    schema: ConditionalTABDLMSchema,
    num_hash_buckets: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    foreign_keys = [
        [
            stable_hash_bucket(column, value, num_hash_buckets)
            for value in frame[column].to_numpy()
        ]
        for column in schema.foreign_key_columns
    ]
    datetimes = []
    for column in schema.datetime_columns:
        timestamps = pd.to_datetime(frame[column], errors="coerce")
        if timestamps.isna().any():
            raise ValueError(f"Cannot encode datetime condition {column!r}; found unparsable values")
        timestamp_ns = timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
        datetimes.append((timestamp_ns.astype(np.float64) / 1_000_000_000.0).astype(np.float32))
    return (
        torch.tensor(np.asarray(foreign_keys, dtype=np.int64).T, dtype=torch.long, device=device),
        torch.tensor(np.asarray(datetimes, dtype=np.float32).T, dtype=torch.float32, device=device),
    )


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    logits = logits / max(float(temperature), 1e-6)
    if top_k is not None and int(top_k) > 0 and int(top_k) < logits.shape[-1]:
        k = int(top_k)
        top_values, top_indices = torch.topk(logits, k=k, dim=-1)
        if top_p < 1.0:
            sorted_logits, sorted_local_idx = torch.sort(top_values, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative > float(top_p)
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
            filtered_top = torch.full_like(top_values, -float("inf"))
            filtered_top.scatter_(dim=-1, index=sorted_local_idx, src=sorted_logits)
            top_values = filtered_top
        probs = torch.softmax(top_values, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(row_sums > 0, probs / row_sums.clamp_min(1e-12), torch.full_like(probs, 1.0 / probs.shape[-1]))
        local = torch.multinomial(probs, num_samples=1)
        return top_indices.gather(dim=-1, index=local).squeeze(-1)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        filtered = torch.full_like(logits, -float("inf"))
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        logits = filtered
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(row_sums > 0, probs / row_sums.clamp_min(1e-12), torch.full_like(probs, 1.0 / probs.shape[-1]))
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def validate_spine(spine: pd.DataFrame, schema: ConditionalTABDLMSchema) -> None:
    missing = [column for column in schema.condition_columns if column not in spine.columns]
    if missing:
        raise ValueError(f"Synthetic spine is missing condition columns: {missing}")
