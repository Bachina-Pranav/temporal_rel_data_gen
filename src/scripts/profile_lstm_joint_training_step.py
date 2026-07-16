#!/usr/bin/env python3
"""Profile a bounded fixed-step LSTM training run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tempdir_bootstrap import configure_tempdir  # noqa: E402

configure_tempdir(Path(__file__).resolve().parents[2])

import torch

from attribute_generation.conditional_tabdlm.lstm_joint import train_lstm_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMConfig,
    ConditionalTABDLMSchema,
    resolve_auto_review_text_config,
)
from attribute_generation.conditional_tabdlm.utils import ensure_dir, load_yaml, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile fixed-step LSTM training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-table", default=None)
    parser.add_argument("--pretokenized-dir", default=None)
    parser.add_argument("--neighbor-cache-dir", default=None)
    parser.add_argument("--physical-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument("--validation-max-batches", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    train_output = ensure_dir(output_dir / "profile_training_run")
    raw = load_yaml(args.config)
    paths = raw.setdefault("paths", {})
    training = raw.setdefault("training", {})
    if args.real_table:
        paths["train_data_path"] = args.real_table
    paths["output_dir"] = str(train_output)
    if args.pretokenized_dir:
        paths["pretokenized_dir"] = args.pretokenized_dir
        training["pretokenized_dir"] = args.pretokenized_dir
    if args.neighbor_cache_dir:
        paths["neighbor_cache_dir"] = args.neighbor_cache_dir
        training["neighbor_cache_dir"] = args.neighbor_cache_dir
    training.update(
        {
            "epoch_mode": False,
            "max_steps": int(args.warmup_steps + args.profile_steps),
            "steps_per_eval": max(1, int(args.profile_steps)),
            "steps_per_checkpoint": max(1, int(args.profile_steps)),
            "physical_batch_size": int(args.physical_batch_size),
            "batch_size": int(args.physical_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "mixed_precision": bool(args.mixed_precision),
            "profile": True,
            "validation_max_batches": int(args.validation_max_batches),
        }
    )
    if args.device is not None:
        training["device"] = args.device
    raw = resolve_auto_review_text_config(raw)
    schema = ConditionalTABDLMSchema.from_config_dict(raw)
    config = ConditionalTABDLMConfig(raw=raw, schema=schema, config_path=Path(args.config))
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_path = output_dir / "torch_profiler_trace.json"
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir / "tensorboard_trace")),
        ) as profiler:
            train_lstm_from_config(config)
            profiler.export_chrome_trace(str(trace_path))
            op_table = profiler.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=30)
    except Exception as exc:
        op_table = f"Profiler failed after/before training: {exc}"
    runtime_path = train_output / "metadata" / "training_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    summary = {
        "avg_step_seconds": runtime.get("avg_step_seconds"),
        "avg_batch_load_seconds": runtime.get("avg_batch_load_seconds"),
        "avg_h2d_seconds": runtime.get("avg_h2d_seconds"),
        "avg_forward_seconds": runtime.get("avg_forward_seconds"),
        "avg_backward_seconds": runtime.get("avg_backward_seconds"),
        "avg_optimizer_seconds": runtime.get("avg_optimizer_seconds"),
        "avg_graph_context_seconds": runtime.get("avg_graph_context_seconds"),
        "avg_token_collate_seconds": runtime.get("avg_batch_load_seconds"),
        "gpu_peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None,
        "gpu_utilization_if_available": None,
        "bottleneck_guess": bottleneck_guess(runtime),
        "torch_profiler_trace": str(trace_path) if trace_path.exists() else None,
    }
    save_json(summary, output_dir / "profiling_summary.json")
    (output_dir / "operator_table.txt").write_text(str(op_table), encoding="utf-8")
    print(f"Wrote {output_dir / 'profiling_summary.json'}")


def bottleneck_guess(runtime: dict[str, object]) -> str:
    candidates = {
        "batch_load": runtime.get("avg_batch_load_seconds") or 0.0,
        "h2d": runtime.get("avg_h2d_seconds") or 0.0,
        "graph_context": runtime.get("avg_graph_context_seconds") or 0.0,
        "forward": runtime.get("avg_forward_seconds") or 0.0,
        "backward": runtime.get("avg_backward_seconds") or 0.0,
        "optimizer": runtime.get("avg_optimizer_seconds") or 0.0,
    }
    return max(candidates, key=lambda key: float(candidates[key]))


if __name__ == "__main__":
    main()
