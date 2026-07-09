#!/usr/bin/env python3
"""Sample a full-size event table from the current best attribute model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MODEL_VARIANT = "v51_length_preserving_exact_block"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample the best model at real-table row count.")
    parser.add_argument("--real-table", required=True)
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-output", required=True)
    parser.add_argument("--num-rows", default="auto")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.set_defaults(length_preserving_exact_blocking=True, auto_batch_size=True, mixed_precision=True)
    parser.add_argument("--length-preserving-exact-blocking", dest="length_preserving_exact_blocking", action="store_true")
    parser.add_argument("--no-length-preserving-exact-blocking", dest="length_preserving_exact_blocking", action="store_false")
    parser.add_argument("--disable-review-text-ngram-blocking", action="store_true")
    parser.add_argument("--auto-batch-size", dest="auto_batch_size", action="store_true")
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--write-chunk-size", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    run_full_sampling(parse_args())


def run_full_sampling(
    args: argparse.Namespace,
    *,
    sampler_fn: Callable[..., Path] | None = None,
    load_config_fn: Callable[[str | Path], Any] | None = None,
    options_cls: type | None = None,
) -> Path:
    if sampler_fn is None or options_cls is None:
        from attribute_generation.conditional_tabdlm.lstm_fast_sampler import FastSamplerOptions, sample_lstm_fast_from_config

        sampler_fn = sampler_fn or sample_lstm_fast_from_config
        options_cls = options_cls or FastSamplerOptions
    if load_config_fn is None:
        from attribute_generation.conditional_tabdlm.schema import load_config

        load_config_fn = load_config
    real_rows = count_rows(args.real_table)
    num_rows = resolve_num_rows(args.num_rows, real_rows)
    output = Path(args.output)
    profile_output = Path(args.profile_output)
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    options = options_cls(
        profile=True,
        profile_output=str(profile_output),
        decode_mode="bucketed",
        auto_batch_size=bool(args.auto_batch_size),
        mixed_precision=bool(args.mixed_precision),
        cache_graph_context=True,
        graph_context_cache_mode="batch",
        cache_condition_embeddings=True,
        active_row_masking=True,
        length_bucketed_decoding=True,
        detokenize_after_generation=True,
        write_chunk_size=int(args.write_chunk_size),
        seed=args.seed,
        use_config_privacy_controls=False,
        exact_train_overlap_blocking_enabled=bool(args.length_preserving_exact_blocking),
        length_preserving_exact_blocking_enabled=bool(args.length_preserving_exact_blocking),
        no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_enabled=False,
        review_text_no_repeat_ngram_enabled=False if args.disable_review_text_ngram_blocking else None,
        summary_no_repeat_ngram_size=0,
        review_text_no_repeat_ngram_size=0,
        max_summary_resample_attempts=5,
        max_review_text_resample_attempts=3,
    )
    sampled_path = sampler_fn(
        load_config_fn(args.config),
        checkpoint_path=args.checkpoint,
        output_path=output,
        num_rows=num_rows,
        batch_size=args.batch_size,
        device=args.device,
        synthetic_spine_path=args.synthetic_spine,
        options=options,
    )
    elapsed = time.perf_counter() - start
    generated_rows = count_rows(sampled_path) if Path(sampled_path).exists() else None
    rows_per_second = float(generated_rows / elapsed) if generated_rows is not None and elapsed > 0 else None
    write_runtime_metadata(
        profile_output,
        {
            "dataset_name": getattr(args, "dataset_name", None),
            "num_requested_rows": int(num_rows),
            "num_generated_rows": int(generated_rows) if generated_rows is not None else None,
            "real_table_row_count": int(real_rows),
            "full_table_sampling": True,
            "model_variant": MODEL_VARIANT,
            "architecture_changed": False,
            "length_preserving_exact_blocking_enabled": bool(args.length_preserving_exact_blocking),
            "exact_train_overlap_blocking_enabled": bool(args.length_preserving_exact_blocking),
            "review_text_no_repeat_ngram_enabled": False if args.disable_review_text_ngram_blocking else None,
            "runtime_seconds_wall_clock": float(elapsed),
            "total_sampling_seconds": float(elapsed),
            "rows_per_second": rows_per_second,
            "projected_hours_for_10m_rows": float(10_000_000.0 / rows_per_second / 3600.0) if rows_per_second else None,
            "peak_gpu_memory_mb": None,
            "mixed_precision_used": bool(args.mixed_precision),
            "auto_batch_size_used": bool(args.auto_batch_size),
            "write_chunk_size": int(args.write_chunk_size),
            "output": str(sampled_path),
            "synthetic_spine": str(args.synthetic_spine),
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
        },
    )
    print(f"Wrote {sampled_path}")
    print(f"Wrote {profile_output}")
    return Path(sampled_path)


def resolve_num_rows(value: Any, real_table_row_count: int) -> int:
    if value in (None, "auto"):
        return int(real_table_row_count)
    return int(value)


def count_rows(path: str | Path) -> int:
    return int(sum(len(chunk) for chunk in pd.read_csv(path, usecols=[0], chunksize=1_000_000)))


def write_runtime_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
