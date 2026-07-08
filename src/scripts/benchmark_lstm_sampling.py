#!/usr/bin/env python3
"""Benchmark naive vs optimized LSTM joint full-review-text sampling."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.constrained import normalize_rating_value  # noqa: E402
from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    FastSamplerOptions,
    sample_lstm_fast_from_config,
)
from attribute_generation.conditional_tabdlm.lstm_joint import sample_lstm_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp5_lstm_joint_full_review_text.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LSTM joint sampler speed.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--row-counts", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--modes", nargs="+", choices=["naive", "optimized"], default=["naive", "optimized"])
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--v4-sampling-seconds-50k", type=float, default=None)
    parser.add_argument("--old-lstm-sampling-seconds-75k", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row_count in args.row_counts:
        for mode in args.modes:
            run_dir = output_dir / f"{mode}_{row_count}"
            run_dir.mkdir(parents=True, exist_ok=True)
            output_path = run_dir / "synthetic_review_attrs.csv"
            reset_cuda_peak()
            start = time.perf_counter()
            if mode == "naive":
                sample_lstm_from_config(
                    config,
                    checkpoint_path=args.checkpoint,
                    output_path=output_path,
                    num_rows=row_count,
                    device=args.device,
                    synthetic_spine_path=args.synthetic_spine,
                )
                runtime_path = output_path.parent / "metadata" / "runtime_sampling.json"
            else:
                sample_lstm_fast_from_config(
                    config,
                    checkpoint_path=args.checkpoint,
                    output_path=output_path,
                    num_rows=row_count,
                    device=args.device,
                    synthetic_spine_path=args.synthetic_spine,
                    options=FastSamplerOptions(
                        profile=True,
                        profile_output=run_dir / "runtime_sampling_fast.json",
                        max_batch_size=args.max_batch_size,
                    ),
                )
                runtime_path = run_dir / "runtime_sampling_fast.json"
            wall_seconds = float(time.perf_counter() - start)
            runtime = load_json(runtime_path) if runtime_path.exists() else {}
            structural = structural_metrics(output_path)
            total_seconds = float(runtime.get("total_sampling_seconds", wall_seconds))
            rows_per_second = float(row_count / max(total_seconds, 1e-9))
            rows.append(
                {
                    "mode": mode,
                    "row_count": int(row_count),
                    "total_seconds": total_seconds,
                    "wall_seconds": wall_seconds,
                    "rows_per_second": rows_per_second,
                    "seconds_per_1000_rows": float(1000.0 / max(rows_per_second, 1e-9)),
                    "projected_hours_for_10m_rows": float((10_000_000 / max(rows_per_second, 1e-9)) / 3600.0),
                    "peak_gpu_memory_mb": runtime.get("cuda_memory_peak_allocated_mb", cuda_peak_allocated_mb()),
                    "average_generated_review_text_length": structural["average_review_text_length"],
                    **structural,
                    "output_path": str(output_path),
                    "runtime_path": str(runtime_path),
                }
            )
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "synthetic_spine": args.synthetic_spine,
        "baselines": {
            "v4_sampling_seconds_50k": args.v4_sampling_seconds_50k,
            "old_lstm_sampling_seconds_75k": args.old_lstm_sampling_seconds_75k,
        },
        "runs": rows,
    }
    write_json(output_dir / "benchmark.json", payload)
    write_markdown(output_dir / "benchmark.md", rows, payload["baselines"])
    print(f"Wrote {output_dir / 'benchmark.json'}")
    print(f"Wrote {output_dir / 'benchmark.md'}")


def structural_metrics(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    rating_valid = frame["rating"].map(lambda value: normalize_rating_value(value) is not None) if "rating" in frame else pd.Series(dtype=bool)
    if "verified" in frame:
        verified_valid = frame["verified"].astype(str).str.strip().str.lower().isin({"true", "false", "0", "1"})
    else:
        verified_valid = pd.Series(dtype=bool)
    summary = frame["summary"].fillna("").astype(str) if "summary" in frame else pd.Series(dtype=str)
    review_text = frame["review_text"].fillna("").astype(str) if "review_text" in frame else pd.Series(dtype=str)
    return {
        "invalid_rating_rate": float(1.0 - rating_valid.mean()) if len(rating_valid) else None,
        "invalid_verified_rate": float(1.0 - verified_valid.mean()) if len(verified_valid) else None,
        "empty_summary_rate": float((summary.str.strip() == "").mean()) if len(summary) else None,
        "empty_review_text_rate": float((review_text.str.strip() == "").mean()) if len(review_text) else None,
        "average_summary_length": float(summary.map(lambda text: len(text.split())).mean()) if len(summary) else None,
        "average_review_text_length": float(review_text.map(lambda text: len(text.split())).mean()) if len(review_text) else None,
    }


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            pass


def cuda_peak_allocated_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    except RuntimeError:
        return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown(path: Path, rows: list[dict[str, Any]], baselines: dict[str, Any]) -> None:
    lines = ["# LSTM Sampling Benchmark", ""]
    if any(value is not None for value in baselines.values()):
        lines.append("## Baselines")
        lines.append("")
        for key, value in baselines.items():
            if value is not None:
                lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.extend(
        [
            "## Runs",
            "",
            "| mode | rows | seconds | rows/sec | sec/1k | projected 10M hours | peak GPU MB | avg review len | invalid rating | invalid verified | empty summary | empty review |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {mode} | {row_count} | {total_seconds:.2f} | {rows_per_second:.2f} | {seconds_per_1000_rows:.2f} | "
            "{projected_hours_for_10m_rows:.2f} | {peak_gpu_memory_mb} | {average_review_text_length} | "
            "{invalid_rating_rate} | {invalid_verified_rate} | {empty_summary_rate} | {empty_review_text_rate} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return round(value, 4)
    return value


if __name__ == "__main__":
    main()
