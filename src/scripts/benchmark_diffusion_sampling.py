#!/usr/bin/env python3
"""Benchmark Conditional TABDLM masked-diffusion sampling settings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.constrained import normalize_rating_value  # noqa: E402
from attribute_generation.conditional_tabdlm.sample import sample_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Conditional TABDLM diffusion sampler speed.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--synthetic-spine", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-rows", type=int, default=1000)
    parser.add_argument("--sampling-steps", nargs="+", default=["50", "25", "10"], help="Step counts or 'full'.")
    parser.add_argument("--sample-batch-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--inference-dtypes", nargs="+", choices=["float32", "float16", "bfloat16"], default=["float32"])
    parser.add_argument("--text-top-k", nargs="+", default=["none"], help="Use 'none' for full-vocab top-p sampling.")
    parser.add_argument("--timestep-spacing", choices=["uniform", "quadratic", "leading", "trailing"], default="uniform")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    variants = list(build_variants(args))
    if args.max_runs is not None:
        variants = variants[: int(args.max_runs)]
    for variant in variants:
        run_name = variant_name(variant)
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        synthetic_path = run_dir / "synthetic_review_attrs.csv"
        profile_path = run_dir / "runtime_diffusion_sampling.json"
        print(f"[benchmark] {run_name}", flush=True)
        start = time.perf_counter()
        try:
            sample_from_config(
                config,
                checkpoint_path=args.checkpoint,
                synthetic_spine_path=args.synthetic_spine,
                output_path=synthetic_path,
                num_rows=args.num_rows,
                sample_batch_size=variant["sample_batch_size"],
                sampling_steps=variant["sampling_steps"],
                timestep_spacing=args.timestep_spacing,
                inference_dtype=variant["inference_dtype"],
                text_top_k=variant["text_top_k"],
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
                seed=args.seed,
                compile_model=args.compile_model,
                profile=True,
                profile_output=profile_path,
            )
            status = "ok"
            error = None
        except RuntimeError as exc:
            status = "runtime_error"
            error = str(exc)
            if "out of memory" in error.lower():
                status = "oom"
            print(f"[benchmark] {run_name} failed: {status}: {error}", flush=True)
        wall_seconds = float(time.perf_counter() - start)
        runtime = load_json(profile_path) if profile_path.exists() else {}
        structural = structural_metrics(synthetic_path) if synthetic_path.exists() else {}
        total_seconds = float(runtime.get("total_sampling_seconds", wall_seconds))
        rows_per_second = float(args.num_rows / max(total_seconds, 1e-9)) if status == "ok" else None
        rows.append(
            {
                "run": run_name,
                "status": status,
                "error": error,
                "num_rows": int(args.num_rows),
                **variant,
                "timestep_spacing": args.timestep_spacing,
                "compile_model": bool(args.compile_model),
                "total_seconds": total_seconds,
                "wall_seconds": wall_seconds,
                "rows_per_second": rows_per_second,
                "seconds_per_1000_rows": float(1000.0 / max(rows_per_second, 1e-9)) if rows_per_second else None,
                "peak_gpu_memory_mb": runtime.get("cuda_memory_peak_allocated_mb"),
                "denoising_loop_seconds": runtime.get("denoising_loop_seconds"),
                "denoising_step_seconds": runtime.get("denoising_step_seconds"),
                "length_enforcement_seconds": runtime.get("length_enforcement_seconds"),
                "text_decoding_seconds": runtime.get("text_decoding_seconds"),
                "csv_writing_seconds": runtime.get("csv_writing_seconds"),
                **structural,
                "output_path": str(synthetic_path),
                "profile_path": str(profile_path),
            }
        )
    payload = {"config": args.config, "checkpoint": args.checkpoint, "synthetic_spine": args.synthetic_spine, "runs": rows}
    write_json(output_dir / "benchmark_diffusion_sampling.json", payload)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "benchmark_diffusion_sampling.csv", index=False)
    write_markdown(output_dir / "benchmark_diffusion_sampling.md", rows)
    print(f"[done] wrote {output_dir / 'benchmark_diffusion_sampling.json'}")
    print(f"[done] wrote {output_dir / 'benchmark_diffusion_sampling.csv'}")
    print(f"[done] wrote {output_dir / 'benchmark_diffusion_sampling.md'}")


def build_variants(args: argparse.Namespace):
    for steps in args.sampling_steps:
        for batch_size in args.sample_batch_sizes:
            for dtype in args.inference_dtypes:
                for top_k in args.text_top_k:
                    yield {
                        "sampling_steps": str(steps),
                        "sample_batch_size": int(batch_size),
                        "inference_dtype": str(dtype),
                        "text_top_k": parse_top_k(top_k),
                    }


def parse_top_k(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "full", "0"}:
        return None
    parsed = int(text)
    return parsed if parsed > 0 else None


def variant_name(variant: dict[str, Any]) -> str:
    top_k = "full" if variant["text_top_k"] is None else str(variant["text_top_k"])
    return (
        f"steps_{variant['sampling_steps']}"
        f"_bs_{variant['sample_batch_size']}"
        f"_dtype_{variant['inference_dtype']}"
        f"_topk_{top_k}"
    )


def structural_metrics(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    rating_valid = frame["rating"].map(lambda value: normalize_rating_value(value) is not None) if "rating" in frame else pd.Series(dtype=bool)
    verified_valid = (
        frame["verified"].astype(str).str.strip().str.lower().isin({"true", "false", "0", "1"})
        if "verified" in frame
        else pd.Series(dtype=bool)
    )
    summary = frame["summary"].fillna("").astype(str) if "summary" in frame else pd.Series(dtype=str)
    review_text = frame["review_text"].fillna("").astype(str) if "review_text" in frame else pd.Series(dtype=str)
    return {
        "invalid_rating_rate": float(1.0 - rating_valid.mean()) if len(rating_valid) else None,
        "invalid_verified_rate": float(1.0 - verified_valid.mean()) if len(verified_valid) else None,
        "empty_summary_rate": float((summary.str.strip() == "").mean()) if len(summary) else None,
        "empty_review_text_rate": float((review_text.str.strip() == "").mean()) if len(review_text) else None,
        "avg_summary_tokens": float(summary.map(lambda text: len(text.split())).mean()) if len(summary) else None,
        "avg_review_text_tokens": float(review_text.map(lambda text: len(text.split())).mean()) if len(review_text) else None,
        "duplicate_summary_rate": float(summary.duplicated().mean()) if len(summary) else None,
        "duplicate_review_text_rate": float(review_text.duplicated().mean()) if len(review_text) else None,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Conditional TABDLM Diffusion Sampling Benchmark",
        "",
        "| run | status | rows | steps | batch | dtype | top_k | seconds | rows/sec | peak GPU MB | denoise s | length s | text decode s | empty review | dup review |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {status} | {num_rows} | {sampling_steps} | {sample_batch_size} | {inference_dtype} | {text_top_k} | "
            "{total_seconds} | {rows_per_second} | {peak_gpu_memory_mb} | {denoising_loop_seconds} | "
            "{length_enforcement_seconds} | {text_decoding_seconds} | {empty_review_text_rate} | {duplicate_review_text_rate} |".format(
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
