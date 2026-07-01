#!/usr/bin/env python3
"""Run ct_2k_sbm_temporal_kde_stubs across seeds and summarize stability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_all_structure_methods import evaluate_method  # noqa: E402
from evaluate_temporal_sbm_event_spine import load_reviews  # noqa: E402
from reldiff.generation import ContinuousTime2KSBMTemporalKDEStubsGenerator  # noqa: E402


METHOD_NAME = "ct_2k_sbm_temporal_kde_stubs"
REPORT_METRICS = [
    "product_degree_ks",
    "customer_degree_ks",
    "global_timestamp_ks",
    "timestamp_count_l1_by_date",
    "product_inter_event_time_ks",
    "customer_inter_event_time_ks",
    "top_product_trajectory_corr",
    "edge_overlap_rate",
    "duplicate_customer_product_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ct_2k_sbm_temporal_kde_stubs across several seeds and report "
            "mean/std stability metrics."
        )
    )
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument(
        "--output-root",
        default="outputs/amazon-toy/ct_2k_sbm_temporal_kde_stubs_seed_sweep",
    )
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--sbm-block-level", default="bottom")
    parser.add_argument(
        "--timestamp-model",
        choices=[
            "auto",
            "smoothed_date_pmf",
            "block_pair_kde",
            "global_kde",
            "bootstrap_jitter",
        ],
        default="auto",
    )
    parser.add_argument("--timestamp-smoothing-alpha", default="auto")
    parser.add_argument("--timestamp-bandwidth", default="scott")
    parser.add_argument("--timestamp-min-block-count", type=int, default=20)
    parser.add_argument(
        "--pairing-mode",
        choices=["random", "temporal_sorted", "temporal_window_shuffle"],
        default="temporal_window_shuffle",
    )
    parser.add_argument("--temporal-window-size", default="auto")
    parser.add_argument("--avoid-real-edge-prob", type=float, default=0.95)
    parser.add_argument("--max-swap-attempts", type=int, default=20)
    parser.add_argument("--trajectory-bins", default="M")
    parser.add_argument("--block-pair-min-count", type=int, default=5)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only evaluate existing seed directories under output-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reviews_raw = pd.read_csv(args.real_reviews)
    real = load_reviews(args.real_reviews, args.timestamp_col)

    rows = []
    for seed in args.seeds:
        seed_dir = output_root / f"seed_{seed}"
        synthetic_path = seed_dir / "synthetic_review.csv"
        debug_dir = seed_dir / "debug"
        if not args.skip_generation:
            run_seed(args, reviews_raw, seed, synthetic_path, debug_dir)
        if not synthetic_path.exists():
            raise FileNotFoundError(
                f"Missing {synthetic_path}; rerun without --skip-generation."
            )
        row = evaluate_method(
            METHOD_NAME,
            synthetic_path,
            debug_dir,
            real,
            args.customer_id_col,
            args.product_id_col,
            args.timestamp_col,
            args.trajectory_bins,
            args.block_pair_min_count,
        )
        row["seed"] = int(seed)
        row["duplicate_customer_product_rate"] = row.get(
            "duplicate_customer_product_rate_synthetic"
        )
        rows.append(row)

    per_seed = pd.DataFrame(rows)
    summary = summarize_metrics(per_seed, REPORT_METRICS)
    per_seed_path = output_root / "per_seed_metrics.csv"
    summary_path = output_root / "seed_summary.csv"
    json_path = output_root / "seed_summary.json"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    with json_path.open("w") as handle:
        json.dump(
            {
                "seeds": [int(seed) for seed in args.seeds],
                "metrics": summary.to_dict(orient="records"),
                "top_product_trajectory_corr_mean": metric_mean(
                    per_seed, "top_product_trajectory_corr"
                ),
                "structural_module_ready_by_corr_0_85": bool(
                    (metric_mean(per_seed, "top_product_trajectory_corr") or 0.0)
                    >= 0.85
                ),
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print(f"Wrote {per_seed_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")
    print("\nSeed stability summary:")
    for _, row in summary.iterrows():
        mean = row["mean"]
        std = row["std"]
        if pd.isna(mean):
            print(f"  {row['metric']}: NA")
        else:
            print(f"  {row['metric']}: {mean:.6f} +/- {std:.6f}")


def run_seed(
    args: argparse.Namespace,
    reviews: pd.DataFrame,
    seed: int,
    synthetic_path: Path,
    debug_dir: Path,
) -> None:
    print(f"\n=== Running {METHOD_NAME} seed={seed} ===")
    generator = ContinuousTime2KSBMTemporalKDEStubsGenerator.from_reviews(
        reviews,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        seed=int(seed),
        sbm_block_level=args.sbm_block_level,
        timestamp_model=args.timestamp_model,
        timestamp_smoothing_alpha=parse_alpha(args.timestamp_smoothing_alpha),
        timestamp_bandwidth=parse_bandwidth(args.timestamp_bandwidth),
        timestamp_min_block_count=args.timestamp_min_block_count,
        pairing_mode=args.pairing_mode,
        temporal_window_size=parse_optional_int(args.temporal_window_size),
        avoid_real_edge_prob=args.avoid_real_edge_prob,
        max_swap_attempts=args.max_swap_attempts,
    )
    generator.fit()
    generator.generate(output_path=synthetic_path, debug_dir=debug_dir)


def summarize_metrics(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        values = pd.to_numeric(df.get(metric), errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def metric_mean(df: pd.DataFrame, metric: str) -> float | None:
    values = pd.to_numeric(df.get(metric), errors="coerce").dropna()
    if len(values) == 0:
        return None
    return float(values.mean())


def parse_optional_int(value: str) -> int | None:
    if value == "auto":
        return None
    return int(value)


def parse_alpha(value: str) -> Any:
    if value == "auto":
        return value
    return float(value)


def parse_bandwidth(value: str) -> Any:
    if value in {"scott", "silverman"}:
        return value
    return float(value)


if __name__ == "__main__":
    main()
