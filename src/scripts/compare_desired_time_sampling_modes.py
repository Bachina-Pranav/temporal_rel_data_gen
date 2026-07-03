#!/usr/bin/env python3
"""Compare desired-time sampling modes for time-biased stub matching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.fast_event_spine_metrics import write_metrics  # noqa: E402
from generators.time_biased_block_stub_matching import TimeBiasedBlockStubMatchingGenerator  # noqa: E402


METRIC_KEYS = [
    "product_first_time_corr",
    "product_last_time_corr",
    "product_peak_time_corr",
    "product_active_span_ks",
    "product_activity_entropy_ks",
    "product_time_activity_distribution_ks",
    "product_relative_age_ks",
    "product_active_window_rate",
    "customer_first_time_corr",
    "customer_last_time_corr",
    "customer_peak_time_corr",
    "customer_active_span_ks",
    "customer_activity_entropy_ks",
    "customer_time_activity_distribution_ks",
    "customer_relative_age_ks",
    "customer_active_window_rate",
    "joint_coactive_window_rate",
    "duplicate_customer_product_rate",
    "real_edge_overlap_rate",
    "exact_event_overlap_rate",
    "mean_dynamic_affinity_synthetic",
    "dynamic_affinity_distribution_ks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare desired-time sampling modes without synthetic-metric tuning.")
    parser.add_argument("--real-reviews", default="data/original/rel-amazon-toy/review.csv")
    parser.add_argument("--structure-debug-dir", default="outputs/amazon-toy/ct_2k_sbm_temporal_kde_stubs/debug")
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--output-root", default="outputs/amazon-toy")
    parser.add_argument("--output-csv", default="outputs/amazon-toy/time_biased_block_stub_matching_desired_time_comparison.csv")
    parser.add_argument("--output-json", default="outputs/amazon-toy/time_biased_block_stub_matching_desired_time_comparison.json")
    parser.add_argument("--time-granularity", default="day", choices=["day"])
    parser.add_argument("--time-gate-granularity", default="month", choices=["day", "month"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha-time-gate", default="auto")
    parser.add_argument("--kernel-bandwidth-mode", choices=["auto_block_iqr", "auto_global_iqr", "fixed"], default="auto_block_iqr")
    parser.add_argument("--kernel-bandwidth-scale", type=float, default=0.25)
    parser.add_argument("--kernel-min-bandwidth-days", type=float, default=1.0)
    parser.add_argument("--kernel-max-bandwidth-days", type=float, default=None)
    parser.add_argument("--kernel-fixed-bandwidth-days", type=float, default=7.0)
    parser.add_argument("--kernel-type", choices=["discrete_laplace", "discrete_gaussian", "none"], default="discrete_laplace")
    parser.add_argument("--max-exact-affinity-cell-size", type=int, default=128)
    parser.add_argument("--compute-c2st", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.real_reviews)
    runs = [
        ("median_degree_mixture", "mixture_shrinkage", "median_degree"),
        ("empirical_bayes_mixture", "empirical_bayes", "empirical_bayes"),
        ("empirical_exact", "empirical_exact", "median_degree"),
        ("local_kernel", "local_kernel", "median_degree"),
    ]
    nested: Dict[str, Dict[str, Any]] = {}
    rows = []
    for name, desired_mode, shrinkage_mode in runs:
        output_dir = Path(args.output_root) / f"time_biased_block_stub_matching_{name}"
        metrics = run_one(args, real, output_dir, desired_mode, shrinkage_mode)
        nested[name] = metrics
        rows.append(
            {
                "run": name,
                "desired_time_sampling_mode": metrics.get("desired_time_sampling_mode"),
                "temporal_shrinkage_mode": metrics.get("temporal_shrinkage_mode"),
                "temporal_alpha_used": metrics.get("temporal_alpha_used"),
                "alpha_customer_time_selected": metrics.get("alpha_customer_time_selected"),
                "alpha_product_time_selected": metrics.get("alpha_product_time_selected"),
                **{key: metrics.get(key) for key in METRIC_KEYS},
            }
        )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(nested, handle, indent=2)
        handle.write("\n")
    table = pd.DataFrame(rows)
    table.to_csv(output_csv, index=False)
    print(table.to_string(index=False))
    print(f"[done] wrote {output_csv}")
    print(f"[done] wrote {output_json}")


def run_one(
    args: argparse.Namespace,
    real: pd.DataFrame,
    output_dir: Path,
    desired_time_sampling_mode: str,
    temporal_shrinkage_mode: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = TimeBiasedBlockStubMatchingGenerator(
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        structure_debug_dir=args.structure_debug_dir,
        time_granularity=args.time_granularity,
        time_gate_granularity=args.time_gate_granularity,
        rank=args.rank,
        temporal_shrinkage_mode=temporal_shrinkage_mode,
        desired_time_sampling_mode=desired_time_sampling_mode,
        alpha_time_gate=args.alpha_time_gate,
        kernel_bandwidth_mode=args.kernel_bandwidth_mode,
        kernel_bandwidth_scale=args.kernel_bandwidth_scale,
        kernel_min_bandwidth_days=args.kernel_min_bandwidth_days,
        kernel_max_bandwidth_days=args.kernel_max_bandwidth_days,
        kernel_fixed_bandwidth_days=args.kernel_fixed_bandwidth_days,
        kernel_type=args.kernel_type,
        pairing_mode="dynamic_exact_penalized",
        max_exact_affinity_cell_size=args.max_exact_affinity_cell_size,
        seed=args.seed,
    )
    synthetic = generator.fit(real).sample(seed=args.seed)
    synthetic_path = output_dir / "synthetic_review.csv"
    synthetic.to_csv(synthetic_path, index=False)
    generator.save_debug(output_dir / "debug")
    generator.save_metadata(output_dir / "metadata.json")
    print(f"[evaluation] {output_dir.name}: starting event-spine metrics", flush=True)
    metrics = generator.evaluate(real, synthetic, compute_c2st=args.compute_c2st)
    print(f"[evaluation] {output_dir.name}: done", flush=True)
    metadata = generator.metadata()
    metrics.update(
        {
            "desired_time_sampling_mode": metadata["desired_time_sampling_mode"],
            "temporal_shrinkage_mode": metadata["temporal_shrinkage_mode"],
            "temporal_alpha_used": metadata["temporal_alpha_used"],
            "empirical_bayes_used": metadata["empirical_bayes_used"],
            "alpha_customer_time_selected": metadata["alpha_customer_time_selected"],
            "alpha_product_time_selected": metadata["alpha_product_time_selected"],
            "bandwidth_selection_uses_synthetic_metrics": metadata["bandwidth_selection_uses_synthetic_metrics"],
            "pairing_mode": metadata["pairing_mode"],
            "pairing_penalties_fixed_defaults": metadata["pairing_penalties_fixed_defaults"],
        }
    )
    write_metrics(metrics, output_dir / "metrics.json")
    return metrics


if __name__ == "__main__":
    main()
