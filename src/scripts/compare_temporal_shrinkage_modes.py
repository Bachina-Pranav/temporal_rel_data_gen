#!/usr/bin/env python3
"""Run and compare temporal shrinkage modes for time-biased stub matching."""

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
    "product_relative_age_ks",
    "product_time_activity_distribution_ks",
    "customer_first_time_corr",
    "customer_last_time_corr",
    "customer_peak_time_corr",
    "customer_active_span_ks",
    "customer_time_activity_distribution_ks",
    "joint_coactive_window_rate",
    "duplicate_customer_product_rate",
    "real_edge_overlap_rate",
    "exact_event_overlap_rate",
    "mean_dynamic_affinity_synthetic",
    "dynamic_affinity_distribution_ks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare temporal shrinkage modes without synthetic-metric alpha tuning.")
    parser.add_argument("--real-reviews", default="data/original/rel-amazon-toy/review.csv")
    parser.add_argument("--structure-debug-dir", default="outputs/amazon-toy/ct_2k_sbm_temporal_kde_stubs/debug")
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--output-root", default="outputs/amazon-toy")
    parser.add_argument("--output-csv", default="outputs/amazon-toy/time_biased_block_stub_matching_shrinkage_comparison.csv")
    parser.add_argument("--output-json", default="outputs/amazon-toy/time_biased_block_stub_matching_shrinkage_comparison.json")
    parser.add_argument("--time-granularity", default="day", choices=["day"])
    parser.add_argument("--time-gate-granularity", default="month", choices=["day", "month"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha-time-gate", default="auto")
    parser.add_argument("--fixed-customer-alpha", type=float, default=0.1)
    parser.add_argument("--fixed-product-alpha", type=float, default=0.0)
    parser.add_argument("--max-exact-affinity-cell-size", type=int, default=128)
    parser.add_argument("--compute-c2st", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.real_reviews)
    output_root = Path(args.output_root)
    runs = [
        ("median_degree", "median_degree", "auto", "auto"),
        ("empirical_bayes", "empirical_bayes", "auto", "auto"),
        ("fixed_alpha_c01_p00_oracle_debug", "fixed", args.fixed_customer_alpha, args.fixed_product_alpha),
    ]
    nested: Dict[str, Dict[str, Any]] = {}
    rows = []
    for name, mode, customer_alpha, product_alpha in runs:
        output_dir = output_root / f"time_biased_block_stub_matching_{name}"
        metrics = run_one(args, real, output_dir, mode, customer_alpha, product_alpha)
        nested[name] = metrics
        rows.append(
            {
                "run": name,
                "temporal_shrinkage_mode": mode,
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
    temporal_shrinkage_mode: str,
    alpha_customer_time: Any,
    alpha_product_time: Any,
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
        alpha_customer_time=alpha_customer_time,
        alpha_product_time=alpha_product_time,
        temporal_shrinkage_mode=temporal_shrinkage_mode,
        alpha_time_gate=args.alpha_time_gate,
        pairing_mode="dynamic_exact_penalized",
        max_exact_affinity_cell_size=args.max_exact_affinity_cell_size,
        seed=args.seed,
    )
    synthetic = generator.fit(real).sample(seed=args.seed)
    synthetic_path = output_dir / "synthetic_review.csv"
    synthetic.to_csv(synthetic_path, index=False)
    generator.save_debug(output_dir / "debug")
    generator.save_metadata(output_dir / "metadata.json")
    metrics = generator.evaluate(real, synthetic, compute_c2st=args.compute_c2st)
    metadata = generator.metadata()
    metrics.update(
        {
            "temporal_shrinkage_mode": metadata["temporal_shrinkage_mode"],
            "alpha_customer_time_selected": metadata["alpha_customer_time_selected"],
            "alpha_product_time_selected": metadata["alpha_product_time_selected"],
            "alpha_selection_uses_synthetic_metrics": metadata["alpha_selection_uses_synthetic_metrics"],
            "pairing_mode": metadata["pairing_mode"],
            "pairing_penalties_fixed_defaults": metadata["pairing_penalties_fixed_defaults"],
        }
    )
    write_metrics(metrics, output_dir / "metrics.json")
    return metrics


if __name__ == "__main__":
    main()
