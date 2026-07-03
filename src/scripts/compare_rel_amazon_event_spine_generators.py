#!/usr/bin/env python3
"""Compare full Rel-Amazon event-spine generator outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from generators.fast_event_spine_metrics import evaluate_fast_event_spine, load_metadata, write_metrics  # noqa: E402
from evaluate_time_biased_block_stub_matching import dynamic_affinity_diagnostics, metadata_value  # noqa: E402


DEFAULT_RUNS = [
    ("static_degree", "outputs/rel-amazon/static_degree/synthetic_review.csv"),
    ("ct_2k_sbm_temporal_kde_stubs", "outputs/rel-amazon/ct_2k_sbm_temporal_kde_stubs/synthetic_review.csv"),
    ("time_biased_median_mixture", "outputs/rel-amazon/time_biased_block_stub_matching_median_mixture/synthetic_review.csv"),
    ("time_biased_empirical_exact", "outputs/rel-amazon/time_biased_block_stub_matching_empirical_exact/synthetic_review.csv"),
    ("time_biased_local_kernel_main", "outputs/rel-amazon/time_biased_block_stub_matching_kernel_main/synthetic_review.csv"),
]

COMPARISON_KEYS = [
    "num_reviews_synthetic",
    "customer_degree_ks",
    "product_degree_ks",
    "customer_degree_exact_match",
    "product_degree_exact_match",
    "daily_count_l1",
    "block_pair_time_count_l1",
    "product_first_time_corr",
    "product_last_time_corr",
    "product_peak_time_corr",
    "product_relative_age_ks",
    "product_time_activity_distribution_ks",
    "customer_first_time_corr",
    "customer_last_time_corr",
    "customer_peak_time_corr",
    "customer_relative_age_ks",
    "customer_time_activity_distribution_ks",
    "joint_coactive_window_rate",
    "duplicate_customer_product_rate",
    "real_edge_overlap_rate",
    "exact_event_overlap_rate",
    "mean_dynamic_affinity_synthetic",
    "dynamic_affinity_distribution_ks",
    "total_seconds",
    "events_per_second",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full Rel-Amazon event-spine generator outputs.")
    parser.add_argument("--real-reviews", default="data/original/rel-amazon/review.csv")
    parser.add_argument("--structure-debug-dir", default="outputs/rel-amazon/ct_2k_sbm_temporal_kde_stubs/debug")
    parser.add_argument("--runs", nargs="*", default=None, help="Optional name=synthetic_review.csv entries.")
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--time-granularity", default="day", choices=["day"])
    parser.add_argument("--time-gate-granularity", default="month", choices=["day", "month"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha-time-gate", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dynamic-affinity", action="store_true")
    parser.add_argument("--reuse-existing-metrics", action="store_true")
    parser.add_argument("--output-json", default="outputs/rel-amazon/event_spine_generator_comparison.json")
    parser.add_argument("--output-csv", default="outputs/rel-amazon/event_spine_generator_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_path = Path(args.real_reviews)
    if not real_path.exists():
        raise SystemExit(f"Missing real review CSV: {real_path}")
    real = pd.read_csv(real_path)
    rows = []
    nested: Dict[str, Dict[str, Any]] = {}
    for name, synthetic_path in parse_runs(args.runs):
        path = Path(synthetic_path)
        if not path.exists():
            print(f"WARNING: skipping missing output for {name}: {path}", file=sys.stderr)
            continue
        metadata_path = path.parent / "metadata.json"
        metrics_path = path.parent / "metrics.json"
        metadata = load_metadata(metadata_path)
        if args.reuse_existing_metrics and metrics_path.exists():
            with metrics_path.open() as handle:
                metrics = json.load(handle)
        else:
            print(f"[compare] evaluating {name}: {path}", flush=True)
            synthetic = pd.read_csv(path)
            metrics = evaluate_fast_event_spine(
                real,
                synthetic,
                structure_debug_dir=args.structure_debug_dir,
                customer_col=args.customer_id_col,
                product_col=args.product_id_col,
                timestamp_col=args.timestamp_col,
                compute_c2st=False,
                metadata=metadata,
            )
            if not args.skip_dynamic_affinity:
                metrics.update(
                    dynamic_affinity_diagnostics(
                        real,
                        synthetic,
                        args.customer_id_col,
                        args.product_id_col,
                        args.timestamp_col,
                        args.time_granularity,
                        metadata_value(metadata, "time_gate_granularity", args.time_gate_granularity),
                        int(metadata_value(metadata, "rank", args.rank)),
                        metadata_value(metadata, "alpha_time_gate", args.alpha_time_gate),
                        args.seed,
                    )
                )
            write_metrics(metrics, path.parent / "comparison_eval_metrics.json")
        nested[name] = metrics
        rows.append({"model": name, **{key: metrics.get(key) for key in COMPARISON_KEYS}})
    if not rows:
        raise SystemExit("No comparison outputs were found.")
    write_json(nested, args.output_json)
    table = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    print(table.to_string(index=False))
    print(f"[done] wrote {args.output_json}")
    print(f"[done] wrote {args.output_csv}")


def parse_runs(items: Iterable[str] | None) -> list[tuple[str, str]]:
    if not items:
        return list(DEFAULT_RUNS)
    runs = []
    for item in items:
        name, path = item.split("=", 1) if "=" in item else (Path(item).parent.name, item)
        runs.append((name, path))
    return runs


def write_json(data: Dict[str, Dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
