#!/usr/bin/env python3
"""Run the ultrafast low-rank temporal event-spine generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.fast_event_spine_metrics import evaluate_fast_event_spine, write_metrics  # noqa: E402
from generators.ultrafast_lowrank_temporal_event import UltraFastLowRankTemporalEventGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ultrafast slot-based low-rank temporal event spine.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-granularity", default="day", choices=["day"])
    parser.add_argument("--time-gate-granularity", default="month", choices=["day", "month"])
    parser.add_argument("--block-pair-time-mode", choices=["exact", "sampled", "none"], default="exact")
    parser.add_argument("--preserve-degrees", action="store_true", default=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha-customer-time", default="auto")
    parser.add_argument("--alpha-product-time", default="auto")
    parser.add_argument("--alpha-time-gate", default="auto")
    parser.add_argument("--block-time-smoothing", type=float, default=5.0)
    parser.add_argument(
        "--pairing-mode",
        choices=["random", "static_projection", "dynamic_projection", "dynamic_exact_small"],
        default="dynamic_projection",
    )
    parser.add_argument("--max-exact-affinity-cell-size", type=int, default=128)
    parser.add_argument("--enable-degree-repair", action="store_true", default=True)
    parser.add_argument("--disable-degree-repair", dest="enable_degree_repair", action="store_false")
    parser.add_argument("--enable-fast-overlap-repair", action="store_true", default=True)
    parser.add_argument("--disable-fast-overlap-repair", dest="enable_fast_overlap_repair", action="store_false")
    parser.add_argument("--repair-max-passes", type=int, default=3)
    parser.add_argument("--fast-repair-attempts", type=int, default=10)
    parser.add_argument("--allow-degree-slack", action="store_true", default=False)
    parser.add_argument("--compute-c2st", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.real_reviews)
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = UltraFastLowRankTemporalEventGenerator(
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        structure_debug_dir=args.structure_debug_dir,
        time_granularity=args.time_granularity,
        time_gate_granularity=args.time_gate_granularity,
        block_pair_time_mode=args.block_pair_time_mode,
        preserve_degrees=args.preserve_degrees,
        rank=args.rank,
        alpha_customer_time=args.alpha_customer_time,
        alpha_product_time=args.alpha_product_time,
        alpha_time_gate=args.alpha_time_gate,
        block_time_smoothing=args.block_time_smoothing,
        pairing_mode=args.pairing_mode,
        max_exact_affinity_cell_size=args.max_exact_affinity_cell_size,
        enable_degree_repair=args.enable_degree_repair,
        enable_fast_overlap_repair=args.enable_fast_overlap_repair,
        repair_max_passes=args.repair_max_passes,
        fast_repair_attempts=args.fast_repair_attempts,
        allow_degree_slack=args.allow_degree_slack,
        seed=args.seed,
    )
    synthetic = generator.fit(real).sample(seed=args.seed)
    synthetic_path = output_dir / "synthetic_review.csv"
    synthetic.to_csv(synthetic_path, index=False)
    generator.save_debug(debug_dir)
    generator.save_metadata(output_dir / "metadata.json")
    metrics = evaluate_fast_event_spine(
        real,
        synthetic,
        structure_debug_dir=debug_dir,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        compute_c2st=args.compute_c2st,
        metadata=generator.metadata(),
    )
    write_metrics(metrics, output_dir / "metrics.json")
    print(f"[done] wrote {synthetic_path}")
    print(f"[done] wrote {output_dir / 'metadata.json'}")
    print(f"[done] wrote {output_dir / 'metrics.json'}")
    print(f"[done] debug files in {debug_dir}")


if __name__ == "__main__":
    main()
