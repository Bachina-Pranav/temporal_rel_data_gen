#!/usr/bin/env python3
"""Run the joint temporal 2K-SBM event-spine generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.event_spine_metrics import evaluate_event_spine, write_metrics  # noqa: E402
from generators.joint_temporal_2k_sbm_event import JointTemporal2KSBMEventGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a joint temporal event spine. Recommended scalable default: "
            "--sampling-mode fast_time_conditioned --alpha-customer-time auto --alpha-product-time auto."
        )
    )
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-granularity", choices=["day"], default="day")
    parser.add_argument("--preserve-daily-counts", action="store_true", default=True)
    parser.add_argument("--preserve-block-pair-time-counts", action="store_true", default=True)
    parser.add_argument("--no-preserve-block-pair-time-counts", dest="preserve_block_pair_time_counts", action="store_false")
    parser.add_argument("--preserve-degrees", action="store_true", default=True)
    parser.add_argument("--no-preserve-degrees", dest="preserve_degrees", action="store_false")
    parser.add_argument("--sample-block-pair-time-counts", action="store_true")
    parser.add_argument("--sampling-mode", choices=["fast_time_conditioned", "candidate_affinity"], default="fast_time_conditioned")
    parser.add_argument("--alpha-customer-time", default="auto")
    parser.add_argument("--alpha-product-time", default="auto")
    parser.add_argument("--block-time-smoothing", type=float, default=5.0)
    parser.add_argument("--age-smoothing", type=float, default=5.0)
    parser.add_argument("--mf-rank", type=int, default=32)
    parser.add_argument("--lambda-static", type=float, default=1.0)
    parser.add_argument("--lambda-ut", type=float, default=1.0)
    parser.add_argument("--lambda-it", type=float, default=1.0)
    parser.add_argument("--lambda-age", type=float, default=0.5)
    parser.add_argument("--lambda-deg", type=float, default=0.1)
    parser.add_argument("--lambda-dup", type=float, default=1.0)
    parser.add_argument("--lambda-mem", type=float, default=2.0)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--customer-candidate-pool-size", type=int, default=256)
    parser.add_argument("--product-candidate-pool-size", type=int, default=256)
    parser.add_argument("--use-static-affinity", action="store_true")
    parser.add_argument("--use-product-lifecycle-affinity", action="store_true")
    parser.add_argument("--enable-fast-repair", action="store_true", default=True)
    parser.add_argument("--disable-fast-repair", dest="enable_fast_repair", action="store_false")
    parser.add_argument("--fast-repair-attempts", type=int, default=10)
    parser.add_argument("--allow-degree-slack", action="store_true")
    parser.add_argument("--compute-c2st", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_block_pair_time_counts:
        raise NotImplementedError("--sample-block-pair-time-counts is reserved for a future smoothed count mode.")
    real = pd.read_csv(args.real_reviews)
    alpha_customer_time = resolve_alpha(args.alpha_customer_time, real[args.customer_id_col].value_counts(), "customer")
    alpha_product_time = resolve_alpha(args.alpha_product_time, real[args.product_id_col].value_counts(), "product")
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = JointTemporal2KSBMEventGenerator(
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        structure_debug_dir=args.structure_debug_dir,
        time_granularity=args.time_granularity,
        alpha_customer_time=alpha_customer_time,
        alpha_product_time=alpha_product_time,
        block_time_smoothing=args.block_time_smoothing,
        age_smoothing=args.age_smoothing,
        mf_rank=args.mf_rank,
        lambda_static=args.lambda_static,
        lambda_ut=args.lambda_ut,
        lambda_it=args.lambda_it,
        lambda_age=args.lambda_age,
        lambda_deg=args.lambda_deg,
        lambda_dup=args.lambda_dup,
        lambda_mem=args.lambda_mem,
        sampling_temperature=args.sampling_temperature,
        customer_candidate_pool_size=args.customer_candidate_pool_size,
        product_candidate_pool_size=args.product_candidate_pool_size,
        sampling_mode=args.sampling_mode,
        use_static_affinity=args.use_static_affinity,
        use_product_lifecycle_affinity=args.use_product_lifecycle_affinity,
        enable_fast_repair=args.enable_fast_repair,
        fast_repair_attempts=args.fast_repair_attempts,
        preserve_block_pair_time_counts=args.preserve_block_pair_time_counts,
        preserve_degrees=args.preserve_degrees,
        allow_degree_slack=args.allow_degree_slack,
        seed=args.seed,
    )
    synthetic = generator.fit(real).sample(seed=args.seed)
    synthetic_path = output_dir / "synthetic_review.csv"
    synthetic.to_csv(synthetic_path, index=False)
    generator.save_debug(debug_dir)
    generator.save_metadata(output_dir / "metadata.json")
    metrics = evaluate_event_spine(
        real,
        synthetic,
        structure_debug_dir=debug_dir,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        compute_c2st=args.compute_c2st,
    )
    write_metrics(metrics, output_dir / "metrics.json")
    print(f"Wrote {synthetic_path}")
    print(f"Wrote {output_dir / 'metadata.json'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote debug files in {debug_dir}")


def resolve_alpha(value, degree_counts: pd.Series, label: str) -> float:
    if str(value).lower() == "auto":
        resolved = float(np.median(degree_counts.to_numpy(dtype=float))) if len(degree_counts) else 1.0
        print(f"[config] alpha_{label}_time=auto -> {resolved:.4g}", flush=True)
        return max(resolved, 1e-6)
    return float(value)


if __name__ == "__main__":
    main()
