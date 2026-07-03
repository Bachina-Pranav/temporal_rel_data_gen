#!/usr/bin/env python3
"""Evaluate a time-biased block-stub matching event spine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.fast_event_spine_metrics import evaluate_fast_event_spine, load_metadata, write_metrics  # noqa: E402
from generators.fast_temporal_activity import canonical_time_bucket  # noqa: E402
from generators.lowrank_time_gated_affinity import LowRankTimeGatedAffinity  # noqa: E402
from generators.time_biased_block_stub_matching import ks_stat  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate time-biased block-stub matching event-spine metrics.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--time-granularity", default="day", choices=["day"])
    parser.add_argument("--time-gate-granularity", default=None, choices=["day", "month"])
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha-time-gate", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dynamic-affinity", action="store_true")
    parser.add_argument("--compute-c2st", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.real_reviews)
    synthetic = pd.read_csv(args.synthetic_reviews)
    metadata = load_metadata(args.metadata)
    metrics = evaluate_fast_event_spine(
        real,
        synthetic,
        structure_debug_dir=args.structure_debug_dir,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        compute_c2st=args.compute_c2st,
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
                args.time_gate_granularity or metadata_value(metadata, "time_gate_granularity", "month"),
                args.rank if args.rank is not None else int(metadata_value(metadata, "rank", 32)),
                args.alpha_time_gate if args.alpha_time_gate is not None else metadata_value(metadata, "alpha_time_gate", "auto"),
                args.seed,
            )
        )
    write_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2))


def dynamic_affinity_diagnostics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    time_granularity: str,
    time_gate_granularity: str,
    rank: int,
    alpha_time_gate: Any,
    seed: int,
) -> Dict[str, float]:
    real_frame = real[[customer_col, product_col, timestamp_col]].copy()
    synthetic_frame = synthetic[[customer_col, product_col, timestamp_col]].copy()
    real_frame[timestamp_col] = canonical_time_bucket(real_frame[timestamp_col], time_granularity)
    synthetic_frame[timestamp_col] = canonical_time_bucket(synthetic_frame[timestamp_col], time_granularity)
    affinity = LowRankTimeGatedAffinity(
        rank=rank,
        alpha_time_gate=alpha_time_gate,
        time_gate_granularity=time_gate_granularity,
        seed=seed,
    ).fit(real_frame, customer_col, product_col, timestamp_col)
    real_scores = score_pairs_by_time(affinity, real_frame, customer_col, product_col, timestamp_col)
    synthetic_scores = score_pairs_by_time(affinity, synthetic_frame, customer_col, product_col, timestamp_col)
    return {
        "mean_dynamic_affinity_real": float(np.mean(real_scores)) if len(real_scores) else 0.0,
        "mean_dynamic_affinity_synthetic": float(np.mean(synthetic_scores)) if len(synthetic_scores) else 0.0,
        "median_dynamic_affinity_real": float(np.median(real_scores)) if len(real_scores) else 0.0,
        "median_dynamic_affinity_synthetic": float(np.median(synthetic_scores)) if len(synthetic_scores) else 0.0,
        "dynamic_affinity_distribution_ks": ks_stat(real_scores, synthetic_scores),
    }


def score_pairs_by_time(
    affinity: LowRankTimeGatedAffinity,
    frame: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
) -> np.ndarray:
    scores = []
    for time_bucket, group in frame.groupby(timestamp_col, sort=False):
        scores.append(
            affinity.score_pairs(
                group[customer_col].to_numpy(dtype=object),
                group[product_col].to_numpy(dtype=object),
                time_bucket,
            )
        )
    return np.concatenate(scores) if scores else np.asarray([], dtype=float)


def metadata_value(metadata: Dict[str, Any] | None, key: str, default: Any) -> Any:
    return metadata.get(key, default) if metadata else default


if __name__ == "__main__":
    main()
