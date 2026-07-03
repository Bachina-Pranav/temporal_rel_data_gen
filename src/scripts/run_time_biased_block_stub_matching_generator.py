#!/usr/bin/env python3
"""Run the time-biased exact block-stub matching event-spine generator."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.fast_event_spine_metrics import evaluate_fast_event_spine, write_metrics  # noqa: E402
from generators.time_biased_block_stub_matching import TimeBiasedBlockStubMatchingGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a time-biased block-stub matched temporal event spine.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-granularity", default="day", choices=["day"])
    parser.add_argument("--time-gate-granularity", default="month", choices=["day", "month"])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha-customer-time", default="auto")
    parser.add_argument("--alpha-product-time", default="auto")
    parser.add_argument(
        "--temporal-shrinkage-mode",
        choices=["median_degree", "empirical_bayes", "fixed"],
        default="median_degree",
    )
    parser.add_argument(
        "--desired-time-sampling-mode",
        choices=["mixture_shrinkage", "empirical_bayes", "local_kernel", "empirical_exact"],
        default="local_kernel",
    )
    parser.add_argument("--alpha-time-gate", default="auto")
    parser.add_argument("--block-time-smoothing", type=float, default=5.0)
    parser.add_argument("--kernel-bandwidth-mode", choices=["auto_block_iqr", "auto_global_iqr", "fixed"], default="auto_block_iqr")
    parser.add_argument("--kernel-bandwidth-scale", type=float, default=0.25)
    parser.add_argument("--kernel-min-bandwidth-days", type=float, default=1.0)
    parser.add_argument("--kernel-max-bandwidth-days", type=float, default=None)
    parser.add_argument("--kernel-fixed-bandwidth-days", type=float, default=7.0)
    parser.add_argument("--kernel-type", choices=["discrete_laplace", "discrete_gaussian", "none"], default="discrete_laplace")
    parser.add_argument(
        "--pairing-mode",
        choices=["random", "static_projection", "dynamic_projection", "dynamic_exact_small", "dynamic_exact_penalized"],
        default="dynamic_exact_penalized",
    )
    parser.add_argument("--max-exact-affinity-cell-size", type=int, default=128)
    parser.add_argument("--large-cell-pairing", choices=["projection_sort", "exact_greedy"], default="projection_sort")
    parser.add_argument("--large-cell-local-swap-attempts", type=int, default=2)
    parser.add_argument("--desired-time-jitter", type=float, default=1e-3)
    parser.add_argument("--verbose-block-logs", action="store_true")
    parser.add_argument("--enable-fast-overlap-repair", action="store_true", default=False)
    parser.add_argument("--overlap-resample-prob", type=float, default=0.0)
    parser.add_argument("--lambda-duplicate-pair", type=float, default=1.0)
    parser.add_argument("--lambda-real-pair-overlap", type=float, default=1.0)
    parser.add_argument("--lambda-exact-event-overlap", type=float, default=3.0)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compute-c2st", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.real_reviews)
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    generator = TimeBiasedBlockStubMatchingGenerator(
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        dataset=infer_dataset(args.dataset, args.real_reviews),
        structure_debug_dir=args.structure_debug_dir,
        time_granularity=args.time_granularity,
        time_gate_granularity=args.time_gate_granularity,
        rank=args.rank,
        alpha_customer_time=args.alpha_customer_time,
        alpha_product_time=args.alpha_product_time,
        temporal_shrinkage_mode=args.temporal_shrinkage_mode,
        desired_time_sampling_mode=args.desired_time_sampling_mode,
        alpha_time_gate=args.alpha_time_gate,
        block_time_smoothing=args.block_time_smoothing,
        kernel_bandwidth_mode=args.kernel_bandwidth_mode,
        kernel_bandwidth_scale=args.kernel_bandwidth_scale,
        kernel_min_bandwidth_days=args.kernel_min_bandwidth_days,
        kernel_max_bandwidth_days=args.kernel_max_bandwidth_days,
        kernel_fixed_bandwidth_days=args.kernel_fixed_bandwidth_days,
        kernel_type=args.kernel_type,
        pairing_mode=args.pairing_mode,
        max_exact_affinity_cell_size=args.max_exact_affinity_cell_size,
        large_cell_pairing=args.large_cell_pairing,
        large_cell_local_swap_attempts=args.large_cell_local_swap_attempts,
        desired_time_jitter=args.desired_time_jitter,
        verbose_block_logs=args.verbose_block_logs,
        enable_fast_overlap_repair=args.enable_fast_overlap_repair,
        overlap_resample_prob=args.overlap_resample_prob,
        lambda_duplicate_pair=args.lambda_duplicate_pair,
        lambda_real_pair_overlap=args.lambda_real_pair_overlap,
        lambda_exact_event_overlap=args.lambda_exact_event_overlap,
        seed=args.seed,
    )
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        synthetic = generator.fit(real).sample(seed=args.seed)
        profiler.disable()
        write_profile_outputs(profiler, debug_dir)
    else:
        synthetic = generator.fit(real).sample(seed=args.seed)
    synthetic_path = output_dir / "synthetic_review.csv"
    synthetic.to_csv(synthetic_path, index=False)
    generator.save_debug(debug_dir)
    generator.save_metadata(output_dir / "metadata.json")
    if args.skip_evaluation:
        print("[evaluation] skipped by --skip-evaluation")
        print(f"[done] wrote {synthetic_path}")
        print(f"[done] wrote {output_dir / 'metadata.json'}")
        print(f"[done] debug files in {debug_dir}")
        return
    print("[eval] computing metrics", flush=True)
    evaluation_start = time.time()
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
    metrics.update(generator.dynamic_affinity_diagnostics(real, synthetic))
    metrics["evaluation_seconds"] = float(time.time() - evaluation_start)
    write_metrics(metrics, output_dir / "metrics.json")
    print(f"[eval] done in {metrics['evaluation_seconds']:.2f}s")
    print(f"[done] wrote {synthetic_path}")
    print(f"[done] wrote {output_dir / 'metadata.json'}")
    print(f"[done] wrote {output_dir / 'metrics.json'}")
    print(f"[done] debug files in {debug_dir}")


def write_profile_outputs(profiler: cProfile.Profile, debug_dir: Path) -> None:
    profile_path = debug_dir / "profile_generation.prof"
    profile_text_path = debug_dir / "profile_generation_top.txt"
    profiler.dump_stats(profile_path)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(50)
    profile_text_path.write_text(stream.getvalue())
    print(f"[profile] wrote {profile_path}")
    print(f"[profile] wrote {profile_text_path}")


def infer_dataset(dataset: str | None, real_reviews: str) -> str | None:
    if dataset:
        return dataset
    text = str(real_reviews)
    if "amazon-toy" in text or "rel-amazon-toy" in text:
        return "rel-amazon-toy"
    if "rel-amazon" in text:
        return "rel-amazon"
    return None


if __name__ == "__main__":
    main()
