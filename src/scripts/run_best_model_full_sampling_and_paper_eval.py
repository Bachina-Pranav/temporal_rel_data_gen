#!/usr/bin/env python3
"""Sample the best model at full event-table size, then run paper metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_single_event_table_paper_metrics import (  # noqa: E402
    evaluate_paper_metrics,
    load_yaml,
    write_legacy_metrics,
)
from evaluation.paper_metrics.reporting import MAIN_ROWS, fmt, write_markdown_report  # noqa: E402
from evaluation.paper_metrics.utils import ensure_dir, write_json  # noqa: E402
from run_sample_best_model_full_event_table import run_full_sampling  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-size best-model sampling and paper-grade evaluation.")
    parser.add_argument("--real-table", required=True)
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--sampler-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--sample-output", required=True)
    parser.add_argument("--eval-output-dir", required=True)
    parser.add_argument("--num-rows", default="auto")
    parser.add_argument("--profile-output", default=None)
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--write-chunk-size", type=int, default=10000)
    parser.set_defaults(
        length_preserving_exact_blocking=True,
        auto_batch_size=True,
        mixed_precision=True,
        disable_review_text_ngram_blocking=True,
    )
    parser.add_argument("--length-preserving-exact-blocking", dest="length_preserving_exact_blocking", action="store_true")
    parser.add_argument("--no-length-preserving-exact-blocking", dest="length_preserving_exact_blocking", action="store_false")
    parser.add_argument("--disable-review-text-ngram-blocking", dest="disable_review_text_ngram_blocking", action="store_true")
    parser.add_argument("--enable-review-text-ngram-blocking", dest="disable_review_text_ngram_blocking", action="store_false")
    parser.add_argument("--auto-batch-size", dest="auto_batch_size", action="store_true")
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_output = args.profile_output or str(Path(args.sample_output).parent / "metadata" / "runtime_sampling_full.json")
    sample_args = argparse.Namespace(
        real_table=args.real_table,
        synthetic_spine=args.synthetic_spine,
        config=args.sampler_config,
        checkpoint=args.checkpoint,
        output=args.sample_output,
        profile_output=profile_output,
        num_rows=args.num_rows,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        length_preserving_exact_blocking=args.length_preserving_exact_blocking,
        disable_review_text_ngram_blocking=args.disable_review_text_ngram_blocking,
        auto_batch_size=args.auto_batch_size,
        mixed_precision=args.mixed_precision,
        write_chunk_size=args.write_chunk_size,
    )
    sampled_path = run_full_sampling(sample_args)
    output_dir = ensure_dir(args.eval_output_dir)
    config = load_yaml(args.eval_config)
    config["real_table_path"] = args.real_table
    config["synthetic_table_path"] = str(sampled_path)
    metrics = evaluate_paper_metrics(config, output_dir)
    write_json(metrics, output_dir / "metrics.json")
    write_json(metrics, output_dir / "paper_metrics.json")
    write_markdown_report(metrics, output_dir / "metrics.md")
    write_legacy_metrics(config, output_dir)
    print_dashboard(metrics)


def print_dashboard(metrics: dict[str, Any]) -> None:
    summary = metrics.get("paper_metrics_summary", {})
    print("# Main Dashboard")
    for _, label, key, _ in MAIN_ROWS:
        print(f"{label}: {fmt(summary.get(key))}")


if __name__ == "__main__":
    main()
