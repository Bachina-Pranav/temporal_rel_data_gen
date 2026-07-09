#!/usr/bin/env python3
"""Run full Rel-Amazon sampling and paper-grade single-event-table evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics, load_yaml, write_legacy_metrics  # noqa: E402
from evaluation.paper_metrics.reporting import write_markdown_report  # noqa: E402
from evaluation.paper_metrics.utils import ensure_dir, write_json  # noqa: E402
from rel_amazon_pipeline_utils import count_csv_rows, dashboard_payload, print_main_dashboard, read_json  # noqa: E402
from run_sample_best_model_full_event_table import run_full_sampling  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run current best v5.3 pipeline on full Rel-Amazon.")
    parser.add_argument("--real-table", required=True)
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--sampler-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--sample-output", required=True)
    parser.add_argument("--runtime-output", required=True)
    parser.add_argument("--eval-output-dir", required=True)
    parser.add_argument("--num-rows", default="auto")
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--write-chunk-size", type=int, default=100000)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(length_preserving_exact_blocking=True, auto_batch_size=True, mixed_precision=True)
    parser.add_argument("--length-preserving-exact-blocking", dest="length_preserving_exact_blocking", action="store_true")
    parser.add_argument("--no-length-preserving-exact-blocking", dest="length_preserving_exact_blocking", action="store_false")
    parser.add_argument("--disable-review-text-ngram-blocking", action="store_true")
    parser.add_argument("--auto-batch-size", dest="auto_batch_size", action="store_true")
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rel_amazon_full(args)


def run_rel_amazon_full(args: argparse.Namespace) -> dict[str, int | str | bool]:
    real_rows = count_csv_rows(args.real_table)
    spine_rows = count_csv_rows(args.synthetic_spine)
    if spine_rows < real_rows:
        raise SystemExit(
            f"Synthetic spine has {spine_rows} rows but real table has {real_rows}; refusing to run full sampling."
        )
    if args.dry_run:
        payload = {"status": "dry_run_ok", "real_rows": real_rows, "spine_rows": spine_rows, "would_sample_rows": real_rows}
        print(payload)
        return payload
    sample_args = argparse.Namespace(
        real_table=args.real_table,
        synthetic_spine=args.synthetic_spine,
        config=args.sampler_config,
        checkpoint=args.checkpoint,
        output=args.sample_output,
        profile_output=args.runtime_output,
        num_rows=args.num_rows,
        dataset_name="rel_amazon",
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
    runtime = read_json(args.runtime_output) if Path(args.runtime_output).exists() else {}
    write_json(dashboard_payload(metrics, runtime), output_dir / "main_dashboard.json")
    print_main_dashboard(metrics)
    return {"status": "complete", "real_rows": real_rows, "spine_rows": spine_rows, "sample_output": str(sampled_path)}


if __name__ == "__main__":
    main()
