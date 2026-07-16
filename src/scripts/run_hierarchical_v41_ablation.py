#!/usr/bin/env python3
"""Run hierarchical v4.1 graph/structured-conditioning ablations."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from attribute_generation.conditional_tabdlm.hierarchical_sample import hierarchical_sample_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics, load_yaml  # noqa: E402
from evaluation.paper_metrics.reporting import write_markdown_report  # noqa: E402
from evaluation.paper_metrics.utils import ensure_dir, write_json  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_hierarchical_v41.yaml"
DEFAULT_EVAL_CONFIG = "configs/evaluation/single_event_table_paper_metrics_amazon_toy.yaml"
DEFAULT_REAL = "data/original/rel-amazon-toy/review.csv"
DEFAULT_SPINE = "outputs/amazon-toy/time_biased_block_stub_matching_kernel_main/synthetic_review.csv"
DEFAULT_OUTPUT_ROOT = "outputs/amazon-toy/conditional_tabdlm_hierarchical_v41"


VARIANTS = {
    "hier_correct_graph": {"graph_mode": "correct", "oracle": False, "label": "valid primary model"},
    "hier_no_graph": {"graph_mode": "no_graph", "oracle": False, "label": "no graph ablation"},
    "hier_zero_graph": {"graph_mode": "zero", "oracle": False, "label": "zero graph ablation"},
    "hier_shuffled_graph": {"graph_mode": "shuffled", "oracle": False, "label": "shuffled graph ablation"},
    "hier_oracle_structured": {"graph_mode": "correct", "oracle": True, "label": "NOT A VALID GENERATIVE BASELINE"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hierarchical v4.1 ablations.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real-table", default=DEFAULT_REAL)
    parser.add_argument("--synthetic-spine", default=DEFAULT_SPINE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--num-rows", default=5000)
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--structured-steps", default=10)
    parser.add_argument("--text-steps", default=25)
    parser.add_argument("--text-top-k", type=int, default=512)
    parser.add_argument("--inference-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    eval_config = load_yaml(args.eval_config)
    output_root = ensure_dir(args.output_root)
    rows = []
    for name in args.variants:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant {name!r}; choices: {sorted(VARIANTS)}")
        spec = VARIANTS[name]
        run_dir = ensure_dir(output_root / "runs" / name)
        synthetic_path = run_dir / "synthetic_review_attrs.csv"
        eval_dir = run_dir / "evaluation" / "paper_grade"
        if not args.skip_sampling and not (args.reuse_existing and synthetic_path.exists()):
            hierarchical_sample_from_config(
                config,
                checkpoint_path=args.checkpoint,
                synthetic_spine_path=args.synthetic_spine,
                output_path=synthetic_path,
                num_rows=args.num_rows,
                sample_batch_size=args.sample_batch_size,
                structured_steps=args.structured_steps,
                text_steps=args.text_steps,
                text_top_k=args.text_top_k,
                graph_mode_override=spec["graph_mode"],
                inference_dtype=args.inference_dtype,
                device=args.device,
                seed=args.seed,
                profile=True,
                profile_output=run_dir / "metadata" / "runtime_hierarchical_sampling.json",
                oracle_structured_table_path=args.real_table if spec["oracle"] else None,
                debug_write_aux_targets=True,
            )
        if not args.skip_evaluation:
            metrics = run_paper_eval(eval_config, args.real_table, synthetic_path, eval_dir, args.seed, args.num_rows)
        else:
            metrics = load_json(eval_dir / "paper_metrics.json")
        runtime = load_json(run_dir / "metadata" / "runtime_hierarchical_sampling.json")
        rows.append(comparison_row(name, spec, synthetic_path, eval_dir / "paper_metrics.json", metrics, runtime))
    pd.DataFrame(rows).to_csv(output_root / "comparison.csv", index=False)
    write_json({"runs": rows}, output_root / "comparison.json")
    write_results_markdown(output_root / "comparison.md", rows)
    print(f"Wrote {output_root / 'comparison.csv'}")


def run_paper_eval(template: dict[str, Any], real_table: str | Path, synthetic_path: Path, output_dir: Path, seed: int, sample_size: Any) -> dict[str, Any]:
    config = copy.deepcopy(template)
    config["real_table_path"] = str(real_table)
    config["synthetic_table_path"] = str(synthetic_path)
    config.setdefault("evaluation", {})["random_seed"] = int(seed)
    if str(sample_size).lower() not in {"all", "none"}:
        config.setdefault("evaluation", {})["sample_size"] = int(sample_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_paper_metrics(config, output_dir)
    write_json(metrics, output_dir / "metrics.json")
    write_json(metrics, output_dir / "paper_metrics.json")
    write_markdown_report(metrics, output_dir / "metrics.md")
    return metrics


def comparison_row(name: str, spec: dict[str, Any], synthetic_path: Path, metrics_path: Path, metrics: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    summary = metrics.get("paper_metrics_summary", {}) or {}
    shape = metrics.get("shape", {}).get("per_column", {}) or {}
    return {
        "model": name,
        "graph": spec["graph_mode"],
        "structured_conditioning": "oracle" if spec["oracle"] else "generated",
        "not_valid_generative_baseline": bool(spec["oracle"]),
        "label": spec["label"],
        "shape_error": summary.get("shape_error"),
        "table_c2st_error": summary.get("single_table_c2st_error"),
        "text_c2st_error": summary.get("text_embedding_c2st_error"),
        "trend_error": summary.get("trend_error"),
        "review_text_token_length_ks": (shape.get("review_text") or {}).get("shape_error"),
        "summary_token_length_ks": (shape.get("summary") or {}).get("shape_error"),
        "runtime_seconds": runtime.get("total_sampling_seconds"),
        "rows_per_second": runtime.get("rows_per_second"),
        "synthetic_path": str(synthetic_path),
        "metrics_path": str(metrics_path),
    }


def write_results_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Hierarchical v4.1 Ablation Results",
        "",
        "| Model | Graph | Structured conditioning | Shape ↓ | Table C2ST ↓ | Text C2ST ↓ | Trend ↓ | Review KS ↓ | Runtime |",
        "| ----- | ----- | ----------------------- | ------: | -----------: | ----------: | ------: | ----------: | ------: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {graph} | {cond} | {shape} | {table} | {text} | {trend} | {review} | {runtime} |".format(
                model=row["model"],
                graph=row["graph"],
                cond=row["structured_conditioning"],
                shape=fmt(row.get("shape_error")),
                table=fmt(row.get("table_c2st_error")),
                text=fmt(row.get("text_c2st_error")),
                trend=fmt(row.get("trend_error")),
                review=fmt(row.get("review_text_token_length_ks")),
                runtime=fmt(row.get("runtime_seconds")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
