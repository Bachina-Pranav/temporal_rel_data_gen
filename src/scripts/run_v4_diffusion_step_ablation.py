#!/usr/bin/env python3
"""Run controlled v4 masked-diffusion step and length diagnostics."""

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

from attribute_generation.conditional_tabdlm.evaluate import evaluate_from_config as legacy_evaluate_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.sample import resolve_sampling_steps, sample_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config as load_sampler_config  # noqa: E402
from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics, load_yaml  # noqa: E402
from evaluation.paper_metrics.reporting import write_markdown_report  # noqa: E402
from evaluation.paper_metrics.utils import ensure_dir, write_json  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml"
DEFAULT_EVAL_CONFIG = "configs/evaluation/single_event_table_paper_metrics_amazon_toy.yaml"
DEFAULT_CHECKPOINT = "outputs/amazon-toy/conditional_tabdlm_exp4_v2_full_review_text/checkpoints/best.pt"
DEFAULT_SPINE = "outputs/amazon-toy/time_biased_block_stub_matching_kernel_main/synthetic_review.csv"
DEFAULT_REAL = "data/original/rel-amazon-toy/review.csv"
DEFAULT_OUTPUT_ROOT = "outputs/amazon-toy/diffusion_step_ablation"


IMPORTANT_TREND_PAIRS = {
    ("review_time", "rating"),
    ("review_time", "verified"),
    ("rating", "verified"),
    ("rating", "summary"),
    ("rating", "review_text"),
    ("verified", "summary"),
    ("verified", "review_text"),
    ("summary", "review_text"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v4 diffusion step-count and length-mode ablations.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--synthetic-spine", default=DEFAULT_SPINE)
    parser.add_argument("--real-table", default=DEFAULT_REAL)
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--diagnosis-doc", default="docs/v4_diffusion_sampler_diagnosis.md")
    parser.add_argument("--steps", nargs="+", default=["25", "50", "100", "250", "full"])
    parser.add_argument("--length-modes", nargs="+", choices=["normal", "empirical_length", "oracle_length"], default=["normal"])
    parser.add_argument("--num-rows", type=int, default=5000)
    parser.add_argument("--full-num-rows", type=int, default=None, help="Optional smaller row count for --steps full.")
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--text-top-k", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--timestep-spacing", choices=["uniform", "quadratic", "leading", "trailing"], default="uniform")
    parser.add_argument("--inference-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    sampler_config = load_sampler_config(args.config)
    eval_template = load_yaml(args.eval_config)
    rows: list[dict[str, Any]] = []

    for length_mode in args.length_modes:
        for requested_steps in args.steps:
            run_dir = output_root / run_name(requested_steps, length_mode, normal_suffix=len(args.length_modes) > 1)
            synthetic_path = run_dir / "synthetic_review_attrs.csv"
            profile_path = run_dir / "metadata" / "runtime_diffusion_sampling.json"
            eval_dir = run_dir / "evaluation" / "paper_grade"
            legacy_output = run_dir / "evaluation" / "legacy_diagnostic_metrics.json"
            row_count = run_num_rows(args, requested_steps)
            resolved_steps = resolve_sampling_steps(requested_steps, train_timesteps(sampler_config.raw))
            print(f"[run] steps={requested_steps} resolved={resolved_steps} length_mode={length_mode} rows={row_count}", flush=True)

            if not args.skip_sampling and not (args.reuse_existing and synthetic_path.exists()):
                sample_from_config(
                    sampler_config,
                    checkpoint_path=args.checkpoint,
                    synthetic_spine_path=args.synthetic_spine,
                    output_path=synthetic_path,
                    num_rows=row_count,
                    sample_batch_size=args.sample_batch_size,
                    sampling_steps=requested_steps,
                    timestep_spacing=args.timestep_spacing,
                    inference_dtype=args.inference_dtype,
                    text_top_k=args.text_top_k,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=args.device,
                    seed=args.seed,
                    profile=True,
                    profile_output=profile_path,
                    length_mode=length_mode,
                    oracle_real_table_path=args.real_table if length_mode == "oracle_length" else None,
                )

            if not synthetic_path.exists():
                rows.append(status_row(args, requested_steps, resolved_steps, length_mode, row_count, "missing_output", run_dir))
                continue

            if not args.skip_evaluation:
                run_paper_eval(eval_template, args.real_table, synthetic_path, eval_dir, seed=args.seed, sample_size=row_count)
                run_legacy_eval(sampler_config, args.real_table, synthetic_path, legacy_output)

            rows.append(
                comparison_row(
                    args=args,
                    requested_steps=requested_steps,
                    resolved_steps=resolved_steps,
                    length_mode=length_mode,
                    row_count=row_count,
                    run_dir=run_dir,
                    synthetic_path=synthetic_path,
                    profile_path=profile_path,
                    paper_metrics_path=eval_dir / "paper_metrics.json",
                    legacy_metrics_path=legacy_output,
                )
            )

    table = pd.DataFrame(rows)
    table.to_csv(output_root / "comparison.csv", index=False)
    write_json({"runs": rows}, output_root / "comparison.json")
    write_comparison_markdown(output_root / "comparison.md", rows)
    update_diagnosis_doc(Path(args.diagnosis_doc), rows)
    print(f"[done] wrote {output_root / 'comparison.csv'}", flush=True)


def run_num_rows(args: argparse.Namespace, requested_steps: str) -> int:
    if str(requested_steps).lower() == "full" and args.full_num_rows is not None:
        return int(args.full_num_rows)
    return int(args.num_rows)


def train_timesteps(raw_config: dict[str, Any]) -> int:
    diffusion = raw_config.get("diffusion", {}) or {}
    return int(diffusion.get("timesteps", diffusion.get("train_timesteps", 50)))


def run_name(requested_steps: str, length_mode: str, *, normal_suffix: bool = False) -> str:
    base = f"steps_{str(requested_steps).lower()}"
    if length_mode != "normal" or normal_suffix:
        return f"{base}_{length_mode}"
    return base


def run_paper_eval(
    eval_template: dict[str, Any],
    real_table: str | Path,
    synthetic_path: Path,
    output_dir: Path,
    seed: int,
    sample_size: int | None = None,
) -> None:
    config = copy.deepcopy(eval_template)
    config["real_table_path"] = str(real_table)
    config["synthetic_table_path"] = str(synthetic_path)
    config.setdefault("evaluation", {})["random_seed"] = int(seed)
    if sample_size is not None:
        config.setdefault("evaluation", {})["sample_size"] = int(sample_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_paper_metrics(config, output_dir)
    write_json(metrics, output_dir / "metrics.json")
    write_json(metrics, output_dir / "paper_metrics.json")
    write_markdown_report(metrics, output_dir / "metrics.md")


def run_legacy_eval(config: Any, real_table: str | Path, synthetic_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = legacy_evaluate_from_config(
        config,
        synthetic_reviews_path=synthetic_path,
        real_reviews_path=real_table,
        output_path=output_path,
        output_dir=output_path.parent / "legacy_diagnostic_report",
    )
    write_json(metrics, output_path.parent / "legacy_metrics.json")


def comparison_row(
    *,
    args: argparse.Namespace,
    requested_steps: str,
    resolved_steps: int,
    length_mode: str,
    row_count: int,
    run_dir: Path,
    synthetic_path: Path,
    profile_path: Path,
    paper_metrics_path: Path,
    legacy_metrics_path: Path,
) -> dict[str, Any]:
    row = status_row(args, requested_steps, resolved_steps, length_mode, row_count, "ok", run_dir)
    row["synthetic_path"] = str(synthetic_path)
    row["paper_metrics_path"] = str(paper_metrics_path)
    row["legacy_metrics_path"] = str(legacy_metrics_path)
    runtime = load_json(profile_path)
    paper = load_json(paper_metrics_path)
    legacy = load_json(legacy_metrics_path)
    summary = paper.get("paper_metrics_summary", {}) or {}
    c2st = paper.get("single_table_c2st", {}) or {}
    text_c2st = paper.get("text_embedding_c2st", {}) or {}
    shape = paper.get("shape", {}) or {}
    row.update(
        {
            "constraint_violation_rate": summary.get("constraint_violation_rate"),
            "shape_error": summary.get("shape_error"),
            "single_table_c2st_accuracy": c2st.get("accuracy"),
            "single_table_c2st_auc": c2st.get("auc"),
            "single_table_c2st_error": summary.get("single_table_c2st_error"),
            "text_embedding_c2st_macro_auc": text_c2st.get("macro_auc"),
            "text_embedding_c2st_error": summary.get("text_embedding_c2st_error"),
            "trend_error": summary.get("trend_error"),
            "runtime_seconds": runtime.get("total_sampling_seconds"),
            "rows_per_second": runtime.get("rows_per_second"),
            "time_per_row_seconds": safe_div(runtime.get("total_sampling_seconds"), row_count),
            "seconds_per_denoising_step": runtime.get("seconds_per_denoising_step"),
            "model_forward_passes": runtime.get("diffusion_model_forward_passes_total"),
            "batch_size": runtime.get("batch_size_used", args.sample_batch_size),
            "peak_gpu_memory_mb": runtime.get("cuda_memory_peak_allocated_mb"),
            "precision": runtime.get("dtype_used", args.inference_dtype),
            "gpu_model": runtime.get("gpu_name"),
            "checkpoint": str(args.checkpoint),
            "seed": int(args.seed),
        }
    )
    row.update(per_column_shape_fields(shape))
    row.update(legacy_text_fields(legacy))
    row.update(important_trend_fields(run_dir / "evaluation" / "paper_grade" / "per_pair_trend_metrics.csv"))
    return row


def status_row(
    args: argparse.Namespace,
    requested_steps: str,
    resolved_steps: int,
    length_mode: str,
    row_count: int,
    status: str,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "status": status,
        "run": run_dir.name,
        "requested_steps": str(requested_steps),
        "resolved_steps": int(resolved_steps),
        "length_mode": str(length_mode),
        "not_valid_generative_baseline": bool(length_mode == "oracle_length"),
        "rows_generated": int(row_count),
        "sample_batch_size": int(args.sample_batch_size),
        "text_top_k": int(args.text_top_k),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timestep_spacing": str(args.timestep_spacing),
        "inference_dtype": str(args.inference_dtype),
        "output_dir": str(run_dir),
    }


def per_column_shape_fields(shape: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    per_column = shape.get("per_column", {}) or {}
    for column in ["rating", "verified", "summary", "review_text"]:
        metrics = per_column.get(column, {}) or {}
        prefix = column
        out[f"{prefix}_shape_error"] = metrics.get("shape_error")
        secondary = metrics.get("secondary_statistics", {}) or {}
        if column in {"summary", "review_text"}:
            out[f"{prefix}_token_length_ks"] = metrics.get("shape_error")
            out[f"{prefix}_char_length_ks"] = secondary.get("char_length_ks")
            out[f"{prefix}_mean_token_length_real"] = secondary.get("token_length_mean_real")
            out[f"{prefix}_mean_token_length_synthetic"] = secondary.get("token_length_mean_synthetic")
    return out


def legacy_text_fields(legacy: dict[str, Any]) -> dict[str, Any]:
    validity = legacy.get("validity", {}) or {}
    text = legacy.get("text", {}) or {}
    out: dict[str, Any] = {}
    for column in ["summary", "review_text"]:
        prefix = "summary" if column == "summary" else "review_text"
        out[f"{prefix}_empty_text_rate"] = validity.get(f"empty_{column}_rate")
        out[f"{prefix}_invalid_text_rate"] = None
        out[f"{prefix}_unique_text_ratio"] = text.get(f"{prefix}_unique_rate")
        unique = text.get(f"{prefix}_unique_rate")
        out[f"{prefix}_exact_duplicate_rate"] = None if unique is None else max(0.0, 1.0 - float(unique))
        out[f"{prefix}_average_length"] = validity.get(f"{column}_length_mean_synthetic")
    return out


def important_trend_fields(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    frame = pd.read_csv(path)
    for _, row in frame.iterrows():
        pair = (str(row.get("col_a")), str(row.get("col_b")))
        reverse = (pair[1], pair[0])
        if pair not in IMPORTANT_TREND_PAIRS and reverse not in IMPORTANT_TREND_PAIRS:
            continue
        key = f"trend_{pair[0]}__{pair[1]}"
        out[key] = row.get("trend_error")
    return out


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def safe_div(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)


def write_comparison_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# V4 Diffusion Step Ablation",
        "",
        "| Steps | Length mode | Runtime | Rows/s | Shape | Review KS | Summary KS | Text C2ST error | Table C2ST error | Trend |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {steps} | {mode} | {runtime} | {rps} | {shape} | {review} | {summary} | {text_c2st} | {table_c2st} | {trend} |".format(
                steps=row.get("requested_steps"),
                mode=row.get("length_mode"),
                runtime=fmt(row.get("runtime_seconds")),
                rps=fmt(row.get("rows_per_second")),
                shape=fmt(row.get("shape_error")),
                review=fmt(row.get("review_text_token_length_ks")),
                summary=fmt(row.get("summary_token_length_ks")),
                text_c2st=fmt(row.get("text_embedding_c2st_error")),
                table_c2st=fmt(row.get("single_table_c2st_error")),
                trend=fmt(row.get("trend_error")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_diagnosis_doc(path: Path, rows: list[dict[str, Any]]) -> None:
    start = "<!-- DIFFUSION_STEP_ABLATION_TABLE_START -->"
    end = "<!-- DIFFUSION_STEP_ABLATION_TABLE_END -->"
    table = [
        start,
        "",
        "| Steps | Length Mode | Runtime | Rows/s | Shape ↓ | Review KS ↓ | Summary KS ↓ | Text C2ST Error ↓ | Table C2ST Error ↓ | Trend ↓ |",
        "| ----: | ----------- | ------: | -----: | ------: | ----------: | -----------: | ----------------: | -----------------: | ------: |",
    ]
    for row in rows:
        table.append(
            "| {steps} | {mode} | {runtime} | {rps} | {shape} | {review} | {summary} | {text_c2st} | {table_c2st} | {trend} |".format(
                steps=row.get("requested_steps"),
                mode=row.get("length_mode"),
                runtime=fmt(row.get("runtime_seconds")),
                rps=fmt(row.get("rows_per_second")),
                shape=fmt(row.get("shape_error")),
                review=fmt(row.get("review_text_token_length_ks")),
                summary=fmt(row.get("summary_token_length_ks")),
                text_c2st=fmt(row.get("text_embedding_c2st_error")),
                table_c2st=fmt(row.get("single_table_c2st_error")),
                trend=fmt(row.get("trend_error")),
            )
        )
    table.extend(["", end])
    replacement = "\n".join(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# V4 Diffusion Sampler Diagnosis\n\n" + replacement + "\n", encoding="utf-8")
        return
    text = path.read_text(encoding="utf-8")
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        path.write_text(before.rstrip() + "\n\n" + replacement + after, encoding="utf-8")
    else:
        path.write_text(text.rstrip() + "\n\n" + replacement + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
