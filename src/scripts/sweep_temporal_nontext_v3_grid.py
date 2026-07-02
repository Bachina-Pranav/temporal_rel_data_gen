#!/usr/bin/env python3
"""Sampling-only grid sweeps for TemporalNonTextAttributeDiffusionV3."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_temporal_nontext_attrs import evaluate_nontext_attrs, flatten, load_reviews  # noqa: E402
from reldiff.attributes import TemporalNonTextAttributeDiffusionV3  # noqa: E402
from reldiff.attributes.temporal_nontext_diffusion import to_jsonable  # noqa: E402
from reldiff.generation.block_diagnostics import load_block_maps_from_debug_dir  # noqa: E402


METRIC_KEYS: List[Tuple[str, str]] = [
    ("monthly_rating_corr", "temporal.monthly_average_rating_correlation"),
    ("monthly_verified_corr", "temporal.monthly_verified_rate_correlation"),
    ("rating_js", "categorical.rating_distribution_js"),
    ("verified_js", "categorical.verified_distribution_js"),
    ("product_avg_rating_ks", "entity_distribution.product_avg_rating_distribution_ks"),
    ("c2st", "c2st.c2st_accuracy"),
    ("customer_avg_rating_ks", "entity_distribution.customer_avg_rating_distribution_ks"),
    ("customer_verified_rate_ks", "entity_distribution.customer_verified_rate_distribution_ks"),
    ("customer_avg_rating_var_ratio", "entity_distribution.customer_avg_rating_variance_ratio"),
    ("customer_verified_rate_var_ratio", "entity_distribution.customer_verified_rate_variance_ratio"),
    ("product_avg_rating_var_ratio", "entity_distribution.product_avg_rating_variance_ratio"),
    ("product_verified_rate_var_ratio", "entity_distribution.product_verified_rate_variance_ratio"),
    ("monthly_rating_js_mean", "temporal.monthly_rating_distribution_js_mean"),
    ("monthly_verified_mae", "temporal.monthly_verified_rate_mae"),
    ("block_rating_corr", "block.block_pair_average_rating_correlation"),
    ("block_verified_corr", "block.block_pair_verified_rate_correlation"),
]

KEEP_TARGETS = {
    "monthly_rating_corr": ("gt", 0.75),
    "monthly_verified_corr": ("gt", 0.90),
    "rating_js": ("lt", 0.002),
    "verified_js": ("lt", 0.001),
    "product_avg_rating_ks": ("lt", 0.04),
    "c2st": ("lt", 0.525),
}

IMPROVEMENT_TARGETS = {
    "customer_avg_rating_ks": ("lt", 0.12),
    "customer_verified_rate_ks": ("lt", 0.15),
    "customer_avg_rating_var_ratio": ("band", (0.8, 1.1)),
    "customer_verified_rate_var_ratio": ("band", (0.8, 1.1)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resample/evaluate V3 from one checkpoint across calibration_strength, "
            "customer_effect_scale, and lambda_customer_effect grids."
        )
    )
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--calibration-strengths", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.25])
    parser.add_argument("--customer-effect-scales", nargs="+", type=float, default=[1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--lambda-customer-effects", nargs="+", type=float, default=[0.7, 1.0, 1.25, 1.5])
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--cat-cols", nargs="+", default=["rating", "verified"])
    parser.add_argument("--num-cols", nargs="*", default=[])
    parser.add_argument("--num-diffusion-steps", type=int, default=50)
    parser.add_argument("--cat-sampling-strategy", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-time-group", choices=["date", "exact", "window"], default="date")
    parser.add_argument("--sampling-window-days", type=float, default=1.0)
    parser.add_argument("--diagnostic-row-sample-size", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Resample runs even when metrics.json already exists.",
    )
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--summary-html", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "v3_sampling_sweep_summary.csv"
    summary_html = Path(args.summary_html) if args.summary_html else output_root / "v3_sampling_sweep_comparison.html"

    real = load_reviews(args.real_reviews, args.timestamp_col)
    spine = pd.read_csv(args.synthetic_spine)
    customer_blocks = product_blocks = None
    if args.structure_debug_dir:
        customer_blocks, product_blocks, _, _ = load_block_maps_from_debug_dir(
            args.structure_debug_dir, args.customer_id_col, args.product_id_col
        )

    run_grid = list(product(args.calibration_strengths, args.customer_effect_scales, args.lambda_customer_effects))
    write_run_config(args, output_root, len(run_grid))
    print(f"Running {len(run_grid)} sampling-only V3 sweep runs.")
    print(f"checkpoint={args.checkpoint}")
    print(f"output_root={output_root}")

    rows: List[Dict[str, Any]] = []
    for index, (calibration_strength, customer_effect_scale, lambda_customer_effect) in enumerate(run_grid, start=1):
        row = run_one(
            args,
            output_root,
            real,
            spine,
            customer_blocks,
            product_blocks,
            index,
            len(run_grid),
            calibration_strength,
            customer_effect_scale,
            lambda_customer_effect,
        )
        rows.append(row)
        summary = pd.DataFrame(rows).sort_values(["selection_score", "run_name"])
        summary.to_csv(summary_csv, index=False)
        write_html_report(summary, summary_html, partial=True)

    summary = pd.DataFrame(rows).sort_values(["selection_score", "run_name"])
    summary.insert(0, "rank", range(1, len(summary) + 1))
    summary.to_csv(summary_csv, index=False)
    write_html_report(summary, summary_html, partial=False)
    print(summary[display_columns()].head(20).to_string(index=False))
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_html}")


def run_one(
    args: argparse.Namespace,
    output_root: Path,
    real: pd.DataFrame,
    spine: pd.DataFrame,
    customer_blocks: Optional[Dict[Any, int]],
    product_blocks: Optional[Dict[Any, int]],
    run_index: int,
    num_runs: int,
    calibration_strength: float,
    customer_effect_scale: float,
    lambda_customer_effect: float,
) -> Dict[str, Any]:
    run_name = (
        f"cal{format_value(calibration_strength)}"
        f"_custscale{format_value(customer_effect_scale)}"
        f"_lambdacust{format_value(lambda_customer_effect)}"
    )
    run_dir = output_root / run_name
    diagnostics_dir = run_dir / "diagnostics"
    synthetic_path = run_dir / "synthetic_review.csv"
    metrics_path = run_dir / "metrics.json"

    print()
    print(f"[{run_index}/{num_runs}] {run_name}")
    if metrics_path.exists() and synthetic_path.exists() and not args.overwrite:
        print(f"  reusing existing metrics: {metrics_path}")
        with metrics_path.open() as handle:
            metrics = json.load(handle)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        generator = TemporalNonTextAttributeDiffusionV3.load_checkpoint(args.checkpoint, device=args.device)
        generator.customer_prior_v3.effect_scale = float(customer_effect_scale)
        generator.lambda_customer_effect = float(lambda_customer_effect)
        generator.config["customer_effect_scale_sampling_override"] = float(customer_effect_scale)
        generator.config["lambda_customer_effect_sampling_override"] = float(lambda_customer_effect)

        synthetic = generator.sample(
            spine,
            structure_debug_dir=args.structure_debug_dir,
            sampled_effects_output_dir=run_dir,
            seed=args.seed,
            num_steps=args.num_diffusion_steps,
            cat_sampling_strategy=args.cat_sampling_strategy,
            temperature=args.temperature,
            sampling_time_group=args.sampling_time_group,
            sampling_window_days=args.sampling_window_days,
            use_temporal_calibration=True,
            temporal_calibration_strength=float(calibration_strength),
            debug_use_posterior_effects=False,
            diagnostics_dir=diagnostics_dir,
            diagnostic_row_sample_size=args.diagnostic_row_sample_size,
            device=args.device,
        )
        synthetic.to_csv(synthetic_path, index=False)
        synthetic_for_eval = synthetic.copy()
        synthetic_for_eval[args.timestamp_col] = pd.to_datetime(
            synthetic_for_eval[args.timestamp_col], errors="coerce"
        )
        metrics = evaluate_nontext_attrs(
            real,
            synthetic_for_eval,
            customer_col=args.customer_id_col,
            product_col=args.product_id_col,
            timestamp_col=args.timestamp_col,
            cat_cols=args.cat_cols,
            num_cols=args.num_cols,
            customer_blocks=customer_blocks,
            product_blocks=product_blocks,
            temporal_bucket_level="year_month",
            diagnostics_dir=diagnostics_dir,
        )
        with metrics_path.open("w") as handle:
            json.dump(to_jsonable(metrics), handle, indent=2)
            handle.write("\n")
        with (run_dir / "synthetic_review_metadata.json").open("w") as handle:
            metadata = generator.synthetic_metadata(synthetic_path, args.checkpoint, args.seed)
            metadata["sweep_calibration_strength"] = float(calibration_strength)
            metadata["sweep_customer_effect_scale"] = float(customer_effect_scale)
            metadata["sweep_lambda_customer_effect"] = float(lambda_customer_effect)
            json.dump(to_jsonable(metadata), handle, indent=2)
            handle.write("\n")

    flat = flatten(metrics)
    row: Dict[str, Any] = {
        "run_name": run_name,
        "calibration_strength": float(calibration_strength),
        "customer_effect_scale": float(customer_effect_scale),
        "lambda_customer_effect": float(lambda_customer_effect),
        "metrics_path": str(metrics_path),
        "synthetic_path": str(synthetic_path),
    }
    for alias, key in METRIC_KEYS:
        row[alias] = finite_or_none(flat.get(key))
    score_row(row)
    print(
        "  score={score:.6f} keep={keep}/6 improve={improve}/4 "
        "monthly_rating={mr:.4f} monthly_verified={mv:.4f} "
        "cust_rating_ks={crks:.4f} cust_verified_ks={cvks:.4f}".format(
            score=row["selection_score"],
            keep=row["keep_pass_count"],
            improve=row["improvement_pass_count"],
            mr=row.get("monthly_rating_corr") or float("nan"),
            mv=row.get("monthly_verified_corr") or float("nan"),
            crks=row.get("customer_avg_rating_ks") or float("nan"),
            cvks=row.get("customer_verified_rate_ks") or float("nan"),
        )
    )
    return row


def score_row(row: Dict[str, Any]) -> None:
    keep_passes = sum(1 for key, target in KEEP_TARGETS.items() if passes_target(row.get(key), target))
    improvement_passes = sum(
        1 for key, target in IMPROVEMENT_TARGETS.items() if passes_target(row.get(key), target)
    )
    row["keep_pass_count"] = int(keep_passes)
    row["keep_all_targets"] = bool(keep_passes == len(KEEP_TARGETS))
    row["improvement_pass_count"] = int(improvement_passes)
    row["improvement_all_targets"] = bool(improvement_passes == len(IMPROVEMENT_TARGETS))
    row["customer_avg_rating_var_band_distance"] = band_distance(
        row.get("customer_avg_rating_var_ratio"), 0.8, 1.1
    )
    row["customer_verified_rate_var_band_distance"] = band_distance(
        row.get("customer_verified_rate_var_ratio"), 0.8, 1.1
    )
    customer_distance = (
        safe_distance(row.get("customer_avg_rating_ks"), 0.0)
        + safe_distance(row.get("customer_verified_rate_ks"), 0.0)
        + row["customer_avg_rating_var_band_distance"]
        + row["customer_verified_rate_var_band_distance"]
    )
    row["customer_improvement_distance"] = float(customer_distance)
    row["selection_score"] = float(100.0 * (len(KEEP_TARGETS) - keep_passes) + customer_distance)


def passes_target(value: Any, target: Tuple[str, Any]) -> bool:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    kind, bound = target
    value = float(value)
    if kind == "gt":
        return value > float(bound)
    if kind == "lt":
        return value < float(bound)
    if kind == "band":
        low, high = bound
        return float(low) <= value <= float(high)
    raise ValueError(f"Unknown target kind {kind!r}")


def band_distance(value: Any, low: float, high: float) -> float:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 1_000.0
    value = float(value)
    if low <= value <= high:
        return 0.0
    return min(abs(value - low), abs(value - high))


def safe_distance(value: Any, optimum: float) -> float:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return 1_000.0
    return abs(float(value) - float(optimum))


def finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def format_value(value: float) -> str:
    return ("%g" % float(value)).replace("-", "m").replace(".", "p")


def write_run_config(args: argparse.Namespace, output_root: Path, num_runs: int) -> None:
    config = {
        "num_runs": int(num_runs),
        "checkpoint": args.checkpoint,
        "real_reviews": args.real_reviews,
        "synthetic_spine": args.synthetic_spine,
        "structure_debug_dir": args.structure_debug_dir,
        "calibration_strengths": args.calibration_strengths,
        "customer_effect_scales": args.customer_effect_scales,
        "lambda_customer_effects": args.lambda_customer_effects,
        "seed": args.seed,
        "device": args.device,
        "num_diffusion_steps": args.num_diffusion_steps,
        "cat_sampling_strategy": args.cat_sampling_strategy,
        "temperature": args.temperature,
        "keep_targets": KEEP_TARGETS,
        "improvement_targets": IMPROVEMENT_TARGETS,
    }
    with (output_root / "sweep_config.json").open("w") as handle:
        json.dump(to_jsonable(config), handle, indent=2)
        handle.write("\n")


def write_html_report(summary: pd.DataFrame, output_path: Path, partial: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = summary.copy()
    if "rank" not in display.columns:
        display.insert(0, "rank", range(1, len(display) + 1))
    display = display[display_columns()]
    rows_html = "\n".join(render_html_row(row) for _, row in display.iterrows())
    status = "partial/in-progress" if partial else "complete"
    best = display.head(10)
    best_rows = "\n".join(render_html_row(row) for _, row in best.iterrows())
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>V3 Sampling Sweep Comparison</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ margin: 6px 0 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: right; white-space: nowrap; }}
    th {{ background: #f0f4f8; position: sticky; top: 0; z-index: 1; }}
    td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {{ text-align: left; }}
    .pass {{ background: #e3f9e5; color: #1f7a1f; }}
    .fail {{ background: #ffe3e3; color: #b42318; }}
    .neutral {{ background: #fffbea; color: #7c5e10; }}
    .target-list {{ columns: 2; max-width: 980px; }}
    .small {{ color: #52606d; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>V3 Sampling Sweep Comparison</h1>
  <p class="small">Status: {html.escape(status)}. Rows are sorted by selection_score: all keep-target failures receive a large penalty, then customer-distribution distance is minimized.</p>
  <h2>Targets</h2>
  <ul class="target-list">
    <li>monthly_rating_corr &gt; 0.75</li>
    <li>monthly_verified_corr &gt; 0.90</li>
    <li>rating_js &lt; 0.002</li>
    <li>verified_js &lt; 0.001</li>
    <li>product_avg_rating_ks &lt; 0.04</li>
    <li>c2st &lt; 0.525</li>
    <li>customer_avg_rating_ks &lt; 0.12</li>
    <li>customer_verified_rate_ks &lt; 0.15</li>
    <li>customer_avg_rating_var_ratio in [0.8, 1.1]</li>
    <li>customer_verified_rate_var_ratio in [0.8, 1.1]</li>
  </ul>
  <h2>Top 10</h2>
  <table>
    {render_header(display.columns)}
    <tbody>{best_rows}</tbody>
  </table>
  <h2>All Runs</h2>
  <table>
    {render_header(display.columns)}
    <tbody>{rows_html}</tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(content)


def render_header(columns: Iterable[str]) -> str:
    cells = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    return f"<thead><tr>{cells}</tr></thead>"


def render_html_row(row: pd.Series) -> str:
    cells = []
    for col, value in row.items():
        css = css_class_for_cell(col, value)
        text = format_cell(value)
        cells.append(f'<td class="{css}">{html.escape(text)}</td>')
    return "<tr>" + "".join(cells) + "</tr>"


def css_class_for_cell(col: str, value: Any) -> str:
    if col in KEEP_TARGETS:
        return "pass" if passes_target(value, KEEP_TARGETS[col]) else "fail"
    if col in IMPROVEMENT_TARGETS:
        return "pass" if passes_target(value, IMPROVEMENT_TARGETS[col]) else "neutral"
    if col in {"keep_all_targets", "improvement_all_targets"}:
        return "pass" if bool(value) else "fail"
    return ""


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        if abs(float(value)) >= 1000:
            return f"{float(value):.3f}"
        return f"{float(value):.6g}"
    return str(value)


def display_columns() -> List[str]:
    return [
        "rank",
        "run_name",
        "selection_score",
        "keep_pass_count",
        "keep_all_targets",
        "improvement_pass_count",
        "improvement_all_targets",
        "calibration_strength",
        "customer_effect_scale",
        "lambda_customer_effect",
        "monthly_rating_corr",
        "monthly_verified_corr",
        "rating_js",
        "verified_js",
        "product_avg_rating_ks",
        "c2st",
        "customer_avg_rating_ks",
        "customer_verified_rate_ks",
        "customer_avg_rating_var_ratio",
        "customer_verified_rate_var_ratio",
        "customer_improvement_distance",
        "monthly_rating_js_mean",
        "monthly_verified_mae",
        "block_rating_corr",
        "block_verified_corr",
        "metrics_path",
    ]


if __name__ == "__main__":
    main()
