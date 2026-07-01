#!/usr/bin/env python3
"""Aggregate diagnostics for TemporalNonTextAttributeDiffusionV3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_temporal_nontext_attrs import evaluate_nontext_attrs, load_reviews  # noqa: E402
from reldiff.attributes import TemporalNonTextAttributeDiffusionV3  # noqa: E402
from reldiff.attributes.temporal_priors import check_temporal_bucket_consistency  # noqa: E402
from reldiff.attributes.temporal_nontext_diffusion import to_jsonable  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose V3 non-text temporal attribute sampling.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--sample-metadata", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--cat-cols", nargs="+", default=["rating", "verified"])
    parser.add_argument("--num-cols", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics_dir = Path(args.diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    real = load_reviews(args.real_reviews, args.timestamp_col)
    synthetic = load_reviews(args.synthetic_reviews, args.timestamp_col)
    metrics = evaluate_nontext_attrs(
        real,
        synthetic,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        cat_cols=args.cat_cols,
        num_cols=args.num_cols,
        diagnostics_dir=diagnostics_dir,
    )
    checkpoint_report = checkpoint_bucket_report(args.checkpoint, real, synthetic, args.timestamp_col)
    if checkpoint_report:
        write_json(diagnostics_dir / "temporal_bucket_consistency.json", checkpoint_report)
    prior_summary = enrich_temporal_prior_diagnostics(diagnostics_dir)
    if prior_summary:
        write_json(diagnostics_dir / "temporal_prior_diagnostics.json", prior_summary)
    report = {
        "metrics": metrics,
        "sample_metadata": load_json_optional(args.sample_metadata),
        "temporal_bucket_consistency": load_json_optional(diagnostics_dir / "temporal_bucket_consistency.json"),
        "decomposition": load_json_optional(diagnostics_dir / "decomposition_diagnostics.json"),
        "temporal_prior_diagnostics": prior_summary or load_json_optional(diagnostics_dir / "temporal_prior_diagnostics.json"),
        "temporal_calibration_summary": summarize_calibration(diagnostics_dir / "temporal_calibration_by_group.csv"),
        "component_curve_summary": summarize_component_curve(diagnostics_dir / "component_curve_by_month.csv"),
    }
    report["top_level_recommendations"] = recommendations(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, to_jsonable(report))
    print(json.dumps(to_jsonable(report["top_level_recommendations"]), indent=2))
    print(f"Wrote {output}")


def checkpoint_bucket_report(
    checkpoint: str | Path,
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    timestamp_col: str,
) -> Optional[Dict[str, Any]]:
    try:
        generator = TemporalNonTextAttributeDiffusionV3.load_checkpoint(checkpoint, device="cpu")
    except Exception as exc:
        return {
            "diagnostic_status": "checkpoint_load_failed",
            "diagnostic_reason": str(exc),
        }
    return check_temporal_bucket_consistency(
        generator.temporal_prior,
        synthetic[timestamp_col],
        real[timestamp_col],
    )


def enrich_temporal_prior_diagnostics(diagnostics_dir: Path) -> Dict[str, Any]:
    curve_path = diagnostics_dir / "temporal_rating_prior_monthly_avg_curve.csv"
    monthly_path = diagnostics_dir / "monthly_real_vs_synthetic.csv"
    summary_path = diagnostics_dir / "temporal_prior_diagnostics.json"
    summary = load_json_optional(summary_path) or {}
    if not curve_path.exists() or not monthly_path.exists():
        if summary:
            summary.setdefault("diagnostic_status", "missing_monthly_or_prior_curve")
        return summary
    curve = pd.read_csv(curve_path)
    monthly = pd.read_csv(monthly_path)
    if "month" not in curve.columns or "month" not in monthly.columns:
        summary["diagnostic_status"] = "missing_month_column"
        return summary
    merged = curve.merge(monthly, on="month", how="inner")
    summary.update(
        {
            "prior_monthly_avg_rating_std": std_or_none(curve.get("prior_avg_rating")),
            "real_monthly_avg_rating_std": std_or_none(monthly.get("real_avg_rating")),
            "prior_to_real_monthly_avg_rating_corr": corr_or_none(
                merged.get("prior_avg_rating"), merged.get("real_avg_rating")
            ),
            "prior_monthly_verified_std": std_or_none(curve.get("prior_verified_rate")),
            "real_monthly_verified_std": std_or_none(monthly.get("real_verified_rate")),
            "prior_to_real_monthly_verified_corr": corr_or_none(
                merged.get("prior_verified_rate"), merged.get("real_verified_rate")
            ),
            "diagnostic_status": "loaded",
        }
    )
    return summary


def summarize_calibration(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"diagnostic_status": "missing_diagnostics_file"}
    frame = pd.read_csv(path)
    if frame.empty:
        return {"diagnostic_status": "empty", "num_groups": 0}
    return {
        "diagnostic_status": "loaded",
        "num_groups": int(len(frame)),
        "average_precal_rating_target_js": mean_or_none(frame.get("precal_rating_target_js")),
        "average_postcal_rating_target_js": mean_or_none(frame.get("postcal_rating_target_js")),
        "average_precal_verified_target_abs_error": mean_or_none(frame.get("precal_verified_target_abs_error")),
        "average_postcal_verified_target_abs_error": mean_or_none(frame.get("postcal_verified_target_abs_error")),
        "average_rating_correction_norm": mean_or_none(frame.get("rating_correction_norm")),
    }


def summarize_component_curve(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"diagnostic_status": "missing_diagnostics_file"}
    frame = pd.read_csv(path)
    if frame.empty:
        return {"diagnostic_status": "empty", "num_months": 0}
    summary: Dict[str, Any] = {"diagnostic_status": "loaded", "num_months": int(len(frame))}
    for col in [
        "temporal_only_avg_rating",
        "temporal_block_avg_rating",
        "temporal_block_product_avg_rating",
        "full_base_avg_rating",
        "final_precal_avg_rating",
        "final_postcal_avg_rating",
        "synthetic_sampled_avg_rating",
    ]:
        summary[f"{col}_std"] = std_or_none(frame.get(col))
    return summary


def recommendations(report: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    prior = report.get("temporal_prior_diagnostics") or {}
    calibration = report.get("temporal_calibration_summary") or {}
    decomposition = report.get("decomposition") or report.get("metrics", {}).get("decomposition", {})
    component = report.get("component_curve_summary") or {}
    metrics = report.get("metrics", {})
    temporal_diag = metrics.get("temporal_diagnostics", {})

    prior_corr = prior.get("prior_to_real_monthly_avg_rating_corr")
    if is_number(prior_corr) and prior_corr < 0.5:
        recs.append("Temporal priors are weak or bucketed incorrectly.")
    if is_number(prior_corr) and prior_corr >= 0.5:
        pre_js = calibration.get("average_precal_rating_target_js")
        post_js = calibration.get("average_postcal_rating_target_js")
        if is_number(pre_js) and is_number(post_js) and post_js >= pre_js:
            recs.append("Calibration implementation is likely wrong or too weak.")
    post_std = component.get("final_postcal_avg_rating_std")
    sampled_std = temporal_diag.get("monthly_avg_rating_synthetic_std")
    if is_number(post_std) and is_number(sampled_std) and post_std > 0 and sampled_std < 0.5 * post_std:
        recs.append("Sampling stochasticity or group sampling order may be washing out calibrated temporal trends.")
    ratio = decomposition.get("residual_to_base_norm_ratio")
    if is_number(ratio) and ratio > 1.0:
        recs.append("Residual model may be overpowering base temporal/entity priors.")
    temporal_std = component.get("temporal_only_avg_rating_std")
    full_base_std = component.get("full_base_avg_rating_std")
    if is_number(temporal_std) and is_number(full_base_std) and temporal_std > 0 and full_base_std < 0.5 * temporal_std:
        recs.append("Entity effects flatten the temporal curve; reduce product/customer lambdas or strengthen calibration.")
    if not recs:
        recs.append("Diagnostics do not isolate one clear failure mode; inspect monthly and row-level component CSVs.")
    return recs


def load_json_optional(path: str | Path | None) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(to_jsonable(data), handle, indent=2)
        handle.write("\n")


def is_number(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float, np.number)) and np.isfinite(float(value))


def mean_or_none(values: Any) -> Optional[float]:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    value = float(series.mean())
    return value if np.isfinite(value) else None


def std_or_none(values: Any) -> Optional[float]:
    if values is None:
        return None
    series = pd.to_numeric(values, errors="coerce").dropna()
    if len(series) < 2:
        return None
    value = float(series.std())
    return value if np.isfinite(value) else None


def corr_or_none(left: Any, right: Any) -> Optional[float]:
    if left is None or right is None:
        return None
    frame = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 2 or frame["left"].std() == 0 or frame["right"].std() == 0:
        return None
    return float(np.corrcoef(frame["left"], frame["right"])[0, 1])


if __name__ == "__main__":
    main()
