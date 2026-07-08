#!/usr/bin/env python3
"""Compare v2, v4 full-text diffusion, and v5 joint LSTM experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SHARED_METRICS = [
    "marginal_categorical.rating_distribution_l1",
    "marginal_categorical.verified_distribution_l1",
    "joint.rating_verified_joint_l1",
    "conditional_fidelity.customer_rating_top_1000_mae",
    "conditional_fidelity.customer_verified_top_1000_mae",
    "conditional_fidelity.product_rating_top_1000_mae",
    "conditional_fidelity.product_verified_top_1000_mae",
    "length_diagnostics.summary_length_ks",
]

REVIEW_TEXT_METRICS = [
    "length_diagnostics.review_text_length_ks",
    "text.review_text_distinct_1",
    "text.review_text_distinct_2",
    "text.review_text_unique_rate",
    "text_privacy.review_text_exact_train_overlap_rate",
    "text_consistency.rating_review_text_consistency_accuracy",
    "text_consistency.verified_review_text_predictor_auc",
    "text_consistency.summary_review_text_rougeL_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Conditional TABDLM v2/v4 and joint LSTM v5.")
    parser.add_argument("--v2", required=True)
    parser.add_argument("--v4", required=True)
    parser.add_argument("--v5", required=True)
    parser.add_argument("--v5-runtime", required=True)
    parser.add_argument("--v4-runtime", default=None)
    parser.add_argument("--v4-sampling-seconds", type=float, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v2 = load_json(args.v2)
    v4 = load_json(args.v4)
    v5 = load_json(args.v5)
    v5_runtime = load_json(args.v5_runtime)
    v4_runtime = load_json(args.v4_runtime) if args.v4_runtime else {}
    v4_seconds = args.v4_sampling_seconds
    if v4_seconds is None:
        v4_seconds = v4_runtime.get("total_sampling_seconds")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_rows = [
        {
            "metric": path,
            "v2": get_path(v2, path),
            "v4": get_path(v4, path),
            "v5": get_path(v5, path),
        }
        for path in SHARED_METRICS
    ]
    review_rows = [
        {
            "metric": path,
            "v4": get_path(v4, path),
            "v5": get_path(v5, path),
        }
        for path in REVIEW_TEXT_METRICS
    ]
    runtime = {
        "v4_sampling_seconds": v4_seconds,
        "v5_sampling_seconds": v5_runtime.get("total_sampling_seconds"),
        "speedup_factor_v5_over_v4": None,
        "projected_hours_for_10m_rows": v5_runtime.get("projected_hours_for_10m_rows"),
        "v5_rows_per_second": v5_runtime.get("rows_per_second"),
    }
    if v4_seconds is not None and v5_runtime.get("total_sampling_seconds"):
        runtime["speedup_factor_v5_over_v4"] = float(v4_seconds) / max(float(v5_runtime["total_sampling_seconds"]), 1e-9)
    payload = {
        "paths": {
            "v2": str(args.v2),
            "v4": str(args.v4),
            "v5": str(args.v5),
            "v5_runtime": str(args.v5_runtime),
        },
        "shared_metrics": shared_rows,
        "review_text_metrics": review_rows,
        "runtime": runtime,
        "verdict": verdict(v5, runtime),
    }
    with (output_dir / "comparison.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(payload, output_dir / "comparison.md")
    print(output_dir / "comparison.json")
    print(output_dir / "comparison.md")


def verdict(v5: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    validity = v5.get("validity", {})
    marginal = v5.get("marginal_categorical", {})
    joint = v5.get("joint", {})
    length = v5.get("length_diagnostics", {})
    privacy = v5.get("text_privacy", {})
    failures = []
    if validity.get("invalid_rating_rate") not in (0, 0.0):
        failures.append("invalid_rating_rate is nonzero")
    if validity.get("invalid_verified_rate") not in (0, 0.0):
        failures.append("invalid_verified_rate is nonzero")
    if runtime.get("v5_sampling_seconds") is not None and float(runtime["v5_sampling_seconds"]) > 1200:
        failures.append("sampling did not meet minimum 20 minute target")
    checks = {
        "rating_distribution_l1": (marginal.get("rating_distribution_l1"), 0.25),
        "rating_verified_joint_l1": (joint.get("rating_verified_joint_l1"), 0.30),
        "summary_length_ks": (length.get("summary_length_ks"), 0.30),
        "review_text_length_ks": (length.get("review_text_length_ks"), 0.40),
        "empty_review_text_rate": (validity.get("empty_review_text_rate"), 0.01),
    }
    for key, (value, threshold) in checks.items():
        if value is not None and float(value) > float(threshold):
            failures.append(f"{key}={float(value):.4g} exceeds {threshold}")
    exact_overlap = privacy.get("review_text_exact_train_overlap_rate")
    return {
        "valid_scalable_experiment": not failures,
        "failures": failures,
        "review_text_exact_train_overlap_rate": exact_overlap,
        "recommended_next_step": "Keep v5 as scalable baseline and tune quality." if failures else "Run seed sweep and consider v5 as main scaling path.",
    }


def load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open() as handle:
        return json.load(handle)


def get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    lines = ["# v2 vs v4 vs v5 Joint LSTM Comparison", ""]
    lines.append("## Runtime")
    for key, value in payload["runtime"].items():
        lines.append(f"- {key}: {format_value(value)}")
    lines.append("")
    lines.append("## Verdict")
    for key, value in payload["verdict"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Shared Metrics", "", "| Metric | v2 | v4 | v5 |", "|---|---:|---:|---:|"])
    for row in payload["shared_metrics"]:
        lines.append(f"| {row['metric']} | {format_value(row.get('v2'))} | {format_value(row.get('v4'))} | {format_value(row.get('v5'))} |")
    lines.extend(["", "## Review Text Metrics", "", "| Metric | v4 | v5 |", "|---|---:|---:|"])
    for row in payload["review_text_metrics"]:
        lines.append(f"| {row['metric']} | {format_value(row.get('v4'))} | {format_value(row.get('v5'))} |")
    Path(path).write_text("\n".join(lines) + "\n")


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
