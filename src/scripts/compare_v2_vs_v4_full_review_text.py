#!/usr/bin/env python3
"""Compare v2 structure graph vs v4 full-review-text Conditional TABDLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SHARED_METRICS = [
    "marginal_categorical.rating_distribution_l1",
    "marginal_categorical.verified_distribution_l1",
    "joint.rating_verified_joint_l1",
    "joint.rating_distribution_given_verified_l1",
    "joint.verified_rate_by_rating_mae",
    "conditional_fidelity.customer_rating_top_1000_mae",
    "conditional_fidelity.customer_verified_top_1000_mae",
    "conditional_fidelity.product_rating_top_1000_mae",
    "conditional_fidelity.product_verified_top_1000_mae",
    "temporal.monthly_rating_mean_corr",
    "temporal.monthly_rating_mean_mae",
    "temporal.monthly_verified_rate_corr",
    "temporal.monthly_verified_rate_mae",
    "temporal.monthly_summary_length_corr",
    "temporal.monthly_summary_length_mae",
    "length_diagnostics.summary_length_mean_synthetic",
    "length_diagnostics.summary_length_ks",
    "length_diagnostics.summary_length_bucket_l1",
    "text.distinct_1",
    "text.distinct_2",
    "text.unique_summary_rate",
    "text_privacy.exact_summary_train_overlap_rate",
    "text_consistency.rating_text_consistency_accuracy",
    "text_consistency.verified_text_predictor_auc",
]

V4_ONLY_METRICS = [
    "validity.empty_review_text_rate",
    "length_diagnostics.review_text_length_mean_synthetic",
    "length_diagnostics.review_text_length_ks",
    "length_diagnostics.review_text_length_bucket_l1",
    "length_diagnostics.review_text_generated_to_max_length_rate",
    "length_diagnostics.review_text_truncation_rate_train",
    "length_diagnostics.review_text_coverage_rate_train",
    "text.review_text_distinct_1",
    "text.review_text_distinct_2",
    "text.review_text_unique_rate",
    "text_privacy.review_text_exact_train_overlap_rate",
    "text_privacy.review_text_nearest_neighbor_rougeL_mean",
    "text_consistency.rating_review_text_consistency_accuracy",
    "text_consistency.verified_review_text_predictor_auc",
    "text_consistency.summary_review_text_rougeL_mean",
    "text_consistency.summary_review_text_token_jaccard_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v2 and v4 full-review-text Conditional TABDLM metrics.")
    parser.add_argument("--v2", required=True)
    parser.add_argument("--v4", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v2 = load_json(args.v2)
    v4 = load_json(args.v4)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shared = []
    for path in SHARED_METRICS:
        v2_value = get_path(v2, path)
        v4_value = get_path(v4, path)
        shared.append(
            {
                "metric": path,
                "v2": v2_value,
                "v4": v4_value,
                "delta_v4_minus_v2": None if v2_value is None or v4_value is None else float(v4_value) - float(v2_value),
            }
        )
    v4_only = [{"metric": path, "v4": get_path(v4, path)} for path in V4_ONLY_METRICS]
    payload = {
        "paths": {"v2": str(args.v2), "v4": str(args.v4)},
        "shared_metrics": shared,
        "v4_only_metrics": v4_only,
        "safety": safety_metadata(v4),
    }
    payload["verdict"] = verdict(payload)
    with (output_dir / "comparison.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(payload, output_dir / "comparison.md")
    print(output_dir / "comparison.json")
    print(output_dir / "comparison.md")


def safety_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    meta = dict(payload.get("graph_conditioning", {}) or {})
    validity = dict(payload.get("validity", {}) or {})
    expected = {
        "graph_conditioning_mode": "structure_only_temporal",
        "temporal_filter_enabled": True,
        "temporal_filter_mode": "past_only",
        "graph_uses_future_events": False,
        "graph_uses_target_attributes": False,
        "graph_uses_clean_target_attributes": False,
        "real_graph_used_at_sampling": False,
        "joint_generation": True,
        "review_text_generated_jointly": True,
        "review_text_separate_stage": False,
    }
    failures = []
    for key, value in expected.items():
        if meta.get(key) != value:
            failures.append(f"{key} expected {value!r}, got {meta.get(key)!r}")
    for key in ["invalid_rating_rate", "invalid_verified_rate"]:
        if validity.get(key) not in (0, 0.0):
            failures.append(f"{key} expected 0.0, got {validity.get(key)!r}")
    return {"valid_experiment": not failures, "failures": failures, **{key: meta.get(key) for key in expected}}


def verdict(payload: dict[str, Any]) -> dict[str, Any]:
    shared_degradation = []
    for row in payload["shared_metrics"]:
        metric = row["metric"]
        delta = row["delta_v4_minus_v2"]
        if delta is None:
            continue
        if metric.endswith("_l1") or metric.endswith("_mae") or metric.endswith("_ks"):
            if float(delta) > 0.05:
                shared_degradation.append({"metric": metric, "delta": delta})
        if metric.endswith("_corr"):
            if float(delta) < -0.05:
                shared_degradation.append({"metric": metric, "delta": delta})
    v4_lookup = {row["metric"]: row["v4"] for row in payload["v4_only_metrics"]}
    empty = v4_lookup.get("validity.empty_review_text_rate")
    maxed = v4_lookup.get("length_diagnostics.review_text_generated_to_max_length_rate")
    exact = v4_lookup.get("text_privacy.review_text_exact_train_overlap_rate")
    quality_flags = []
    if empty is not None and float(empty) > 0.1:
        quality_flags.append("review_text empty rate is high")
    if maxed is not None and float(maxed) > 0.2:
        quality_flags.append("review_text often decodes to max length")
    if exact is not None and float(exact) > 0.2:
        quality_flags.append("review_text exact train overlap is high")
    return {
        "shared_metrics_degradation": shared_degradation,
        "review_text_quality_status": "needs_attention" if quality_flags else "non_degenerate_basic_checks_passed",
        "review_text_quality_flags": quality_flags,
        "recommended_next_step": "Inspect qualitative samples and tune review_text length/repetition settings."
        if quality_flags or shared_degradation
        else "Run a longer seed sweep for stability.",
    }


def load_json(path: str | Path) -> dict[str, Any]:
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
    lines = ["# Conditional TABDLM v2 vs v4 Full Review Text", ""]
    lines.append("## Verdict")
    for key, value in payload["verdict"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety"])
    lines.append(f"- valid_experiment: `{payload['safety']['valid_experiment']}`")
    for failure in payload["safety"].get("failures", []):
        lines.append(f"- {failure}")
    lines.extend(["", "## Shared Metrics", "", "| Metric | v2 | v4 | Delta |", "|---|---:|---:|---:|"])
    for row in payload["shared_metrics"]:
        lines.append(f"| {row['metric']} | {fmt(row['v2'])} | {fmt(row['v4'])} | {fmt(row['delta_v4_minus_v2'])} |")
    lines.extend(["", "## v4-Only Metrics", "", "| Metric | v4 |", "|---|---:|"])
    for row in payload["v4_only_metrics"]:
        lines.append(f"| {row['metric']} | {fmt(row['v4'])} |")
    Path(path).write_text("\n".join(lines) + "\n")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
