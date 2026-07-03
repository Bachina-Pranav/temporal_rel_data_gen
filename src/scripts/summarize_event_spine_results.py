#!/usr/bin/env python3
"""Summarize event-spine comparison results with cautious interpretation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


MAIN_METHOD = "time_biased_local_kernel_main"
RANDOM_METHOD = "time_biased_local_kernel_random_pairing"
MEDIAN_METHOD = "time_biased_median_mixture"
EMPIRICAL_EXACT_METHOD = "time_biased_empirical_exact"

METHOD_LABELS = {
    "static_degree": "StaticDegree",
    "ct_2k_sbm_temporal_kde_stubs": "CT-2K-SBM",
    MEDIAN_METHOD: "TBSM-median-mixture",
    EMPIRICAL_EXACT_METHOD: "TBSM-empirical-exact",
    RANDOM_METHOD: "TBSM-local-kernel-random",
    MAIN_METHOD: "TBSM-local-kernel-dynamic",
}

LIFECYCLE_KEYS = [
    "product_first_time_corr",
    "product_last_time_corr",
    "product_peak_time_corr",
    "customer_first_time_corr",
    "customer_last_time_corr",
    "customer_peak_time_corr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize event-spine comparison results.")
    parser.add_argument("--input-json", default="outputs/rel-amazon/event_spine_generator_comparison.json")
    parser.add_argument("--output", default="outputs/rel-amazon/event_spine_result_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.input_json).open() as handle:
        comparison = json.load(handle)
    lines = build_summary(comparison)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[done] wrote {output}")


def build_summary(comparison: Dict[str, Dict[str, Any]]) -> list[str]:
    lines = ["# Event-Spine Validation Summary", ""]
    lines.append("This summary is metric-based and intentionally cautious; it does not tune or change the generator.")
    lines.append("")
    lines.extend(section("Best lifecycle metrics", best_by_score(comparison, lifecycle_score, higher_is_better=True)))
    lines.extend(section("Lowest copying / memorization", best_by_metric(comparison, "exact_event_overlap_rate", higher_is_better=False)))
    lines.extend(section("Best dynamic affinity KS", best_by_metric(comparison, "dynamic_affinity_distribution_ks", higher_is_better=False)))
    lines.extend(section("Lowest sampled C2ST distinguishability", best_by_score(comparison, c2st_distance_from_chance, higher_is_better=False)))
    lines.extend(pairwise_statement(comparison, MAIN_METHOD, RANDOM_METHOD, "dynamic_affinity_distribution_ks", lower_is_better=True, label="Main vs random pairing on dynamic affinity KS"))
    lines.extend(pairwise_statement(comparison, MAIN_METHOD, MEDIAN_METHOD, "lifecycle_score", lower_is_better=False, label="Local kernel vs median mixture on lifecycle"))
    lines.extend(pairwise_statement(comparison, EMPIRICAL_EXACT_METHOD, MAIN_METHOD, "exact_event_overlap_rate", lower_is_better=False, label="Empirical exact vs main on exact-event memorization"))
    lines.extend(duplicate_ratio_statement(comparison.get(MAIN_METHOD, {})))
    return lines


def section(title: str, result: Optional[tuple[str, float]]) -> list[str]:
    if result is None:
        return [f"## {title}", "", "Not enough available metrics to identify a method.", ""]
    method, value = result
    return [f"## {title}", "", f"{label(method)} is best among available methods by this criterion ({format_float(value)}).", ""]


def best_by_metric(comparison: Dict[str, Dict[str, Any]], metric: str, higher_is_better: bool) -> Optional[tuple[str, float]]:
    return best_by_score(comparison, lambda values: to_float(values.get(metric)), higher_is_better)


def best_by_score(comparison: Dict[str, Dict[str, Any]], scorer, higher_is_better: bool) -> Optional[tuple[str, float]]:
    scored = []
    for method, metrics in comparison.items():
        score = scorer(metrics)
        if score is None or not np.isfinite(score):
            continue
        scored.append((method, score))
    if not scored:
        return None
    return max(scored, key=lambda item: item[1]) if higher_is_better else min(scored, key=lambda item: item[1])


def lifecycle_score(metrics: Dict[str, Any]) -> Optional[float]:
    values = [to_float(metrics.get(key)) for key in LIFECYCLE_KEYS]
    values = [value for value in values if value is not None and np.isfinite(value)]
    if not values:
        return None
    return float(np.mean(values))


def c2st_distance_from_chance(metrics: Dict[str, Any]) -> Optional[float]:
    auc = to_float(metrics.get("event_tuple_c2st_auc"))
    if auc is None:
        return None
    return abs(auc - 0.5)


def pairwise_statement(
    comparison: Dict[str, Dict[str, Any]],
    left: str,
    right: str,
    metric: str,
    lower_is_better: bool,
    label: str,
) -> list[str]:
    left_metrics = comparison.get(left)
    right_metrics = comparison.get(right)
    if not left_metrics or not right_metrics:
        return [f"## {label}", "", "One or both methods are missing, so this comparison is not available.", ""]
    left_value = lifecycle_score(left_metrics) if metric == "lifecycle_score" else to_float(left_metrics.get(metric))
    right_value = lifecycle_score(right_metrics) if metric == "lifecycle_score" else to_float(right_metrics.get(metric))
    if left_value is None or right_value is None:
        return [f"## {label}", "", "The required metric is missing for one or both methods.", ""]
    condition = left_value < right_value if lower_is_better else left_value > right_value
    direction = ("lower" if lower_is_better else "higher") if condition else ("not lower" if lower_is_better else "not higher")
    metric_name = "mean lifecycle correlation" if metric == "lifecycle_score" else metric
    return [
        f"## {label}",
        "",
        f"{label_method(left)} is {direction} than {label_method(right)} on {metric_name}: "
        f"{format_float(left_value)} vs {format_float(right_value)}.",
        "",
    ]


def duplicate_ratio_statement(main_metrics: Dict[str, Any]) -> list[str]:
    ratio = to_float(main_metrics.get("duplicate_rate_ratio"))
    real_rate = to_float(main_metrics.get("real_duplicate_customer_product_rate"))
    synthetic_rate = to_float(main_metrics.get("synthetic_duplicate_customer_product_rate"))
    lines = ["## Duplicate ratio", ""]
    if ratio is None:
        lines.append("Duplicate ratio is unavailable, so no acceptability statement is made.")
    elif 0.8 <= ratio <= 1.25:
        lines.append(
            "The main method's duplicate rate is close to the real duplicate rate "
            f"(real {format_float(real_rate)}, synthetic {format_float(synthetic_rate)}, ratio {format_float(ratio)})."
        )
    else:
        lines.append(
            "The main method's duplicate rate differs from the real duplicate rate and should be reported "
            f"(real {format_float(real_rate)}, synthetic {format_float(synthetic_rate)}, ratio {format_float(ratio)})."
        )
    lines.append("")
    return lines


def label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def label_method(method: str) -> str:
    return label(method)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.4g}"


if __name__ == "__main__":
    main()
