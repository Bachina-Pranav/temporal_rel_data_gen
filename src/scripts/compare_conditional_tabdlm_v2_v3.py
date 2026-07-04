#!/usr/bin/env python3
"""Compare graph-conditioned Conditional TABDLM v2 and v3 metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


METRICS = [
    ("validity.invalid_rating_rate", "lower"),
    ("validity.invalid_verified_rate", "lower"),
    ("conditional_fidelity.customer_rating_top_1000_mae", "lower"),
    ("conditional_fidelity.customer_verified_top_1000_mae", "lower"),
    ("conditional_fidelity.product_rating_top_1000_mae", "lower"),
    ("conditional_fidelity.product_verified_top_1000_mae", "lower"),
    ("marginal_categorical.rating_distribution_l1", "lower"),
    ("marginal_categorical.verified_distribution_l1", "lower"),
    ("joint.rating_verified_joint_l1", "lower"),
    ("temporal.monthly_rating_mean_corr", "higher"),
    ("temporal.monthly_rating_mean_mae", "lower"),
    ("temporal.monthly_verified_rate_corr", "higher"),
    ("temporal.monthly_verified_rate_mae", "lower"),
    ("temporal.monthly_summary_length_corr", "higher"),
    ("temporal.monthly_summary_length_mae", "lower"),
    ("length_diagnostics.summary_length_mean_synthetic", "reference"),
    ("length_diagnostics.summary_length_ks", "lower"),
    ("text.distinct_1", "higher"),
    ("text.distinct_2", "higher"),
    ("text.unique_summary_rate", "higher"),
    ("text_privacy.exact_summary_train_overlap_rate", "lower"),
    ("text_privacy.nearest_neighbor_rougeL_mean", "lower"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Conditional TABDLM v2 and v3 metrics.")
    parser.add_argument("--baseline-v2", required=True)
    parser.add_argument("--attr-denoise-v3", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v2 = load_json(args.baseline_v2)
    v3 = load_json(args.attr_denoise_v3)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, direction in METRICS:
        base = get_path(v2, path)
        new = get_path(v3, path)
        delta = None if base is None or new is None else float(new) - float(base)
        rows.append(
            {
                "metric": path,
                "v2": base,
                "v3": new,
                "delta_v3_minus_v2": delta,
                "preferred_direction": direction,
                "improved": improved(delta, direction),
            }
        )
    meta = v3.get("graph_conditioning", {})
    invalid = unsafe_reasons(meta)
    payload = {
        "v2_metrics_path": str(args.baseline_v2),
        "v3_metrics_path": str(args.attr_denoise_v3),
        "v3_graph_conditioning": meta,
        "valid_v3_experiment": not invalid,
        "invalid_reasons": invalid,
        "metric_comparison": rows,
    }
    with (output_dir / "comparison_v2_vs_v3.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(payload, output_dir / "comparison_v2_vs_v3.md")
    print(output_dir / "comparison_v2_vs_v3.json")
    if invalid:
        print("WARNING: v3 experiment marked invalid: " + "; ".join(invalid), file=sys.stderr)


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


def improved(delta: float | None, direction: str) -> bool | None:
    if delta is None or direction == "reference":
        return None
    if direction == "lower":
        return delta < 0
    if direction == "higher":
        return delta > 0
    return None


def unsafe_reasons(meta: dict[str, Any]) -> list[str]:
    checks = {
        "graph_uses_future_events": False,
        "graph_uses_clean_target_attributes": False,
        "graph_uses_clean_future_attributes": False,
        "real_graph_used_at_sampling": False,
        "temporal_filter_enabled": True,
        "temporal_filter_mode": "past_only",
        "history_source_sampling": "generated_past_synthetic_attributes",
        "sampling_chronological": True,
    }
    reasons = []
    for key, expected in checks.items():
        if meta.get(key) != expected:
            reasons.append(f"{key} expected {expected!r}, got {meta.get(key)!r}")
    return reasons


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    lines = ["# Conditional TABDLM v2 vs v3", ""]
    lines.append(f"Valid v3 experiment: `{payload['valid_v3_experiment']}`")
    if payload["invalid_reasons"]:
        lines.extend(["", "## Warnings"])
        for reason in payload["invalid_reasons"]:
            lines.append(f"- {reason}")
    lines.extend(["", "## Metrics", "", "| Metric | v2 | v3 | Delta | Direction | Improved |", "|---|---:|---:|---:|---|---|"])
    for row in payload["metric_comparison"]:
        lines.append(
            "| {metric} | {v2} | {v3} | {delta} | {direction} | {improved} |".format(
                metric=row["metric"],
                v2=format_value(row["v2"]),
                v3=format_value(row["v3"]),
                delta=format_value(row["delta_v3_minus_v2"]),
                direction=row["preferred_direction"],
                improved=row["improved"],
            )
        )
    Path(path).write_text("\n".join(lines) + "\n")


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
