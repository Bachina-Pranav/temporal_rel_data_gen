#!/usr/bin/env python3
"""Compare Conditional TABDLM v1.2 and graph-conditioned v2 metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Conditional TABDLM v1.2 and v2 metrics.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


METRICS = [
    ("conditional_fidelity.product_rating_top_1000_mae", "lower"),
    ("conditional_fidelity.product_verified_top_1000_mae", "lower"),
    ("conditional_fidelity.customer_rating_top_1000_mae", "lower"),
    ("conditional_fidelity.customer_verified_top_1000_mae", "lower"),
    ("marginal_categorical.rating_distribution_l1", "lower"),
    ("marginal_categorical.verified_distribution_l1", "lower"),
    ("joint.rating_verified_joint_l1", "lower"),
    ("temporal.monthly_rating_mean_corr", "higher"),
    ("temporal.monthly_rating_mean_mae", "lower"),
    ("temporal.monthly_verified_rate_corr", "higher"),
    ("temporal.monthly_verified_rate_mae", "lower"),
    ("length_diagnostics.summary_length_mean_synthetic", "reference"),
    ("length_diagnostics.summary_length_ks", "lower"),
    ("text.unique_summary_rate", "higher"),
    ("text_privacy.exact_summary_train_overlap_rate", "lower"),
]


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    graph = load_json(args.graph)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, direction in METRICS:
        base = get_path(baseline, path)
        new = get_path(graph, path)
        delta = None if base is None or new is None else float(new) - float(base)
        rows.append(
            {
                "metric": path,
                "baseline_v1_2": base,
                "graph_v2": new,
                "delta_v2_minus_v1_2": delta,
                "preferred_direction": direction,
                "improved": improved(delta, direction),
            }
        )
    graph_meta = graph.get("graph_conditioning", {})
    invalid_reasons = unsafe_graph_reasons(graph_meta)
    payload = {
        "baseline_metrics_path": str(args.baseline),
        "graph_metrics_path": str(args.graph),
        "graph_conditioning": graph_meta,
        "valid_graph_experiment": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "metric_comparison": rows,
    }
    with (output_dir / "comparison_v1_2_vs_v2.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(payload, output_dir / "comparison_v1_2_vs_v2.md")
    print(output_dir / "comparison_v1_2_vs_v2.json")
    if invalid_reasons:
        print("WARNING: graph experiment marked invalid:", "; ".join(invalid_reasons), file=sys.stderr)


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


def unsafe_graph_reasons(meta: dict[str, Any]) -> list[str]:
    reasons = []
    if meta.get("graph_uses_future_events") is not False:
        reasons.append("graph_uses_future_events must be false")
    if meta.get("graph_uses_target_attributes") is not False:
        reasons.append("graph_uses_target_attributes must be false")
    if meta.get("real_graph_used_at_sampling") is not False:
        reasons.append("real_graph_used_at_sampling must be false")
    if meta.get("temporal_filter_enabled") is not True:
        reasons.append("temporal_filter_enabled must be true")
    if meta.get("temporal_filter_mode") != "past_only":
        reasons.append("temporal_filter_mode must be past_only")
    return reasons


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    lines = ["# Conditional TABDLM v1.2 vs v2", ""]
    lines.append(f"Valid graph experiment: `{payload['valid_graph_experiment']}`")
    if payload["invalid_reasons"]:
        lines.append("")
        lines.append("## Warnings")
        for reason in payload["invalid_reasons"]:
            lines.append(f"- {reason}")
    lines.extend(["", "## Metrics", "", "| Metric | v1.2 | v2 | Delta | Direction | Improved |", "|---|---:|---:|---:|---|---|"])
    for row in payload["metric_comparison"]:
        lines.append(
            "| {metric} | {baseline} | {graph} | {delta} | {direction} | {improved} |".format(
                metric=row["metric"],
                baseline=format_value(row["baseline_v1_2"]),
                graph=format_value(row["graph_v2"]),
                delta=format_value(row["delta_v2_minus_v1_2"]),
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
