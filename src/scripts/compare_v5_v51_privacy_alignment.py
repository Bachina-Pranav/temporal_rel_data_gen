#!/usr/bin/env python3
"""Compare v5 LSTM against v5.1 privacy/alignment patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = {
    "speed": [
        ("total_sampling_seconds", "lower"),
        ("rows_per_second", "higher"),
        ("projected_hours_for_10m_rows", "lower"),
    ],
    "core_quality": [
        ("marginal_categorical.rating_distribution_l1", "lower"),
        ("marginal_categorical.verified_distribution_l1", "lower"),
        ("joint.rating_verified_joint_l1", "lower"),
        ("length_diagnostics.summary_length_ks", "lower"),
        ("length_diagnostics.review_text_length_ks", "lower"),
    ],
    "privacy": [
        ("text_privacy.summary_exact_train_overlap_rate", "lower"),
        ("text_privacy.review_text_exact_train_overlap_rate", "lower"),
        ("text_privacy.summary_nearest_neighbor_rougeL_mean", "lower"),
        ("text_privacy.review_text_nearest_neighbor_rougeL_mean", "lower"),
    ],
    "alignment": [
        ("text_consistency.synthetic_summary_review_text_rougeL_mean", "higher"),
        ("text_consistency.synthetic_summary_review_text_token_jaccard_mean", "higher"),
        ("alignment_gap.gap_to_real_summary_review_text_rougeL", "lower"),
        ("alignment_gap.gap_to_real_summary_review_text_jaccard", "lower"),
    ],
    "consistency": [
        ("text_consistency.rating_text_consistency_accuracy", "higher"),
        ("text_consistency.rating_review_text_consistency_accuracy", "higher"),
        ("text_consistency.verified_review_text_predictor_auc", "higher"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v5 and v5.1 privacy/alignment metrics.")
    parser.add_argument("--v5", required=True)
    parser.add_argument("--v51", required=True)
    parser.add_argument("--v5-runtime", required=True)
    parser.add_argument("--v51-runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v5 = load_json(args.v5)
    v51 = load_json(args.v51)
    v5_runtime = load_json(args.v5_runtime)
    v51_runtime = load_json(args.v51_runtime)
    add_alignment_gaps(v5)
    add_alignment_gaps(v51)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sections: dict[str, list[dict[str, Any]]] = {}
    for section, metrics in METRICS.items():
        rows = []
        for path, direction in metrics:
            left_source = v5_runtime if section == "speed" else v5
            right_source = v51_runtime if section == "speed" else v51
            v5_value = get_path(left_source, path)
            v51_value = get_path(right_source, path)
            rows.append(
                {
                    "metric": path,
                    "direction": direction,
                    "v5": v5_value,
                    "v51": v51_value,
                    "delta": numeric_delta(v5_value, v51_value),
                    "status": status(v5_value, v51_value, direction),
                }
            )
        sections[section] = rows
    payload = {
        "inputs": {
            "v5": args.v5,
            "v51": args.v51,
            "v5_runtime": args.v5_runtime,
            "v51_runtime": args.v51_runtime,
        },
        "sections": sections,
    }
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(payload, output_dir / "comparison.md")
    print(output_dir / "comparison.json")
    print(output_dir / "comparison.md")


def add_alignment_gaps(metrics: dict[str, Any]) -> None:
    consistency = metrics.get("text_consistency", {})
    real_rouge = consistency.get("real_summary_review_text_rougeL_mean")
    syn_rouge = consistency.get("synthetic_summary_review_text_rougeL_mean", consistency.get("summary_review_text_rougeL_mean"))
    real_jaccard = consistency.get("real_summary_review_text_token_jaccard_mean")
    syn_jaccard = consistency.get(
        "synthetic_summary_review_text_token_jaccard_mean",
        consistency.get("summary_review_text_token_jaccard_mean"),
    )
    metrics["alignment_gap"] = {
        "gap_to_real_summary_review_text_rougeL": abs(float(real_rouge) - float(syn_rouge)) if real_rouge is not None and syn_rouge is not None else None,
        "gap_to_real_summary_review_text_jaccard": abs(float(real_jaccard) - float(syn_jaccard)) if real_jaccard is not None and syn_jaccard is not None else None,
    }


def status(v5_value: Any, v51_value: Any, direction: str) -> str:
    if v5_value is None or v51_value is None:
        return "unchanged"
    delta = float(v51_value) - float(v5_value)
    if abs(delta) <= 1e-9:
        return "unchanged"
    if direction == "lower":
        return "improved" if delta < 0 else "worsened"
    return "improved" if delta > 0 else "worsened"


def numeric_delta(v5_value: Any, v51_value: Any) -> float | None:
    if v5_value is None or v51_value is None:
        return None
    return float(v51_value) - float(v5_value)


def get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = ["# v5 vs v5.1 Privacy Alignment", ""]
    for section, rows in payload["sections"].items():
        lines.extend([f"## {section.replace('_', ' ').title()}", "", "| Metric | v5 | v5.1 | Delta | Status |", "|---|---:|---:|---:|---|"])
        for row in rows:
            lines.append(
                f"| {row['metric']} | {fmt(row['v5'])} | {fmt(row['v51'])} | {fmt(row['delta'])} | {row['status']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
