"""Reporting helpers for single-event-table paper metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


MAIN_ROWS = [
    ("Validity", "Constraint Violation Rate", "constraint_violation_rate", "down"),
    ("Relational/temporal fidelity", "FK Cardinality Similarity", "fk_cardinality_similarity", "up"),
    ("Relational/temporal fidelity", "Temporal Event Distance", "temporal_event_distance", "down"),
    ("Attribute/text fidelity", "Shape Error", "shape_error", "down"),
    ("Attribute/text fidelity", "Trend Error", "trend_error", "down"),
    ("Attribute/text fidelity", "Text Embedding C2ST Error", "text_embedding_c2st_error", "down"),
    ("Attribute/text fidelity", "Single-Table C2ST Error", "single_table_c2st_error", "down"),
]


def write_markdown_report(metrics: dict[str, Any], output_path: str | Path) -> None:
    summary = metrics.get("paper_metrics_summary", {})
    dataset = metrics.get("dataset", {})
    top_features = (metrics.get("single_table_c2st") or {}).get("top_features") or []
    lines = [
        "# Main Dashboard",
        "",
        f"Dataset: {dataset.get('dataset_name', '')}",
        f"Real table: {dataset.get('real_table_path', '')}",
        f"Synthetic table: {dataset.get('synthetic_table_path', '')}",
        f"Number of real rows: {dataset.get('num_real_rows', '')}",
        f"Number of synthetic rows: {dataset.get('num_synthetic_rows', '')}",
        f"Row count match: {dataset.get('row_count_match', '')}",
        "",
        "| Axis | Metric | Value | Direction | Verdict |",
        "|---|---|---:|---|---|",
    ]
    for axis, label, key, direction in MAIN_ROWS:
        value = summary.get(key)
        lines.append(f"| {axis} | {label} | {fmt(value)} | {direction} | {verdict(key, value)} |")
    lines.extend(["", "# What To Inspect Next", ""])
    notes = inspection_notes(summary)
    if notes:
        lines.extend([f"- {note}" for note in notes])
    else:
        lines.append("- No immediate red flags from the main dashboard thresholds.")
    if top_features:
        lines.extend(["", "Top Single-Table C2ST features:", ""])
        lines.append("| Rank | Classifier | Feature | Importance |")
        lines.append("|---:|---|---|---:|")
        for idx, item in enumerate(top_features[:10], start=1):
            lines.append(
                f"| {idx} | {item.get('classifier', '')} | {item.get('feature_name', '')} | {fmt(item.get('abs_importance'))} |"
            )
    lines.extend(
        [
            "",
            "# Skipped Full-Relational Metrics",
            "",
            "- k-hop Relational Correlation: skipped because evaluation_level = single_event_table; requires full multi-table relational generation.",
            "- C2ST-Agg: skipped because evaluation_level = single_event_table; requires full multi-table relational generation.",
        ]
    )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None or frame.empty:
        pd.DataFrame().to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def verdict(key: str, value: Any) -> str:
    if value is None:
        return "skipped"
    value = float(value)
    if key == "constraint_violation_rate":
        return "ok" if value == 0 else "inspect"
    if key == "fk_cardinality_similarity":
        return "ok" if value >= 0.9 else "inspect"
    if key in {"temporal_event_distance", "shape_error", "trend_error", "text_embedding_c2st_error", "single_table_c2st_error"}:
        return "ok" if value <= 0.1 else "inspect"
    return "computed"


def inspection_notes(summary: dict[str, Any]) -> list[str]:
    notes = []
    if gt(summary.get("constraint_violation_rate"), 0.0):
        notes.append("Constraint violations found; inspect the `validity` section in metrics.json.")
    if lt(summary.get("fk_cardinality_similarity"), 0.9):
        notes.append("FK Cardinality Similarity is low; inspect `per_fk_metrics.csv`.")
    if gt(summary.get("temporal_event_distance"), 0.1):
        notes.append("Temporal Event Distance is high; inspect `per_temporal_metrics.csv`.")
    if gt(summary.get("shape_error"), 0.1):
        notes.append("Shape Error is high; inspect `per_column_metrics.csv`.")
    if gt(summary.get("trend_error"), 0.1):
        notes.append("Trend Error is high; inspect `per_pair_trend_metrics.csv`.")
    if gt(summary.get("text_embedding_c2st_error"), 0.1):
        notes.append("Text Embedding C2ST Error is high; inspect `text_embedding_c2st_report.json`.")
    if gt(summary.get("single_table_c2st_error"), 0.1):
        notes.append("Single-Table C2ST Error is high; inspect `c2st_feature_importance.csv`.")
    return notes


def gt(value: Any, threshold: float) -> bool:
    return value is not None and float(value) > threshold


def lt(value: Any, threshold: float) -> bool:
    return value is not None and float(value) < threshold
