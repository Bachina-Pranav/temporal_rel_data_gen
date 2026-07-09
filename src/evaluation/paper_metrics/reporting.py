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
    lines = [
        "# Single Event Table Paper Metrics",
        "",
        f"Dataset: {dataset.get('dataset_name', '')}",
        f"Real table: {dataset.get('real_table_path', '')}",
        f"Synthetic table: {dataset.get('synthetic_table_path', '')}",
        f"Number of real rows: {dataset.get('num_real_rows', '')}",
        f"Number of synthetic rows: {dataset.get('num_synthetic_rows', '')}",
        "",
        "## Main Metrics Table",
        "",
        "| Axis | Metric | Value | Direction | Status |",
        "|---|---|---:|---|---|",
    ]
    for axis, label, key, direction in MAIN_ROWS:
        value = summary.get(key)
        lines.append(f"| {axis} | {label} | {fmt(value)} | {direction} | {status(value)} |")
    lines.extend(
        [
            "",
            "## Skipped Metrics",
            "",
            "- k-hop Relational Correlation: skipped because evaluation_level = single_event_table; requires full multi-table relational generation.",
            "- C2ST-Agg: skipped because evaluation_level = single_event_table; requires full multi-table relational generation.",
            "",
            "## Notes",
            "",
            "The old diagnostic evaluator is intentionally kept separate. This report contains dataset-agnostic single-event-table paper metrics.",
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


def status(value: Any) -> str:
    return "skipped" if value is None else "computed"

