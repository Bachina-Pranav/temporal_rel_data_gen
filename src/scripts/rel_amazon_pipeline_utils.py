"""Shared helpers for Rel-Amazon full single-event-table pipeline scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_DASHBOARD_KEYS = [
    "constraint_violation_rate",
    "fk_cardinality_similarity",
    "temporal_event_distance",
    "shape_error",
    "trend_error",
    "text_embedding_c2st_error",
    "single_table_c2st_error",
]


def count_csv_rows(path: str | Path) -> int:
    path = Path(path)
    return int(sum(len(chunk) for chunk in pd.read_csv(path, usecols=[0], chunksize=1_000_000)))


def read_csv_head(path: str | Path, nrows: int = 1000, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, usecols=usecols)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def dashboard_payload(metrics: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    dataset = metrics.get("dataset", {}) or {}
    summary = metrics.get("paper_metrics_summary", {}) or {}
    return {
        "dataset": dataset.get("dataset_name", "rel_amazon"),
        "model": "v51_length_preserving_exact_block",
        "architecture_changed": False,
        "num_real_rows": dataset.get("num_real_rows"),
        "num_synthetic_rows": dataset.get("num_synthetic_rows"),
        "row_count_match": dataset.get("row_count_match"),
        "main_dashboard": {key: summary.get(key) for key in REQUIRED_DASHBOARD_KEYS},
        "runtime": runtime_summary(runtime or {}),
    }


def runtime_summary(runtime: dict[str, Any]) -> dict[str, Any]:
    total = runtime.get("total_sampling_seconds", runtime.get("runtime_seconds_wall_clock"))
    rows_per_second = runtime.get("rows_per_second")
    projected = runtime.get("projected_hours_for_10m_rows")
    if rows_per_second is None and total and runtime.get("num_generated_rows"):
        rows_per_second = float(runtime["num_generated_rows"]) / max(float(total), 1e-9)
    if projected is None and rows_per_second:
        projected = 10_000_000.0 / float(rows_per_second) / 3600.0
    return {
        "total_sampling_seconds": total,
        "rows_per_second": rows_per_second,
        "projected_hours_for_10m_rows": projected,
    }


def print_main_dashboard(metrics: dict[str, Any]) -> None:
    summary = metrics.get("paper_metrics_summary", {}) or {}
    print("# Main Dashboard")
    for key in REQUIRED_DASHBOARD_KEYS:
        print(f"{key}: {summary.get(key)}")
