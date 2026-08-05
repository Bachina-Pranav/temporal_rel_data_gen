"""Integrity controls for classifier two-sample tests."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from .c2st import single_table_c2st_metrics


def c2st_integrity_audit(
    real: pd.DataFrame,
    config: dict[str, Any],
    *,
    max_rows_per_side: int = 5000,
    seed: int = 42,
    chance_tolerance: float = 0.15,
    corruption_auc_minimum: float = 0.80,
) -> dict[str, Any]:
    """Run chance and positive controls through the production C2ST pipeline."""

    if len(real) < 8:
        raise ValueError("C2ST integrity audit requires at least 8 real rows")
    shuffled = real.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    side = min(int(max_rows_per_side), len(shuffled) // 2)
    left = shuffled.iloc[:side].reset_index(drop=True)
    right = shuffled.iloc[side : 2 * side].reset_index(drop=True)
    audit_config = copy.deepcopy(config)
    audit_config.setdefault("evaluation", {})["random_seed"] = int(seed)
    audit_config["evaluation"].setdefault("c2st", {})["max_rows"] = int(side)

    controls = {
        "real_vs_disjoint_real": (left, right, "chance"),
        "real_vs_row_shuffled_real": (
            left,
            left.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True),
            "chance",
        ),
        "identical_copies": (left, left.copy(), "chance"),
        "real_vs_obvious_corruption": (
            left,
            obviously_corrupt(left, audit_config.get("table") or {}),
            "separable",
        ),
    }
    results: dict[str, Any] = {}
    all_passed = True
    for name, (first, second, expectation) in controls.items():
        metrics, _ = single_table_c2st_metrics(first, second, audit_config)
        auc = metrics.get("auc")
        if auc is None:
            passed = False
        elif expectation == "chance":
            passed = abs(float(auc) - 0.5) <= float(chance_tolerance)
        else:
            passed = float(auc) >= float(corruption_auc_minimum)
        all_passed = all_passed and passed
        results[name] = {
            "expectation": expectation,
            "passed": bool(passed),
            "auc": auc,
            "accuracy": metrics.get("accuracy"),
            "c2st_error": metrics.get("error"),
            "num_rows": metrics.get("num_rows"),
            "best_classifier": metrics.get("best_classifier"),
        }
    return {
        "status": "passed" if all_passed else "failed",
        "all_controls_passed": bool(all_passed),
        "chance_auc_target": 0.5,
        "chance_tolerance": float(chance_tolerance),
        "corruption_auc_minimum": float(corruption_auc_minimum),
        "metric_semantics": {
            "auc": "0.5 is chance; higher means easier real/synthetic discrimination",
            "accuracy": "0.5 is chance for the balanced audit",
            "c2st_error": "2 * abs(AUC - 0.5); lower is better and 0 is chance",
        },
        "num_rows_per_side": int(side),
        "controls": results,
    }


def obviously_corrupt(
    frame: pd.DataFrame, table_config: dict[str, Any]
) -> pd.DataFrame:
    """Apply schema-driven, deliberately conspicuous corruption."""

    corrupted = frame.copy()
    for column, column_config in (table_config.get("columns", {}) or {}).items():
        if column not in corrupted:
            continue
        column_type = str((column_config or {}).get("type", "categorical")).lower()
        if column_type in {"numerical", "numeric", "number"}:
            values = pd.to_numeric(corrupted[column], errors="coerce")
            scale = float(values.std()) if np.isfinite(values.std()) else 1.0
            corrupted[column] = values.fillna(0.0) + max(scale, 1.0) * 20.0
        elif column_type == "datetime":
            values = pd.to_datetime(corrupted[column], errors="coerce", utc=True)
            corrupted[column] = values + pd.Timedelta(days=3650)
        elif column_type == "text":
            corrupted[column] = (
                "__c2st_corruption__ "
                + corrupted[column].fillna("").astype(str).str.slice(0, 16)
            )
        elif column_type not in {"primary_key", "foreign_key", "id"}:
            corrupted[column] = "__c2st_corruption_category__"
    return corrupted
