"""Config-driven constraint validation for generated event tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import datetime_series, is_null_like, normalize_value, numeric_series


def constraint_violation_metrics(synthetic: pd.DataFrame, table_config: dict[str, Any]) -> dict[str, Any]:
    columns_cfg = dict(table_config.get("columns", {}) or {})
    num_rows = int(len(synthetic))
    row_violating = pd.Series(False, index=synthetic.index)
    per_column: dict[str, dict[str, Any]] = {}
    per_constraint_counts = {
        "dtype": 0,
        "null": 0,
        "categorical_domain": 0,
        "primary_key": 0,
        "foreign_key": 0,
        "datetime_parse": 0,
    }
    total_checked = 0
    total_violations = 0

    for column, cfg in columns_cfg.items():
        cfg = dict(cfg or {})
        checked: list[str] = []
        violation_mask = pd.Series(False, index=synthetic.index)
        values = synthetic[column] if column in synthetic.columns else pd.Series([pd.NA] * num_rows, index=synthetic.index)
        if not bool(cfg.get("nullable", True)):
            checked.append("null")
            mask = values.map(is_null_like)
            violation_mask |= mask
            per_constraint_counts["null"] += int(mask.sum())
            total_checked += num_rows
            total_violations += int(mask.sum())
        col_type = str(cfg.get("type", "categorical")).lower()
        if col_type in {"numerical", "numeric", "number"}:
            checked.append("dtype")
            parsed = numeric_series(values)
            mask = parsed.isna() & ~values.map(is_null_like)
            violation_mask |= mask
            per_constraint_counts["dtype"] += int(mask.sum())
            total_checked += num_rows
            total_violations += int(mask.sum())
        elif col_type == "datetime":
            checked.extend(["dtype", "datetime_parse"])
            parsed = datetime_series(values)
            mask = parsed.isna() & ~values.map(is_null_like)
            violation_mask |= mask
            per_constraint_counts["dtype"] += int(mask.sum())
            per_constraint_counts["datetime_parse"] += int(mask.sum())
            total_checked += num_rows
            total_violations += int(mask.sum())
        elif col_type == "categorical":
            valid_values = cfg.get("valid_values")
            if valid_values is not None:
                checked.append("categorical_domain")
                valid = {normalize_value(value) for value in valid_values}
                mask = ~values.map(normalize_value).isin(valid) & ~values.map(is_null_like)
                violation_mask |= mask
                per_constraint_counts["categorical_domain"] += int(mask.sum())
                total_checked += num_rows
                total_violations += int(mask.sum())
        elif col_type == "foreign_key":
            checked.append("foreign_key")
            parent_values = load_parent_values(cfg)
            if parent_values is not None:
                mask = ~values.map(normalize_value).isin(parent_values) & ~values.map(is_null_like)
                violation_mask |= mask
                per_constraint_counts["foreign_key"] += int(mask.sum())
                total_checked += num_rows
                total_violations += int(mask.sum())
        elif col_type == "text":
            checked.append("dtype")
            mask = values.map(lambda value: not (is_null_like(value) or isinstance(value, str)))
            violation_mask |= mask
            per_constraint_counts["dtype"] += int(mask.sum())
            total_checked += num_rows
            total_violations += int(mask.sum())
        row_violating |= violation_mask
        per_column[column] = {
            "violation_rate": float(violation_mask.mean()) if num_rows else None,
            "violation_count": int(violation_mask.sum()),
            "checked_constraints": checked,
        }

    pk = table_config.get("primary_key")
    if pk:
        pk_cols = [pk] if isinstance(pk, str) else list(pk)
        pk_present = all(column in synthetic.columns for column in pk_cols)
        if pk_present:
            missing = synthetic[pk_cols].isna().any(axis=1)
            duplicate = synthetic.duplicated(subset=pk_cols, keep=False)
            mask = missing | duplicate
        else:
            mask = pd.Series(True, index=synthetic.index)
        row_violating |= mask
        per_constraint_counts["primary_key"] += int(mask.sum())
        total_checked += num_rows
        total_violations += int(mask.sum())

    return {
        "constraint_violation_rate": float(row_violating.mean()) if num_rows else None,
        "violation_count_rate": float(total_violations / total_checked) if total_checked else None,
        "num_rows": num_rows,
        "num_violating_rows": int(row_violating.sum()),
        "num_total_violations": int(total_violations),
        "num_total_checked_constraints": int(total_checked),
        "per_constraint": {
            "dtype_violation_rate": rate(per_constraint_counts["dtype"], total_checked),
            "null_violation_rate": rate(per_constraint_counts["null"], total_checked),
            "categorical_domain_violation_rate": rate(per_constraint_counts["categorical_domain"], total_checked),
            "primary_key_violation_rate": rate(per_constraint_counts["primary_key"], total_checked),
            "foreign_key_violation_rate": rate(per_constraint_counts["foreign_key"], total_checked),
            "datetime_parse_violation_rate": rate(per_constraint_counts["datetime_parse"], total_checked),
            "counts": per_constraint_counts,
        },
        "per_column": per_column,
    }


def rate(count: int, denom: int) -> float | None:
    return float(count / denom) if denom else None


def load_parent_values(column_config: dict[str, Any]) -> set[str] | None:
    parent_path = column_config.get("parent_table_path")
    ref_col = (column_config.get("references") or {}).get("column")
    if not parent_path or not ref_col or not Path(parent_path).exists():
        return None
    parent = pd.read_csv(parent_path, usecols=[ref_col])
    return set(parent[ref_col].map(normalize_value))

