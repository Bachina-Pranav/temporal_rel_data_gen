#!/usr/bin/env python3
"""Audit Rel-Amazon real review table and frozen synthetic event spine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from rel_amazon_pipeline_utils import count_csv_rows, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit full Rel-Amazon single-event-table generation inputs.")
    parser.add_argument("--real-table", required=True)
    parser.add_argument("--customer-table", required=True)
    parser.add_argument("--product-table", required=True)
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-spine-row-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_inputs(args)
    write_json(args.output, report)
    print(f"Wrote {args.output}")
    if report["fatal_errors"]:
        for error in report["fatal_errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


def audit_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "real_table": Path(args.real_table),
        "customer_table": Path(args.customer_table),
        "product_table": Path(args.product_table),
        "synthetic_spine": Path(args.synthetic_spine),
    }
    fatal_errors = [f"Missing {name}: {path}" for name, path in paths.items() if not path.exists()]
    if fatal_errors:
        return {"paths": {key: str(value) for key, value in paths.items()}, "fatal_errors": fatal_errors, "warnings": []}
    real_rows = count_csv_rows(paths["real_table"])
    spine_rows = count_csv_rows(paths["synthetic_spine"])
    real = pd.read_csv(paths["real_table"])
    customer = pd.read_csv(paths["customer_table"])
    product = pd.read_csv(paths["product_table"])
    spine = pd.read_csv(paths["synthetic_spine"])
    warnings: list[str] = []
    if spine_rows != real_rows:
        message = f"Synthetic spine row count ({spine_rows}) does not match real review row count ({real_rows})."
        warnings.append(message)
        if not args.allow_spine_row_mismatch:
            fatal_errors.append(message)
    report = {
        "paths": {key: str(value) for key, value in paths.items()},
        "real_review": table_profile(real),
        "customer": table_profile(customer),
        "product": table_profile(product),
        "synthetic_spine": table_profile(spine),
        "rating_distribution": distribution(real, "rating"),
        "verified_distribution": distribution(real, "verified"),
        "summary_length_distribution": length_distribution(real, "summary"),
        "review_text_length_distribution": length_distribution(real, "review_text"),
        "timestamp": timestamp_profile(real, "review_time"),
        "fk_parent_coverage": {
            "customer_id": fk_coverage(real, "customer_id", customer, "customer_id"),
            "product_id": fk_coverage(real, "product_id", product, "product_id"),
        },
        "spine_fk_parent_coverage": {
            "customer_id": fk_coverage(spine, "customer_id", customer, "customer_id"),
            "product_id": fk_coverage(spine, "product_id", product, "product_id"),
        },
        "spine_timestamp": timestamp_profile(spine, "review_time"),
        "spine_row_count_matches_real": bool(spine_rows == real_rows),
        "fatal_errors": fatal_errors,
        "warnings": warnings,
    }
    return report


def table_profile(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "null_rates": {column: float(frame[column].isna().mean()) for column in frame.columns},
        "dtype_inference": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }


def distribution(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame:
        return {"status": "missing_column"}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().items()}


def length_distribution(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame:
        return {"status": "missing_column"}
    lengths = frame[column].fillna("").astype(str).map(lambda text: len(text.split()))
    return {
        "min": float(lengths.min()) if len(lengths) else None,
        "median": float(lengths.median()) if len(lengths) else None,
        "mean": float(lengths.mean()) if len(lengths) else None,
        "p95": float(lengths.quantile(0.95)) if len(lengths) else None,
        "max": float(lengths.max()) if len(lengths) else None,
        "empty_rate": float((lengths == 0).mean()) if len(lengths) else None,
    }


def timestamp_profile(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame:
        return {"status": "missing_column"}
    parsed = pd.to_datetime(frame[column], errors="coerce")
    return {
        "parse_error_rate": float(parsed.isna().mean()),
        "min": parsed.min().isoformat() if parsed.notna().any() else None,
        "max": parsed.max().isoformat() if parsed.notna().any() else None,
    }


def fk_coverage(child: pd.DataFrame, child_col: str, parent: pd.DataFrame, parent_col: str) -> dict[str, Any]:
    if child_col not in child or parent_col not in parent:
        return {"status": "missing_column"}
    parent_values = set(parent[parent_col].dropna().astype(str))
    values = child[child_col].dropna().astype(str)
    valid = values.isin(parent_values)
    return {
        "num_non_null_child_values": int(len(values)),
        "invalid_count": int((~valid).sum()),
        "valid_rate": float(valid.mean()) if len(values) else None,
    }


if __name__ == "__main__":
    main()
