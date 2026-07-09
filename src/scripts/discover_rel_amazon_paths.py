#!/usr/bin/env python3
"""Discover likely Rel-Amazon input/output paths for the full pipeline."""

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


DEFAULT_ROOTS = [
    "data/original/rel-amazon",
    "data/original/rel-amazon-full",
    "outputs/rel-amazon",
    "outputs/amazon",
    "outputs/relbench-amazon",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover Rel-Amazon tables and synthetic event spines.")
    parser.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = discover_paths([Path(root) for root in args.roots])
    print_report(report)
    if args.output:
        write_json(args.output, report)


def discover_paths(roots: list[Path]) -> dict[str, Any]:
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(path for path in root.rglob("*.csv") if path.is_file())
        candidates.extend(path for path in root.rglob("*.csv.gz") if path.is_file())
    report = {
        "real_review_path": first_matching(candidates, ["review.csv", "review.csv.gz"], under_data=True),
        "customer_path": first_matching(candidates, ["customer.csv", "customer.csv.gz"], under_data=True),
        "product_path": first_matching(candidates, ["product.csv", "product.csv.gz"], under_data=True),
        "synthetic_spine_path": first_matching(candidates, ["synthetic_review.csv", "synthetic_review.csv.gz"], under_outputs=True),
        "searched_roots": [str(root) for root in roots],
    }
    for key in ["real_review_path", "customer_path", "product_path", "synthetic_spine_path"]:
        path = report.get(key)
        if path:
            report[f"{key}_profile"] = profile_csv(Path(path))
    return report


def first_matching(candidates: list[Path], names: list[str], *, under_data: bool = False, under_outputs: bool = False) -> str | None:
    matches = []
    for path in candidates:
        if path.name not in names:
            continue
        text = str(path)
        if under_data and "data/" not in text:
            continue
        if under_outputs and "outputs/" not in text:
            continue
        matches.append(path)
    if not matches:
        return None
    matches = sorted(matches, key=lambda path: (0 if "rel-amazon" in str(path) else 1, len(str(path))))
    return str(matches[0])


def profile_csv(path: Path) -> dict[str, Any]:
    try:
        head = pd.read_csv(path, nrows=1000)
    except Exception as exc:
        return {"path": str(path), "error": str(exc)}
    return {
        "path": str(path),
        "row_count": count_csv_rows(path),
        "columns": list(head.columns),
        "timestamp_column_candidates": [col for col in head.columns if "time" in col.lower() or "date" in col.lower()],
        "text_column_candidates": [
            col
            for col in head.columns
            if "text" in col.lower() or "summary" in col.lower() or "description" in col.lower()
        ],
    }


def print_report(report: dict[str, Any]) -> None:
    for key in ["real_review_path", "customer_path", "product_path", "synthetic_spine_path"]:
        print(f"{key}: {report.get(key)}")
        profile = report.get(f"{key}_profile") or {}
        if profile:
            print(f"  rows: {profile.get('row_count')}")
            print(f"  columns: {profile.get('columns')}")
            print(f"  timestamp candidates: {profile.get('timestamp_column_candidates')}")
            print(f"  text candidates: {profile.get('text_column_candidates')}")


if __name__ == "__main__":
    main()
