#!/usr/bin/env python3
"""Validate processed interaction benchmark subsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preprocessing.interaction_datasets.registry import get_adapter, list_datasets  # noqa: E402
from data_preprocessing.interaction_datasets.validation import validate_subset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate induced interaction subsets.")
    parser.add_argument("--processed-root", default="data/processed/interaction_benchmarks")
    parser.add_argument("--datasets", nargs="+", default=None, choices=[f"{name}_100k" for name in list_datasets()] + list_datasets())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.processed_root)
    names = args.datasets or [f"{name}_100k" for name in list_datasets()]
    failed = False
    for name in names:
        adapter = get_adapter(name)
        report = validate_subset(adapter, root / adapter.benchmark_name)
        (root / adapter.benchmark_name / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        failed = failed or not report["valid"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
