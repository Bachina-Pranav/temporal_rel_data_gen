#!/usr/bin/env python3
"""Summarize processed interaction benchmark datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preprocessing.interaction_datasets.registry import get_adapter, list_datasets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cross-dataset benchmark summary.")
    parser.add_argument("--processed-root", default="data/processed/interaction_benchmarks")
    parser.add_argument("--output-csv", default="outputs/benchmark_dataset_summary.csv")
    parser.add_argument("--output-md", default="docs/benchmark_dataset_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    root = Path(args.processed_root)
    for name in list_datasets():
        adapter = get_adapter(name)
        manifest_path = root / adapter.benchmark_name / "subset_manifest.json"
        if not manifest_path.exists():
            rows.append({"Dataset": adapter.benchmark_name, "Domain": adapter.domain, "Status": "missing"})
            continue
        manifest = json.loads(manifest_path.read_text())
        attrs = manifest.get("attribute_types", {})
        rows.append(
            {
                "Dataset": adapter.benchmark_name,
                "Domain": adapter.domain,
                "Interactions": manifest.get("actual_interactions"),
                "Source entities": manifest.get("selected_source_entities"),
                "Destination entities": manifest.get("selected_destination_entities"),
                "Categorical": ", ".join([k for k, v in attrs.items() if v in {"categorical", "ordinal_categorical", "boolean"}]),
                "Numerical": ", ".join([k for k, v in attrs.items() if v in {"continuous_numerical", "count_numerical"}]),
                "Text": ", ".join([k for k, v in attrs.items() if v == "text"]),
                "Status": "ready",
            }
        )
    frame = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    write_markdown(frame, args.output_md)
    print(frame.to_string(index=False))


def write_markdown(frame: pd.DataFrame, path: str | Path) -> None:
    lines = ["# Benchmark Dataset Summary", ""]
    lines.append("| Dataset | Domain | Interactions | Source entities | Destination entities | Categorical | Numerical | Text |")
    lines.append("| ------- | ------ | -----------: | --------------: | -------------------: | ----------- | --------- | ---- |")
    for _, row in frame.iterrows():
        lines.append(
            f"| {row.get('Dataset', '')} | {row.get('Domain', '')} | {row.get('Interactions', '')} | {row.get('Source entities', '')} | {row.get('Destination entities', '')} | {row.get('Categorical', '')} | {row.get('Numerical', '')} | {row.get('Text', '')} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
