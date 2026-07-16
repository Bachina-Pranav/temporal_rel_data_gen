#!/usr/bin/env python3
"""Build a compact comparison CSV from paper-grade metric JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize interaction LSTM/baseline paper metrics.")
    parser.add_argument("--metric", action="append", nargs=3, metavar=("MODEL", "SPINE", "PATH"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for model, spine, path in args.metric:
        metrics = load_json(path)
        rows.append(
            {
                "model": model,
                "spine": spine,
                "metrics_path": path,
                "rating_tv": find_rating_tv(metrics),
                "rating_wasserstein": find_rating_wasserstein(metrics),
                "c2st_auc": get(metrics, ["single_table_c2st", "auc"]),
                "c2st_error": get(metrics, ["single_table_c2st", "error"]),
                "trend_error": get(metrics, ["paper_metrics_summary", "trend_error"]),
                "validity_violations": get(metrics, ["paper_metrics_summary", "constraint_violation_rate"]),
                "runtime": get(metrics, ["runtime", "total_seconds"]),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def get(data: dict[str, Any], path: list[str]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def find_rating_tv(metrics: dict[str, Any]) -> Any:
    rating = get(metrics, ["shape", "per_column", "rating"])
    if isinstance(rating, dict) and rating.get("primary_statistic") == "total_variation":
        return rating.get("shape_error")
    return None


def find_rating_wasserstein(metrics: dict[str, Any]) -> Any:
    rating = get(metrics, ["shape", "per_column", "rating"])
    if not isinstance(rating, dict):
        return None
    secondary = rating.get("secondary_statistics")
    if isinstance(secondary, dict):
        return secondary.get("ordinal_wasserstein_distance", secondary.get("wasserstein_distance"))
    return None


if __name__ == "__main__":
    main()
