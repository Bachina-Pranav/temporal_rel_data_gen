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
        diagnostics = load_optional_diagnostics(Path(path).parent / "rating_diagnostics.json")
        rows.append(
            {
                "Model": model,
                "Setting": spine,
                "metrics_path": path,
                "Rating domain": domain_label(diagnostics),
                "Rating TV ↓": first_not_none(get(diagnostics, ["rating_marginal", "total_variation"]), find_rating_tv(metrics)),
                "Rating JS ↓": get(diagnostics, ["rating_marginal", "js"]),
                "Ordinal Wasserstein ↓": first_not_none(get(diagnostics, ["rating_marginal", "ordinal_wasserstein"]), find_rating_wasserstein(metrics)),
                "User MAE ↓": get(diagnostics, ["user_conditional", "user_mean_rating_mae_weighted"]),
                "Movie MAE ↓": get(diagnostics, ["movie_conditional", "movie_mean_rating_mae_weighted"]),
                "Monthly corr ↑": get(diagnostics, ["temporal", "monthly_mean_rating_corr"]),
                "Monthly MAE ↓": get(diagnostics, ["temporal", "monthly_mean_rating_mae"]),
                "C2ST AUC ↓": first_not_none(get(diagnostics, ["c2st", "auc"]), get(metrics, ["single_table_c2st", "auc"])),
                "C2ST error ↓": first_not_none(get(diagnostics, ["c2st", "error"]), get(metrics, ["single_table_c2st", "error"])),
                "Invalid rate ↓": first_not_none(
                    get(diagnostics, ["validity", "synthetic", "canonicalized_invalid_rating_rate"]),
                    get(metrics, ["paper_metrics_summary", "constraint_violation_rate"]),
                ),
                "Trend error ↓": get(metrics, ["paper_metrics_summary", "trend_error"]),
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


def load_optional_diagnostics(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return load_json(path)


def get(data: dict[str, Any], path: list[str]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def domain_label(metrics: dict[str, Any]) -> str | None:
    domain = get(metrics, ["rating_domain"])
    return ",".join(str(value) for value in domain) if isinstance(domain, list) else None


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
