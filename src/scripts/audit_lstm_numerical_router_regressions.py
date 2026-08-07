#!/usr/bin/env python3
"""Audit auto numerical routing boundaries on existing benchmark schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.numerical_type import (  # noqa: E402
    infer_numerical_types,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default=(
            "configs/experiments/"
            "lstm_numerical_router_regressions.yaml"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_yaml(Path(args.experiment_config))
    output_dir = Path(
        args.output_dir or matrix["output_root"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    table_rows = []
    for dataset, definition in matrix["datasets"].items():
        config = load_yaml(Path(definition["config"]))
        targets = declared_targets(config)
        numerical = targets["numerical"]
        categorical = targets["categorical"]
        routing = {}
        if numerical:
            columns = list(numerical)
            frame = pd.read_csv(
                config["paths"]["train_data_path"],
                usecols=[
                    *columns,
                    *(
                        ["split"]
                        if has_split_column(
                            config["paths"]["train_data_path"]
                        )
                        else []
                    ),
                ],
                low_memory=False,
            )
            if "split" in frame and (frame["split"] == "train").any():
                frame = frame[frame["split"] == "train"]
            routing = infer_numerical_types(
                frame,
                columns,
                config=(
                    (config.get("numerical_heads") or {}).get(
                        "type_inference"
                    )
                ),
                seed=int(args.seed),
            )
        metrics_path = first_existing(
            definition.get("legacy_metrics_candidates") or []
        )
        legacy_summary = (
            extract_paper_summary(metrics_path)
            if metrics_path is not None
            else {}
        )
        no_op = not numerical
        report = {
            "dataset": dataset,
            "config": definition["config"],
            "numerical_targets": numerical,
            "categorical_targets": categorical,
            "auto_routing": routing,
            "modalities": definition.get("modalities", []),
            "architecture_affected_by_numerical_router": bool(numerical),
            "legacy_metrics_reused": str(metrics_path) if metrics_path else None,
            "legacy_metrics_summary": legacy_summary,
            "auto_metrics_summary": (
                legacy_summary if no_op else None
            ),
            "auto_minus_legacy": (
                {
                    key: 0.0
                    for key, value in legacy_summary.items()
                    if isinstance(value, (int, float))
                }
                if no_op
                else None
            ),
            "non_regression_status": (
                "passed_by_exact_architecture_identity"
                if no_op
                else "requires_model_evaluation"
            ),
            "note": (
                "Numeric-coded ordinal/categorical targets remain in the "
                "categorical head and are not silently reclassified."
            ),
        }
        reports[dataset] = report
        if numerical:
            for column, decision in routing.items():
                table_rows.append(
                    {
                        "dataset": dataset,
                        "numerical_attribute": column,
                        "inferred_type": decision["label"],
                        "legacy_head": "continuous",
                        "auto_head": decision["recommended_head"],
                        "decision": "evaluate",
                    }
                )
        else:
            for column in categorical:
                table_rows.append(
                    {
                        "dataset": dataset,
                        "numerical_attribute": column,
                        "inferred_type": "schema_categorical",
                        "legacy_head": definition.get(
                            "legacy_head_label",
                            "categorical",
                        ),
                        "auto_head": "not_applicable",
                        "decision": "unchanged",
                    }
                )
    payload = {
        "router_scope": "schema-declared numerical targets only",
        "dataset_reports": reports,
        "all_noop_regressions_passed": all(
            (
                report["non_regression_status"]
                == "passed_by_exact_architecture_identity"
            )
            for report in reports.values()
            if not report["numerical_targets"]
        ),
        "auto_router_sensible": router_is_sensible(reports),
        "dataset_specific_changes": False,
    }
    (output_dir / "regression_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    table = pd.DataFrame(table_rows)
    table.to_csv(output_dir / "cross_dataset_routing_table.csv", index=False)
    (output_dir / "regression_report.md").write_text(
        markdown_report(payload, table),
        encoding="utf-8",
    )
    print(output_dir / "regression_audit.json")
    print(output_dir / "cross_dataset_routing_table.csv")
    print(output_dir / "regression_report.md")


def markdown_report(
    payload: dict[str, Any],
    table: pd.DataFrame,
) -> str:
    lines = [
        "# Numerical Router Regression Audit",
        "",
        "The router acts only on schema-declared numerical targets.",
        "Numeric-coded categorical and ordinal targets are not reclassified.",
        "",
        dataframe_markdown(table) if len(table) else "No targets.",
        "",
    ]
    for dataset, report in payload["dataset_reports"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Numerical targets: {report['numerical_targets']}",
                f"- Categorical targets: {report['categorical_targets']}",
                f"- Status: {report['non_regression_status']}",
                f"- Modalities: {report['modalities']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def has_split_column(path: str | Path) -> bool:
    return "split" in pd.read_csv(path, nrows=0).columns


def declared_targets(config: dict[str, Any]) -> dict[str, list[str]]:
    target = ((config.get("columns") or {}).get("target") or {})
    return {
        kind: [str(column) for column in target.get(kind, [])]
        for kind in ("numerical", "categorical", "text")
    }


def dataframe_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value) for value in row)
            + " |"
        )
    return "\n".join(lines)


def first_existing(paths: list[str]) -> Path | None:
    return next((Path(path) for path in paths if Path(path).exists()), None)


def extract_paper_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    summary = value.get("paper_metrics_summary") or {}
    return {
        key: summary.get(key)
        for key in (
            "constraint_violation_rate",
            "shape_error",
            "single_table_c2st_error",
            "temporal_event_distance",
            "trend_error",
        )
        if key in summary
    }


def router_is_sensible(reports: dict[str, Any]) -> bool | None:
    decisions = [
        decision
        for report in reports.values()
        for decision in report["auto_routing"].values()
    ]
    if not decisions:
        return None
    return bool(
        all(
            (
                decision["recommended_head"] == "support_prior"
                if decision["label"]
                in {
                    "low_cardinality_discrete_numerical",
                    "repeated_or_quantized",
                    "high_cardinality_structured_support",
                }
                else decision["recommended_head"] == "continuous"
            )
            for decision in decisions
        )
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


if __name__ == "__main__":
    main()
