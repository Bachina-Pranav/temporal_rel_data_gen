#!/usr/bin/env python3
"""Audit raw versus canonical categorical validity in completed LSTM runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.utils import (  # noqa: E402
    canonicalize_categorical_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous-root",
        default="outputs/architecture_finalization",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/architecture_finalization_compact/validity_audit.md"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    previous = Path(args.previous_root)
    all_runs_path = previous / "all_runs.csv"
    if not all_runs_path.is_file():
        raise FileNotFoundError(f"Missing previous run table: {all_runs_path}")
    rows = pd.read_csv(all_runs_path)
    suspects = rows[
        pd.to_numeric(
            rows.get("invalid_categorical_rate"),
            errors="coerce",
        ).fillna(0.0)
        > 0.0
    ]
    findings = []
    for _, row in suspects.iterrows():
        findings.extend(audit_run(row))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_audit(findings, len(suspects)), encoding="utf-8")
    print(output)


def audit_run(row: pd.Series) -> list[dict[str, Any]]:
    metrics_path = Path(str(row["metrics_path"]))
    run_root = metrics_path.parents[2]
    config_path = run_root / "config_resolved.yaml"
    eval_path = run_root / "evaluation_config_resolved.yaml"
    attribute_path = run_root / "evaluation/attribute_diagnostics.json"
    synthetic_path = run_root / "samples/synthetic_interactions.csv"
    required = [config_path, eval_path, attribute_path, synthetic_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return [
            {
                "dataset": row.get("dataset"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "status": "incomplete_audit",
                "missing": missing,
            }
        ]
    config = load_yaml(config_path)
    evaluation = load_yaml(eval_path)
    attribute = load_json(attribute_path)
    columns = (((config.get("columns") or {}).get("target") or {}).get(
        "categorical"
    ) or [])
    eval_columns = ((evaluation.get("table") or {}).get("columns") or {})
    synthetic = pd.read_csv(
        synthetic_path,
        usecols=[column for column in columns if column],
        low_memory=False,
    )
    findings = []
    for column in columns:
        stored = (
            ((attribute.get("categorical_attributes") or {}).get(column) or {})
            .get("invalid_category_rate")
        )
        if not finite_positive(stored):
            continue
        training_domain = (
            ((attribute.get("categorical_attributes") or {}).get(column) or {})
            .get("train_domain")
            or []
        )
        findings.append(
            audit_field(
                synthetic[column],
                training_domain,
                eval_columns.get(column) or {},
                dataset=row.get("dataset"),
                model=row.get("model"),
                seed=row.get("seed"),
                column=column,
                stored_invalid_rate=float(stored),
            )
        )
    if not findings:
        findings.append(
            {
                "dataset": row.get("dataset"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "column": "NA",
                "stored_invalid_rate": row.get(
                    "invalid_categorical_rate"
                ),
                "raw_invalid_rate": None,
                "canonical_invalid_rate": None,
                "status": "non_applicable_aggregation_bug",
            }
        )
    return findings


def audit_field(
    synthetic: pd.Series,
    training_domain: list[Any],
    column_config: dict[str, Any],
    **identity: Any,
) -> dict[str, Any]:
    raw_values = synthetic.dropna().astype(str)
    raw_domain = {str(value) for value in training_domain}
    raw_invalid = ~raw_values.isin(raw_domain)
    canonical_values = canonicalize_categorical_series(
        synthetic,
        column_config,
    ).dropna()
    canonical_domain = set(
        canonicalize_categorical_series(
            pd.Series(training_domain),
            column_config,
        ).dropna()
    )
    canonical_invalid = (
        ~canonical_values.isin(canonical_domain)
    )
    raw_rate = float(raw_invalid.mean()) if len(raw_values) else None
    canonical_rate = (
        float(canonical_invalid.mean()) if len(canonical_values) else None
    )
    return {
        **identity,
        "status": (
            "evaluator_representation_bug"
            if (raw_rate or 0.0) > 0.0
            and (canonical_rate or 0.0) == 0.0
            else "genuine_invalid_values"
        ),
        "raw_invalid_rate": raw_rate,
        "canonical_invalid_rate": canonical_rate,
        "raw_training_domain": sorted(raw_domain)[:25],
        "canonical_training_domain": sorted(
            canonical_domain,
            key=lambda value: str(value),
        )[:25],
        "raw_invalid_values": sorted(
            set(raw_values[raw_invalid]),
            key=str,
        )[:25],
        "canonical_invalid_values": sorted(
            set(canonical_values[canonical_invalid].dropna()),
            key=str,
        )[:25],
    }


def render_audit(findings: list[dict[str, Any]], suspect_runs: int) -> str:
    bugs = [
        item
        for item in findings
        if item.get("status") == "evaluator_representation_bug"
    ]
    genuine = [
        item
        for item in findings
        if item.get("status") == "genuine_invalid_values"
    ]
    incomplete = [
        item
        for item in findings
        if item.get("status") == "incomplete_audit"
    ]
    non_applicable = [
        item
        for item in findings
        if item.get("status") == "non_applicable_aggregation_bug"
    ]
    status = "PASS" if not genuine and not incomplete else "FAIL"
    lines = [
        "# Categorical Validity Audit",
        "",
        f"Status: **{status}**",
        "",
        f"- Prior runs flagged: `{suspect_runs}`",
        f"- Representation/canonicalization bugs: `{len(bugs)}`",
        f"- Genuine invalid-domain findings: `{len(genuine)}`",
        f"- Non-applicable aggregation bugs: `{len(non_applicable)}`",
        f"- Incomplete audits: `{len(incomplete)}`",
        "",
        "The old auxiliary attribute diagnostic converted categories with "
        "`astype(str)`. Integer-equivalent values such as real `1.0` and "
        "generated `1` therefore appeared different. The canonical paper "
        "constraint evaluator already treated them as the same category. "
        "The auxiliary evaluator now uses the same schema-driven "
        "canonicalization. Non-applicable categorical metrics remain `NA`.",
        "",
        "| Dataset | Model | Seed | Field | Stored | Raw | Canonical | Finding |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for item in findings:
        lines.append(
            "| {dataset} | {model} | {seed} | {column} | {stored} | "
            "{raw} | {canonical} | {status} |".format(
                dataset=item.get("dataset", "NA"),
                model=item.get("model", "NA"),
                seed=item.get("seed", "NA"),
                column=item.get("column", "NA"),
                stored=fmt(item.get("stored_invalid_rate")),
                raw=fmt(item.get("raw_invalid_rate")),
                canonical=fmt(item.get("canonical_invalid_rate")),
                status=item.get("status"),
            )
        )
        if item.get("raw_invalid_values"):
            lines.append(
                "\nRaw values flagged for this field: `" 
                + json.dumps(item["raw_invalid_values"])
                + "`."
            )
        if item.get("canonical_invalid_values"):
            lines.append(
                "\nCanonical values still invalid: `"
                + json.dumps(item["canonical_invalid_values"])
                + "`."
            )
    return "\n".join(lines) + "\n"


def finite_positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
