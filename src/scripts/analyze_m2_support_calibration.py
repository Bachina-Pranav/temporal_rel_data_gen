#!/usr/bin/env python3
"""Analyze M2 support-frequency calibration without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.support_calibration import (  # noqa: E402
    support_calibration_metrics,
    support_probability_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-real", required=True)
    parser.add_argument("--validation-real", required=True)
    parser.add_argument(
        "--synthetic",
        nargs="+",
        required=True,
        help="Entries may be PATH or LABEL=PATH.",
    )
    parser.add_argument("--numerical-columns", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_real, low_memory=False)
    validation = pd.read_csv(args.validation_real, low_memory=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, Any] = {}
    for entry in args.synthetic:
        label, path = parse_labeled_path(entry)
        generated = pd.read_csv(path, low_memory=False)
        run: dict[str, Any] = {}
        for column in args.numerical_columns:
            require_columns(train, validation, generated, column)
            table = support_probability_table(
                train[column],
                validation[column],
                generated[column],
                epsilon=float(args.epsilon),
            )
            metrics = support_calibration_metrics(
                table,
                epsilon=float(args.epsilon),
            )
            table_path = (
                output_dir
                / label
                / f"{column}_support_probabilities.csv"
            )
            table_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(table_path, index=False)
            run[column] = {
                **metrics,
                "support_probability_table": str(table_path),
            }
        runs[label] = run
    aggregate = aggregate_runs(runs)
    report = {
        "training_only_target_distribution": True,
        "validation_labels_used_for_diagnosis_only": True,
        "test_labels_used_for_calibration": False,
        "runs": runs,
        "aggregate_mean_std": aggregate,
    }
    json_path = output_dir / "m2_support_calibration_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "m2_support_calibration_report.md"
    markdown_path.write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print(json_path)
    print(markdown_path)


def aggregate_runs(runs: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    columns = sorted(
        {
            column
            for run in runs.values()
            for column in run
        }
    )
    for column in columns:
        records = [
            run[column]
            for run in runs.values()
            if column in run
        ]
        keys = sorted(
            {
                key
                for record in records
                for key, value in record.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        )
        output[column] = {
            key: {
                "mean": float(
                    np.mean([record[key] for record in records])
                ),
                "sample_std": float(
                    np.std(
                        [record[key] for record in records],
                        ddof=1,
                    )
                )
                if len(records) > 1
                else 0.0,
                "num_runs": int(len(records)),
            }
            for key in keys
            if all(record.get(key) is not None for record in records)
        }
    return output


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# M2 Support Calibration Report",
        "",
        "The target support distribution is fitted from the training split only.",
        "",
    ]
    for label, columns in report["runs"].items():
        lines.extend([f"## {label}", ""])
        for column, metrics in columns.items():
            diagnosis = metrics["diagnosis"]
            lines.extend(
                [
                    f"### {column}",
                    "",
                    f"- Support TV: {metrics['total_variation_train_vs_generated']:.6g}",
                    f"- Jensen-Shannon: {metrics['jensen_shannon_train_vs_generated']:.6g}",
                    f"- KL(train || generated): {metrics['kl_train_to_generated']:.6g}",
                    f"- Head-frequency correlation: {format_value(metrics['head_value_frequency_correlation'])}",
                    f"- Support-rank correlation: {format_value(metrics['support_value_rank_correlation'])}",
                    f"- Top-10 mass error: {metrics['top_10_support_mass_error']:.6g}",
                    f"- Top-50 mass error: {metrics['top_50_support_mass_error']:.6g}",
                    f"- Top-100 mass error: {metrics['top_100_support_mass_error']:.6g}",
                    f"- Tail mass error after top 100: {metrics['tail_mass_error_after_top_100']:.6g}",
                    f"- Training entropy: {metrics['train_entropy_nats']:.6g}",
                    f"- Generated entropy: {metrics['generated_entropy_nats']:.6g}",
                    f"- Entropy difference: {metrics['entropy_difference_generated_minus_train']:.6g}",
                    f"- Overproduces rare values: {diagnosis['overproduces_rare_values']}",
                    f"- Underproduces dominant values: {diagnosis['underproduces_dominant_values']}",
                    f"- Flattens support: {diagnosis['flattens_support_distribution']}",
                    f"- Concentrates too strongly: {diagnosis['concentrates_too_strongly']}",
                    f"- Shifts toward central values: {diagnosis['shifts_toward_central_values']}",
                    "",
                    "Support-frequency buckets:",
                    "",
                    "| Bucket | Values | Target mass | Generated mass | Absolute error | Ratio |",
                    "|---:|---:|---:|---:|---:|---:|",
                ]
            )
            lines.extend(
                "| {bucket} | {support_values} | {target_mass:.6g} | "
                "{generated_mass:.6g} | {absolute_mass_error:.6g} | "
                "{generated_to_target_ratio:.6g} |".format(**bucket)
                for bucket in metrics[
                    "calibration_by_support_frequency_bucket"
                ]
            )
            lines.append("")
    lines.extend(["## Aggregate Mean +/- Sample Standard Deviation", ""])
    for column, metrics in report["aggregate_mean_std"].items():
        lines.extend([f"### {column}", ""])
        for metric, values in metrics.items():
            lines.append(
                f"- {metric}: {values['mean']:.6g} +/- "
                f"{values['sample_std']:.3g} "
                f"(n={values['num_runs']})"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    else:
        raw_path = value
        label = Path(raw_path).stem
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing synthetic table: {path}")
    return label, path


def require_columns(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    generated: pd.DataFrame,
    column: str,
) -> None:
    missing = [
        name
        for name, frame in (
            ("train", train),
            ("validation", validation),
            ("generated", generated),
        )
        if column not in frame
    ]
    if missing:
        raise KeyError(
            f"Column {column!r} is missing from: {missing}"
        )


def format_value(value: Any) -> str:
    return "not available" if value is None else f"{float(value):.6g}"


if __name__ == "__main__":
    main()
