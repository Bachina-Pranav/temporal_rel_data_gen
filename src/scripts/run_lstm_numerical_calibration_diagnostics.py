#!/usr/bin/env python3
"""Run Q0-Q4 training-only rank calibration on fixed LSTM outputs."""

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

from attribute_generation.conditional_tabdlm.numerical_calibration import (  # noqa: E402
    CalibrationOptions,
    calibrate_numerical_column,
)
from attribute_generation.conditional_tabdlm.numerical_support import (  # noqa: E402
    finite_array,
    nearest_support_distances,
)
from attribute_generation.conditional_tabdlm.posthoc_diagnostics import (  # noqa: E402
    assert_aligned_spine,
    repeated_c2st,
    weighted_entity_numerical_errors,
)
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import (  # noqa: E402
    ensure_dir,
    load_yaml,
    save_json,
)
from evaluation.paper_metrics.shape_trend import (  # noqa: E402
    shape_metrics,
    trend_metrics,
)
from scripts.evaluate_lstm_attribute_diagnostics import numerical_metrics  # noqa: E402


MODE_LABELS = {
    "Q0": "original",
    "Q1": "global",
    "Q2": "time_bucket",
    "Q3": "destination_frequency_bucket",
    "Q4": "destination_hierarchy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="Completed run_lstm_multiseed_experiment.py output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to EXPERIMENT_ROOT/numerical_calibration.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_LABELS),
        default=list(MODE_LABELS),
    )
    parser.add_argument("--classifier-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-c2st-rows", type=int, default=20000)
    parser.add_argument("--min-destination-rows", type=int, default=20)
    parser.add_argument("--min-bucket-rows", type=int, default=100)
    parser.add_argument("--num-time-buckets", type=int, default=8)
    parser.add_argument("--num-frequency-buckets", type=int, default=4)
    parser.add_argument(
        "--project-support",
        choices=["auto", "yes", "no"],
        default="auto",
    )
    parser.add_argument(
        "--real-numerical-oracle-c2st",
        type=float,
        default=None,
        help="Optional C2ST error from replacing generated numerical values with aligned real values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_root)
    output_dir = ensure_dir(
        args.output_dir or root / "numerical_calibration"
    )
    shared = root / "shared"
    train_path = shared / "train_real.csv"
    real_path = shared / "test_real.csv"
    require_files([train_path, real_path])
    train = pd.read_csv(train_path, low_memory=False)
    real = pd.read_csv(real_path, low_memory=False)
    project_support = {
        "auto": None,
        "yes": True,
        "no": False,
    }[args.project_support]
    options = CalibrationOptions(
        min_destination_rows=int(args.min_destination_rows),
        min_bucket_rows=int(args.min_bucket_rows),
        num_time_buckets=int(args.num_time_buckets),
        num_frequency_buckets=int(args.num_frequency_buckets),
        project_to_training_support=project_support,
    )

    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "experiment_root": str(root),
        "mapping_fit_scope": "training_split_only",
        "training_table": str(train_path),
        "evaluation_table": str(real_path),
        "modes": {
            label: MODE_LABELS[label]
            for label in args.modes
        },
        "seeds": [int(seed) for seed in args.seeds],
        "options": options.__dict__,
        "runs": {},
    }
    for seed in args.seeds:
        run_root = root / "runs" / f"seed_{seed}"
        config_path = run_root / "config_resolved.yaml"
        evaluation_config_path = (
            run_root / "evaluation_config_resolved.yaml"
        )
        synthetic_path = (
            run_root / "samples" / "synthetic_interactions.csv"
        )
        require_files(
            [config_path, evaluation_config_path, synthetic_path]
        )
        model_config = load_config(config_path)
        evaluation_config = load_yaml(evaluation_config_path)
        synthetic = pd.read_csv(synthetic_path, low_memory=False)
        assert_aligned_spine(real, synthetic, model_config.schema)
        source = resolve_event_role(
            model_config.raw,
            "source_fk",
            "source_foreign_key",
        )
        destination = resolve_event_role(
            model_config.raw,
            "destination_fk",
            "destination_foreign_key",
        )
        timestamp = resolve_event_role(
            model_config.raw,
            "timestamp",
            "timestamp",
        )
        run_metadata: dict[str, Any] = {}
        for label in args.modes:
            mode = MODE_LABELS[label]
            print(
                f"[calibration] seed={seed} {label} ({mode})",
                flush=True,
            )
            calibrated = synthetic.copy()
            column_metadata: dict[str, Any] = {}
            if mode != "original":
                for column in model_config.schema.numerical_targets:
                    values, column_report = calibrate_numerical_column(
                        train,
                        calibrated,
                        value_column=column,
                        destination_column=destination,
                        timestamp_column=timestamp,
                        mode=mode,
                        options=options,
                    )
                    calibrated[column] = values
                    column_metadata[column] = column_report
            output_path = (
                output_dir
                / f"seed_{seed}"
                / f"{label}_{mode}.csv"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            calibrated.to_csv(output_path, index=False)
            row = evaluate_calibration(
                train,
                real,
                calibrated,
                model_config,
                evaluation_config,
                source=source,
                destination=destination,
                label=label,
                mode=mode,
                seed=int(seed),
                classifier_seeds=args.classifier_seeds,
                max_c2st_rows=args.max_c2st_rows,
            )
            row["synthetic_table"] = str(output_path)
            rows.append(row)
            run_metadata[label] = column_metadata
            pd.DataFrame(rows).to_csv(
                output_dir / "calibration_results_progress.csv",
                index=False,
            )
        metadata["runs"][str(seed)] = run_metadata

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "calibration_results.csv", index=False)
    aggregate = aggregate_results(results)
    aggregate.to_csv(
        output_dir / "calibration_results_aggregate.csv",
        index=False,
    )
    interpretation = calibration_interpretation(
        aggregate,
        real_numerical_oracle_c2st=args.real_numerical_oracle_c2st,
    )
    save_json(metadata, output_dir / "calibration_metadata.json")
    save_json(interpretation, output_dir / "calibration_interpretation.json")
    write_markdown(
        aggregate,
        interpretation,
        output_dir / "calibration_report.md",
    )
    print(output_dir / "calibration_results.csv")
    print(output_dir / "calibration_results_aggregate.csv")
    print(output_dir / "calibration_interpretation.json")


def evaluate_calibration(
    train: pd.DataFrame,
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    model_config: Any,
    evaluation_config: dict[str, Any],
    *,
    source: str,
    destination: str,
    label: str,
    mode: str,
    seed: int,
    classifier_seeds: list[int],
    max_c2st_rows: int,
) -> dict[str, Any]:
    shape, _ = shape_metrics(
        real,
        synthetic,
        evaluation_config["table"],
        evaluation_config,
    )
    trend, _ = trend_metrics(
        real,
        synthetic,
        evaluation_config["table"],
        evaluation_config,
    )
    full_c2st, _ = repeated_c2st(
        real,
        synthetic,
        evaluation_config,
        classifier_seeds=classifier_seeds,
        max_rows=max_c2st_rows,
        generator_seed=seed,
        label=label,
    )
    numerical_columns = list(model_config.schema.numerical_targets)
    numerical_c2st, _ = repeated_c2st(
        real,
        synthetic,
        evaluation_config,
        classifier_seeds=classifier_seeds,
        columns=numerical_columns,
        max_rows=max_c2st_rows,
        generator_seed=seed,
        label=f"{label}_numerical_only",
    )
    row: dict[str, Any] = {
        "seed": int(seed),
        "calibration": label,
        "mode": mode,
        "shape_error": shape.get(
            "macro_attribute_shape_error"
        ),
        "trend_error": trend.get("macro_headline_trend_error"),
        "full_row_c2st": full_c2st.get("c2st_error_mean"),
        "full_row_auc": full_c2st.get("auc_mean"),
        "numerical_only_c2st": numerical_c2st.get(
            "c2st_error_mean"
        ),
        "numerical_only_auc": numerical_c2st.get("auc_mean"),
    }
    for column in numerical_columns:
        metrics = numerical_metrics(
            train[column],
            real[column],
            synthetic[column],
        )
        support = np.unique(finite_array(train[column]))
        syn_values = finite_array(synthetic[column])
        distances = nearest_support_distances(
            syn_values,
            support,
        )
        destination_metrics = weighted_entity_numerical_errors(
            real,
            synthetic,
            destination,
            column,
        )
        source_metrics = weighted_entity_numerical_errors(
            real,
            synthetic,
            source,
            column,
        )
        prefix = f"{column}."
        row.update(
            {
                prefix + "ks": metrics["ks_distance"],
                prefix + "wasserstein": metrics[
                    "wasserstein_distance"
                ],
                prefix + "mean_error": metrics[
                    "mean_absolute_error"
                ],
                prefix + "std_error": metrics[
                    "std_absolute_error"
                ],
                prefix + "quantile_mae": metrics["quantile_mae"],
                prefix + "support_overlap": (
                    float(np.mean(np.isin(syn_values, support)))
                    if len(syn_values)
                    else None
                ),
                prefix + "nearest_support_mean": (
                    float(np.mean(distances))
                    if len(distances)
                    else None
                ),
                prefix + "unique_value_ratio": (
                    float(len(np.unique(syn_values)) / len(syn_values))
                    if len(syn_values)
                    else None
                ),
                prefix
                + "destination_conditioned_standardized_mae": (
                    destination_metrics[
                        "weighted_group_mean_standardized_mae"
                    ]
                ),
                prefix + "source_conditioned_standardized_mae": (
                    source_metrics[
                        "weighted_group_mean_standardized_mae"
                    ]
                ),
            }
        )
    return row


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in results.columns
        if column not in {
            "seed",
            "calibration",
            "mode",
            "synthetic_table",
        }
        and pd.api.types.is_numeric_dtype(results[column])
    ]
    rows: list[dict[str, Any]] = []
    for (label, mode), frame in results.groupby(
        ["calibration", "mode"],
        sort=False,
    ):
        row: dict[str, Any] = {
            "calibration": label,
            "mode": mode,
            "num_seeds": int(frame["seed"].nunique()),
        }
        for column in numeric:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_std"] = float(frame[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_interpretation(
    aggregate: pd.DataFrame,
    *,
    real_numerical_oracle_c2st: float | None,
) -> dict[str, Any]:
    indexed = aggregate.set_index("calibration")
    q0 = value_at(indexed, "Q0", "full_row_c2st_mean")
    q1 = value_at(indexed, "Q1", "full_row_c2st_mean")
    conditional = {
        label: value_at(indexed, label, "full_row_c2st_mean")
        for label in ("Q2", "Q3", "Q4")
        if label in indexed.index
    }
    best_label = (
        min(
            conditional,
            key=lambda key: conditional[key],
        )
        if conditional
        else None
    )
    best_value = (
        conditional[best_label]
        if best_label is not None
        else None
    )
    denominator = (
        q0 - float(real_numerical_oracle_c2st)
        if q0 is not None and real_numerical_oracle_c2st is not None
        else None
    )
    q1_fraction = safe_fraction(
        None if q0 is None or q1 is None else q0 - q1,
        denominator,
    )
    best_fraction = safe_fraction(
        None if q0 is None or best_value is None else q0 - best_value,
        denominator,
    )
    additional = (
        None
        if q1_fraction is None or best_fraction is None
        else best_fraction - q1_fraction
    )
    ranking_useful = (
        bool(best_value < q1 - 0.01)
        if best_value is not None and q1 is not None
        else None
    )
    return {
        "q0_original_full_row_c2st": q0,
        "q1_global_full_row_c2st": q1,
        "best_conditional_mode": best_label,
        "best_conditional_full_row_c2st": best_value,
        "real_numerical_oracle_c2st": real_numerical_oracle_c2st,
        "fraction_oracle_improvement_recovered_by_q1": q1_fraction,
        "fraction_oracle_improvement_recovered_by_best_q2_q4": (
            best_fraction
        ),
        "additional_fraction_recovered_beyond_q1": additional,
        "conditional_row_ranking_appears_useful": ranking_useful,
        "interpretation_rule": (
            "Q1 isolates global marginal calibration. Incremental Q2-Q4 "
            "improvement measures usable temporal/destination ranking. "
            "Remaining distance to the aligned real-numerical oracle "
            "requires a stronger trained conditional head."
        ),
    }


def resolve_event_role(
    raw: dict[str, Any],
    event_key: str,
    schema_role: str,
) -> str:
    explicit = (raw.get("event_spine") or {}).get(event_key)
    if explicit:
        return str(explicit)
    fields = (raw.get("schema") or {}).get("fields") or {}
    matches = [
        str(column)
        for column, metadata in fields.items()
        if str((metadata or {}).get("role", "")) == schema_role
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not resolve event role {event_key!r}; "
            f"schema role {schema_role!r} matched {matches}"
        )
    return matches[0]


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing completed experiment files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def value_at(
    indexed: pd.DataFrame,
    row: str,
    column: str,
) -> float | None:
    if row not in indexed.index or column not in indexed.columns:
        return None
    value = indexed.loc[row, column]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return float(value) if pd.notna(value) else None


def safe_fraction(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    if (
        numerator is None
        or denominator is None
        or abs(float(denominator)) < 1e-12
    ):
        return None
    return float(numerator / denominator)


def write_markdown(
    aggregate: pd.DataFrame,
    interpretation: dict[str, Any],
    path: Path,
) -> None:
    columns = [
        column
        for column in [
            "calibration",
            "full_row_c2st_mean",
            "numerical_only_c2st_mean",
            "shape_error_mean",
            "trend_error_mean",
        ]
        if column in aggregate
    ]
    lines = [
        "# Numerical calibration Q0-Q4",
        "",
        aggregate[columns].to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "```json",
        json.dumps(interpretation, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
