#!/usr/bin/env python3
"""Consolidate M0-M4 LSTM numerical-head results and decision gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default=(
            "configs/experiments/"
            "lstm_numerical_heads_hm_10k.yaml"
        ),
    )
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_mapping(args.experiment_config)
    output_root = Path(matrix["output_root"])
    output_dir = Path(
        args.output_dir
        or output_root / "results_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = list(
        args.variants
        or (matrix.get("variants") or {}).keys()
    )
    roots = {
        "M0_original_lstm_v53": Path(
            matrix["baseline_experiment_root"]
        ),
        **{
            variant: output_root / variant
            for variant in variants
        },
    }
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for model, root in roots.items():
        for seed in matrix["seeds"]:
            run_root = root / "runs" / f"seed_{int(seed)}"
            row, absent = collect_run(
                model,
                int(seed),
                run_root,
            )
            if row is not None:
                rows.append(row)
            if absent:
                missing.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "missing": absent,
                    }
                )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(
        output_dir / "per_model_per_seed_metrics.csv",
        index=False,
    )
    aggregate = aggregate_rows(per_seed)
    aggregate.to_csv(
        output_dir / "aggregate_mean_std_long.csv",
        index=False,
    )
    wide = aggregate_wide(aggregate)
    wide.to_csv(
        output_dir / "aggregate_mean_std_wide.csv",
        index=False,
    )
    decision = build_decision(
        per_seed,
        matrix,
        output_root,
        missing,
    )
    write_json(decision, output_dir / "final_decision.json")
    write_json(
        {
            "models_requested": list(roots),
            "rows_found": int(len(per_seed)),
            "missing_runs": missing,
            "files": {
                "per_seed": "per_model_per_seed_metrics.csv",
                "aggregate_long": "aggregate_mean_std_long.csv",
                "aggregate_wide": "aggregate_mean_std_wide.csv",
                "decision": "final_decision.json",
            },
        },
        output_dir / "comparison_manifest.json",
    )
    (output_dir / "report.md").write_text(
        markdown_report(per_seed, aggregate, decision),
        encoding="utf-8",
    )
    for path in (
        output_dir / "per_model_per_seed_metrics.csv",
        output_dir / "aggregate_mean_std_wide.csv",
        output_dir / "final_decision.json",
        output_dir / "report.md",
    ):
        print(path)


def collect_run(
    model: str,
    seed: int,
    run_root: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    paths = {
        "paper": (
            run_root
            / "evaluation"
            / "paper_grade"
            / "metrics.json"
        ),
        "attribute": preferred_attribute_path(run_root),
        "training": run_root / "training_metadata.json",
        "sampling": (
            run_root
            / "samples"
            / "metadata"
            / "runtime_sampling_fast.json"
        ),
        "context": (
            run_root
            / "evaluation"
            / "numerical_context_usage.json"
        ),
        "config": run_root / "config_resolved.yaml",
    }
    required = [
        "paper",
        "attribute",
        "training",
        "sampling",
        "config",
    ]
    missing = [
        str(paths[name])
        for name in required
        if not paths[name].exists()
    ]
    if missing:
        return None, missing
    paper = load_json(paths["paper"])
    attribute = load_json(paths["attribute"])
    training = load_json(paths["training"])
    sampling = load_json(paths["sampling"])
    context = (
        load_json(paths["context"])
        if paths["context"].exists()
        else {}
    )
    config = load_mapping(paths["config"])
    summary = paper.get("paper_metrics_summary") or {}
    row: dict[str, Any] = {
        "model": model,
        "seed": int(seed),
        "constraint_violation": summary.get(
            "constraint_violation_rate"
        ),
        "fk_similarity": summary.get(
            "fk_cardinality_similarity"
        ),
        "shape_error": summary.get("shape_error"),
        "full_row_c2st": summary.get(
            "single_table_c2st_error"
        ),
        "temporal_event_distance": summary.get(
            "temporal_event_distance"
        ),
        "trend_error": summary.get("trend_error"),
        "training_seconds": first_present(
            training,
            "total_training_seconds",
            "train_time_seconds",
        ),
        "sampling_seconds": sampling.get(
            "total_sampling_seconds"
        ),
        "rows_per_second": sampling.get("rows_per_second"),
        "parameter_count": training.get("parameter_count"),
        "peak_training_gpu_memory_mb": training.get(
            "peak_gpu_memory_mb"
        ),
        "peak_sampling_gpu_memory_mb": sampling.get(
            "peak_gpu_memory_mb"
        ),
    }
    row.update(flatten_numeric(attribute, "attribute"))
    row.update(flatten_numeric(context, "context"))
    add_metric_aliases(row, attribute, config)
    return row, []


def add_metric_aliases(
    row: dict[str, Any],
    attribute: dict[str, Any],
    config: dict[str, Any],
) -> None:
    group_c2st = attribute.get("attribute_group_c2st") or {}
    row["numerical_only_c2st"] = (
        (group_c2st.get("numerical_only") or {}).get(
            "c2st_error_mean"
        )
    )
    row["categorical_only_c2st"] = (
        (group_c2st.get("categorical_only") or {}).get(
            "c2st_error_mean"
        )
    )
    for column, metrics in (
        attribute.get("numerical_attributes") or {}
    ).items():
        prefix = f"numerical.{column}"
        aliases = {
            "ks": "ks_distance",
            "wasserstein": "wasserstein_distance",
            "mean_error": "mean_absolute_error",
            "std_error": "std_absolute_error",
            "quantile_mae": "quantile_mae",
            "support_overlap": (
                "synthetic_training_support_overlap_rate"
            ),
            "nearest_support_mean": (
                "nearest_training_support_distance_mean"
            ),
            "unique_value_ratio": (
                "unique_value_ratio_synthetic"
            ),
            "support_entropy": "support_entropy_synthetic",
            "invalid_rate": "invalid_rate",
            "out_of_range_rate": "out_of_train_range_rate",
        }
        for alias, key in aliases.items():
            row[f"{prefix}.{alias}"] = metrics.get(key)
    for column, metrics in (
        attribute.get("categorical_attributes") or {}
    ).items():
        prefix = f"categorical.{column}"
        row[f"{prefix}.tv"] = metrics.get(
            "total_variation_distance"
        )
        row[f"{prefix}.js"] = metrics.get(
            "jensen_shannon_distance"
        )
        row[f"{prefix}.invalid_rate"] = metrics.get(
            "invalid_category_rate"
        )
    conditional = attribute.get("conditional_fidelity") or {}
    event_spine = config.get("event_spine") or {}
    source = event_spine.get("source_fk")
    destination = event_spine.get("destination_fk")
    for condition, targets in conditional.items():
        for target, metrics in targets.items():
            if not isinstance(metrics, dict):
                continue
            if "group_mean_standardized_mae" in metrics:
                row[
                    "conditional."
                    f"{condition}.{target}."
                    "standardized_mae"
                ] = metrics.get(
                    "group_mean_standardized_mae"
                )
            if "weighted_group_total_variation" in metrics:
                row[
                    "conditional."
                    f"{condition}.{target}.tv"
                ] = metrics.get(
                    "weighted_group_total_variation"
                )
    for column in (
        attribute.get("numerical_attributes") or {}
    ):
        if destination:
            row[
                f"numerical.{column}.destination_standardized_mae"
            ] = (
                (
                    conditional.get(str(destination)) or {}
                ).get(column)
                or {}
            ).get("group_mean_standardized_mae")
        if source:
            row[
                f"numerical.{column}.source_standardized_mae"
            ] = (
                (conditional.get(str(source)) or {}).get(column)
                or {}
            ).get("group_mean_standardized_mae")


def preferred_attribute_path(run_root: Path) -> Path:
    refreshed = (
        run_root
        / "evaluation"
        / "attribute_diagnostics_numerical_head_comparison.json"
    )
    if refreshed.exists():
        return refreshed
    return (
        run_root
        / "evaluation"
        / "attribute_diagnostics.json"
    )


def aggregate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["model", "metric", "mean", "std", "num_seeds"]
        )
    rows = []
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column != "seed"
    ]
    for model, group in frame.groupby("model", sort=False):
        for metric in numeric_columns:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()
            if not len(values):
                continue
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": (
                        float(values.std(ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "num_seeds": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def aggregate_wide(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    output: list[dict[str, Any]] = []
    for model, group in aggregate.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model}
        for item in group.to_dict(orient="records"):
            row[f"{item['metric']}_mean"] = item["mean"]
            row[f"{item['metric']}_std"] = item["std"]
            row[f"{item['metric']}_num_seeds"] = item[
                "num_seeds"
            ]
        output.append(row)
    return pd.DataFrame(output)


def build_decision(
    frame: pd.DataFrame,
    matrix: dict[str, Any],
    output_root: Path,
    missing: list[dict[str, Any]],
) -> dict[str, Any]:
    calibration_path = (
        output_root
        / "calibration_q0_q4"
        / "calibration_interpretation.json"
    )
    calibration = (
        load_json(calibration_path)
        if calibration_path.exists()
        else {
            "status": "not_run",
            "path": str(calibration_path),
        }
    )
    means = (
        frame.groupby("model").mean(numeric_only=True)
        if not frame.empty
        else pd.DataFrame()
    )
    ablations = {
        "conditioning_only_M1_minus_M0": model_delta(
            means,
            "M1_destination_continuous",
            "M0_original_lstm_v53",
        ),
        "support_only_M2_minus_M0": model_delta(
            means,
            "M2_global_support",
            "M0_original_lstm_v53",
        ),
        "destination_given_support_M3_minus_M2": model_delta(
            means,
            "M3_destination_support",
            "M2_global_support",
        ),
        "prior_given_destination_M4_minus_M3": model_delta(
            means,
            "M4_destination_support_prior",
            "M3_destination_support",
        ),
        "weak_residual_R1_minus_pure_prior_R0": model_delta(
            means,
            "M2P_R1_weak_residual",
            "M2P_R0_global_prior",
        ),
        "moderate_residual_R2_minus_pure_prior_R0": model_delta(
            means,
            "M2P_R2_moderate_residual",
            "M2P_R0_global_prior",
        ),
        "full_residual_R3_minus_pure_prior_R0": model_delta(
            means,
            "M2P_R3_full_residual",
            "M2P_R0_global_prior",
        ),
        "full_prior_residual_R3_minus_ordinary_M2": model_delta(
            means,
            "M2P_R3_full_residual",
            "M2_global_support",
        ),
    }
    candidate_scores: dict[str, float] = {}
    if not means.empty:
        for model, values in means.iterrows():
            if model == "M0_original_lstm_v53":
                continue
            score_terms = [
                values.get("full_row_c2st"),
                values.get("numerical_only_c2st"),
                mean_matching(
                    values,
                    "numerical.",
                    ".destination_standardized_mae",
                ),
                mean_matching(
                    values,
                    "numerical.",
                    ".ks",
                ),
            ]
            finite = [
                float(value)
                for value in score_terms
                if value is not None
                and np.isfinite(float(value))
            ]
            trend = values.get("trend_error")
            if finite:
                candidate_scores[model] = float(
                    sum(finite)
                    + (
                        0.1 * float(trend)
                        if trend is not None
                        and np.isfinite(float(trend))
                        else 0.0
                    )
                )
    winner = (
        min(candidate_scores, key=candidate_scores.get)
        if candidate_scores
        else None
    )
    seeds = [int(value) for value in matrix["seeds"]]
    complete_models = (
        {
            model
            for model, group in frame.groupby("model")
            if sorted(group["seed"].astype(int).tolist())
            == sorted(seeds)
        }
        if not frame.empty
        else set()
    )
    complete_candidates = {
        model: score
        for model, score in candidate_scores.items()
        if model in complete_models
    }
    comparable_winner = (
        min(complete_candidates, key=complete_candidates.get)
        if complete_candidates
        and "M0_original_lstm_v53" in complete_models
        else None
    )
    acceptance = acceptance_checks(
        means.loc[comparable_winner]
        if comparable_winner is not None
        else None
    )
    if comparable_winner is not None:
        acceptance.update(
            comparative_acceptance_checks(
                frame,
                comparable_winner,
                "M0_original_lstm_v53",
            )
        )
    replace_default = bool(
        comparable_winner
        and all(
            check["status"] == "passed"
            for check in acceptance.values()
            if check["required"]
        )
    )
    return {
        "calibration_interpretation": calibration,
        "ablation_deltas": ablations,
        "delta_semantics": (
            "candidate minus reference; negative is an improvement "
            "for error metrics"
        ),
        "candidate_composite_scores_lower_is_better": (
            candidate_scores
        ),
        "best_observed_candidate": winner,
        "three_seed_comparable_candidate": comparable_winner,
        "complete_three_seed_models": sorted(complete_models),
        "acceptance_checks": acceptance,
        "replace_existing_numerical_head": replace_default,
        "recommendation": (
            f"Promote {comparable_winner} as the default numerical "
            "head."
            if replace_default
            else (
                "Do not replace the default yet; complete the staged "
                "runs and satisfy all required acceptance checks."
            )
        ),
        "sales_channel_next": (
            "Address categorical generation next only after the "
            "numerical-head decision, because the established "
            "categorical-only C2ST contribution is much smaller."
        ),
        "missing_runs": missing,
    }


def model_delta(
    means: pd.DataFrame,
    candidate: str,
    reference: str,
) -> dict[str, float] | None:
    if candidate not in means.index or reference not in means.index:
        return None
    metrics = [
        "full_row_c2st",
        "numerical_only_c2st",
        "shape_error",
        "trend_error",
        "rows_per_second",
    ]
    return {
        metric: float(
            means.loc[candidate, metric]
            - means.loc[reference, metric]
        )
        for metric in metrics
        if metric in means
        and np.isfinite(means.loc[candidate, metric])
        and np.isfinite(means.loc[reference, metric])
    }


def acceptance_checks(
    values: pd.Series | None,
) -> dict[str, dict[str, Any]]:
    checks = {
        "constraint_zero": (
            "constraint_violation",
            "eq",
            0.0,
            True,
        ),
        "fk_similarity_one": (
            "fk_similarity",
            "eq",
            1.0,
            True,
        ),
        "full_c2st_target": (
            "full_row_c2st",
            "lt",
            0.65,
            True,
        ),
        "numerical_c2st_target": (
            "numerical_only_c2st",
            "lt",
            0.70,
            True,
        ),
        "numerical_ks_directional_target": (
            "__mean_numerical_ks__",
            "lt",
            0.12,
            False,
        ),
        "destination_numerical_mae_target": (
            "__mean_destination_numerical_mae__",
            "lt",
            0.50,
            True,
        ),
        "no_invalid_numerical_values": (
            "__max_numerical_invalid__",
            "eq",
            0.0,
            True,
        ),
        "no_invalid_categorical_values": (
            "__max_categorical_invalid__",
            "eq",
            0.0,
            True,
        ),
        "practical_sampling_speed": (
            "rows_per_second",
            "gt",
            1000.0,
            True,
        ),
    }
    output = {}
    for name, (metric, operation, threshold, required) in checks.items():
        if values is None:
            value = None
        elif metric == "__mean_numerical_ks__":
            value = mean_matching(
                values,
                "numerical.",
                ".ks",
            )
        elif metric == "__mean_destination_numerical_mae__":
            value = mean_matching(
                values,
                "numerical.",
                ".destination_standardized_mae",
            )
        elif metric == "__max_numerical_invalid__":
            value = max_matching(
                values,
                "numerical.",
                ".invalid_rate",
            )
        elif metric == "__max_categorical_invalid__":
            value = max_matching(
                values,
                "categorical.",
                ".invalid_rate",
            )
        else:
            value = values.get(metric) if metric in values else None
        evaluable = (
            value is not None and np.isfinite(float(value))
        )
        passed = (
            (
                math.isclose(
                    float(value),
                    float(threshold),
                    abs_tol=1e-9,
                )
                if operation == "eq"
                else (
                    float(value) < float(threshold)
                    if operation == "lt"
                    else float(value) > float(threshold)
                )
            )
            if evaluable
            else None
        )
        output[name] = {
            "metric": metric,
            "value": (
                float(value)
                if value is not None and np.isfinite(float(value))
                else None
            ),
            "operation": operation,
            "threshold": float(threshold),
            "required": bool(required),
            "status": (
                "passed"
                if passed is True
                else "failed"
                if passed is False
                else "not_evaluable"
            ),
            "passed": passed,
        }
    return output


def comparative_acceptance_checks(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
) -> dict[str, dict[str, Any]]:
    left = frame[frame["model"] == candidate].set_index("seed")
    right = frame[frame["model"] == baseline].set_index("seed")
    common = left.index.intersection(right.index)
    full_improvements = [
        float(left.loc[seed, "full_row_c2st"])
        < float(right.loc[seed, "full_row_c2st"])
        for seed in common
        if pd.notna(left.loc[seed, "full_row_c2st"])
        and pd.notna(right.loc[seed, "full_row_c2st"])
    ]
    trend_delta = (
        float(left["trend_error"].mean())
        - float(right["trend_error"].mean())
        if len(common)
        else float("nan")
    )
    categorical_columns = [
        column
        for column in frame.columns
        if column.startswith("categorical.")
        and column.endswith(".tv")
    ]
    categorical_regression = max(
        [
            float(left[column].mean())
            - float(right[column].mean())
            for column in categorical_columns
            if column in left
            and column in right
            and left[column].notna().any()
            and right[column].notna().any()
        ]
        or [0.0]
    )
    return {
        "full_c2st_improves_every_seed": {
            "metric": "paired_seed_full_row_c2st",
            "value": (
                float(np.mean(full_improvements))
                if full_improvements
                else None
            ),
            "operation": "all_true",
            "threshold": 1.0,
            "required": True,
            "status": (
                "not_evaluable"
                if len(common) < 3 or not full_improvements
                else "passed"
                if all(full_improvements)
                else "failed"
            ),
            "passed": (
                None
                if len(common) < 3 or not full_improvements
                else bool(all(full_improvements))
            ),
        },
        "trend_does_not_materially_regress": {
            "metric": "trend_error_candidate_minus_baseline",
            "value": (
                trend_delta if np.isfinite(trend_delta) else None
            ),
            "operation": "lte",
            "threshold": 0.02,
            "required": True,
            "status": (
                "not_evaluable"
                if not np.isfinite(trend_delta)
                else "passed"
                if trend_delta <= 0.02
                else "failed"
            ),
            "passed": (
                None
                if not np.isfinite(trend_delta)
                else bool(trend_delta <= 0.02)
            ),
        },
        "categorical_tv_does_not_materially_regress": {
            "metric": "max_categorical_tv_delta",
            "value": categorical_regression,
            "operation": "lte",
            "threshold": 0.05,
            "required": True,
            "status": (
                "passed"
                if categorical_regression <= 0.05
                else "failed"
            ),
            "passed": bool(categorical_regression <= 0.05),
        },
    }


def mean_matching(
    values: pd.Series,
    prefix: str,
    suffix: str,
) -> float | None:
    selected = []
    for key, value in values.items():
        if not str(key).startswith(prefix):
            continue
        if not str(key).endswith(suffix):
            continue
        if value is not None and np.isfinite(float(value)):
            selected.append(float(value))
    return float(np.mean(selected)) if selected else None


def max_matching(
    values: pd.Series,
    prefix: str,
    suffix: str,
) -> float | None:
    selected = [
        float(value)
        for key, value in values.items()
        if str(key).startswith(prefix)
        and str(key).endswith(suffix)
        and value is not None
        and np.isfinite(float(value))
    ]
    return max(selected) if selected else None


def markdown_report(
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    key_metrics = [
        "constraint_violation",
        "fk_similarity",
        "shape_error",
        "full_row_c2st",
        "numerical_only_c2st",
        "temporal_event_distance",
        "trend_error",
        "rows_per_second",
        "training_seconds",
        "sampling_seconds",
        "parameter_count",
    ]
    lines = [
        "# LSTM Numerical-Head Comparison",
        "",
        "All values are mean +/- sample standard deviation across "
        "available generator seeds.",
        "",
        "| Model | " + " | ".join(key_metrics) + " |",
        "|---|" + "|".join(["---:"] * len(key_metrics)) + "|",
    ]
    models = (
        per_seed["model"].drop_duplicates().tolist()
        if not per_seed.empty
        else []
    )
    for model in models:
        cells = [
            format_aggregate(aggregate, model, metric)
            for metric in key_metrics
        ]
        lines.append(
            f"| {model} | " + " | ".join(cells) + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best observed candidate: "
            f"{decision.get('best_observed_candidate')}",
            f"- Three-seed comparable candidate: "
            f"{decision.get('three_seed_comparable_candidate')}",
            f"- Replace existing head: "
            f"{decision.get('replace_existing_numerical_head')}",
            f"- Recommendation: {decision.get('recommendation')}",
            "",
            "## Missing Runs",
            "",
        ]
    )
    missing = decision.get("missing_runs") or []
    lines.extend(
        [
            f"- {item['model']} seed {item['seed']}"
            for item in missing
        ]
        or ["- None"]
    )
    return "\n".join(lines) + "\n"


def format_aggregate(
    aggregate: pd.DataFrame,
    model: str,
    metric: str,
) -> str:
    selected = aggregate[
        (aggregate["model"] == model)
        & (aggregate["metric"] == metric)
    ]
    if selected.empty:
        return ""
    row = selected.iloc[0]
    return f"{row['mean']:.6g} +/- {row['std']:.3g}"


def flatten_numeric(
    value: Any,
    prefix: str,
) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            output.update(
                flatten_numeric(
                    child,
                    f"{prefix}.{key}" if prefix else str(key),
                )
            )
    elif isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric):
            output[prefix] = (
                int(value)
                if isinstance(value, (int, np.integer))
                and not isinstance(value, bool)
                else numeric
            )
    return output


def first_present(
    mapping: dict[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def load_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
