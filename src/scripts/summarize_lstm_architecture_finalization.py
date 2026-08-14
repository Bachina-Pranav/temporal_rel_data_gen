#!/usr/bin/env python3
"""Build the paper-facing report for the validation-locked LSTM study."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/experiments/lstm_architecture_finalization.yaml"
if not __package__:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.run_lstm_architecture_finalization import (  # noqa: E402
    collect_scope_rows,
    evaluator_fingerprint,
    finite_mean,
    finite_std,
    load_json,
    load_json_optional,
    load_yaml,
    nested,
    object_sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_yaml(Path(args.experiment_config))
    output = Path(matrix["output_root"])
    lock = require_complete_lock(output)
    final_model = str(lock["selected_model"])

    validation = load_validation_rows(output, matrix)
    test = collect_scope_rows(
        output / "rel_hm/test",
        "rel_hm",
        test_models(output, matrix, final_model),
        matrix,
    )
    test["split"] = "test"
    validation["split"] = "validation"
    transfer = collect_transfer_rows(output, matrix)
    all_runs = pd.concat([validation, test, transfer], ignore_index=True)
    if all_runs.empty:
        raise RuntimeError("No completed runs are available for reporting")
    all_runs.to_csv(output / "all_runs.csv", index=False)

    deltas = paired_deltas(test, final_model, [
        "M0_original_lstm_v53",
        "M2_global_support",
    ])
    deltas.to_csv(output / "per_seed_deltas.csv", index=False)
    support = collect_support_calibration(output)
    support.to_csv(output / "support_calibration.csv", index=False)
    routing = collect_routing(output)
    routing.to_csv(output / "numerical_routing.csv", index=False)

    evaluator = evaluator_comparability(output, matrix, all_runs)
    write_json(evaluator, output / "evaluator_comparability.json")
    failure_path = output / "evaluator_comparability_failure.json"
    if evaluator["status"] != "passed":
        write_json(evaluator, failure_path)
        raise RuntimeError(
            "Evaluator comparability failed; refusing to create a mixed final "
            "table. See evaluator_comparability_failure.json."
        )
    if failure_path.is_file():
        failure_path.unlink()
    (output / "evaluator_audit.md").write_text(
        evaluator_markdown(evaluator), encoding="utf-8"
    )
    checks = acceptance_checks(test, transfer, final_model, matrix)
    freeze = bool(all(item["passed"] for item in checks.values()))
    aggregate = aggregate_metrics(all_runs)
    uncertainty = paired_bootstrap_intervals(test, final_model, matrix)
    decision = build_decision(
        matrix=matrix,
        output=output,
        lock=lock,
        final_model=final_model,
        all_runs=all_runs,
        aggregate=aggregate,
        deltas=deltas,
        support=support,
        routing=routing,
        evaluator=evaluator,
        checks=checks,
        uncertainty=uncertainty,
        freeze=freeze,
    )
    write_json(decision, output / "final_architecture_decision.json")
    (output / "final_architecture_decision.md").write_text(
        decision_markdown(decision, aggregate, deltas),
        encoding="utf-8",
    )
    generate_figures(output, all_runs, support, final_model)
    print_console_summary(decision, aggregate)
    for name in (
        "final_architecture_decision.md",
        "final_architecture_decision.json",
        "all_runs.csv",
        "per_seed_deltas.csv",
        "support_calibration.csv",
        "numerical_routing.csv",
        "evaluator_audit.md",
        "loss_audit.md",
        "support_head_loss_audit.md",
    ):
        print(output / name)


def require_complete_lock(output: Path) -> dict[str, Any]:
    path = output / "architecture_lock.json"
    if not path.is_file():
        raise RuntimeError("Missing validation architecture lock")
    lock = load_json(path)
    if lock.get("status") != "fully_frozen_on_validation":
        raise RuntimeError("Architecture is not fully frozen on validation")
    if lock.get("test_metrics_consulted") is not False:
        raise RuntimeError("Architecture lock is contaminated by test selection")
    if not (output / "test_evaluation_manifest.json").is_file():
        raise RuntimeError("Rel-HM test evaluation has not completed")
    return lock


def load_validation_rows(
    output: Path,
    matrix: dict[str, Any],
) -> pd.DataFrame:
    path = output / "validation_all_runs.csv"
    if path.is_file():
        return pd.read_csv(path)
    return collect_scope_rows(
        output / "rel_hm/validation",
        "rel_hm",
        list(matrix["variants"]),
        matrix,
    )


def test_models(
    output: Path,
    matrix: dict[str, Any],
    final_model: str,
) -> list[str]:
    names = [
        name
        for name in matrix["variants"]
        if (output / "rel_hm/test" / name).exists()
    ]
    if final_model not in names and (output / "rel_hm/test" / final_model).exists():
        names.append(final_model)
    return names


def collect_transfer_rows(
    output: Path,
    matrix: dict[str, Any],
) -> pd.DataFrame:
    frames = []
    single_seed_matrix = {**matrix, "seeds": [42]}
    for dataset in matrix["transfer"]["datasets"]:
        frame = collect_scope_rows(
            output / "transfer" / dataset,
            dataset,
            ["M2_global_support", "final"],
            single_seed_matrix,
        )
        if len(frame):
            frame["split"] = "test"
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_deltas(
    frame: pd.DataFrame,
    final_model: str,
    baselines: list[str],
) -> pd.DataFrame:
    metrics = [
        "full_row_c2st",
        "numerical_only_c2st",
        "categorical_only_c2st",
        "shape_error",
        "trend_error",
        "support_tv",
        "sampling_seconds",
        "rows_per_second",
        "parameter_count",
        "peak_training_gpu_memory_mb",
        "peak_sampling_gpu_memory_mb",
    ]
    final = frame[frame["model"] == final_model].set_index("seed")
    rows = []
    for baseline in baselines:
        reference = frame[frame["model"] == baseline].set_index("seed")
        for seed in sorted(set(final.index) & set(reference.index)):
            for metric in metrics:
                left = final.loc[seed].get(metric)
                right = reference.loc[seed].get(metric)
                if is_finite(left) and is_finite(right):
                    rows.append(
                        {
                            "dataset": "rel_hm",
                            "seed": int(seed),
                            "candidate": final_model,
                            "baseline": baseline,
                            "metric": metric,
                            "candidate_value": float(left),
                            "baseline_value": float(right),
                            "candidate_minus_baseline": float(left) - float(right),
                            "lower_is_better": metric != "rows_per_second",
                        }
                    )
    return pd.DataFrame(rows)


def collect_support_calibration(output: Path) -> pd.DataFrame:
    rows = []
    for report_path in output.rglob(
        "diagnostics/support_calibration/m2_support_calibration_report.json"
    ):
        report = load_json(report_path)
        relative = report_path.relative_to(output).parts
        dataset, split, model = infer_artifact_identity(relative)
        for seed_label, columns in (report.get("runs") or {}).items():
            for column, metrics in columns.items():
                row = {
                    "dataset": dataset,
                    "split": split,
                    "model": model,
                    "seed": parse_seed(seed_label),
                    "column": column,
                    "report_path": str(report_path),
                }
                row.update(
                    {
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    }
                )
                rows.append(row)
    frame = pd.DataFrame(rows)
    calibration_rows = collect_logit_calibration(output)
    if frame.empty:
        return calibration_rows
    if calibration_rows.empty:
        return frame
    keys = ["dataset", "split", "model", "seed", "column"]
    return frame.merge(calibration_rows, on=keys, how="outer")


def collect_logit_calibration(output: Path) -> pd.DataFrame:
    rows = []
    for path in output.rglob("evaluation/numerical_context_usage.json"):
        report = load_json(path)
        parts = path.relative_to(output).parts
        dataset, split, model = infer_artifact_identity(parts)
        seed = next(
            (parse_seed(part) for part in parts if part.startswith("seed_")),
            None,
        )
        for column, metrics in (
            report.get("support_head_calibration") or {}
        ).items():
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "model": model,
                    "seed": seed,
                    "column": column,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    },
                }
            )
    return pd.DataFrame(rows)


def collect_routing(output: Path) -> pd.DataFrame:
    rows = []
    for path in output.rglob("metadata/numerical_head_routing.json"):
        report = load_json(path)
        parts = path.relative_to(output).parts
        dataset, split, model = infer_artifact_identity(parts)
        seed = next(
            (parse_seed(part) for part in parts if part.startswith("seed_")),
            None,
        )
        for column in report.get("columns") or []:
            rows.append(
                {
                    "dataset": report.get("dataset") or dataset,
                    "split": split,
                    "model": model,
                    "seed": seed,
                    "column": column.get("column"),
                    "train_rows": column.get("train_rows"),
                    "unique_count": column.get("unique_count"),
                    "unique_ratio": column.get("unique_ratio"),
                    "inferred_type": column.get("inferred_type"),
                    "chosen_head": column.get("chosen_head"),
                    "implementation_mode": column.get("implementation_mode"),
                    "training_only": report.get("training_only"),
                    "path": str(path),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["dataset", "split", "model", "seed", "column"]
    ) if rows else pd.DataFrame()


def infer_artifact_identity(parts: tuple[str, ...]) -> tuple[str, str, str]:
    if parts[0] == "rel_hm":
        return "rel_hm", parts[1], parts[2]
    if parts[0] == "transfer":
        return parts[1], "test", parts[2]
    return parts[0], "unknown", parts[1] if len(parts) > 1 else "unknown"


def evaluator_comparability(
    output: Path,
    matrix: dict[str, Any],
    all_runs: pd.DataFrame,
) -> dict[str, Any]:
    config_paths: dict[str, Path] = {
        "rel_hm": Path(matrix["rel_hm"]["evaluation_config"])
    }
    for dataset, definition in matrix["transfer"]["datasets"].items():
        config_paths[dataset] = Path(definition["evaluation_config"])
    expected = {
        dataset: evaluator_runtime_normalized_fingerprint(path)
        for dataset, path in config_paths.items()
    }
    missing = []
    hash_mismatches = []
    per_run_hashes = []
    resolved_hashes_by_dataset: dict[str, set[str]] = {
        dataset: set() for dataset in expected
    }
    fixed_seed = int(matrix["evaluator_seed"])
    resolved_seed_mismatches = []
    for row in all_runs.to_dict(orient="records"):
        metrics_path = Path(str(row.get("metrics_path", "")))
        if not metrics_path.is_file():
            missing.append(str(metrics_path))
            continue
        evaluation_path = metrics_path.parents[2] / "evaluation_config_resolved.yaml"
        if not evaluation_path.is_file():
            missing.append(str(evaluation_path))
            continue
        dataset = str(row["dataset"])
        actual = evaluator_fingerprint(evaluation_path)["evaluator_hash"]
        actual_method = evaluator_runtime_normalized_fingerprint(
            evaluation_path
        )
        expected_method = expected.get(dataset)
        resolved_hashes_by_dataset.setdefault(dataset, set()).add(actual)
        per_run_hashes.append(
            {
                "dataset": dataset,
                "model": row.get("model"),
                "seed": row.get("seed"),
                "evaluator_hash": actual,
                "runtime_normalized_evaluator_hash": actual_method,
                "evaluation_config": str(evaluation_path),
            }
        )
        if actual_method != expected_method:
            hash_mismatches.append(str(evaluation_path))
        raw = load_yaml(evaluation_path)
        seed = nested(raw, "evaluation", "random_seed")
        if seed is not None and int(seed) != fixed_seed:
            resolved_seed_mismatches.append(str(evaluation_path))
    within_dataset_mismatches = {
        dataset: sorted(hashes)
        for dataset, hashes in resolved_hashes_by_dataset.items()
        if len(hashes) > 1
    }
    resolved_hashes = {
        dataset: next(iter(hashes)) if len(hashes) == 1 else None
        for dataset, hashes in resolved_hashes_by_dataset.items()
    }
    return {
        "status": (
            "passed"
            if not missing
            and not resolved_seed_mismatches
            and not hash_mismatches
            and not within_dataset_mismatches
            else "failed"
        ),
        "fixed_evaluator_seed": fixed_seed,
        "evaluator_hash_by_dataset": resolved_hashes,
        "template_runtime_normalized_hash_by_dataset": expected,
        "missing_metrics": missing,
        "hash_mismatches": hash_mismatches,
        "within_dataset_hash_mismatches": within_dataset_mismatches,
        "per_run_hashes": per_run_hashes,
        "resolved_seed_mismatches": resolved_seed_mismatches,
        "generator_seed_decoupled_from_evaluator_seed": True,
        "runtime_resolved_fields_excluded_from_method_hash": [
            "table.columns.*.valid_values",
            "table.columns.*.support",
        ],
    }


def evaluator_runtime_normalized_fingerprint(path: Path) -> str:
    """Hash evaluator method settings, excluding fitted data domains."""

    policy = load_yaml(path)
    policy.pop("real_table_path", None)
    policy.pop("synthetic_table_path", None)
    columns = ((policy.get("table") or {}).get("columns") or {})
    for config in columns.values():
        if isinstance(config, dict):
            config.pop("valid_values", None)
            config.pop("support", None)
    controlled = evaluator_fingerprint(path)["controlled_files"]
    return object_sha256(
        {
            "runtime_normalized_policy": policy,
            "controlled_files": controlled,
        }
    )


def acceptance_checks(
    test: pd.DataFrame,
    transfer: pd.DataFrame,
    final_model: str,
    matrix: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    final = test[test["model"] == final_model]
    m0 = test[test["model"] == "M0_original_lstm_v53"]
    m2 = test[test["model"] == "M2_global_support"]
    required = set(int(seed) for seed in matrix["seeds"])
    checks: dict[str, tuple[bool, Any]] = {
        "adaptive_numerical_router_enabled": (
            final_model != "M0_original_lstm_v53",
            final_model,
        ),
        "three_test_seeds_complete": (
            set(final["seed"].astype(int)) == required,
            sorted(final["seed"].astype(int).tolist()),
        ),
        "constraint_violation_zero": (
            bool(len(final) and (final["constraint_violation"].fillna(np.inf).abs() <= 1e-12).all()),
            finite_mean(final["constraint_violation"]),
        ),
        "fk_similarity_one": (
            bool(len(final) and (final["fk_similarity"].fillna(-np.inf) >= 1 - 1e-12).all()),
            finite_mean(final["fk_similarity"]),
        ),
        "all_domains_valid": (
            bool(
                len(final)
                and (final["invalid_categorical_rate"].fillna(0) <= 1e-12).all()
                and (final["invalid_numerical_rate"].fillna(0) <= 1e-12).all()
                and (final["invalid_support_rate"].fillna(0) <= 1e-12).all()
            ),
            None,
        ),
        "sampling_over_1000_rows_per_second": (
            bool(len(final) and (final["rows_per_second"].fillna(-np.inf) > 1000).all()),
            finite_mean(final["rows_per_second"]),
        ),
        "full_c2st_improves_m0_every_seed": (
            paired_all_better(final, m0, "full_row_c2st"),
            paired_values(final, m0, "full_row_c2st"),
        ),
        "numerical_c2st_improves_m0_every_seed": (
            paired_all_better(final, m0, "numerical_only_c2st"),
            paired_values(final, m0, "numerical_only_c2st"),
        ),
        "full_c2st_not_worse_than_m2": (
            below_or_equal(
                difference_of_means(final, m2, "full_row_c2st"),
                0.005,
            ),
            difference_of_means(final, m2, "full_row_c2st"),
        ),
        "numerical_c2st_not_worse_than_m2": (
            below_or_equal(
                difference_of_means(final, m2, "numerical_only_c2st"),
                0.005,
            ),
            difference_of_means(final, m2, "numerical_only_c2st"),
        ),
        "support_calibration_not_worse_than_m2": (
            below_or_equal(
                difference_of_means(final, m2, "support_tv"),
                0.005,
            ),
            difference_of_means(final, m2, "support_tv"),
        ),
        "mean_full_c2st_below_0_65": (
            below(finite_mean(final["full_row_c2st"]), 0.65),
            finite_mean(final["full_row_c2st"]),
        ),
        "mean_numerical_c2st_below_0_70": (
            below(finite_mean(final["numerical_only_c2st"]), 0.70),
            finite_mean(final["numerical_only_c2st"]),
        ),
        "trend_not_regressed_over_m0": (
            below_or_equal(
                difference_of_means(final, m0, "trend_error"),
                float(matrix["selection"]["m0_trend_regression_tolerance"]),
            ),
            difference_of_means(final, m0, "trend_error"),
        ),
        "categorical_not_materially_regressed_vs_m2": (
            below_or_equal(
                difference_of_means(final, m2, "categorical_tv"),
                float(matrix["selection"]["categorical_tv_regression_tolerance"]),
            ),
            difference_of_means(final, m2, "categorical_tv"),
        ),
        "cross_dataset_transfer_complete": (
            set(transfer["dataset"].unique())
            == set(matrix["transfer"]["datasets"])
            and all(
                set(group["model"]) == {"M2_global_support", "final"}
                for _, group in transfer.groupby("dataset")
            ),
            sorted(transfer["dataset"].unique().tolist()),
        ),
        "cross_dataset_validity": (
            transfer_validity(transfer, matrix),
            transfer_validity_details(transfer),
        ),
        "cross_dataset_full_c2st_not_regressed": (
            transfer_not_regressed(
                transfer,
                matrix,
                "full_row_c2st",
                tolerance=0.02,
            ),
            transfer_deltas(transfer, "full_row_c2st"),
        ),
        "cross_dataset_attribute_c2st_not_regressed": (
            all(
                transfer_not_regressed(
                    transfer,
                    matrix,
                    metric,
                    tolerance=0.02,
                    skip_unavailable=True,
                )
                for metric in (
                    "numerical_only_c2st",
                    "categorical_only_c2st",
                    "text_embedding_c2st",
                )
            ),
            {
                metric: transfer_deltas(transfer, metric)
                for metric in (
                    "numerical_only_c2st",
                    "categorical_only_c2st",
                    "text_embedding_c2st",
                )
            },
        ),
    }
    return {
        name: {"passed": bool(passed), "observed": observed}
        for name, (passed, observed) in checks.items()
    }


def transfer_validity(
    frame: pd.DataFrame,
    matrix: dict[str, Any],
) -> bool:
    required = set(matrix["transfer"]["datasets"])
    if set(frame.get("dataset", pd.Series(dtype=str)).unique()) != required:
        return False
    return bool(
        len(frame)
        and (frame["constraint_violation"].fillna(np.inf).abs() <= 1e-12).all()
        and (frame["fk_similarity"].fillna(-np.inf) >= 1 - 1e-12).all()
        and (frame["invalid_categorical_rate"].fillna(0) <= 1e-12).all()
        and (frame["invalid_numerical_rate"].fillna(0) <= 1e-12).all()
    )


def transfer_validity_details(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "datasets": sorted(
            frame.get("dataset", pd.Series(dtype=str)).unique().tolist()
        ),
        "max_constraint_violation": maximum_value(
            frame.get("constraint_violation", [])
        ),
        "min_fk_similarity": minimum_value(frame.get("fk_similarity", [])),
        "max_invalid_categorical_rate": maximum_value(
            frame.get("invalid_categorical_rate", [])
        ),
        "max_invalid_numerical_rate": maximum_value(
            frame.get("invalid_numerical_rate", [])
        ),
    }


def transfer_not_regressed(
    frame: pd.DataFrame,
    matrix: dict[str, Any],
    metric: str,
    *,
    tolerance: float,
    skip_unavailable: bool = False,
) -> bool:
    for dataset in matrix["transfer"]["datasets"]:
        delta = transfer_dataset_delta(frame, dataset, metric)
        if delta is None:
            if skip_unavailable:
                continue
            return False
        if delta > tolerance:
            return False
    return True


def transfer_deltas(frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    return {
        str(dataset): transfer_dataset_delta(frame, str(dataset), metric)
        for dataset in frame.get("dataset", pd.Series(dtype=str)).unique()
    }


def transfer_dataset_delta(
    frame: pd.DataFrame,
    dataset: str,
    metric: str,
) -> float | None:
    if metric not in frame:
        return None
    subset = frame[frame["dataset"] == dataset]
    candidate = finite_mean(subset.loc[subset["model"] == "final", metric])
    baseline = finite_mean(
        subset.loc[subset["model"] == "M2_global_support", metric]
    )
    if candidate is None or baseline is None:
        return None
    return float(candidate - baseline)


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "constraint_violation",
        "fk_similarity",
        "shape_error",
        "full_row_c2st",
        "numerical_only_c2st",
        "categorical_only_c2st",
        "temporal_event_distance",
        "text_embedding_c2st",
        "trend_error",
        "support_tv",
        "support_js",
        "support_entropy_error",
        "categorical_tv",
        "conditional_numerical_error",
        "conditional_categorical_error",
        "training_seconds",
        "sampling_seconds",
        "rows_per_second",
    ]
    rows = []
    for keys, group in frame.groupby(["dataset", "split", "model"]):
        row = {"dataset": keys[0], "split": keys[1], "model": keys[2], "num_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            if metric not in group:
                continue
            row[f"{metric}_mean"] = finite_mean(group[metric])
            row[f"{metric}_std"] = finite_std(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_intervals(
    test: pd.DataFrame,
    final_model: str,
    matrix: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    samples = int(matrix["selection"].get("bootstrap_samples", 2000))
    output: dict[str, Any] = {}
    final = test[test["model"] == final_model].set_index("seed")
    for baseline in ("M0_original_lstm_v53", "M2_global_support"):
        reference = test[test["model"] == baseline].set_index("seed")
        common = sorted(set(final.index) & set(reference.index))
        output[baseline] = {}
        for metric in ("full_row_c2st", "numerical_only_c2st", "shape_error"):
            differences = np.asarray(
                [float(final.loc[seed, metric]) - float(reference.loc[seed, metric]) for seed in common if is_finite(final.loc[seed, metric]) and is_finite(reference.loc[seed, metric])],
                dtype=float,
            )
            if not len(differences):
                output[baseline][metric] = None
                continue
            draws = rng.choice(differences, size=(samples, len(differences)), replace=True).mean(axis=1)
            output[baseline][metric] = {
                "paired_mean_delta": float(differences.mean()),
                "bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                "num_paired_seeds": int(len(differences)),
                "caution": "Only three generator seeds; interval is descriptive.",
            }
    return output


def build_decision(**kwargs: Any) -> dict[str, Any]:
    matrix = kwargs["matrix"]
    output = kwargs["output"]
    lock = kwargs["lock"]
    all_runs = kwargs["all_runs"]
    final_model = kwargs["final_model"]
    validation_selection = load_json(output / "validation_architecture_selection.json")
    temperature = load_json_optional(output / "temperature_selection.json")
    categorical = load_json_optional(output / "categorical_selection.json")
    final_config = load_yaml(
        Path(lock.get("deployment_config", lock["selected_config"]))
    )
    per_seed = all_runs.to_dict(orient="records")
    return {
        "final_model_name": final_model,
        "freeze_architecture": bool(kwargs["freeze"]),
        "freeze_recommendation": "FREEZE" if kwargs["freeze"] else "DO NOT FREEZE",
        "selection_policy": {
            "architecture_selected_on": "Rel-HM validation only",
            "test_data_used_for_selection": False,
            "validation_selection": validation_selection,
            "temperature_selection": temperature,
            "categorical_selection": categorical,
        },
        "final_architecture": {
            "shared_row_representation": True,
            "temporal_relational_context": "past-only",
            "numerical_routing": (
                "training-only schema/data-driven auto router"
                if lock.get("adaptive_router_enabled", True)
                else "continuous-only M0 control; adaptive architecture not frozen"
            ),
            "continuous_head": "Gaussian location/scale",
            "support_head_equation": "logit_k(x) = log(p_train(v_k) + eps) + gamma * delta_k(x)",
            "numerical_head_config": final_config.get("numerical_heads"),
            "numerical_temperature": lock["numerical_temperature"],
            "categorical_prior": lock["categorical_prior"],
            "text_architecture_changed": False,
        },
        "routing_parameters": nested(final_config, "numerical_heads", "type_inference"),
        "chosen_hyperparameters": {
            "support_residual_weight": nested(final_config, "numerical_heads", "global_prior", "residual_weight"),
            "support_prior_alpha": nested(final_config, "numerical_heads", "global_prior", "alpha"),
            "support_sampling_temperature": lock["numerical_temperature"],
            "categorical_prior": lock["categorical_prior"],
        },
        "evaluator_audit": kwargs["evaluator"],
        "evaluator_hashes": kwargs["evaluator"].get("evaluator_hash_by_dataset"),
        "acceptance_checks": kwargs["checks"],
        "bootstrap_intervals": kwargs["uncertainty"],
        "per_seed_metrics": per_seed,
        "aggregate_metrics": kwargs["aggregate"].to_dict(orient="records"),
        "paired_deltas": kwargs["deltas"].to_dict(orient="records"),
        "support_calibration": kwargs["support"].to_dict(orient="records"),
        "numerical_routing": kwargs["routing"].to_dict(orient="records"),
        "per_dataset_status": dataset_status(all_runs, matrix),
        "remaining_weaknesses": remaining_weaknesses(all_runs, final_model),
        "artifacts": {
            "all_runs": str(output / "all_runs.csv"),
            "per_seed_deltas": str(output / "per_seed_deltas.csv"),
            "support_calibration": str(output / "support_calibration.csv"),
            "numerical_routing": str(output / "numerical_routing.csv"),
            "evaluator_audit": str(output / "evaluator_audit.md"),
            "loss_audit": str(output / "loss_audit.md"),
            "support_head_loss_audit": str(
                output / "support_head_loss_audit.md"
            ),
            "figures": str(output / "figures"),
        },
        "git_commit": load_json_optional(output / "experiment_manifest.json").get("git_commit"),
    }


def decision_markdown(
    decision: dict[str, Any],
    aggregate: pd.DataFrame,
    deltas: pd.DataFrame,
) -> str:
    final = decision["final_model_name"]
    checks = decision["acceptance_checks"]
    lines = [
        "# Final LSTM Architecture Decision",
        "",
        "## Executive Conclusion",
        "",
        f"Recommendation: **{decision['freeze_recommendation']}**.",
        f"Validation-selected model: **{final}**.",
        "Architecture, support temperature, and optional categorical prior were selected using training and validation data only. Test results were opened only after the architecture lock was written.",
        "",
        "## Exact Final Architecture",
        "",
        "The model retains the leakage-safe past-only temporal-relational encoder, shared stochastic row representation, schema-specific categorical heads, and unchanged autoregressive text decoders. Numerical fields are routed using training-only schema/statistics: genuinely continuous fields use the Gaussian location/scale head; repeated or quantized fields use the empirical-prior support head.",
        "",
        "## Numerical Routing Rule",
        "",
        "Routing uses unique count and ratio, repeated observation mass, training-only holdout support recurrence, decimal precision concentration, support spacing, and common-value mass. No dataset or column names participate.",
        "",
        "## Winning Support Head",
        "",
        "```text",
        "logit_k(x) = log(p_train(v_k) + eps) + gamma * delta_k(x)",
        "```",
        "",
        f"Selected support residual weight: `{decision['chosen_hyperparameters']['support_residual_weight']}`. Selected validation temperature: `{decision['chosen_hyperparameters']['support_sampling_temperature']}`.",
        "",
        "## Rel-HM Three-Seed Results",
        "",
        markdown_table(aggregate[(aggregate["dataset"] == "rel_hm") & (aggregate["split"] == "test")]),
        "",
        "## Paired Comparisons Against M0 and M2",
        "",
        markdown_table(deltas),
        "",
        "## M3/M4 Completed Results",
        "",
        markdown_table(aggregate[(aggregate["dataset"] == "rel_hm") & (aggregate["split"] == "validation") & (aggregate["model"].isin(["M3_destination_support", "M4_destination_support_prior"]))]),
        "",
        "## M2P Residual-Strength Sweep",
        "",
        markdown_table(aggregate[(aggregate["dataset"] == "rel_hm") & (aggregate["split"] == "validation") & (aggregate["model"].str.startswith("M2P_"))]),
        "",
        "## Support Calibration Diagnostics",
        "",
        "See `support_calibration.csv` and `figures/`. Both KL directions, TV/JS, support-rank/head-frequency correlations, entropy, head/tail mass, rare/dominant errors, and missing/invalid support rates are retained per seed and field.",
        "",
        "## Loss Audit",
        "",
        "See `loss_audit.md`. The controlled M2U run separates ordinary CE from the old inverse-square-root weighting plus label smoothing; M2P uses ordinary CE.",
        "",
        "## Temperature Sweep",
        "",
        f"Validation-only selection: `{json.dumps(decision['selection_policy']['temperature_selection'], sort_keys=True)}`.",
        "",
        "## Conditional-Context Ablation",
        "",
        "R0 is the empirical-prior-only control. R1-R3 add the same full temporal-relational context at increasing residual strengths. Source, destination, time, and history-status conditioned errors are included in `all_runs.csv`; fixed-latent logit decomposition is saved in each run's `numerical_context_usage.json`.",
        "",
        "## MovieLens Transfer",
        "",
        markdown_table(aggregate[(aggregate["dataset"] == "movielens_100k") & (aggregate["split"] == "test")]),
        "",
        "## Amazon-Toy Transfer",
        "",
        markdown_table(aggregate[(aggregate["dataset"] == "amazon_toy") & (aggregate["split"] == "test")]),
        "",
        "## Runtime Comparison",
        "",
        markdown_table(aggregate[[column for column in aggregate.columns if column in {"dataset", "split", "model", "training_seconds_mean", "sampling_seconds_mean", "rows_per_second_mean", "peak_training_gpu_memory_mb_mean", "peak_sampling_gpu_memory_mb_mean"}]]),
        "",
        "## Acceptance Checks",
        "",
        "| Check | Passed | Observed |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {name} | {value['passed']} | {value['observed']} |"
        for name, value in checks.items()
    )
    lines.extend(
        [
            "",
            "## Remaining Weaknesses",
            "",
            *[f"- {item}" for item in decision["remaining_weaknesses"]],
            "",
            "## Final Recommendation",
            "",
            f"**{decision['freeze_recommendation']}** the structured attribute architecture described above.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluator_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evaluator Consistency Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        f"- Fixed evaluator seed: `{report['fixed_evaluator_seed']}`",
        "- Generator seed is decoupled from evaluator/classifier randomness.",
        "- Primary-key identifiers are excluded by the common paper evaluator.",
        "- Standardization is fitted inside classifier CV pipelines.",
        "- Semantic feature, categorical hashing, and text handling policies are fixed per dataset config.",
        "",
        "## Hashes",
        "",
    ]
    lines.extend(
        f"- {dataset}: `{value}`"
        for dataset, value in report["evaluator_hash_by_dataset"].items()
    )
    if report["resolved_seed_mismatches"]:
        lines.extend(["", "Seed mismatches:", *[f"- `{path}`" for path in report["resolved_seed_mismatches"]]])
    return "\n".join(lines) + "\n"


def generate_figures(
    output: Path,
    all_runs: pd.DataFrame,
    support: pd.DataFrame,
    final_model: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output / "figures_skipped.txt").write_text(str(exc), encoding="utf-8")
        return
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    hm = all_runs[(all_runs["dataset"] == "rel_hm") & (all_runs["split"] == "test")]
    for metric, filename, ylabel in (
        ("full_row_c2st", "rel_hm_full_row_c2st.png", "Full-row C2ST error"),
        ("numerical_only_c2st", "rel_hm_numerical_c2st.png", "Numerical-only C2ST error"),
    ):
        if metric in hm and hm[metric].notna().any():
            plot_model_metric(plt, hm, metric, figures / filename, ylabel)
    if len(support) and "generated_entropy_nats" in support:
        plot_model_metric(
            plt,
            support.rename(columns={"generated_entropy_nats": "entropy"}),
            "entropy",
            figures / "support_entropy_by_model_seed.png",
            "Generated support entropy (nats)",
        )
    transfer = all_runs[all_runs["dataset"].isin(["movielens_100k", "amazon_toy"])]
    if len(transfer) and transfer["full_row_c2st"].notna().any():
        plot_cross_dataset(plt, transfer, figures / "cross_dataset_c2st.png")
    plot_seed_variance(plt, hm, figures / "rel_hm_seed_variance.png")
    plot_support_tables(plt, output, figures, final_model)
    plot_ordinal_transfer_distribution(
        plt,
        output,
        "movielens_100k",
        figures / "movielens_rating_distribution.png",
    )


def plot_model_metric(plt: Any, frame: pd.DataFrame, metric: str, path: Path, ylabel: str) -> None:
    models = list(dict.fromkeys(frame["model"].astype(str)))
    values = [pd.to_numeric(frame.loc[frame["model"] == model, metric], errors="coerce").dropna() for model in models]
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(models)), 4.5))
    ax.boxplot(values, labels=models, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cross_dataset(plt: Any, frame: pd.DataFrame, path: Path) -> None:
    pivot = frame.groupby(["dataset", "model"])["full_row_c2st"].mean().unstack()
    ax = pivot.plot(kind="bar", figsize=(7, 4.5), rot=0)
    ax.set_ylabel("Full-row C2ST error")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=220)
    plt.close(ax.figure)


def plot_seed_variance(plt: Any, frame: pd.DataFrame, path: Path) -> None:
    metrics = [
        metric
        for metric in (
            "full_row_c2st",
            "numerical_only_c2st",
            "shape_error",
        )
        if metric in frame and frame[metric].notna().any()
    ]
    if not metrics:
        return
    rows = []
    for model, group in frame.groupby("model"):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "sample_std": (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ),
                }
            )
    pivot = pd.DataFrame(rows).pivot(
        index="model",
        columns="metric",
        values="sample_std",
    )
    ax = pivot.plot(
        kind="bar",
        figsize=(max(8, 0.8 * len(pivot)), 4.5),
    )
    ax.set_ylabel("Sample standard deviation across seeds")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=220)
    plt.close(ax.figure)


def plot_support_tables(plt: Any, output: Path, figures: Path, final_model: str) -> None:
    root = output / "rel_hm/test" / final_model / "diagnostics/support_calibration"
    tables = sorted(root.rglob("*_support_probabilities.csv"))
    if not tables:
        return
    table = pd.read_csv(tables[0]).sort_values("p_train", ascending=False).head(100)
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, table["p_train"], label="training", linewidth=2)
    ax.plot(x, table["p_generated"], label="synthetic", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Support values ranked by training frequency")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "rel_hm_support_head_frequency.png", dpi=220)
    plt.close(fig)

    bucket_rows = []
    labels = ["head", "upper-middle", "middle", "lower-middle", "tail"]
    for path in tables:
        probabilities = pd.read_csv(path).sort_values(
            "p_train",
            ascending=False,
        )
        if probabilities.empty:
            continue
        bucket_count = min(5, len(probabilities))
        probabilities["bucket"] = pd.qcut(
            np.arange(len(probabilities)),
            q=bucket_count,
            labels=labels[:bucket_count],
        )
        masses = probabilities.groupby("bucket", observed=True)[
            ["p_train", "p_generated"]
        ].sum()
        for bucket, row in masses.iterrows():
            bucket_rows.append(
                {
                    "bucket": str(bucket),
                    "training": float(row["p_train"]),
                    "synthetic": float(row["p_generated"]),
                }
            )
    if bucket_rows:
        bucket_frame = (
            pd.DataFrame(bucket_rows)
            .groupby("bucket", sort=False)
            .mean()
        )
        bucket_frame = bucket_frame.reindex(
            [label for label in labels if label in bucket_frame.index]
        )
        ax = bucket_frame.plot(kind="bar", figsize=(8, 4.5), rot=20)
        ax.set_ylabel("Probability mass")
        ax.set_xlabel("Training-support frequency bucket")
        ax.grid(axis="y", alpha=0.25)
        ax.figure.tight_layout()
        ax.figure.savefig(
            figures / "rel_hm_support_frequency_bucket_mass.png",
            dpi=220,
        )
        plt.close(ax.figure)


def plot_ordinal_transfer_distribution(
    plt: Any,
    output: Path,
    dataset: str,
    path: Path,
) -> None:
    """Plot an ordered generated field selected from schema metadata."""

    config_path = (
        output / "resolved_configs" / "transfer" / dataset / "final.yaml"
    )
    synthetic_path = (
        output
        / "transfer"
        / dataset
        / "final"
        / "runs"
        / "seed_42"
        / "samples"
        / "synthetic_interactions.csv"
    )
    if not config_path.is_file() or not synthetic_path.is_file():
        return
    config = load_yaml(config_path)
    generated = config.get("generated_attributes") or {}
    fields = (config.get("schema") or {}).get("fields") or {}
    candidates = []
    for column in generated:
        metadata = {
            **dict(fields.get(column) or {}),
            **dict(generated.get(column) or {}),
        }
        semantic = str(metadata.get("semantic_type", "")).lower()
        domain = metadata.get("valid_domain")
        if semantic not in {"ordinal", "ordinal_categorical"} or not isinstance(
            domain,
            list,
        ):
            continue
        try:
            support = [float(value) for value in domain]
        except (TypeError, ValueError):
            continue
        candidates.append((column, support))
    if not candidates:
        return
    column, support = candidates[0]
    real_path = Path((config.get("paths") or {}).get("train_data_path", ""))
    if not real_path.is_file():
        return
    real = pd.read_csv(real_path, usecols=[column])[column]
    synthetic = pd.read_csv(synthetic_path, usecols=[column])[column]
    real_mass = pd.to_numeric(real, errors="coerce").value_counts(
        normalize=True
    )
    synthetic_mass = pd.to_numeric(
        synthetic,
        errors="coerce",
    ).value_counts(normalize=True)
    frame = pd.DataFrame(
        {
            "real": [float(real_mass.get(value, 0.0)) for value in support],
            "synthetic": [
                float(synthetic_mass.get(value, 0.0)) for value in support
            ],
        },
        index=support,
    )
    ax = frame.plot(kind="bar", figsize=(8, 4.5), rot=0)
    ax.set_xlabel(column)
    ax.set_ylabel("Probability")
    ax.set_title(f"{dataset}: real vs synthetic {column}")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=220)
    plt.close(ax.figure)


def plot_model_summary(aggregate: pd.DataFrame, dataset: str, model: str) -> str:
    selected = aggregate[(aggregate["dataset"] == dataset) & (aggregate["split"] == "test") & (aggregate["model"] == model)]
    if selected.empty:
        return "not available"
    row = selected.iloc[0]
    return (
        f"full C2ST={fmt(row.get('full_row_c2st_mean'))}, "
        f"numerical C2ST={fmt(row.get('numerical_only_c2st_mean'))}, "
        f"shape={fmt(row.get('shape_error_mean'))}, trend={fmt(row.get('trend_error_mean'))}"
    )


def print_console_summary(decision: dict[str, Any], aggregate: pd.DataFrame) -> None:
    final = decision["final_model_name"]
    print("\nFINAL ARCHITECTURE: " + final)
    print("\nFREEZE:\n" + ("YES" if decision["freeze_architecture"] else "NO"))
    print("\nWHY:")
    passed = [name for name, value in decision["acceptance_checks"].items() if value["passed"]]
    failed = [name for name, value in decision["acceptance_checks"].items() if not value["passed"]]
    for item in passed[:4]:
        print(f"- Passed: {item}")
    for item in failed[:2]:
        print(f"- Failed: {item}")
    print("\nREL-HM:")
    for model in ("M0_original_lstm_v53", "M2_global_support", final):
        print(f"{model}: {plot_model_summary(aggregate, 'rel_hm', model)}")
    print("\nMOVIELENS:")
    for model in ("M2_global_support", "final"):
        print(f"{model}: {plot_model_summary(aggregate, 'movielens_100k', model)}")
    print("\nAMAZON:")
    for model in ("M2_global_support", "final"):
        print(f"{model}: {plot_model_summary(aggregate, 'amazon_toy', model)}")
    checks = decision["acceptance_checks"]
    print("\nVALIDITY:\n" + ("PASS" if all(value["passed"] for key, value in checks.items() if "transfer" not in key) else "FAIL"))
    print("\nSAMPLING SPEED:\n" + str(checks["sampling_over_1000_rows_per_second"]["observed"]) + " rows/sec mean")
    print("\nPRIMARY REMAINING FAILURE:\n" + decision["remaining_weaknesses"][0])


def dataset_status(frame: pd.DataFrame, matrix: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for dataset in ["rel_hm", *matrix["transfer"]["datasets"]]:
        selected = frame[frame["dataset"] == dataset]
        output[dataset] = {
            "completed_runs": int(len(selected)),
            "models": sorted(selected["model"].unique().tolist()),
            "status": "completed" if len(selected) else "missing",
        }
    return output


def remaining_weaknesses(frame: pd.DataFrame, final_model: str) -> list[str]:
    final = frame[(frame["dataset"] == "rel_hm") & (frame["split"] == "test") & (frame["model"] == final_model)]
    candidates = {
        "full-row discrimination remains above chance": finite_mean(final["full_row_c2st"]),
        "numerical-only discrimination remains above chance": finite_mean(final["numerical_only_c2st"]),
        "support marginal mismatch remains": finite_mean(final["support_tv"]),
        "temporal trend mismatch remains": finite_mean(final["trend_error"]),
    }
    ranked = sorted(candidates.items(), key=lambda item: -(item[1] if item[1] is not None else -1))
    return [f"{name} (mean={fmt(value)})" for name, value in ranked]


def paired_all_better(candidate: pd.DataFrame, baseline: pd.DataFrame, metric: str) -> bool:
    left = candidate.set_index("seed")
    right = baseline.set_index("seed")
    common = set(left.index) & set(right.index)
    pairs = [
        (float(left.loc[seed, metric]), float(right.loc[seed, metric]))
        for seed in common
        if is_finite(left.loc[seed, metric])
        and is_finite(right.loc[seed, metric])
    ]
    return bool(pairs and all(candidate < reference for candidate, reference in pairs))


def paired_values(candidate: pd.DataFrame, baseline: pd.DataFrame, metric: str) -> dict[str, float]:
    left = candidate.set_index("seed")
    right = baseline.set_index("seed")
    return {
        str(seed): float(left.loc[seed, metric]) - float(right.loc[seed, metric])
        for seed in sorted(set(left.index) & set(right.index))
        if is_finite(left.loc[seed, metric]) and is_finite(right.loc[seed, metric])
    }


def difference_of_means(candidate: pd.DataFrame, baseline: pd.DataFrame, metric: str) -> float | None:
    left = finite_mean(candidate[metric])
    right = finite_mean(baseline[metric])
    return left - right if left is not None and right is not None else None


def below(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def below_or_equal(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def maximum_value(values: Any) -> float | None:
    finite = [float(value) for value in values if is_finite(value)]
    return max(finite) if finite else None


def minimum_value(values: Any) -> float | None:
    finite = [float(value) for value in values if is_finite(value)]
    return min(finite) if finite else None


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_seed(label: str) -> int | None:
    try:
        return int(str(label).replace("seed_", ""))
    except ValueError:
        return None


def markdown_table(frame: pd.DataFrame, max_columns: int = 12) -> str:
    if frame.empty:
        return "_Not available._"
    preferred = [
        "dataset", "split", "model", "seed", "num_seeds",
        "full_row_c2st_mean", "numerical_only_c2st_mean",
        "categorical_only_c2st_mean", "shape_error_mean",
        "trend_error_mean", "rows_per_second_mean",
        "metric", "candidate_minus_baseline",
    ]
    columns = [column for column in preferred if column in frame][:max_columns]
    if not columns:
        columns = list(frame.columns[:max_columns])
    shown = frame[columns].copy()
    headers = [str(column) for column in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


if __name__ == "__main__":
    main()
