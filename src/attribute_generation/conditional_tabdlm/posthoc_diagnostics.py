"""Schema-driven post-hoc diagnostics for generated event attributes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.paper_metrics.c2st import (
    featurize_real_synthetic,
    single_table_c2st_metrics,
)
from evaluation.paper_metrics.shape_trend import (
    shape_metrics,
    trend_metrics,
)
from evaluation.paper_metrics.utils import total_variation

from .numerical_support import (
    distance_summary,
    finite_array,
    frequency_bucket_mapping,
    infer_support_tolerance,
    nearest_support_distances,
    numerical_support_profile,
    project_numerical_support,
)


DEFAULT_CLASSIFIER_SEEDS = (11, 23, 37, 53, 71)


def repeated_c2st(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: dict[str, Any],
    *,
    classifier_seeds: list[int] | tuple[int, ...],
    columns: list[str] | None = None,
    max_rows: int | None = None,
    generator_seed: int | None = None,
    label: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation_config = c2st_config_for_columns(
        config,
        columns=columns,
        max_rows=max_rows,
    )
    rows: list[dict[str, Any]] = []
    for seed in classifier_seeds:
        seeded = copy.deepcopy(evaluation_config)
        seeded.setdefault("evaluation", {})["random_seed"] = int(seed)
        metrics, _ = single_table_c2st_metrics(real, synthetic, seeded)
        rows.append(
            {
                "label": label,
                "generator_seed": generator_seed,
                "classifier_seed": int(seed),
                "auc": metrics.get("auc"),
                "accuracy": metrics.get("accuracy"),
                "c2st_error": metrics.get("error"),
                "best_classifier": metrics.get("best_classifier"),
                "num_rows": metrics.get("num_rows"),
                "balanced_n_per_class": metrics.get(
                    "balanced_eval_n_real"
                ),
                "num_features": metrics.get("num_features"),
                "feature_names": metrics.get("feature_names"),
                "top_features": metrics.get("top_features"),
                "preprocessing_pipeline": metrics.get(
                    "preprocessing_fit_scope"
                ),
            }
        )
    return aggregate_c2st_rows(rows), rows


def aggregate_c2st_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_runs": int(len(rows)),
        "classifier_seeds": [
            int(row["classifier_seed"])
            for row in rows
            if row.get("classifier_seed") is not None
        ],
    }
    for key in ("auc", "accuracy", "c2st_error"):
        values = np.asarray(
            [
                float(row[key])
                for row in rows
                if row.get(key) is not None
                and np.isfinite(float(row[key]))
            ],
            dtype=float,
        )
        result[f"{key}_mean"] = (
            float(np.mean(values)) if len(values) else None
        )
        result[f"{key}_std"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
        half_width = (
            1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
        result[f"{key}_ci95_low"] = (
            float(np.mean(values) - half_width) if len(values) else None
        )
        result[f"{key}_ci95_high"] = (
            float(np.mean(values) + half_width) if len(values) else None
        )
    result["feature_count"] = next(
        (
            int(row["num_features"])
            for row in rows
            if row.get("num_features") is not None
        ),
        None,
    )
    result["balanced_n_per_class"] = next(
        (
            int(row["balanced_n_per_class"])
            for row in rows
            if row.get("balanced_n_per_class") is not None
        ),
        None,
    )
    result["preprocessing_pipeline"] = next(
        (
            row["preprocessing_pipeline"]
            for row in rows
            if row.get("preprocessing_pipeline")
        ),
        None,
    )
    return result


def c2st_config_for_columns(
    config: dict[str, Any],
    *,
    columns: list[str] | None,
    max_rows: int | None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    table = resolved.setdefault("table", {})
    configured = table.setdefault("columns", {})
    if columns is not None:
        table["columns"] = {
            column: copy.deepcopy(configured[column])
            for column in columns
            if column in configured
        }
        primary_key = table.get("primary_key")
        if primary_key and primary_key not in table["columns"]:
            table["primary_key"] = None
    c2st = resolved.setdefault("evaluation", {}).setdefault("c2st", {})
    if max_rows is not None:
        c2st["max_rows"] = int(max_rows)
        c2st["max_rows_for_c2st"] = int(max_rows)
        resolved["evaluation"]["max_rows_for_c2st"] = int(max_rows)
    return resolved


def c2st_sanity_suite(
    train: pd.DataFrame,
    real: pd.DataFrame,
    config: dict[str, Any],
    schema: Any,
    *,
    classifier_seeds: list[int] | tuple[int, ...],
    max_rows: int | None,
    chance_tolerance: float,
    progress_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(2027)
    shuffled_rows = real.sample(
        frac=1.0,
        random_state=2027,
    ).reset_index(drop=True)
    side = len(shuffled_rows) // 2
    split_a = shuffled_rows.iloc[:side].reset_index(drop=True)
    split_b = shuffled_rows.iloc[side : side * 2].reset_index(drop=True)

    shuffled_attributes = real.copy()
    for index, column in enumerate(schema.target_columns):
        shuffled_attributes[column] = (
            shuffled_attributes[column]
            .sample(frac=1.0, random_state=2030 + index)
            .reset_index(drop=True)
        )

    globally_resampled = real.copy()
    for column in schema.target_columns:
        candidates = train[column].dropna().to_numpy()
        if not len(candidates):
            raise ValueError(
                f"Cannot build global empirical baseline for empty target {column!r}"
            )
        globally_resampled[column] = rng.choice(
            candidates,
            size=len(real),
            replace=True,
        )

    corrupted = real.copy()
    for column in schema.numerical_targets:
        values = pd.to_numeric(corrupted[column], errors="coerce")
        train_values = pd.to_numeric(train[column], errors="coerce")
        scale = max(float(train_values.std()), 1e-12)
        corrupted[column] = values + 20.0 * scale

    scenarios = {
        "S1_identical_real_copy": (real, real.copy(), "chance"),
        "S2_disjoint_real_splits": (split_a, split_b, "chance"),
        "S3_shuffled_real_attributes": (
            real,
            shuffled_attributes,
            "separable",
        ),
        "S4_global_empirical_attributes": (
            real,
            globally_resampled,
            "separable",
        ),
        "S5_corrupted_numerical_attribute": (
            real,
            corrupted,
            "separable",
        ),
    }
    details: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for label, (first, second, expectation) in scenarios.items():
        print(f"[c2st-sanity] {label}", flush=True)
        aggregate, rows = repeated_c2st(
            first,
            second,
            config,
            classifier_seeds=classifier_seeds,
            max_rows=max_rows,
            label=label,
        )
        aggregate["expectation"] = expectation
        auc = aggregate.get("auc_mean")
        if expectation == "chance":
            aggregate["passed"] = bool(
                auc is not None
                and abs(float(auc) - 0.5) <= float(chance_tolerance)
            )
        else:
            aggregate["passed"] = bool(
                auc is not None and float(auc) >= 0.60
            )
        summary[label] = aggregate
        details.extend(rows)
        if progress_dir is not None:
            progress_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                {
                    "status": "running",
                    "chance_tolerance": float(chance_tolerance),
                    "scenarios": summary,
                },
                progress_dir / "sanity_progress.json",
            )
            pd.DataFrame(details).to_csv(
                progress_dir / "sanity_classifier_runs_progress.csv",
                index=False,
            )
    controls_passed = bool(
        summary["S1_identical_real_copy"]["passed"]
        and summary["S2_disjoint_real_splits"]["passed"]
    )
    return {
        "status": "passed" if controls_passed else "failed",
        "chance_controls_passed": controls_passed,
        "chance_tolerance": float(chance_tolerance),
        "metric_semantics": {
            "auc": "0.5 is chance; higher is more distinguishable",
            "c2st_error": "2 * abs(AUC - 0.5); zero is chance and lower is better",
        },
        "scenarios": summary,
    }, pd.DataFrame(details)


def c2st_feature_ablation_suite(
    train: pd.DataFrame,
    real: pd.DataFrame,
    synthetic_by_seed: dict[int, pd.DataFrame],
    config: dict[str, Any],
    schema: Any,
    *,
    classifier_seeds: list[int] | tuple[int, ...],
    max_rows: int | None,
    progress_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = schema.foreign_key_columns[0]
    destination = (
        schema.foreign_key_columns[1]
        if len(schema.foreign_key_columns) > 1
        else schema.foreign_key_columns[0]
    )
    timestamp = schema.datetime_columns[0]
    numerical = list(schema.numerical_targets)
    categorical = list(schema.categorical_targets)
    targets = [*numerical, *categorical]
    all_configured = list(
        (config.get("table", {}).get("columns", {}) or {}).keys()
    )
    feature_sets: dict[str, list[str]] = {
        "F1_numerical_only": numerical,
        "F2_categorical_only": categorical,
        "F3_generated_attributes": targets,
        "F4_numerical_plus_destination": [destination, *numerical],
        "F5_categorical_plus_source": [source, *categorical],
        "F6_generated_attributes_plus_time": [*targets, timestamp],
        "F7_full_transaction_row": all_configured,
        "F8_full_without_entity_ids": [
            column
            for column in all_configured
            if column not in set(schema.foreign_key_columns)
        ],
    }
    real_augmented, synthetic_augmented, derived_config = (
        add_frequency_bucket_features(
            train,
            real,
            synthetic_by_seed,
            config,
            schema,
        )
    )
    derived_columns = [
        column
        for column in derived_config["table"]["columns"]
        if column.startswith("__entity_frequency_")
    ]
    feature_sets["F9_frequency_buckets_instead_of_ids"] = [
        *targets,
        timestamp,
        *derived_columns,
    ]

    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for generator_seed, synthetic in synthetic_augmented.items():
        for label, columns in feature_sets.items():
            print(
                f"[c2st-feature-ablation] seed={generator_seed} {label}",
                flush=True,
            )
            aggregate, details = repeated_c2st(
                real_augmented,
                synthetic,
                derived_config,
                classifier_seeds=classifier_seeds,
                columns=columns,
                max_rows=max_rows,
                generator_seed=generator_seed,
                label=label,
            )
            rows.append(
                {
                    "feature_set": label,
                    "generator_seed": int(generator_seed),
                    "columns": json.dumps(columns),
                    **aggregate,
                }
            )
            detail_rows.extend(details)
            if progress_dir is not None:
                progress_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(
                    progress_dir / "feature_ablation_progress.csv",
                    index=False,
                )
                pd.DataFrame(detail_rows).to_csv(
                    progress_dir
                    / "feature_ablation_classifier_runs_progress.csv",
                    index=False,
                )
    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def add_frequency_bucket_features(
    train: pd.DataFrame,
    real: pd.DataFrame,
    synthetic_by_seed: dict[int, pd.DataFrame],
    config: dict[str, Any],
    schema: Any,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[str, Any]]:
    real_output = real.copy()
    synthetic_output = {
        seed: frame.copy()
        for seed, frame in synthetic_by_seed.items()
    }
    resolved = copy.deepcopy(config)
    columns = resolved.setdefault("table", {}).setdefault("columns", {})
    for index, entity_column in enumerate(schema.foreign_key_columns):
        feature = f"__entity_frequency_{index}"
        counts = train[entity_column].astype(str).value_counts()
        mapping = frequency_bucket_mapping(counts)
        real_output[feature] = (
            real_output[entity_column].astype(str).map(mapping).fillna("cold")
        )
        for frame in synthetic_output.values():
            frame[feature] = (
                frame[entity_column].astype(str).map(mapping).fillna("cold")
            )
        columns[feature] = {
            "type": "categorical",
            "nullable": False,
            "c2st_hash_buckets": min(16, max(4, len(set(mapping.values())) + 1)),
        }
    return real_output, synthetic_output, resolved


def projection_ablation_suite(
    train: pd.DataFrame,
    real: pd.DataFrame,
    synthetic_by_seed: dict[int, pd.DataFrame],
    history_prefix: pd.DataFrame,
    config: dict[str, Any],
    model_config: Any,
    *,
    output_dir: Path,
    c2st_seed: int,
    max_c2st_rows: int | None,
    stochastic_neighbors: int,
    stochastic_temperature: float,
    min_entity_rows: int,
    progress_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[tuple[int, str], pd.DataFrame]]:
    from scripts.evaluate_lstm_attribute_diagnostics import (
        add_condition_groups,
        conditional_metrics,
        numerical_metrics,
    )

    mode_names = {
        "P0": "none",
        "P1": "global_nearest",
        "P2": "global_stochastic",
        "P3": "entity_nearest",
        "P4": "learned_bins",
    }
    destination = (
        model_config.schema.foreign_key_columns[1]
        if len(model_config.schema.foreign_key_columns) > 1
        else model_config.schema.foreign_key_columns[0]
    )
    projected_tables: dict[tuple[int, str], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for generator_seed, original in synthetic_by_seed.items():
        for label, mode in mode_names.items():
            print(
                f"[price-projection] seed={generator_seed} {label} ({mode})",
                flush=True,
            )
            projected = original.copy()
            projection_metadata: dict[str, Any] = {}
            for target_index, column in enumerate(
                model_config.schema.numerical_targets
            ):
                values, metadata = project_numerical_support(
                    train[column],
                    projected[column],
                    mode=mode,
                    seed=int(generator_seed) + target_index,
                    train_entities=train[destination],
                    query_entities=projected[destination],
                    stochastic_neighbors=stochastic_neighbors,
                    stochastic_temperature=stochastic_temperature,
                    min_entity_rows=min_entity_rows,
                )
                projected[column] = values
                projection_metadata[column] = metadata
            projected_path = (
                output_dir
                / f"seed_{generator_seed}"
                / f"{label}_{mode}.csv"
            )
            projected_path.parent.mkdir(parents=True, exist_ok=True)
            projected.to_csv(projected_path, index=False)
            projected_tables[(int(generator_seed), label)] = projected
            assert_aligned_spine(real, projected, model_config.schema)

            c2st, _ = repeated_c2st(
                real,
                projected,
                config,
                classifier_seeds=[int(c2st_seed)],
                max_rows=max_c2st_rows,
                generator_seed=int(generator_seed),
                label=label,
            )
            shape, _ = shape_metrics(
                real,
                projected,
                config["table"],
                config,
            )
            trend, _ = trend_metrics(
                real,
                projected,
                config["table"],
                config,
            )
            conditioned_real, conditioned_syn, _ = add_condition_groups(
                real,
                projected,
                history_prefix,
                model_config.schema,
            )
            source_condition = conditional_metrics(
                conditioned_real,
                conditioned_syn,
                model_config.schema.foreign_key_columns[0],
                model_config.schema,
            )
            destination_condition = conditional_metrics(
                conditioned_real,
                conditioned_syn,
                destination,
                model_config.schema,
            )
            row: dict[str, Any] = {
                "projection": label,
                "mode": mode,
                "generator_seed": int(generator_seed),
                "synthetic_table": str(projected_path),
                "shape_error": shape.get("macro_attribute_shape_error"),
                "trend_error": trend.get("macro_headline_trend_error"),
                "single_table_c2st_error": c2st.get("c2st_error_mean"),
                "single_table_c2st_auc": c2st.get("auc_mean"),
                "projection_metadata": json.dumps(
                    projection_metadata,
                    sort_keys=True,
                ),
            }
            for column in model_config.schema.numerical_targets:
                numerical = numerical_metrics(
                    train[column],
                    real[column],
                    projected[column],
                )
                support = np.unique(finite_array(train[column]))
                distances = nearest_support_distances(
                    finite_array(projected[column]),
                    support,
                )
                row.update(
                    {
                        f"{column}.ks_distance": numerical["ks_distance"],
                        f"{column}.wasserstein_distance": numerical[
                            "wasserstein_distance"
                        ],
                        f"{column}.mean_absolute_error": numerical[
                            "mean_absolute_error"
                        ],
                        f"{column}.std_absolute_error": numerical[
                            "std_absolute_error"
                        ],
                        f"{column}.quantile_mae": numerical["quantile_mae"],
                        f"{column}.unique_values": int(
                            projected[column].nunique(dropna=True)
                        ),
                        f"{column}.exact_support_overlap_rate": float(
                            np.mean(
                                np.isin(
                                    finite_array(projected[column]),
                                    support,
                                )
                            )
                        ),
                        f"{column}.nearest_support_mean": float(
                            np.mean(distances)
                        ),
                        f"{column}.nearest_support_p95": float(
                            np.quantile(distances, 0.95)
                        ),
                        f"{column}.source_conditioned_standardized_mae": (
                            source_condition[column][
                                "group_mean_standardized_mae"
                            ]
                        ),
                        f"{column}.destination_conditioned_standardized_mae": (
                            destination_condition[column][
                                "group_mean_standardized_mae"
                            ]
                        ),
                    }
                )
            rows.append(row)
            if progress_path is not None:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(progress_path, index=False)
    return pd.DataFrame(rows), projected_tables


def oracle_ablation_suite(
    real: pd.DataFrame,
    synthetic_by_seed: dict[int, pd.DataFrame],
    projected_tables: dict[tuple[int, str], pd.DataFrame],
    projection_results: pd.DataFrame,
    config: dict[str, Any],
    schema: Any,
    *,
    c2st_seed: int,
    max_c2st_rows: int | None,
    progress_path: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    nonzero = projection_results.loc[
        projection_results["projection"] != "P0"
    ].copy()
    best_mode = (
        str(
            nonzero.groupby("projection")["single_table_c2st_error"]
            .mean()
            .idxmin()
        )
        if len(nonzero)
        else "P0"
    )
    rows: list[dict[str, Any]] = []
    for generator_seed, synthetic in synthetic_by_seed.items():
        variants: dict[str, pd.DataFrame] = {
            "O1_both_generated": synthetic.copy(),
            "O2_real_numerical_generated_categorical": synthetic.copy(),
            "O3_generated_numerical_real_categorical": synthetic.copy(),
            "O4_both_real": real.copy(),
            "O5_projected_numerical_generated_categorical": (
                projected_tables[(int(generator_seed), best_mode)].copy()
            ),
        }
        for column in schema.numerical_targets:
            variants[
                "O2_real_numerical_generated_categorical"
            ][column] = real[column].to_numpy()
        for column in schema.categorical_targets:
            variants[
                "O3_generated_numerical_real_categorical"
            ][column] = real[column].to_numpy()
        for label, frame in variants.items():
            print(
                f"[oracle-ablation] seed={generator_seed} {label}",
                flush=True,
            )
            assert_aligned_spine(real, frame, schema)
            c2st, _ = repeated_c2st(
                real,
                frame,
                config,
                classifier_seeds=[int(c2st_seed)],
                max_rows=max_c2st_rows,
                generator_seed=int(generator_seed),
                label=label,
            )
            shape, _ = shape_metrics(
                real,
                frame,
                config["table"],
                config,
            )
            trend, _ = trend_metrics(
                real,
                frame,
                config["table"],
                config,
            )
            rows.append(
                {
                    "oracle_variant": label,
                    "generator_seed": int(generator_seed),
                    "support_projection": (
                        best_mode
                        if label
                        == "O5_projected_numerical_generated_categorical"
                        else None
                    ),
                    "auc": c2st.get("auc_mean"),
                    "c2st_error": c2st.get("c2st_error_mean"),
                    "shape_error": shape.get("macro_attribute_shape_error"),
                    "trend_error": trend.get("macro_headline_trend_error"),
                }
            )
            if progress_path is not None:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(progress_path, index=False)
    return pd.DataFrame(rows), best_mode


def classifier_importance_analysis(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: dict[str, Any],
    schema: Any,
    *,
    seed: int,
    max_rows: int | None,
    permutation_repeats: int,
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}

    x, y, feature_names, balanced_n = featurize_real_synthetic(
        real,
        synthetic,
        config["table"],
        max_rows=max_rows,
        seed=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(x_train, y_train)
    baseline_auc = float(
        roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    )
    importance = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=int(permutation_repeats),
        random_state=seed,
    )
    feature_rows = sorted(
        [
            {
                "feature": str(name),
                "group": feature_group(name, schema),
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
            for name, mean, std in zip(
                feature_names,
                importance.importances_mean,
                importance.importances_std,
            )
        ],
        key=lambda row: row["importance_mean"],
        reverse=True,
    )
    groups = sorted({row["group"] for row in feature_rows})
    group_ablation = []
    for group in groups:
        keep = np.asarray(
            [
                index
                for index, name in enumerate(feature_names)
                if feature_group(name, schema) != group
            ],
            dtype=int,
        )
        if not len(keep):
            continue
        ablated = GradientBoostingClassifier(random_state=seed)
        ablated.fit(x_train[:, keep], y_train)
        auc = float(
            roc_auc_score(
                y_test,
                ablated.predict_proba(x_test[:, keep])[:, 1],
            )
        )
        group_ablation.append(
            {
                "group": group,
                "auc_without_group": auc,
                "auc_drop": float(baseline_auc - auc),
                "features_removed": int(len(feature_names) - len(keep)),
            }
        )
    price_thresholds: list[float] = []
    numerical_prefixes = tuple(schema.numerical_targets)
    for estimator_row in model.estimators_:
        for estimator in np.asarray(estimator_row).reshape(-1):
            tree = estimator.tree_
            for feature_index, threshold in zip(
                tree.feature,
                tree.threshold,
            ):
                if feature_index < 0:
                    continue
                name = feature_names[int(feature_index)]
                if name.startswith(numerical_prefixes):
                    price_thresholds.append(float(threshold))
    return {
        "status": "completed",
        "classifier": "gradient_boosting",
        "classifier_seed": int(seed),
        "balanced_rows_per_class": int(balanced_n),
        "baseline_holdout_auc": baseline_auc,
        "permutation_importance": feature_rows,
        "feature_group_ablation": sorted(
            group_ablation,
            key=lambda row: row["auc_drop"],
            reverse=True,
        ),
        "numerical_split_thresholds": {
            "count": int(len(price_thresholds)),
            "sample": price_thresholds[:100],
        },
    }


def feature_group(feature_name: str, schema: Any) -> str:
    if feature_name.endswith("_missing"):
        return "missingness"
    for column in schema.numerical_targets:
        if feature_name.startswith(column):
            return "numerical_attributes"
    for column in schema.categorical_targets:
        if feature_name.startswith(column):
            return "categorical_attributes"
    if schema.foreign_key_columns:
        if feature_name.startswith(schema.foreign_key_columns[0]):
            return "source_identity"
    if len(schema.foreign_key_columns) > 1:
        if feature_name.startswith(schema.foreign_key_columns[1]):
            return "destination_identity"
    for column in schema.datetime_columns:
        if feature_name.startswith(column):
            return "timestamp"
    if "__entity_frequency_" in feature_name:
        return "history_or_frequency"
    return "other"


def conditional_support_suite(
    train: pd.DataFrame,
    real: pd.DataFrame,
    synthetic_by_seed: dict[int, pd.DataFrame],
    history_prefix: pd.DataFrame,
    model_config: Any,
) -> dict[str, Any]:
    schema = model_config.schema
    output: dict[str, Any] = {
        "numerical_by_entity": {},
        "categorical_by_entity": {},
        "temporal": {},
        "entity_temporal": {},
        "history": {},
    }
    for seed, synthetic in synthetic_by_seed.items():
        seed_key = f"seed_{seed}"
        output["numerical_by_entity"][seed_key] = {}
        output["categorical_by_entity"][seed_key] = {}
        for entity in schema.foreign_key_columns:
            for column in schema.numerical_targets:
                output["numerical_by_entity"][seed_key][
                    f"{entity}__{column}"
                ] = entity_numerical_support_report(
                    train,
                    real,
                    synthetic,
                    entity,
                    column,
                )
            for column in schema.categorical_targets:
                output["categorical_by_entity"][seed_key][
                    f"{entity}__{column}"
                ] = entity_categorical_support_report(
                    train,
                    real,
                    synthetic,
                    entity,
                    column,
                )
        output["temporal"][seed_key] = temporal_conditional_report(
            real,
            synthetic,
            schema,
        )
        output["entity_temporal"][seed_key] = (
            entity_temporal_conditional_report(
                real,
                synthetic,
                schema,
            )
        )
        output["history"][seed_key] = history_conditional_report(
            real,
            synthetic,
            history_prefix,
            schema,
        )
    return output


def entity_numerical_support_report(
    train: pd.DataFrame,
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity: str,
    target: str,
) -> dict[str, Any]:
    train_frame = pd.DataFrame(
        {
            "entity": train[entity].astype(str),
            "value": pd.to_numeric(train[target], errors="coerce"),
        }
    ).dropna()
    support_by_entity = {
        str(key): np.sort(group["value"].unique().astype(float))
        for key, group in train_frame.groupby("entity", sort=False)
    }
    train_pairs = set(
        zip(
            train_frame["entity"],
            train_frame["value"],
        )
    )
    real_entity = real[entity].astype(str).to_numpy()
    syn_entity = synthetic[entity].astype(str).to_numpy()
    real_values = pd.to_numeric(real[target], errors="coerce").to_numpy(float)
    syn_values = pd.to_numeric(
        synthetic[target],
        errors="coerce",
    ).to_numpy(float)
    real_pair_seen = np.asarray(
        [
            (key, value) in train_pairs
            for key, value in zip(real_entity, real_values)
        ],
        dtype=bool,
    )
    syn_pair_seen = np.asarray(
        [
            (key, value) in train_pairs
            for key, value in zip(syn_entity, syn_values)
        ],
        dtype=bool,
    )
    real_distance, real_seen = entity_nearest_distances(
        real_entity,
        real_values,
        support_by_entity,
    )
    syn_distance, syn_seen = entity_nearest_distances(
        syn_entity,
        syn_values,
        support_by_entity,
    )
    group_metrics = weighted_entity_numerical_errors(
        real,
        synthetic,
        entity,
        target,
    )
    counts = train_frame["entity"].value_counts()
    frequency_mapping = frequency_bucket_mapping(counts)
    frequency_metrics = conditioned_numerical_by_mapping(
        real,
        synthetic,
        entity,
        target,
        frequency_mapping,
    )
    diversity = train_frame.groupby("entity")["value"].nunique()
    diversity_mapping = quantile_bucket_mapping(
        diversity,
        prefix="support_diversity",
    )
    diversity_metrics = conditioned_numerical_by_mapping(
        real,
        synthetic,
        entity,
        target,
        diversity_mapping,
    )
    return {
        "synthetic_entity_value_pair_seen_in_train_rate": float(
            np.mean(syn_pair_seen)
        ),
        "real_test_entity_value_pair_seen_in_train_rate": float(
            np.mean(real_pair_seen)
        ),
        "synthetic_seen_entity_rate": float(np.mean(syn_seen)),
        "real_test_seen_entity_rate": float(np.mean(real_seen)),
        "synthetic_nearest_entity_support": distance_summary(
            syn_distance[syn_seen]
        ),
        "real_test_nearest_entity_support": distance_summary(
            real_distance[real_seen]
        ),
        "entity_conditioned_errors": group_metrics,
        "by_entity_frequency_bucket": frequency_metrics,
        "by_entity_support_diversity": diversity_metrics,
    }


def entity_categorical_support_report(
    train: pd.DataFrame,
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity: str,
    target: str,
) -> dict[str, Any]:
    train_entity = train[entity].astype(str)
    train_target = train[target].astype(str)
    train_pairs = set(zip(train_entity, train_target))
    real_entity = real[entity].astype(str)
    syn_entity = synthetic[entity].astype(str)
    real_target = real[target].astype(str)
    syn_target = synthetic[target].astype(str)
    real_seen = pd.Series(
        [
            pair in train_pairs
            for pair in zip(real_entity, real_target)
        ]
    )
    syn_seen = pd.Series(
        [
            pair in train_pairs
            for pair in zip(syn_entity, syn_target)
        ]
    )
    modes = (
        pd.DataFrame({"entity": train_entity, "target": train_target})
        .groupby("entity")["target"]
        .agg(lambda values: values.value_counts().index[0])
    )
    real_mode = real_entity.map(modes)
    syn_mode = syn_entity.map(modes)
    counts = train_entity.value_counts()
    frequency_mapping = frequency_bucket_mapping(counts)
    by_frequency = conditioned_categorical_by_mapping(
        real,
        synthetic,
        entity,
        target,
        frequency_mapping,
    )
    return {
        "synthetic_entity_category_pair_seen_in_train_rate": float(
            syn_seen.mean()
        ),
        "real_test_entity_category_pair_seen_in_train_rate": float(
            real_seen.mean()
        ),
        "train_mode_available_rate": float(real_mode.notna().mean()),
        "real_matches_train_entity_mode_rate": float(
            (real_target[real_mode.notna()] == real_mode.dropna()).mean()
        )
        if real_mode.notna().any()
        else None,
        "synthetic_matches_train_entity_mode_rate": float(
            (syn_target[syn_mode.notna()] == syn_mode.dropna()).mean()
        )
        if syn_mode.notna().any()
        else None,
        "weighted_entity_total_variation": weighted_entity_categorical_tv(
            real,
            synthetic,
            entity,
            target,
        ),
        "by_entity_frequency_bucket": by_frequency,
    }


def entity_nearest_distances(
    entities: np.ndarray,
    values: np.ndarray,
    support_by_entity: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full(len(values), np.nan, dtype=float)
    seen = np.zeros(len(values), dtype=bool)
    grouped = pd.DataFrame(
        {
            "entity": entities.astype(str),
            "position": np.arange(len(entities), dtype=np.int64),
        }
    ).groupby("entity", sort=False)["position"]
    for entity, position_series in grouped:
        positions = position_series.to_numpy(dtype=np.int64)
        support = support_by_entity.get(str(entity))
        finite = positions[np.isfinite(values[positions])]
        if support is None or not len(support) or not len(finite):
            continue
        distances[finite] = nearest_support_distances(
            values[finite],
            support,
        )
        seen[finite] = True
    return distances, seen


def weighted_entity_numerical_errors(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity: str,
    target: str,
) -> dict[str, Any]:
    real_group = real.groupby(entity)[target]
    syn_group = synthetic.groupby(entity)[target]
    real_mean = real_group.mean()
    syn_mean = syn_group.mean()
    common = real_mean.index.intersection(syn_mean.index)
    weights = real[entity].value_counts().reindex(common).to_numpy(float)
    mean_errors = np.abs(
        real_mean.loc[common].to_numpy(float)
        - syn_mean.loc[common].to_numpy(float)
    )
    quantile_errors = []
    quantile_weights = []
    real_groups = real.groupby(entity, sort=False)[target]
    synthetic_groups = synthetic.groupby(entity, sort=False)[target]
    for key in common:
        real_values = pd.to_numeric(
            real_groups.get_group(key),
            errors="coerce",
        ).dropna()
        try:
            synthetic_group = synthetic_groups.get_group(key)
        except KeyError:
            continue
        syn_values = pd.to_numeric(synthetic_group, errors="coerce").dropna()
        if len(real_values) < 2 or len(syn_values) < 2:
            continue
        quantile_errors.append(
            float(
                np.mean(
                    np.abs(
                        real_values.quantile([0.25, 0.5, 0.75]).to_numpy()
                        - syn_values.quantile([0.25, 0.5, 0.75]).to_numpy()
                    )
                )
            )
        )
        quantile_weights.append(len(real_values))
    scale = max(
        float(pd.to_numeric(real[target], errors="coerce").std()),
        1e-12,
    )
    return {
        "num_entities_compared": int(len(common)),
        "weighted_group_mean_mae": float(
            np.average(mean_errors, weights=weights)
        )
        if len(mean_errors)
        else None,
        "weighted_group_mean_standardized_mae": float(
            np.average(mean_errors, weights=weights) / scale
        )
        if len(mean_errors)
        else None,
        "weighted_group_quantile_mae": float(
            np.average(quantile_errors, weights=quantile_weights)
        )
        if quantile_errors
        else None,
    }


def weighted_entity_categorical_tv(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity: str,
    target: str,
) -> float | None:
    distances = []
    weights = []
    synthetic_groups = synthetic.groupby(entity, sort=False)[target]
    for key, real_group in real.groupby(entity, sort=False):
        try:
            syn_group = synthetic_groups.get_group(key)
        except KeyError:
            continue
        if len(real_group) < 2 or not len(syn_group):
            continue
        distances.append(total_variation(real_group[target], syn_group))
        weights.append(len(real_group))
    return (
        float(np.average(distances, weights=weights))
        if distances
        else None
    )


def conditioned_numerical_by_mapping(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity: str,
    target: str,
    mapping: dict[str, str],
) -> dict[str, Any]:
    real_augmented = real.copy()
    synthetic_augmented = synthetic.copy()
    real_augmented["__bucket"] = (
        real_augmented[entity].astype(str).map(mapping).fillna("cold")
    )
    synthetic_augmented["__bucket"] = (
        synthetic_augmented[entity].astype(str).map(mapping).fillna("cold")
    )
    output = {}
    for bucket, real_group in real_augmented.groupby("__bucket"):
        synthetic_group = synthetic_augmented.loc[
            synthetic_augmented["__bucket"] == bucket
        ]
        if not len(synthetic_group):
            continue
        real_values = pd.to_numeric(real_group[target], errors="coerce")
        synthetic_values = pd.to_numeric(
            synthetic_group[target],
            errors="coerce",
        )
        output[str(bucket)] = {
            "rows": int(len(real_group)),
            "mean_absolute_error": float(
                abs(real_values.mean() - synthetic_values.mean())
            ),
            "quantile_mae": float(
                np.mean(
                    np.abs(
                        real_values.quantile([0.25, 0.5, 0.75]).to_numpy()
                        - synthetic_values.quantile(
                            [0.25, 0.5, 0.75]
                        ).to_numpy()
                    )
                )
            ),
        }
    return output


def conditioned_categorical_by_mapping(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity: str,
    target: str,
    mapping: dict[str, str],
) -> dict[str, Any]:
    output = {}
    real_bucket = real[entity].astype(str).map(mapping).fillna("cold")
    syn_bucket = synthetic[entity].astype(str).map(mapping).fillna("cold")
    for bucket in sorted(set(real_bucket)):
        real_values = real.loc[real_bucket == bucket, target]
        syn_values = synthetic.loc[syn_bucket == bucket, target]
        if not len(syn_values):
            continue
        output[str(bucket)] = {
            "rows": int(len(real_values)),
            "total_variation": total_variation(real_values, syn_values),
        }
    return output


def temporal_conditional_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Any,
) -> dict[str, Any]:
    timestamp = schema.datetime_columns[0]
    bins, bin_metadata = adaptive_time_bins(real[timestamp])
    real_augmented = real.copy()
    synthetic_augmented = synthetic.copy()
    real_augmented["__time_bin"] = bins
    synthetic_augmented["__time_bin"] = bins
    output: dict[str, Any] = {"binning": bin_metadata, "attributes": {}}
    for target in schema.numerical_targets:
        output["attributes"][target] = conditioned_numerical_by_mapping(
            real_augmented,
            synthetic_augmented,
            "__time_bin",
            target,
            {str(value): str(value) for value in set(bins)},
        )
    for target in schema.categorical_targets:
        output["attributes"][target] = conditioned_categorical_by_mapping(
            real_augmented,
            synthetic_augmented,
            "__time_bin",
            target,
            {str(value): str(value) for value in set(bins)},
        )
    return output


def entity_temporal_conditional_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Any,
) -> dict[str, Any]:
    """Measure target fidelity jointly by entity and adaptive time bin."""

    timestamp = schema.datetime_columns[0]
    bins, bin_metadata = adaptive_time_bins(real[timestamp])
    real_augmented = real.copy()
    synthetic_augmented = synthetic.copy()
    real_augmented["__time_bin"] = bins.to_numpy()
    synthetic_augmented["__time_bin"] = bins.to_numpy()
    output: dict[str, Any] = {
        "binning": bin_metadata,
        "numerical": {},
        "categorical": {},
    }
    for entity in schema.foreign_key_columns:
        for target in schema.numerical_targets:
            key = f"{entity}__{target}"
            output["numerical"][key] = weighted_entity_time_numerical_error(
                real_augmented,
                synthetic_augmented,
                entity,
                target,
            )
        for target in schema.categorical_targets:
            key = f"{entity}__{target}"
            output["categorical"][key] = (
                weighted_entity_time_categorical_error(
                    real_augmented,
                    synthetic_augmented,
                    entity,
                    target,
                )
            )
    return output


def weighted_entity_time_numerical_error(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity: str,
    target: str,
    *,
    minimum_cell_rows: int = 2,
) -> dict[str, Any]:
    keys = [entity, "__time_bin"]
    real_stats = real.groupby(keys)[target].agg(["mean", "size"])
    syn_stats = synthetic.groupby(keys)[target].agg(["mean", "size"])
    common = real_stats.index.intersection(syn_stats.index)
    eligible = common[
        (real_stats.loc[common, "size"] >= int(minimum_cell_rows))
        & (syn_stats.loc[common, "size"] >= int(minimum_cell_rows))
    ]
    if not len(eligible):
        return {
            "num_entity_time_cells": 0,
            "weighted_mean_mae": None,
            "weighted_standardized_mae": None,
        }
    errors = np.abs(
        real_stats.loc[eligible, "mean"].to_numpy(float)
        - syn_stats.loc[eligible, "mean"].to_numpy(float)
    )
    weights = real_stats.loc[eligible, "size"].to_numpy(float)
    scale = max(
        float(pd.to_numeric(real[target], errors="coerce").std()),
        1e-12,
    )
    weighted = float(np.average(errors, weights=weights))
    return {
        "num_entity_time_cells": int(len(eligible)),
        "weighted_mean_mae": weighted,
        "weighted_standardized_mae": float(weighted / scale),
    }


def weighted_entity_time_categorical_error(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity: str,
    target: str,
    *,
    minimum_cell_rows: int = 2,
) -> dict[str, Any]:
    keys = [entity, "__time_bin"]
    real_groups = real.groupby(keys)
    synthetic_groups = synthetic.groupby(keys)
    distances: list[float] = []
    weights: list[int] = []
    for key, real_group in real_groups:
        if len(real_group) < int(minimum_cell_rows):
            continue
        try:
            synthetic_group = synthetic_groups.get_group(key)
        except KeyError:
            continue
        if len(synthetic_group) < int(minimum_cell_rows):
            continue
        distances.append(
            total_variation(real_group[target], synthetic_group[target])
        )
        weights.append(int(len(real_group)))
    return {
        "num_entity_time_cells": int(len(distances)),
        "weighted_total_variation": (
            float(np.average(distances, weights=weights))
            if distances
            else None
        ),
    }


def adaptive_time_bins(
    timestamps: pd.Series,
    *,
    desired_bins: int = 8,
    minimum_rows: int = 50,
) -> tuple[pd.Series, dict[str, Any]]:
    parsed = pd.to_datetime(timestamps, errors="coerce", utc=True)
    weekly = parsed.dt.to_period("W").astype(str)
    counts = weekly.value_counts()
    if len(counts) >= 4 and int(counts.min()) >= int(minimum_rows):
        return weekly, {
            "mode": "weekly",
            "num_bins": int(len(counts)),
            "minimum_bin_rows": int(counts.min()),
        }
    valid = parsed.notna()
    output = pd.Series("missing", index=timestamps.index, dtype=object)
    if valid.any():
        ranks = parsed.loc[valid].rank(method="first")
        q = min(int(desired_bins), int(valid.sum()))
        labels = pd.qcut(
            ranks,
            q=q,
            labels=False,
            duplicates="drop",
        )
        output.loc[valid] = labels.map(
            lambda value: f"adaptive_q{int(value) + 1}"
        )
    counts = output.value_counts()
    return output, {
        "mode": "adaptive_equal_count",
        "num_bins": int(len(counts)),
        "minimum_bin_rows": int(counts.min()) if len(counts) else 0,
    }


def history_conditional_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    history_prefix: pd.DataFrame,
    schema: Any,
) -> dict[str, Any]:
    from scripts.evaluate_lstm_attribute_diagnostics import (
        add_condition_groups,
        conditional_metrics,
    )

    conditioned_real, conditioned_syn, coverage = add_condition_groups(
        real,
        synthetic,
        history_prefix,
        schema,
    )
    conditions = [
        "_history_status",
        *[
            f"_{column}_history_bucket"
            for column in schema.foreign_key_columns
        ],
    ]
    return {
        "coverage": coverage,
        "metrics": {
            condition: conditional_metrics(
                conditioned_real,
                conditioned_syn,
                condition,
                schema,
            )
            for condition in conditions
        },
    }


def quantile_bucket_mapping(
    values: pd.Series,
    *,
    prefix: str,
) -> dict[str, str]:
    if not len(values):
        return {}
    try:
        buckets = pd.qcut(
            values.rank(method="first"),
            q=min(4, len(values)),
            labels=False,
            duplicates="drop",
        )
        return {
            str(key): f"{prefix}_q{int(bucket) + 1}"
            for key, bucket in buckets.items()
        }
    except ValueError:
        return {str(key): f"{prefix}_positive" for key in values.index}


def assert_aligned_spine(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Any,
) -> None:
    if len(real) != len(synthetic):
        raise ValueError(
            f"Aligned diagnostic requires equal rows: {len(real)} != {len(synthetic)}"
        )
    for column in schema.condition_columns:
        if column in schema.datetime_columns:
            first = pd.to_datetime(real[column], errors="coerce", utc=True)
            second = pd.to_datetime(
                synthetic[column],
                errors="coerce",
                utc=True,
            )
            matches = first.equals(second)
        else:
            matches = np.array_equal(
                real[column].astype(str).to_numpy(),
                synthetic[column].astype(str).to_numpy(),
            )
        if not matches:
            raise ValueError(
                f"Event-spine column {column!r} differs or is reordered"
            )


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(
        pd.util.hash_pandas_object(
            frame,
            index=True,
        ).to_numpy(dtype=np.uint64).tobytes()
    )
    digest.update("\x1f".join(map(str, frame.columns)).encode("utf-8"))
    return digest.hexdigest()


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
