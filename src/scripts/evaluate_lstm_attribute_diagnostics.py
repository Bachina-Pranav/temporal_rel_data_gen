#!/usr/bin/env python3
"""Evaluate schema-driven attribute, conditional, dependency, and privacy diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.experiment_audit import strict_prior_counts  # noqa: E402
from attribute_generation.conditional_tabdlm.posthoc_diagnostics import repeated_c2st  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from evaluation.paper_metrics.shape_trend import pair_trend_error  # noqa: E402
from evaluation.paper_metrics.utils import (  # noqa: E402
    ks_distance,
    total_variation,
    wasserstein_1d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-real", required=True)
    parser.add_argument("--evaluation-real", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--graph-history-prefix", default=None)
    parser.add_argument("--evaluation-config", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--c2st-classifier-seeds",
        nargs="+",
        type=int,
        default=[11, 23, 37],
    )
    parser.add_argument("--max-c2st-rows", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train = pd.read_csv(args.train_real, low_memory=False)
    real = pd.read_csv(args.evaluation_real, low_memory=False)
    synthetic = pd.read_csv(args.synthetic, low_memory=False)
    prefix = (
        pd.read_csv(args.graph_history_prefix, low_memory=False)
        if args.graph_history_prefix
        else pd.DataFrame(columns=config.schema.condition_columns)
    )
    evaluation_config = (
        load_mapping(args.evaluation_config)
        if args.evaluation_config
        else None
    )
    if len(real) != len(synthetic):
        raise ValueError(
            f"Attribute diagnostics require aligned rows: real={len(real)}, synthetic={len(synthetic)}"
        )
    report = evaluate_attribute_diagnostics(
        train,
        real,
        synthetic,
        prefix,
        config,
        seed=int(args.seed),
        evaluation_config=evaluation_config,
        c2st_classifier_seeds=args.c2st_classifier_seeds,
        max_c2st_rows=int(args.max_c2st_rows),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(output)


def evaluate_attribute_diagnostics(
    train: pd.DataFrame,
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    prefix: pd.DataFrame,
    config: Any,
    *,
    seed: int,
    evaluation_config: dict[str, Any] | None = None,
    c2st_classifier_seeds: list[int] | tuple[int, ...] = (
        11,
        23,
        37,
    ),
    max_c2st_rows: int | None = 20000,
) -> dict[str, Any]:
    schema = config.schema
    numerical = {
        column: numerical_metrics(
            train[column],
            real[column],
            synthetic[column],
        )
        for column in schema.numerical_targets
    }
    categorical = {
        column: categorical_metrics(
            train[column],
            real[column],
            synthetic[column],
        )
        for column in schema.categorical_targets
    }
    conditioned_real, conditioned_syn, history = add_condition_groups(
        real,
        synthetic,
        prefix,
        schema,
    )
    condition_columns = [
        *schema.foreign_key_columns,
        "_time_bin",
        *[
            f"_{column}_history_bucket"
            for column in schema.foreign_key_columns
        ],
        "_history_status",
    ]
    conditional = {
        condition: conditional_metrics(
            conditioned_real,
            conditioned_syn,
            condition,
            schema,
        )
        for condition in condition_columns
    }
    dependencies = dependency_metrics(real, synthetic, config)
    privacy = privacy_metrics(
        train,
        synthetic,
        schema,
        seed=seed,
    )
    group_c2st = attribute_group_c2st(
        real,
        synthetic,
        schema,
        evaluation_config,
        classifier_seeds=c2st_classifier_seeds,
        max_rows=max_c2st_rows,
        generator_seed=seed,
    )
    return {
        "dataset_name": config.raw.get("dataset_name"),
        "num_train_rows": int(len(train)),
        "num_evaluation_rows": int(len(real)),
        "numerical_attributes": numerical,
        "categorical_attributes": categorical,
        "conditional_fidelity": conditional,
        "dependency_fidelity": dependencies,
        "history_coverage": history,
        "privacy_memorization": privacy,
        "attribute_group_c2st": group_c2st,
    }


def numerical_metrics(
    train: pd.Series,
    real: pd.Series,
    synthetic: pd.Series,
) -> dict[str, Any]:
    train_num = pd.to_numeric(train, errors="coerce")
    real_num = pd.to_numeric(real, errors="coerce")
    syn_num = pd.to_numeric(synthetic, errors="coerce")
    real_valid = real_num.dropna()
    syn_valid = syn_num.dropna()
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    real_q = real_valid.quantile(quantiles)
    syn_q = syn_valid.quantile(quantiles)
    scale = max(float(real_valid.std()), 1e-12)
    invalid = syn_num.isna() | ~np.isfinite(syn_num)
    out_of_range = (
        (syn_num < train_num.min())
        | (syn_num > train_num.max())
    ) & ~invalid
    train_support = np.sort(train_num.dropna().unique())
    real_support_overlap = support_overlap_rate(
        real_valid.to_numpy(float),
        train_support,
    )
    synthetic_support_overlap = support_overlap_rate(
        syn_valid.to_numpy(float),
        train_support,
    )
    nearest = nearest_support_distances(
        syn_valid.to_numpy(float),
        train_support,
    )
    return {
        "mean_real": float(real_valid.mean()),
        "mean_synthetic": float(syn_valid.mean()),
        "mean_absolute_error": float(abs(real_valid.mean() - syn_valid.mean())),
        "mean_standardized_error": float(
            abs(real_valid.mean() - syn_valid.mean()) / scale
        ),
        "std_real": float(real_valid.std()),
        "std_synthetic": float(syn_valid.std()),
        "std_absolute_error": float(abs(real_valid.std() - syn_valid.std())),
        "quantiles_real": {str(key): float(value) for key, value in real_q.items()},
        "quantiles_synthetic": {
            str(key): float(value) for key, value in syn_q.items()
        },
        "quantile_mae": float(np.mean(np.abs(real_q - syn_q))),
        "ks_distance": ks_distance(real_valid, syn_valid),
        "wasserstein_distance": wasserstein_1d(real_valid, syn_valid),
        "support_total_variation": total_variation(
            real_valid,
            syn_valid,
        ),
        "invalid_rate": float(invalid.mean()),
        "out_of_train_range_rate": float(out_of_range.mean()),
        "training_support_size": int(len(train_support)),
        "real_training_support_overlap_rate": real_support_overlap,
        "synthetic_training_support_overlap_rate": (
            synthetic_support_overlap
        ),
        "nearest_training_support_distance_mean": (
            float(np.mean(nearest)) if len(nearest) else None
        ),
        "nearest_training_support_distance_p95": (
            float(np.quantile(nearest, 0.95))
            if len(nearest)
            else None
        ),
        "unique_value_ratio_real": float(
            real_valid.nunique() / max(len(real_valid), 1)
        ),
        "unique_value_ratio_synthetic": float(
            syn_valid.nunique() / max(len(syn_valid), 1)
        ),
        "num_unique_real": int(real_valid.nunique()),
        "num_unique_synthetic": int(syn_valid.nunique()),
        "support_entropy_real": empirical_entropy(real_valid),
        "support_entropy_synthetic": empirical_entropy(syn_valid),
        "missingness_rate_error": float(
            abs(real_num.isna().mean() - syn_num.isna().mean())
        ),
    }


def attribute_group_c2st(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Any,
    evaluation_config: dict[str, Any] | None,
    *,
    classifier_seeds: list[int] | tuple[int, ...],
    max_rows: int | None,
    generator_seed: int,
) -> dict[str, Any]:
    if evaluation_config is None:
        return {
            "status": "skipped",
            "reason": "No evaluation configuration was supplied.",
        }
    groups = {
        "numerical_only": list(schema.numerical_targets),
        "categorical_only": list(schema.categorical_targets),
    }
    output: dict[str, Any] = {
        "status": "completed",
        "classifier_seeds": [int(value) for value in classifier_seeds],
        "max_rows_per_side": (
            int(max_rows) if max_rows is not None else None
        ),
    }
    for label, columns in groups.items():
        if not columns:
            output[label] = {
                "status": "not_applicable",
                "columns": [],
            }
            continue
        aggregate, _ = repeated_c2st(
            real,
            synthetic,
            evaluation_config,
            classifier_seeds=classifier_seeds,
            columns=columns,
            max_rows=max_rows,
            generator_seed=generator_seed,
            label=label,
        )
        output[label] = {
            "status": "completed",
            "columns": columns,
            **aggregate,
        }
    return output


def support_overlap_rate(
    values: np.ndarray,
    support: np.ndarray,
) -> float | None:
    values = np.asarray(values, dtype=float)
    support = np.asarray(support, dtype=float)
    if not len(values) or not len(support):
        return None
    return float(np.mean(np.isin(values, support)))


def nearest_support_distances(
    values: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    support = np.sort(np.asarray(support, dtype=float))
    if not len(values) or not len(support):
        return np.asarray([], dtype=float)
    right = np.searchsorted(support, values, side="left")
    left = np.clip(right - 1, 0, len(support) - 1)
    right = np.clip(right, 0, len(support) - 1)
    return np.minimum(
        np.abs(values - support[left]),
        np.abs(values - support[right]),
    )


def empirical_entropy(values: pd.Series) -> float | None:
    counts = values.value_counts(dropna=True).to_numpy(float)
    if not len(counts):
        return None
    probability = counts / counts.sum()
    return float(
        -np.sum(
            probability
            * np.log(np.maximum(probability, 1e-12))
        )
    )


def categorical_metrics(
    train: pd.Series,
    real: pd.Series,
    synthetic: pd.Series,
) -> dict[str, Any]:
    train_values = train.dropna().astype(str)
    real_values = real.dropna().astype(str)
    syn_values = synthetic.dropna().astype(str)
    domain = sorted(set(train_values))
    invalid = ~syn_values.isin(domain)
    real_counts = real_values.value_counts()
    rare_cutoff = max(5, int(np.ceil(0.01 * max(len(real_values), 1))))
    rare = set(real_counts[real_counts <= rare_cutoff].index)
    rare_covered = rare.intersection(set(syn_values))
    return {
        "train_domain": domain,
        "total_variation_distance": total_variation(
            real_values,
            syn_values,
            support=domain,
        ),
        "jensen_shannon_distance": jensen_shannon_distance(
            real_values,
            syn_values,
            domain,
        ),
        "invalid_category_rate": float(invalid.mean()) if len(syn_values) else 0.0,
        "missingness_rate_error": float(
            abs(real.isna().mean() - synthetic.isna().mean())
        ),
        "num_rare_real_categories": int(len(rare)),
        "rare_category_coverage": (
            float(len(rare_covered) / len(rare)) if rare else 1.0
        ),
    }


def add_condition_groups(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    prefix: pd.DataFrame,
    schema: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    real = real.copy()
    synthetic = synthetic.copy()
    timestamp = schema.datetime_columns[0]
    real["_time_bin"] = (
        pd.to_datetime(real[timestamp], errors="coerce", utc=True)
        .dt.to_period("M")
        .astype(str)
    )
    synthetic["_time_bin"] = real["_time_bin"].to_numpy()
    combined = pd.concat(
        [
            prefix.loc[:, list(schema.condition_columns)],
            real.loc[:, list(schema.condition_columns)],
        ],
        ignore_index=True,
    )
    times = pd.to_datetime(combined[timestamp], errors="coerce", utc=True)
    history_values: dict[str, np.ndarray] = {}
    summaries = {}
    query_rows = len(real)
    for column in schema.foreign_key_columns:
        prior = strict_prior_counts(combined[column], times)[-query_rows:]
        history_values[column] = prior
        bucket = history_buckets(prior)
        real[f"_{column}_history_bucket"] = bucket
        synthetic[f"_{column}_history_bucket"] = bucket
        summaries[column] = {
            "coverage_rate": float(np.mean(prior > 0)),
            "mean": float(np.mean(prior)),
            "p50": float(np.quantile(prior, 0.50)),
            "p95": float(np.quantile(prior, 0.95)),
        }
    first = history_values[schema.foreign_key_columns[0]]
    second = (
        history_values[schema.foreign_key_columns[1]]
        if len(schema.foreign_key_columns) > 1
        else np.zeros_like(first)
    )
    status = np.where(
        (first == 0) & (second == 0),
        "cold",
        np.where((first > 0) & (second > 0), "warm", "partial"),
    )
    real["_history_status"] = status
    synthetic["_history_status"] = status
    summaries["status_counts"] = {
        str(key): int(value)
        for key, value in pd.Series(status).value_counts().items()
    }
    return real, synthetic, summaries


def history_buckets(values: np.ndarray) -> np.ndarray:
    output = np.full(len(values), "cold", dtype=object)
    positive = values > 0
    if not positive.any():
        return output
    try:
        labels = pd.qcut(
            pd.Series(values[positive]),
            q=4,
            duplicates="drop",
        ).astype(str)
        output[positive] = labels.to_numpy()
    except ValueError:
        output[positive] = "positive"
    return output


def conditional_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    condition: str,
    schema: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_groups_real": int(real[condition].nunique(dropna=False)),
    }
    for column in schema.numerical_targets:
        real_means = real.groupby(condition, dropna=False)[column].mean()
        syn_means = synthetic.groupby(condition, dropna=False)[column].mean()
        common = real_means.index.intersection(syn_means.index)
        scale = max(float(pd.to_numeric(real[column], errors="coerce").std()), 1e-12)
        result[column] = {
            "num_groups_compared": int(len(common)),
            "group_mean_mae": (
                float(np.mean(np.abs(real_means.loc[common] - syn_means.loc[common])))
                if len(common)
                else None
            ),
            "group_mean_standardized_mae": (
                float(
                    np.mean(
                        np.abs(real_means.loc[common] - syn_means.loc[common])
                    )
                    / scale
                )
                if len(common)
                else None
            ),
        }
    for column in schema.categorical_targets:
        group_sizes = real.groupby(condition, dropna=False).size()
        eligible = group_sizes[group_sizes >= 2].index
        distances = []
        weights = []
        for group in eligible:
            real_group = real.loc[real[condition] == group, column]
            syn_group = synthetic.loc[synthetic[condition] == group, column]
            if len(syn_group) == 0:
                continue
            distances.append(total_variation(real_group, syn_group))
            weights.append(len(real_group))
        result[column] = {
            "num_groups_compared": int(len(distances)),
            "weighted_group_total_variation": (
                float(np.average(distances, weights=weights))
                if distances
                else None
            ),
        }
    return result


def dependency_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: Any,
) -> dict[str, Any]:
    schema = config.schema
    targets = list(schema.target_columns)
    columns = {
        column: {"type": "numerical"}
        for column in schema.numerical_targets
    }
    columns.update(
        {
            column: {"type": "categorical"}
            for column in schema.categorical_targets
        }
    )
    timestamp = schema.datetime_columns[0]
    columns[timestamp] = {"type": "datetime"}
    pairs = []
    for index, left in enumerate([*targets, timestamp]):
        for right in [*targets, timestamp][index + 1 :]:
            error, metric = pair_trend_error(
                real,
                synthetic,
                left,
                right,
                columns[left]["type"],
                columns[right]["type"],
                columns[left],
                columns[right],
            )
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "error": error,
                    "metric": metric,
                }
            )
    return {"pairs": pairs}


def privacy_metrics(
    train: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    targets = list(schema.target_columns)
    train_keys = target_keys(train, targets)
    syn_keys = target_keys(synthetic, targets)
    train_set = set(train_keys)
    exact = syn_keys.isin(train_set)
    counts = train_keys.value_counts()
    rare_set = set(counts[counts <= 5].index)
    rare_exact = syn_keys.isin(rare_set)
    return {
        "comparison_scope": "generated attributes only; event-spine identifiers are excluded",
        "exact_generated_attribute_tuple_overlap_rate": float(exact.mean()),
        "rare_train_tuple_overlap_rate": float(rare_exact.mean()),
        "num_unique_train_attribute_tuples": int(train_keys.nunique()),
        "num_unique_synthetic_attribute_tuples": int(syn_keys.nunique()),
        "nearest_neighbor": nearest_neighbor_summary(
            train,
            synthetic,
            schema,
            seed=seed,
        ),
    }


def nearest_neighbor_summary(
    train: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.neighbors import NearestNeighbors
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}
    train_sample = train.sample(
        n=min(50_000, len(train)),
        random_state=seed,
    )
    syn_sample = synthetic.sample(
        n=min(5_000, len(synthetic)),
        random_state=seed + 1,
    )
    numerical = list(schema.numerical_targets)
    categorical = list(schema.categorical_targets)
    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )
    train_features = transformer.fit_transform(train_sample)
    syn_features = transformer.transform(syn_sample)
    model = NearestNeighbors(n_neighbors=1, metric="euclidean")
    model.fit(train_features)
    distances, _ = model.kneighbors(syn_features)
    values = distances[:, 0]
    return {
        "status": "completed",
        "train_rows": int(len(train_sample)),
        "synthetic_rows": int(len(syn_sample)),
        "mean_distance": float(values.mean()),
        "p05_distance": float(np.quantile(values, 0.05)),
        "median_distance": float(np.median(values)),
    }


def target_keys(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame.loc[:, columns].astype(str).agg("\u241f".join, axis=1)


def jensen_shannon_distance(
    real: pd.Series,
    synthetic: pd.Series,
    support: list[str],
) -> float:
    real_counts = real.value_counts().reindex(support, fill_value=0).to_numpy(float)
    syn_counts = synthetic.value_counts().reindex(support, fill_value=0).to_numpy(float)
    p = real_counts / max(real_counts.sum(), 1.0)
    q = syn_counts / max(syn_counts.sum(), 1.0)
    midpoint = 0.5 * (p + q)
    left = np.sum(np.where(p > 0, p * np.log2(p / np.maximum(midpoint, 1e-12)), 0.0))
    right = np.sum(np.where(q > 0, q * np.log2(q / np.maximum(midpoint, 1e-12)), 0.0))
    return float(np.sqrt(max(0.5 * (left + right), 0.0)))


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def load_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


if __name__ == "__main__":
    main()
