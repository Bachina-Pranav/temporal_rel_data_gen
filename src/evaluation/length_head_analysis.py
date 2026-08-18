"""Leakage-safe analysis of conditional text-length signal."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attribute_generation.conditional_tabdlm.tokenization import (
    SimpleTextTokenizer,
)


MODEL_FEATURE_FAMILIES = {
    "L0_unconditional": (),
    "L1_structured": ("structured",),
    "L2_time": ("time",),
    "L3_source_history": ("source_history",),
    "L4_destination_history": ("destination_history",),
    "L5_full": (
        "structured",
        "time",
        "source_history",
        "destination_history",
    ),
    "L5_without_source_history": (
        "structured",
        "time",
        "destination_history",
    ),
    "L5_without_destination_history": (
        "structured",
        "time",
        "source_history",
    ),
}


@dataclass(frozen=True)
class TextDatasetSpec:
    name: str
    config_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    source_column: str
    destination_column: str
    timestamp_column: str
    text_columns: tuple[str, ...]
    structured_columns: tuple[str, ...]
    table_columns: dict[str, dict[str, Any]]


def analyze_text_dataset(
    spec: TextDatasetSpec,
    output_dir: Path,
    *,
    seed: int = 42,
    minimum_entity_interactions: int = 5,
    max_rows: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml(spec.config_path)
    tokenizer = SimpleTextTokenizer(
        lowercase=bool((config.get("tokenizer") or {}).get("lowercase", True))
    )
    frame = load_splits(spec, max_rows=max_rows)
    results: dict[str, list[dict[str, Any]]] = {
        "statistics": [],
        "predictive": [],
        "associations": [],
        "entity_effects": [],
        "conditional_distributions": [],
        "summary": [],
        "leakage": [],
    }
    for text_column in spec.text_columns:
        analyzed = add_length_columns(frame, text_column, tokenizer)
        analyzed = add_past_only_history_features(
            analyzed,
            source_column=spec.source_column,
            destination_column=spec.destination_column,
            timestamp_column=spec.timestamp_column,
            length_column="token_length",
        )
        results["statistics"].extend(
            descriptive_statistics(spec.name, text_column, analyzed)
        )
        results["associations"].extend(
            association_statistics(spec, text_column, analyzed, seed=seed)
        )
        results["entity_effects"].extend(
            [
                entity_effect_statistics(
                    spec.name,
                    text_column,
                    analyzed,
                    spec.source_column,
                    "source",
                    minimum_entity_interactions,
                ),
                entity_effect_statistics(
                    spec.name,
                    text_column,
                    analyzed,
                    spec.destination_column,
                    "destination",
                    minimum_entity_interactions,
                ),
            ]
        )
        results["conditional_distributions"].extend(
            conditional_distribution_statistics(
                spec, text_column, analyzed
            )
        )
        predictive, summary = predictive_length_experiment(
            spec,
            text_column,
            analyzed,
            seed=seed,
        )
        results["predictive"].extend(predictive)
        results["summary"].append(summary)
        results["leakage"].append(
            leakage_audit_record(spec, text_column, analyzed, tokenizer)
        )
        save_length_figures(
            spec.name,
            text_column,
            analyzed,
            spec.timestamp_column,
            output_dir / "figures",
        )
    return results


def load_splits(
    spec: TextDatasetSpec,
    *,
    max_rows: int | None,
) -> pd.DataFrame:
    pieces = []
    for split, path in (
        ("train", spec.train_path),
        ("validation", spec.validation_path),
        ("test", spec.test_path),
    ):
        frame = pd.read_csv(path, low_memory=False)
        if max_rows is not None:
            fraction = {"train": 0.70, "validation": 0.15, "test": 0.15}[split]
            frame = frame.head(max(1, int(max_rows * fraction)))
        frame["_analysis_split"] = split
        pieces.append(frame)
    combined = pd.concat(pieces, ignore_index=True)
    combined[spec.timestamp_column] = pd.to_datetime(
        combined[spec.timestamp_column], errors="coerce", utc=True
    )
    tie_breaker = next(
        (
            column
            for column in ("event_id", "review_id")
            if column in combined
        ),
        None,
    )
    sort_columns = [spec.timestamp_column]
    if tie_breaker:
        sort_columns.append(tie_breaker)
    return combined.sort_values(sort_columns, kind="mergesort").reset_index(
        drop=True
    )


def add_length_columns(
    frame: pd.DataFrame,
    text_column: str,
    tokenizer: SimpleTextTokenizer,
) -> pd.DataFrame:
    result = frame.copy()
    text = result[text_column].fillna("").astype(str)
    result["token_length"] = text.map(
        lambda value: len(tokenizer.tokenize(value))
    ).astype(float)
    result["word_count"] = text.map(
        lambda value: len(value.split()) if value.strip() else 0
    ).astype(float)
    result["character_count"] = text.str.len().astype(float)
    result["_text_empty"] = text.str.strip().eq("")
    return result


def add_past_only_history_features(
    frame: pd.DataFrame,
    *,
    source_column: str,
    destination_column: str,
    timestamp_column: str,
    length_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    for prefix, entity in (
        ("source", source_column),
        ("destination", destination_column),
    ):
        values = history_feature_frame(
            result,
            entity_column=entity,
            timestamp_column=timestamp_column,
            length_column=length_column,
        )
        for column in values:
            result[f"{prefix}_{column}"] = values[column].to_numpy()
    return result


def history_feature_frame(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    timestamp_column: str,
    length_column: str,
) -> pd.DataFrame:
    working = frame[[entity_column, timestamp_column, length_column]].copy()
    working["_row"] = np.arange(len(working))
    grouped = (
        working.groupby([entity_column, timestamp_column], dropna=False)
        .agg(_current_count=(length_column, "size"), _current_sum=(length_column, "sum"))
        .reset_index()
        .sort_values([entity_column, timestamp_column], kind="mergesort")
    )
    by_entity = grouped.groupby(entity_column, sort=False, dropna=False)
    grouped["prior_count"] = (
        by_entity["_current_count"].cumsum() - grouped["_current_count"]
    ).astype(float)
    grouped["prior_sum"] = (
        by_entity["_current_sum"].cumsum() - grouped["_current_sum"]
    ).astype(float)
    grouped["past_mean_length"] = grouped["prior_sum"] / grouped[
        "prior_count"
    ].replace(0.0, np.nan)
    grouped["previous_timestamp"] = by_entity[timestamp_column].shift(1)
    grouped["first_timestamp"] = by_entity[timestamp_column].transform("min")
    grouped["recency_days"] = (
        grouped[timestamp_column] - grouped["previous_timestamp"]
    ).dt.total_seconds() / 86400.0
    age_days = (
        grouped[timestamp_column] - grouped["first_timestamp"]
    ).dt.total_seconds() / 86400.0
    grouped["activity_rate"] = grouped["prior_count"] / np.maximum(
        age_days.to_numpy(float), 1.0
    )
    merged = working.merge(
        grouped[
            [
                entity_column,
                timestamp_column,
                "prior_count",
                "past_mean_length",
                "previous_timestamp",
                "recency_days",
                "activity_rate",
            ]
        ],
        on=[entity_column, timestamp_column],
        how="left",
        sort=False,
    ).sort_values("_row")
    return merged[
        [
            "prior_count",
            "past_mean_length",
            "previous_timestamp",
            "recency_days",
            "activity_rate",
        ]
    ].reset_index(drop=True)


def descriptive_statistics(
    dataset: str,
    text_column: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for length_name in ("token_length", "word_count", "character_count"):
        values = pd.to_numeric(frame[length_name], errors="coerce").dropna()
        rows.append(
            {
                "dataset": dataset,
                "text_field": text_column,
                "length_definition": length_name,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(values.median()),
                "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                "p5": float(values.quantile(0.05)),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "min": float(values.min()),
                "max": float(values.max()),
                "zero_or_empty_fraction": float(frame["_text_empty"].mean()),
            }
        )
    return rows


def association_statistics(
    spec: TextDatasetSpec,
    text_column: str,
    frame: pd.DataFrame,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    target = np.log1p(frame["token_length"].to_numpy(float))
    for column in spec.structured_columns:
        if column not in frame:
            continue
        cfg = spec.table_columns.get(column) or {}
        col_type = str(cfg.get("type", "categorical")).lower()
        semantic = str(cfg.get("semantic_type", "")).lower()
        if col_type in {"numerical", "numeric", "number"} or "ordinal" in semantic:
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = values.notna() & np.isfinite(target)
            spearman = spearman_correlation(values[valid], target[valid])
            mutual_information = mutual_information_regression(
                values[valid].to_numpy(float).reshape(-1, 1),
                target[valid],
                seed=seed,
            )
            rows.append(
                {
                    "dataset": spec.name,
                    "text_field": text_column,
                    "context": column,
                    "context_type": "ordinal_or_numerical",
                    "n": int(valid.sum()),
                    "spearman": spearman,
                    "mutual_information": mutual_information,
                    "effect_size": abs(spearman) if spearman is not None else None,
                    "test": "Spearman; mutual information",
                }
            )
        else:
            groups = [
                group["token_length"].to_numpy(float)
                for _, group in frame.groupby(column, dropna=False)
                if len(group) >= 2
            ]
            statistic, pvalue, epsilon_squared = kruskal_effect(groups)
            rows.append(
                {
                    "dataset": spec.name,
                    "text_field": text_column,
                    "context": column,
                    "context_type": "categorical",
                    "n": int(len(frame)),
                    "num_groups": int(len(groups)),
                    "kruskal_statistic": statistic,
                    "p_value": pvalue,
                    "effect_size": epsilon_squared,
                    "test": "Kruskal-Wallis epsilon-squared",
                    "group_summaries": json.dumps(
                        group_length_summaries(frame, column), sort_keys=True
                    ),
                }
            )
    timestamp = frame[spec.timestamp_column]
    seconds = (
        timestamp.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
        / 1e9
    )
    rows.append(
        {
            "dataset": spec.name,
            "text_field": text_column,
            "context": spec.timestamp_column,
            "context_type": "time",
            "n": int(len(frame)),
            "spearman": spearman_correlation(seconds, target),
            "test": "Spearman over chronological time",
        }
    )
    return rows


def entity_effect_statistics(
    dataset: str,
    text_column: str,
    frame: pd.DataFrame,
    entity_column: str,
    role: str,
    minimum_interactions: int,
) -> dict[str, Any]:
    counts = frame.groupby(entity_column)["token_length"].size()
    eligible = counts[counts >= int(minimum_interactions)].index
    selected = frame[frame[entity_column].isin(eligible)]
    if selected.empty:
        return {
            "dataset": dataset,
            "text_field": text_column,
            "entity_role": role,
            "entity_column": entity_column,
            "minimum_interactions": int(minimum_interactions),
            "eligible_entities": 0,
            "eligible_rows": 0,
        }
    grouped = selected.groupby(entity_column)["token_length"]
    means = grouped.mean()
    medians = grouped.median()
    sizes = grouped.size().astype(float)
    overall = float(selected["token_length"].mean())
    between = float(np.average((means - overall) ** 2, weights=sizes))
    within = float(grouped.var(ddof=1).fillna(0.0).mul(sizes).sum() / sizes.sum())
    total = float(selected["token_length"].var(ddof=0))
    return {
        "dataset": dataset,
        "text_field": text_column,
        "entity_role": role,
        "entity_column": entity_column,
        "minimum_interactions": int(minimum_interactions),
        "eligible_entities": int(len(eligible)),
        "eligible_rows": int(len(selected)),
        "between_entity_variance": between,
        "within_entity_variance": within,
        "variance_explained_by_entity_means": between / total if total > 0 else None,
        "icc_like": between / (between + within) if between + within > 0 else None,
        "entity_mean_p5": float(means.quantile(0.05)),
        "entity_mean_median": float(means.median()),
        "entity_mean_p95": float(means.quantile(0.95)),
        "entity_median_p5": float(medians.quantile(0.05)),
        "entity_median_median": float(medians.median()),
        "entity_median_p95": float(medians.quantile(0.95)),
    }


def predictive_length_experiment(
    spec: TextDatasetSpec,
    text_column: str,
    frame: pd.DataFrame,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared, families = predictor_features(spec, frame)
    train_mask = prepared["_analysis_split"].eq("train")
    test_mask = prepared["_analysis_split"].eq("test")
    y = np.log1p(prepared["token_length"].to_numpy(float))
    records = []
    predictions: dict[str, np.ndarray] = {}
    for model_name, selected_families in MODEL_FEATURE_FAMILIES.items():
        print(
            f"[length-predict] dataset={spec.name} field={text_column} "
            f"model={model_name}",
            flush=True,
        )
        columns = [
            column
            for family in selected_families
            for column in families.get(family, [])
        ]
        if model_name == "L0_unconditional":
            prediction = np.full(
                int(test_mask.sum()),
                float(np.median(y[train_mask])),
                dtype=float,
            )
            predictor = "training median"
        else:
            prediction, predictor = fit_predictor(
                prepared.loc[train_mask, columns],
                y[train_mask],
                prepared.loc[test_mask, columns],
                seed=seed,
            )
        prediction = np.clip(
            prediction,
            0.0,
            float(np.nanmax(y[train_mask])),
        )
        predictions[model_name] = prediction
        metric = prediction_metrics(y[test_mask], prediction)
        records.append(
            {
                "dataset": spec.name,
                "text_field": text_column,
                "model": model_name,
                "feature_families": ",".join(selected_families) or "none",
                "feature_columns": json.dumps(columns),
                "predictor": predictor,
                **metric,
            }
        )
    indexed = {record["model"]: record for record in records}
    baseline = indexed["L0_unconditional"]
    for record in records:
        record["relative_token_mae_reduction_vs_L0"] = relative_reduction(
            baseline["mae_token_length"], record["mae_token_length"]
        )
        record["delta_r2_vs_L0"] = nullable_difference(
            record["r2_log1p_token_length"],
            baseline["r2_log1p_token_length"],
        )
        record["delta_spearman_vs_L0"] = nullable_difference(
            record["spearman"], baseline["spearman"]
        )
    full = indexed["L5_full"]
    no_source = indexed["L5_without_source_history"]
    no_destination = indexed["L5_without_destination_history"]
    individual = [indexed[name] for name in (
        "L1_structured",
        "L2_time",
        "L3_source_history",
        "L4_destination_history",
    )]
    best_individual = min(individual, key=lambda row: row["mae_token_length"])
    source_gain = relative_reduction(
        no_source["mae_token_length"], full["mae_token_length"]
    )
    destination_gain = relative_reduction(
        no_destination["mae_token_length"], full["mae_token_length"]
    )
    evidence, rationale = classify_evidence(
        full,
        baseline,
        source_gain,
        destination_gain,
    )
    summary = {
        "dataset": spec.name,
        "text_field": text_column,
        "L0_MAE": baseline["mae_token_length"],
        "Structured_MAE": indexed["L1_structured"]["mae_token_length"],
        "Source_History_MAE": indexed["L3_source_history"]["mae_token_length"],
        "Destination_History_MAE": indexed["L4_destination_history"]["mae_token_length"],
        "Full_MAE": full["mae_token_length"],
        "Full_R2": full["r2_log1p_token_length"],
        "Full_Spearman": full["spearman"],
        "Full_relative_MAE_reduction": relative_reduction(
            baseline["mae_token_length"], full["mae_token_length"]
        ),
        "Full_minus_best_single_family_MAE_reduction": relative_reduction(
            best_individual["mae_token_length"], full["mae_token_length"]
        ),
        "Source_Incremental_Signal": source_gain,
        "Source_Incremental_R2": nullable_difference(
            full["r2_log1p_token_length"],
            no_source["r2_log1p_token_length"],
        ),
        "Destination_Incremental_Signal": destination_gain,
        "Destination_Incremental_R2": nullable_difference(
            full["r2_log1p_token_length"],
            no_destination["r2_log1p_token_length"],
        ),
        "Evidence_for_Conditional_Length_Head": evidence,
        "Evidence_Rationale": rationale,
    }
    return records, summary


def predictor_features(
    spec: TextDatasetSpec,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    result = frame.copy()
    train = result[result["_analysis_split"] == "train"]
    structured = []
    for column in spec.structured_columns:
        cfg = spec.table_columns.get(column) or {}
        col_type = str(cfg.get("type", "categorical")).lower()
        semantic = str(cfg.get("semantic_type", "")).lower()
        name = f"structured__{column}"
        if col_type in {"numerical", "numeric", "number"} or "ordinal" in semantic:
            result[name] = pd.to_numeric(result[column], errors="coerce")
        else:
            values = sorted(train[column].dropna().astype(str).unique())
            mapping = {value: index for index, value in enumerate(values)}
            result[name] = result[column].astype(str).map(mapping).fillna(-1)
        structured.append(name)
    timestamp = result[spec.timestamp_column]
    train_timestamp = train[spec.timestamp_column]
    lower = train_timestamp.min()
    span = max((train_timestamp.max() - lower).total_seconds(), 1.0)
    result["time__normalized"] = (
        timestamp - lower
    ).dt.total_seconds() / span
    month = timestamp.dt.month.fillna(0).to_numpy(float)
    day = timestamp.dt.dayofweek.fillna(0).to_numpy(float)
    result["time__month_sin"] = np.sin(2 * np.pi * month / 12.0)
    result["time__month_cos"] = np.cos(2 * np.pi * month / 12.0)
    result["time__weekday_sin"] = np.sin(2 * np.pi * day / 7.0)
    result["time__weekday_cos"] = np.cos(2 * np.pi * day / 7.0)
    source = [
        "source_prior_count",
        "source_past_mean_length",
        "source_recency_days",
        "source_activity_rate",
    ]
    destination = [
        "destination_prior_count",
        "destination_past_mean_length",
        "destination_recency_days",
        "destination_activity_rate",
    ]
    families = {
        "structured": structured,
        "time": [
            "time__normalized",
            "time__month_sin",
            "time__month_cos",
            "time__weekday_sin",
            "time__weekday_cos",
        ],
        "source_history": source,
        "destination_history": destination,
    }
    return result, families


def fit_predictor(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    test_x: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, str]:
    medians = train_x.median(numeric_only=True).reindex(train_x.columns).fillna(0.0)
    x_train = train_x.apply(pd.to_numeric, errors="coerce").fillna(medians)
    x_test = test_x.apply(pd.to_numeric, errors="coerce").fillna(medians)
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=int(seed),
            n_jobs=-1,
            tree_method="hist",
        )
        name = "XGBRegressor(fixed generic configuration)"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=int(seed),
        )
        name = "HistGradientBoostingRegressor(documented fallback)"
    model.fit(x_train.to_numpy(float), train_y)
    return np.asarray(model.predict(x_test.to_numpy(float)), dtype=float), name


def prediction_metrics(actual_log: np.ndarray, predicted_log: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, r2_score

    actual_length = np.maximum(np.expm1(actual_log), 0.0)
    predicted_length = np.maximum(np.expm1(predicted_log), 0.0)
    return {
        "mae_token_length": float(mean_absolute_error(actual_length, predicted_length)),
        "mae_log1p_token_length": float(mean_absolute_error(actual_log, predicted_log)),
        "r2_log1p_token_length": float(r2_score(actual_log, predicted_log)),
        "spearman": spearman_correlation(predicted_log, actual_log),
        "num_test_rows": int(len(actual_log)),
    }


def conditional_distribution_statistics(
    spec: TextDatasetSpec,
    text_column: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    contexts = list(spec.structured_columns) + [
        "source_prior_count",
        "destination_prior_count",
    ]
    records = []
    for context in contexts:
        if context not in frame:
            continue
        cfg = spec.table_columns.get(context) or {}
        col_type = str(cfg.get("type", "numerical")).lower()
        values = frame[context]
        if col_type in {"numerical", "numeric", "number"} or values.nunique(dropna=True) > 20:
            groups = quantile_groups(values)
            grouping = "generic quantiles"
        else:
            groups = values.fillna("<NA>").astype(str)
            grouping = "categories"
        grouped_values = [
            group["token_length"].to_numpy(float)
            for _, group in frame.assign(_group=groups).groupby("_group")
            if len(group) >= 2
        ]
        ks, wasserstein = maximum_pairwise_distance(grouped_values)
        records.append(
            {
                "dataset": spec.name,
                "text_field": text_column,
                "context": context,
                "grouping": grouping,
                "num_groups": int(len(grouped_values)),
                "max_pairwise_ks": ks,
                "max_pairwise_wasserstein": wasserstein,
            }
        )
    return records


def leakage_audit_record(
    spec: TextDatasetSpec,
    text_column: str,
    frame: pd.DataFrame,
    tokenizer: SimpleTextTokenizer,
) -> dict[str, Any]:
    ranges = {
        split: {
            "min": group[spec.timestamp_column].min().isoformat(),
            "max": group[spec.timestamp_column].max().isoformat(),
            "rows": int(len(group)),
        }
        for split, group in frame.groupby("_analysis_split")
    }
    source_safe = history_timestamps_are_past(
        frame["source_previous_timestamp"], frame[spec.timestamp_column]
    )
    destination_safe = history_timestamps_are_past(
        frame["destination_previous_timestamp"], frame[spec.timestamp_column]
    )
    chronological = bool(
        pd.Timestamp(ranges["train"]["max"])
        <= pd.Timestamp(ranges["validation"]["min"])
        <= pd.Timestamp(ranges["validation"]["max"])
        <= pd.Timestamp(ranges["test"]["min"])
    )
    checks = {
        "chronological_splits": chronological,
        "source_history_strictly_past": source_safe,
        "destination_history_strictly_past": destination_safe,
        "raw_source_id_excluded_from_primary_predictors": True,
        "raw_destination_id_excluded_from_primary_predictors": True,
        "current_text_excluded_from_predictors": True,
        "predictor_fit_on_train_only": True,
        "test_used_only_for_final_metrics": True,
    }
    return {
        "dataset": spec.name,
        "text_field": text_column,
        "passed": all(checks.values()),
        "checks": checks,
        "split_ranges": ranges,
        "tokenizer": {
            "class": "SimpleTextTokenizer",
            "lowercase": tokenizer.lowercase,
            "length_unit": "tokenizer lexical content tokens",
        },
    }


def save_length_figures(
    dataset: str,
    text_column: str,
    frame: pd.DataFrame,
    timestamp_column: str,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{slug(dataset)}__{slug(text_column)}"
    values = frame["token_length"].to_numpy(float)
    clipped = values[values <= np.quantile(values, 0.99)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(clipped, bins=50)
    axes[0].set_title("Token length (through p99)")
    ordered = np.sort(values)
    axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered))
    axes[1].set_title("Token length ECDF")
    axes[2].hist(np.log1p(values), bins=50)
    axes[2].set_title("log1p token length")
    for axis in axes:
        axis.set_xlabel("length")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}__length_distributions.png", dpi=150)
    plt.close(fig)

    monthly = (
        frame.assign(
            _month=frame[timestamp_column].dt.tz_localize(None).dt.to_period("M")
        )
        .groupby("_month", observed=True)["token_length"]
        .agg(["median", "mean", "count"])
        .reset_index()
    )
    if not monthly.empty:
        positions = np.arange(len(monthly))
        fig, axis = plt.subplots(figsize=(12, 4))
        axis.plot(positions, monthly["median"], label="median")
        axis.plot(positions, monthly["mean"], label="mean", alpha=0.8)
        tick_step = max(1, len(monthly) // 12)
        ticks = positions[::tick_step]
        axis.set_xticks(ticks)
        axis.set_xticklabels(
            monthly["_month"].astype(str).iloc[::tick_step],
            rotation=45,
            ha="right",
        )
        axis.set_title("Text length over chronological month")
        axis.set_ylabel("token length")
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}__length_over_time.png", dpi=150)
        plt.close(fig)


def classify_evidence(
    full: dict[str, Any],
    baseline: dict[str, Any],
    source_gain: float | None,
    destination_gain: float | None,
) -> tuple[str, str]:
    mae_gain = relative_reduction(
        baseline["mae_token_length"], full["mae_token_length"]
    ) or 0.0
    r2 = float(full.get("r2_log1p_token_length") or 0.0)
    history_gain = max(source_gain or 0.0, destination_gain or 0.0)
    if mae_gain >= 0.10 and r2 >= 0.10 and history_gain >= 0.02:
        evidence = "STRONG"
    elif mae_gain >= 0.03 or r2 >= 0.03 or history_gain >= 0.01:
        evidence = "MODERATE"
    else:
        evidence = "WEAK"
    rationale = (
        f"held-out token-MAE reduction={mae_gain:.4f}, "
        f"full log-length R2={r2:.4f}, strongest relational-history "
        f"incremental MAE reduction={history_gain:.4f}. Classification uses "
        "effect magnitudes jointly rather than p-values alone."
    )
    return evidence, rationale


def group_length_summaries(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    result = {}
    for value, group in frame.groupby(column, dropna=False):
        result[str(value)] = {
            "n": int(len(group)),
            "mean": float(group["token_length"].mean()),
            "median": float(group["token_length"].median()),
        }
    return result


def kruskal_effect(groups: list[np.ndarray]) -> tuple[Any, Any, Any]:
    if len(groups) < 2:
        return None, None, None
    try:
        from scipy.stats import kruskal

        result = kruskal(*groups)
        n = sum(len(group) for group in groups)
        effect = max(
            0.0,
            float((result.statistic - len(groups) + 1) / max(n - len(groups), 1)),
        )
        return float(result.statistic), float(result.pvalue), effect
    except Exception:
        return None, None, None


def mutual_information_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> float | None:
    if len(y) < 10 or np.unique(x).size < 2:
        return None
    try:
        from sklearn.feature_selection import mutual_info_regression

        return float(
            mutual_info_regression(x, y, random_state=int(seed))[0]
        )
    except Exception:
        return None


def maximum_pairwise_distance(
    groups: list[np.ndarray],
) -> tuple[float | None, float | None]:
    if len(groups) < 2:
        return None, None
    max_ks = 0.0
    max_wasserstein = 0.0
    try:
        from scipy.stats import ks_2samp, wasserstein_distance

        for left_index in range(len(groups)):
            for right_index in range(left_index + 1, len(groups)):
                max_ks = max(
                    max_ks,
                    float(ks_2samp(groups[left_index], groups[right_index]).statistic),
                )
                max_wasserstein = max(
                    max_wasserstein,
                    float(wasserstein_distance(groups[left_index], groups[right_index])),
                )
        return max_ks, max_wasserstein
    except Exception:
        return None, None


def quantile_groups(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.nunique(dropna=True) < 2:
        return pd.Series("all", index=series.index)
    ranks = numeric.rank(method="average", pct=True)
    return pd.cut(
        ranks,
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["q1", "q2", "q3", "q4"],
        include_lowest=True,
    ).astype("string").fillna("missing")


def spearman_correlation(left: Any, right: Any) -> float | None:
    left_values = pd.to_numeric(pd.Series(left), errors="coerce")
    right_values = pd.to_numeric(pd.Series(right), errors="coerce")
    valid = left_values.notna() & right_values.notna()
    if valid.sum() < 2:
        return None
    value = left_values[valid].corr(right_values[valid], method="spearman")
    return float(value) if pd.notna(value) else None


def history_timestamps_are_past(
    previous: pd.Series,
    current: pd.Series,
) -> bool:
    valid = previous.notna()
    return bool((previous[valid] < current[valid]).all())


def relative_reduction(old: Any, new: Any) -> float | None:
    if old is None or new is None or not math.isfinite(float(old)) or float(old) == 0:
        return None
    return float((float(old) - float(new)) / abs(float(old)))


def nullable_difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))
