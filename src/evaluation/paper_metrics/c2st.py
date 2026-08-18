"""Classifier two-sample tests for single event tables."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .utils import (
    canonicalize_categorical_series,
    char_lengths,
    datetime_series,
    numeric_series,
    text_hash_embedding,
    token_lengths,
)


def structured_c2st_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """C2ST over generated structured attributes only.

    Text, text-derived lengths, identifiers, foreign keys, and timestamps are
    intentionally excluded. The resolved policy is returned with the metric
    so every evaluation can be audited independently of dataset naming.
    """

    table_cfg = config.get("table") or {}
    manifest = structured_c2st_feature_manifest(table_cfg)
    if not manifest["included_columns"]:
        return {
            "status": "skipped",
            "reason": "no_generated_structured_attributes",
            "error": None,
            "metric_name": "structured_c2st_error",
            "feature_manifest": manifest,
        }, pd.DataFrame()
    structured_table = {
        "columns": {
            column: (table_cfg.get("columns") or {})[column]
            for column in manifest["included_columns"]
        }
    }
    metrics, importance = _c2st_metrics(
        real,
        synthetic,
        config,
        structured_table,
    )
    metrics.update(
        {
            "metric_name": "structured_c2st_error",
            "feature_policy": (
                "generated numerical + categorical attributes only"
            ),
            "feature_manifest": manifest,
        }
    )
    return metrics, importance


def single_table_c2st_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Backward-compatible alias for the structured-only primary C2ST."""

    return structured_c2st_metrics(real, synthetic, config)


def legacy_full_row_c2st_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reproduce the historical all-schema-column C2ST for audit only."""

    metrics, importance = _c2st_metrics(
        real,
        synthetic,
        config,
        config.get("table") or {},
    )
    metrics.update(
        {
            "metric_name": "legacy_full_row_c2st_error",
            "feature_policy": (
                "historical policy: all non-primary-key schema columns"
            ),
        }
    )
    return metrics, importance


def _c2st_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: dict[str, Any],
    table_cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    c2st_cfg = ((config.get("evaluation") or {}).get("c2st") or {})
    if not bool(c2st_cfg.get("enabled", True)):
        return {"status": "skipped", "reason": "disabled"}, pd.DataFrame()
    max_rows = c2st_cfg.get("max_rows") or c2st_cfg.get("max_rows_for_c2st") or (config.get("evaluation") or {}).get("max_rows_for_c2st", 100000)
    seed = int((config.get("evaluation") or {}).get("random_seed", 42))
    classifiers = c2st_cfg.get("classifiers") or ["logistic_regression"]
    x, y, feature_names, balanced_n = featurize_real_synthetic(real, synthetic, table_cfg, max_rows=max_rows, seed=seed)
    results = run_binary_classifiers(
        x,
        y,
        classifiers,
        seed=seed,
        n_splits=int(c2st_cfg.get("n_splits", 5)),
    )
    best_name = max(results, key=lambda name: results[name].get("auc", 0.5)) if results else None
    best = results.get(best_name, {}) if best_name else {}
    importance = feature_importance(results, feature_names)
    top_features = top_feature_records(importance, limit=10)
    return {
        "auc": best.get("auc"),
        "accuracy": best.get("accuracy"),
        "error": best.get("error"),
        "best_classifier": best_name,
        "per_classifier": results,
        "num_rows": int(len(y)),
        "balanced_eval_n_real": int(balanced_n),
        "balanced_eval_n_synthetic": int(balanced_n),
        "classes_balanced": True,
        "preprocessing_fit_scope": (
            "StandardScaler is inside each classifier CV fold; categorical "
            "and text hash features are fixed stateless transforms."
        ),
        "row_order_feature_included": False,
        "classifier_random_seed": int(seed),
        "cross_validation_splits": int(c2st_cfg.get("n_splits", 5)),
        "num_features": int(x.shape[1]) if x.ndim == 2 else 0,
        "feature_names": feature_names,
        "top_features": top_features,
    }, importance


def structured_c2st_feature_manifest(
    table_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the structured C2ST columns from schema roles and types."""

    primary_key = table_cfg.get("primary_key")
    primary_keys = (
        {str(value) for value in primary_key}
        if isinstance(primary_key, (list, tuple, set))
        else ({str(primary_key)} if primary_key else set())
    )
    included: list[str] = []
    numerical: list[str] = []
    categorical: list[str] = []
    excluded: list[dict[str, str]] = []
    columns = table_cfg.get("columns", {}) or {}
    for column, raw_cfg in columns.items():
        cfg = dict(raw_cfg or {})
        col_type = str(cfg.get("type", "categorical")).strip().lower()
        semantic = str(cfg.get("semantic_type", "")).strip().lower()
        role = str(cfg.get("role", "")).strip().lower()
        reason = structured_exclusion_reason(
            str(column), cfg, col_type, semantic, role, primary_keys
        )
        if reason is not None:
            excluded.append({"column": str(column), "reason": reason})
            continue
        included.append(str(column))
        if col_type in {"numerical", "numeric", "number"}:
            numerical.append(str(column))
        else:
            categorical.append(str(column))
    return {
        "metric_name": "structured_c2st_error",
        "policy": "generated numerical + categorical attributes only",
        "included_columns": included,
        "included_numerical_columns": numerical,
        "included_categorical_columns": categorical,
        "excluded_columns": excluded,
        "explicitly_excludes": [
            "text and text embeddings",
            "text/token/character/sequence lengths",
            "primary keys and event IDs",
            "source/destination foreign keys and arbitrary identifiers",
            "timestamps and fixed event-spine features",
        ],
    }


def structured_exclusion_reason(
    column: str,
    cfg: dict[str, Any],
    col_type: str,
    semantic: str,
    role: str,
    primary_keys: set[str],
) -> str | None:
    if column in primary_keys or col_type in {"primary_key", "id"}:
        return "primary key / event identifier"
    if col_type in {"foreign_key", "foreignkey"}:
        return "fixed source/destination foreign key"
    if col_type in {"datetime", "timestamp", "date"}:
        return "fixed event-spine timestamp"
    if col_type in {"text", "string_text"}:
        return "free-form text evaluated separately"
    if role in {
        "condition",
        "conditioning",
        "fixed",
        "source_foreign_key",
        "destination_foreign_key",
        "timestamp",
        "primary_key",
    }:
        return f"fixed/conditioning schema role: {role}"
    if bool(cfg.get("text_derived", False)) or "text_length" in semantic:
        return "text-derived feature"
    lowered = column.lower()
    length_markers = (
        "text_length",
        "review_length",
        "summary_length",
        "token_length",
        "word_count",
        "char_length",
        "character_count",
        "sequence_length",
        "padding_length",
        "eos_position",
        "length_bucket",
    )
    if any(marker in lowered for marker in length_markers):
        return "text-derived length feature"
    if col_type not in {
        "numerical",
        "numeric",
        "number",
        "categorical",
        "ordinal",
        "boolean",
        "bool",
    }:
        return f"unsupported/non-generated structured type: {col_type}"
    if cfg.get("generated") is False:
        return "schema marks field as fixed/non-generated"
    return None


def featurize_real_synthetic(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    table_cfg: dict[str, Any],
    max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    n = min(len(real), len(synthetic), int(max_rows or max(len(real), len(synthetic))))
    real_sample = real.sample(n=n, random_state=seed) if len(real) > n else real.head(n)
    syn_sample = synthetic.sample(n=n, random_state=seed + 1) if len(synthetic) > n else synthetic.head(n)
    combined = pd.concat([real_sample, syn_sample], ignore_index=True)
    features, names = featurize_frame(combined, table_cfg)
    y = np.array([1] * len(real_sample) + [0] * len(syn_sample), dtype=int)
    return features, y, names, int(n)


def featurize_frame(frame: pd.DataFrame, table_cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    pieces: list[np.ndarray] = []
    names: list[str] = []
    primary_key = table_cfg.get("primary_key")
    primary_keys = (
        {str(value) for value in primary_key}
        if isinstance(primary_key, (list, tuple, set))
        else ({str(primary_key)} if primary_key else set())
    )
    for column, cfg in (table_cfg.get("columns", {}) or {}).items():
        if column not in frame:
            continue
        col_type = str((cfg or {}).get("type", "categorical")).lower()
        if column in primary_keys or col_type in {"primary_key", "id"}:
            continue
        if col_type in {"numerical", "numeric", "number"}:
            parsed = numeric_series(frame[column])
            values = parsed.fillna(0.0).to_numpy(dtype=float)
            pieces.append(
                np.column_stack(
                    [values, parsed.isna().to_numpy(dtype=float)]
                )
            )
            names.extend([column, f"{column}_missing"])
        elif col_type == "datetime":
            parsed = datetime_series(frame[column])
            seconds = (
                parsed.to_numpy(dtype="datetime64[ns]")
                .astype("int64")
                .astype(float)
                / 1e9
            )
            seconds = np.where(parsed.isna(), 0.0, seconds)
            month = parsed.dt.month.fillna(0).to_numpy(dtype=float)
            day = parsed.dt.dayofweek.fillna(0).to_numpy(dtype=float)
            extras = np.column_stack(
                [
                    seconds,
                    np.sin(2.0 * np.pi * month / 12.0),
                    np.cos(2.0 * np.pi * month / 12.0),
                    np.sin(2.0 * np.pi * day / 7.0),
                    np.cos(2.0 * np.pi * day / 7.0),
                    parsed.isna().to_numpy(dtype=float),
                ]
            )
            pieces.append(extras)
            names.extend(
                [
                    f"{column}_seconds",
                    f"{column}_month_sin",
                    f"{column}_month_cos",
                    f"{column}_dayofweek_sin",
                    f"{column}_dayofweek_cos",
                    f"{column}_missing",
                ]
            )
        elif col_type == "text":
            text_features = np.column_stack([token_lengths(frame[column]), char_lengths(frame[column])])
            pieces.append(text_features)
            names.extend([f"{column}_token_length", f"{column}_char_length"])
            emb = np.vstack([text_hash_embedding(value, dim=8) for value in frame[column]])
            pieces.append(emb)
            names.extend([f"{column}_hash_emb_{idx}" for idx in range(emb.shape[1])])
        else:
            values = categorical_values_for_c2st(frame[column], col_type, cfg or {})
            num_buckets = int((cfg or {}).get("c2st_hash_buckets", 16))
            buckets = values.astype(str).map(
                lambda value: stable_bucket(value, num_buckets)
            ).to_numpy(dtype=np.int64)
            one_hot = np.eye(num_buckets, dtype=float)[buckets]
            pieces.append(one_hot)
            names.extend(
                [
                    f"{column}_hash_bucket_{idx}"
                    for idx in range(num_buckets)
                ]
            )
    if not pieces:
        return np.zeros((len(frame), 1), dtype=float), ["constant"]
    return np.concatenate(pieces, axis=1), names


def run_binary_classifiers(
    x: np.ndarray,
    y: np.ndarray,
    classifiers: list[str],
    seed: int = 42,
    n_splits: int = 5,
) -> dict[str, dict[str, Any]]:
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return {"fallback_mean_difference": fallback_c2st(x, y)}
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }
    splits = min(int(n_splits), int(np.bincount(y).min()))
    if splits < 2:
        return {}
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    results: dict[str, dict[str, Any]] = {}
    for name in classifiers:
        if name not in models:
            continue
        model = models[name]
        try:
            if hasattr(model, "predict_proba"):
                scores = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
            else:
                scores = cross_val_predict(model, x, y, cv=cv, method="decision_function")
            auc = float(roc_auc_score(y, scores))
            pred = (scores >= 0.5).astype(int)
            acc = float(accuracy_score(y, pred))
        except Exception as exc:
            results[name] = {"status": "failed", "reason": str(exc)}
            continue
        fitted = model.fit(x, y)
        with np.errstate(divide="ignore", invalid="ignore"):
            importances = getattr(fitted, "feature_importances_", None)
        if importances is None and hasattr(fitted, "steps"):
            last = fitted.steps[-1][1]
            importances = getattr(last, "coef_", None)
            if importances is not None:
                importances = np.abs(importances).reshape(-1)
        if importances is not None:
            importances = np.nan_to_num(
                np.asarray(importances, dtype=float).reshape(-1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        results[name] = {
            "auc": auc,
            "accuracy": acc,
            "error": float(abs(auc - 0.5) * 2.0),
            "feature_importances": importances.tolist() if importances is not None else None,
        }
    return results


def fallback_c2st(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    real_mean = x[y == 1].mean(axis=0)
    syn_mean = x[y == 0].mean(axis=0)
    diff = float(np.linalg.norm(real_mean - syn_mean) / max(np.linalg.norm(real_mean), 1e-9))
    err = float(min(diff, 1.0))
    return {"auc": 0.5 + err / 2.0, "accuracy": None, "error": err}


def feature_importance(results: dict[str, dict[str, Any]], feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for classifier, result in results.items():
        values = result.get("feature_importances")
        if values is None:
            continue
        for feature_name, value in zip(feature_names, values):
            importance = float(
                np.nan_to_num(
                    float(value),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
            rows.append(
                {
                    "classifier": classifier,
                    "feature_name": feature_name,
                    "importance": importance,
                    "abs_importance": abs(importance),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["classifier", "feature_name", "importance", "abs_importance", "rank"])
    frame["rank"] = frame.groupby("classifier")["abs_importance"].rank(method="first", ascending=False).astype(int)
    return frame.sort_values(["classifier", "rank"]).reset_index(drop=True)


def top_feature_records(importance: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if importance is None or importance.empty:
        return []
    rows = importance.sort_values("abs_importance", ascending=False).head(int(limit))
    return [
        {
            "classifier": str(row["classifier"]),
            "feature_name": str(row["feature_name"]),
            "importance": float(row["importance"]),
            "abs_importance": float(row["abs_importance"]),
            "rank": int(idx + 1),
        }
        for idx, (_, row) in enumerate(rows.iterrows())
    ]


def categorical_values_for_c2st(series: pd.Series, col_type: str, cfg: dict[str, Any]) -> pd.Series:
    if col_type == "categorical":
        return canonicalize_categorical_series(series, cfg)
    return series.astype(str)


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    return np.nan_to_num((values - mean) / std)


def stable_bucket(value: Any, num_buckets: int) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(num_buckets)
