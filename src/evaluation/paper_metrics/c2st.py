"""Classifier two-sample tests for single event tables."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .utils import char_lengths, datetime_numeric, datetime_series, numeric_series, text_hash_embedding, token_lengths


def single_table_c2st_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    c2st_cfg = ((config.get("evaluation") or {}).get("c2st") or {})
    if not bool(c2st_cfg.get("enabled", True)):
        return {"status": "skipped", "reason": "disabled"}, pd.DataFrame()
    max_rows = c2st_cfg.get("max_rows") or (config.get("evaluation") or {}).get("max_rows_for_c2st", 100000)
    seed = int((config.get("evaluation") or {}).get("random_seed", 42))
    table_cfg = config.get("table") or {}
    classifiers = c2st_cfg.get("classifiers") or ["logistic_regression"]
    x, y, feature_names = featurize_real_synthetic(real, synthetic, table_cfg, max_rows=max_rows, seed=seed)
    results = run_binary_classifiers(x, y, classifiers, seed=seed)
    best_name = max(results, key=lambda name: results[name].get("auc", 0.5)) if results else None
    best = results.get(best_name, {}) if best_name else {}
    importance = feature_importance(results, feature_names)
    return {
        "auc": best.get("auc"),
        "accuracy": best.get("accuracy"),
        "error": best.get("error"),
        "best_classifier": best_name,
        "per_classifier": results,
        "num_rows": int(len(y)),
        "num_features": int(x.shape[1]) if x.ndim == 2 else 0,
    }, importance


def featurize_real_synthetic(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    table_cfg: dict[str, Any],
    max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n = min(len(real), len(synthetic), int(max_rows or max(len(real), len(synthetic))))
    real_sample = real.sample(n=n, random_state=seed) if len(real) > n else real.head(n)
    syn_sample = synthetic.sample(n=n, random_state=seed + 1) if len(synthetic) > n else synthetic.head(n)
    combined = pd.concat([real_sample, syn_sample], ignore_index=True)
    features, names = featurize_frame(combined, table_cfg)
    y = np.array([1] * len(real_sample) + [0] * len(syn_sample), dtype=int)
    return features, y, names


def featurize_frame(frame: pd.DataFrame, table_cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    pieces: list[np.ndarray] = []
    names: list[str] = []
    for column, cfg in (table_cfg.get("columns", {}) or {}).items():
        if column not in frame:
            continue
        col_type = str((cfg or {}).get("type", "categorical")).lower()
        if col_type in {"numerical", "numeric", "number"}:
            values = numeric_series(frame[column]).fillna(0.0).to_numpy(dtype=float)[:, None]
            pieces.append(standardize(values))
            names.append(column)
        elif col_type == "datetime":
            parsed = datetime_series(frame[column])
            values = datetime_numeric(frame[column]).fillna(0.0)
            extras = np.column_stack(
                [
                    values.to_numpy(dtype=float),
                    parsed.dt.month.fillna(0).to_numpy(dtype=float),
                    parsed.dt.dayofweek.fillna(0).to_numpy(dtype=float),
                ]
            )
            pieces.append(standardize(extras))
            names.extend([f"{column}_timestamp", f"{column}_month", f"{column}_dayofweek"])
        elif col_type == "text":
            text_features = np.column_stack([token_lengths(frame[column]), char_lengths(frame[column])])
            pieces.append(standardize(text_features))
            names.extend([f"{column}_token_length", f"{column}_char_length"])
            emb = np.vstack([text_hash_embedding(value, dim=8) for value in frame[column]])
            pieces.append(emb)
            names.extend([f"{column}_hash_emb_{idx}" for idx in range(emb.shape[1])])
        else:
            freq = frame[column].astype(str).map(frame[column].astype(str).value_counts(normalize=True)).fillna(0.0)
            buckets = frame[column].astype(str).map(lambda value: stable_bucket(value, 16)).to_numpy(dtype=float)
            pieces.append(standardize(np.column_stack([freq.to_numpy(dtype=float), buckets])))
            names.extend([f"{column}_frequency", f"{column}_hash_bucket"])
    if not pieces:
        return np.zeros((len(frame), 1), dtype=float), ["constant"]
    return np.concatenate(pieces, axis=1), names


def run_binary_classifiers(x: np.ndarray, y: np.ndarray, classifiers: list[str], seed: int = 42) -> dict[str, dict[str, Any]]:
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
    splits = min(5, int(np.bincount(y).min()))
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
        importances = getattr(fitted, "feature_importances_", None)
        if importances is None and hasattr(fitted, "steps"):
            last = fitted.steps[-1][1]
            importances = getattr(last, "coef_", None)
            if importances is not None:
                importances = np.abs(importances).reshape(-1)
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
    for model, result in results.items():
        values = result.get("feature_importances")
        if values is None:
            continue
        for name, value in zip(feature_names, values):
            rows.append({"model": model, "feature": name, "importance": float(value)})
    return pd.DataFrame(rows)


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    return np.nan_to_num((values - mean) / std)


def stable_bucket(value: Any, num_buckets: int) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(num_buckets)
