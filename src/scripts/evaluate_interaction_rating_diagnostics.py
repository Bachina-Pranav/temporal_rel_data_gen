#!/usr/bin/env python3
"""Leakage-safe rating diagnostics for single interaction-table generators."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.constrained import normalize_rating_value  # noqa: E402
from attribute_generation.conditional_tabdlm.evaluate import rating_domain_from_config, rating_validity_details  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import ks_statistic, safe_corr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MovieLens-style generated ratings on fixed rows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-table", required=True)
    parser.add_argument("--synthetic-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--source-col", default=None)
    parser.add_argument("--destination-col", default=None)
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c2st", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    schema = config.schema
    source_col = args.source_col or schema.foreign_key_columns[0]
    destination_col = args.destination_col or schema.foreign_key_columns[1]
    timestamp_col = args.timestamp_col or schema.datetime_columns[0]
    real = pd.read_csv(args.real_table)
    synthetic = pd.read_csv(args.synthetic_table)
    if args.sample_size is not None:
        real = real.sample(n=min(int(args.sample_size), len(real)), random_state=int(args.seed)).reset_index(drop=True)
        synthetic = synthetic.sample(n=min(int(args.sample_size), len(synthetic)), random_state=int(args.seed) + 1).reset_index(drop=True)
    domain = rating_domain_from_config(config, real)
    metrics = evaluate_rating_diagnostics(
        real,
        synthetic,
        rating_col=args.rating_col,
        source_col=source_col,
        destination_col=destination_col,
        timestamp_col=timestamp_col,
        domain=domain,
        compute_c2st=bool(args.c2st),
        seed=int(args.seed),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rating_diagnostics.json").write_text(json.dumps(jsonable(metrics), indent=2, sort_keys=True) + "\n")
    pd.DataFrame([flatten(metrics)]).to_csv(output_dir / "rating_diagnostics_flat.csv", index=False)
    print(output_dir / "rating_diagnostics.json")


def evaluate_rating_diagnostics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    rating_col: str,
    source_col: str,
    destination_col: str,
    timestamp_col: str,
    domain: list[int | float],
    compute_c2st: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    real = real.copy()
    synthetic = synthetic.copy()
    real["_rating_norm"] = normalize_rating_column(real[rating_col], domain)
    synthetic["_rating_norm"] = normalize_rating_column(synthetic[rating_col], domain)
    return {
        "row_counts": {"real": int(len(real)), "synthetic": int(len(synthetic)), "match": bool(len(real) == len(synthetic))},
        "rating_domain": list(domain),
        "validity": {
            "real": rating_validity_details(real[rating_col], domain),
            "synthetic": rating_validity_details(synthetic[rating_col], domain),
        },
        "rating_marginal": marginal_metrics(real["_rating_norm"], synthetic["_rating_norm"], domain),
        "user_conditional": entity_metrics(real, synthetic, source_col, "_rating_norm", domain, label="user"),
        "movie_conditional": entity_metrics(real, synthetic, destination_col, "_rating_norm", domain, label="movie"),
        "temporal": temporal_metrics(real, synthetic, timestamp_col, "_rating_norm", domain),
        "c2st": c2st_metrics(real, synthetic, source_col, destination_col, timestamp_col, "_rating_norm", seed=seed) if compute_c2st else {"status": "skipped"},
    }


def normalize_rating_column(series: pd.Series, domain: list[int | float]) -> pd.Series:
    valid = set(domain)
    return series.map(lambda value: normalize_rating_value(value)).map(lambda value: value if value in valid else np.nan)


def marginal_metrics(real: pd.Series, synthetic: pd.Series, domain: list[int | float]) -> dict[str, Any]:
    r = real.dropna().astype(float)
    s = synthetic.dropna().astype(float)
    real_probs = probabilities(real, domain)
    syn_probs = probabilities(synthetic, domain)
    freq = {
        str(value): {
            "real": float(real_probs[idx]),
            "synthetic": float(syn_probs[idx]),
            "diff": float(syn_probs[idx] - real_probs[idx]),
        }
        for idx, value in enumerate(domain)
    }
    return {
        "total_variation": float(0.5 * np.abs(real_probs - syn_probs).sum()),
        "l1": float(np.abs(real_probs - syn_probs).sum()),
        "js": js_from_probs(real_probs, syn_probs),
        "ks": ks_statistic(r, s),
        "ordinal_wasserstein": wasserstein_1d(r, s),
        "mean_real": float(r.mean()) if len(r) else None,
        "mean_synthetic": float(s.mean()) if len(s) else None,
        "mean_diff": float(s.mean() - r.mean()) if len(r) and len(s) else None,
        "variance_real": float(r.var(ddof=0)) if len(r) else None,
        "variance_synthetic": float(s.var(ddof=0)) if len(s) else None,
        "variance_diff": float(s.var(ddof=0) - r.var(ddof=0)) if len(r) and len(s) else None,
        "per_rating_frequency": freq,
    }


def entity_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity_col: str,
    rating_col: str,
    domain: list[int | float],
    *,
    label: str,
) -> dict[str, Any]:
    if entity_col not in real or entity_col not in synthetic:
        return {"status": "missing_entity_column", "entity_col": entity_col}
    real_group = real.dropna(subset=[rating_col]).groupby(real[entity_col].astype(str), sort=False)[rating_col]
    syn_group = synthetic.dropna(subset=[rating_col]).groupby(synthetic[entity_col].astype(str), sort=False)[rating_col]
    real_mean = real_group.mean()
    syn_mean = syn_group.mean()
    real_count = real_group.size()
    common = real_mean.index.intersection(syn_mean.index)
    diff = (real_mean.reindex(common) - syn_mean.reindex(common)).abs()
    weights = real_count.reindex(common).fillna(0).astype(float)
    top = real_count.sort_values(ascending=False).head(1000).index.intersection(common)
    tvs = []
    entropy_diffs = []
    for key in common:
        rp = probabilities(real.loc[real[entity_col].astype(str) == key, rating_col], domain)
        sp = probabilities(synthetic.loc[synthetic[entity_col].astype(str) == key, rating_col], domain)
        tvs.append(0.5 * np.abs(rp - sp).sum())
        entropy_diffs.append(abs(entropy(rp) - entropy(sp)))
    return {
        "entity_col": entity_col,
        "num_real_entities": int(real_mean.shape[0]),
        "num_synthetic_entities": int(syn_mean.shape[0]),
        "num_common_entities": int(len(common)),
        f"{label}_mean_rating_mae_unweighted": float(diff.mean()) if len(diff) else None,
        f"{label}_mean_rating_mae_weighted": float(np.average(diff, weights=weights)) if len(diff) and float(weights.sum()) > 0 else None,
        f"{label}_top_1000_mean_rating_mae": float((real_mean.reindex(top) - syn_mean.reindex(top)).abs().mean()) if len(top) else None,
        f"{label}_rating_distribution_tv_mean": float(np.mean(tvs)) if tvs else None,
        f"{label}_rating_entropy_abs_diff_mean": float(np.mean(entropy_diffs)) if entropy_diffs else None,
        "activity_buckets": activity_bucket_metrics(real_mean, syn_mean, real_count),
    }


def activity_bucket_metrics(real_mean: pd.Series, syn_mean: pd.Series, real_count: pd.Series) -> dict[str, Any]:
    buckets = {
        "1": real_count[real_count == 1].index,
        "2_4": real_count[(real_count >= 2) & (real_count <= 4)].index,
        "5_9": real_count[(real_count >= 5) & (real_count <= 9)].index,
        "10_plus": real_count[real_count >= 10].index,
    }
    out = {}
    for name, idx in buckets.items():
        common = idx.intersection(syn_mean.index)
        diff = (real_mean.reindex(common) - syn_mean.reindex(common)).abs()
        out[name] = {"num_entities": int(len(common)), "mean_rating_mae": float(diff.mean()) if len(diff) else None}
    return out


def temporal_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, timestamp_col: str, rating_col: str, domain: list[int | float]) -> dict[str, Any]:
    out = {}
    for freq, label in [("M", "monthly"), ("Q", "quarterly")]:
        real_binned = binned_rating(real, timestamp_col, rating_col, freq)
        syn_binned = binned_rating(synthetic, timestamp_col, rating_col, freq)
        idx = real_binned.index.union(syn_binned.index)
        r = real_binned.reindex(idx)
        s = syn_binned.reindex(idx)
        out[f"{label}_mean_rating_corr"] = safe_corr(r.to_numpy(dtype=float), s.to_numpy(dtype=float))
        out[f"{label}_mean_rating_mae"] = float(np.nanmean(np.abs(r - s))) if len(idx) else None
        out[f"{label}_num_bins"] = int(len(idx))
        if len(idx) < 3:
            out[f"{label}_warning"] = "too_few_bins_for_stable_correlation"
        out[f"{label}_rating_distribution_tv_mean"] = time_distribution_tv(real, synthetic, timestamp_col, rating_col, domain, freq)
    out["global_timestamp_rating_corr_real"] = timestamp_rating_corr(real, timestamp_col, rating_col)
    out["global_timestamp_rating_corr_synthetic"] = timestamp_rating_corr(synthetic, timestamp_col, rating_col)
    if out["global_timestamp_rating_corr_real"] is not None and out["global_timestamp_rating_corr_synthetic"] is not None:
        out["global_timestamp_rating_trend_error"] = abs(out["global_timestamp_rating_corr_real"] - out["global_timestamp_rating_corr_synthetic"])
    else:
        out["global_timestamp_rating_trend_error"] = None
    return out


def c2st_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, source_col: str, destination_col: str, timestamp_col: str, rating_col: str, *, seed: int) -> dict[str, Any]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"status": "skipped", "reason": f"sklearn unavailable: {exc}"}
    left = c2st_frame(real, source_col, destination_col, timestamp_col, rating_col)
    right = c2st_frame(synthetic, source_col, destination_col, timestamp_col, rating_col)
    n = min(len(left), len(right), 100_000)
    left = left.sample(n=n, random_state=seed) if len(left) > n else left
    right = right.sample(n=n, random_state=seed + 1) if len(right) > n else right
    data = pd.concat([left, right], ignore_index=True)
    y = np.asarray([0] * len(left) + [1] * len(right), dtype=int)
    categorical = [source_col, destination_col]
    numeric = ["timestamp_numeric", rating_col]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    clf = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=500, random_state=seed))])
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=seed, stratify=y)
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    pred = probs >= 0.5
    return {
        "status": "ok",
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, probs)),
        "error": float(abs(roc_auc_score(y_test, probs) - 0.5) * 2.0),
        "num_rows_per_class": int(n),
    }


def c2st_frame(frame: pd.DataFrame, source_col: str, destination_col: str, timestamp_col: str, rating_col: str) -> pd.DataFrame:
    out = frame[[source_col, destination_col, timestamp_col, rating_col]].copy()
    out[source_col] = out[source_col].astype(str)
    out[destination_col] = out[destination_col].astype(str)
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    out["timestamp_numeric"] = ts.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
    out[rating_col] = pd.to_numeric(out[rating_col], errors="coerce")
    return out.dropna()


def probabilities(series: pd.Series, domain: list[int | float]) -> np.ndarray:
    counts = series.dropna().map(normalize_rating_value).value_counts(normalize=True)
    return counts.reindex(domain, fill_value=0.0).to_numpy(dtype=float)


def js_from_probs(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))


def wasserstein_1d(left: pd.Series, right: pd.Series) -> float | None:
    a = left.dropna().astype(float).to_numpy()
    b = right.dropna().astype(float).to_numpy()
    if len(a) == 0 or len(b) == 0:
        return None
    q = np.linspace(0.0, 1.0, max(len(a), len(b)))
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def binned_rating(frame: pd.DataFrame, timestamp_col: str, rating_col: str, freq: str) -> pd.Series:
    tmp = frame[[timestamp_col, rating_col]].dropna().copy()
    tmp["_bin"] = pd.to_datetime(tmp[timestamp_col], errors="coerce").dt.to_period(freq)
    tmp = tmp.dropna(subset=["_bin"])
    return tmp.groupby("_bin")[rating_col].mean()


def time_distribution_tv(real: pd.DataFrame, synthetic: pd.DataFrame, timestamp_col: str, rating_col: str, domain: list[int | float], freq: str) -> float | None:
    real_tmp = real[[timestamp_col, rating_col]].dropna().copy()
    syn_tmp = synthetic[[timestamp_col, rating_col]].dropna().copy()
    real_tmp["_bin"] = pd.to_datetime(real_tmp[timestamp_col], errors="coerce").dt.to_period(freq)
    syn_tmp["_bin"] = pd.to_datetime(syn_tmp[timestamp_col], errors="coerce").dt.to_period(freq)
    bins = real_tmp["_bin"].dropna().unique()
    values = []
    for bucket in bins:
        rp = probabilities(real_tmp.loc[real_tmp["_bin"] == bucket, rating_col], domain)
        sp = probabilities(syn_tmp.loc[syn_tmp["_bin"] == bucket, rating_col], domain)
        values.append(0.5 * np.abs(rp - sp).sum())
    return float(np.mean(values)) if values else None


def timestamp_rating_corr(frame: pd.DataFrame, timestamp_col: str, rating_col: str) -> float | None:
    tmp = frame[[timestamp_col, rating_col]].dropna().copy()
    timestamps = pd.to_datetime(tmp[timestamp_col], errors="coerce")
    numeric_time = pd.Series(timestamps.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float), index=tmp.index)
    return safe_corr(numeric_time, pd.to_numeric(tmp[rating_col], errors="coerce"))


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
    return out


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
