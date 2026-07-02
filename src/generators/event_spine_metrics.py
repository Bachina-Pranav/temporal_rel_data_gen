"""Metrics for temporal event-spine generators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .event_affinity import day_index
from .joint_temporal_2k_sbm_event import load_blocks
from .temporal_activity_models import canonical_day_bucket


def evaluate_event_spine(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    structure_debug_dir: Optional[str | Path] = None,
    customer_col: str = "customer_id",
    product_col: str = "product_id",
    timestamp_col: str = "review_time",
    compute_c2st: bool = False,
    coactive_margin_days: int = 30,
) -> Dict[str, Any]:
    real = canonicalize(real, customer_col, product_col, timestamp_col)
    synthetic = canonicalize(synthetic, customer_col, product_col, timestamp_col)
    root = Path(structure_debug_dir) if structure_debug_dir else None
    customer_blocks = load_blocks(root, "customer_blocks.csv", [customer_col, "id", "customer_id"], ["customer_block", "block"])
    product_blocks = load_blocks(root, "product_blocks.csv", [product_col, "id", "product_id"], ["product_block", "block"])
    metrics: Dict[str, Any] = {
        "num_reviews_real": int(len(real)),
        "num_reviews_synthetic": int(len(synthetic)),
        "active_customers_real": int(real[customer_col].nunique()),
        "active_customers_synthetic": int(synthetic[customer_col].nunique()),
        "active_products_real": int(real[product_col].nunique()),
        "active_products_synthetic": int(synthetic[product_col].nunique()),
    }
    metrics.update(degree_metrics(real, synthetic, customer_col, product_col))
    metrics.update(overlap_metrics(real, synthetic, customer_col, product_col, "_time_bucket"))
    metrics.update(time_count_metrics(real, synthetic, "_time_bucket"))
    metrics.update(block_metrics(real, synthetic, customer_col, product_col, "_time_bucket", customer_blocks, product_blocks))
    metrics.update(lifecycle_metrics(real, synthetic, product_col, "_time_bucket", prefix="product"))
    metrics.update(lifecycle_metrics(real, synthetic, customer_col, "_time_bucket", prefix="customer"))
    metrics.update(coactivity_metrics(real, synthetic, customer_col, product_col, "_time_bucket", coactive_margin_days))
    metrics.update(relative_age_metrics(real, synthetic, product_col, "_time_bucket", prefix="product"))
    metrics.update(relative_age_metrics(real, synthetic, customer_col, "_time_bucket", prefix="customer"))
    metrics["event_tuple_c2st_accuracy"] = (
        event_tuple_c2st(real, synthetic, customer_col, product_col, "_time_bucket", customer_blocks, product_blocks)
        if compute_c2st
        else None
    )
    return metrics


def canonicalize(frame: pd.DataFrame, customer_col: str, product_col: str, timestamp_col: str) -> pd.DataFrame:
    out = frame[[customer_col, product_col, timestamp_col]].copy()
    out["_time_bucket"] = canonical_day_bucket(out[timestamp_col])
    return out


def degree_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, customer_col: str, product_col: str) -> Dict[str, float]:
    real_c = real[customer_col].value_counts()
    syn_c = synthetic[customer_col].value_counts()
    real_p = real[product_col].value_counts()
    syn_p = synthetic[product_col].value_counts()
    return {
        "customer_degree_ks": ks_stat(real_c.to_numpy(dtype=float), syn_c.reindex(real_c.index, fill_value=0).to_numpy(dtype=float)),
        "product_degree_ks": ks_stat(real_p.to_numpy(dtype=float), syn_p.reindex(real_p.index, fill_value=0).to_numpy(dtype=float)),
    }


def overlap_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, customer_col: str, product_col: str, time_col: str) -> Dict[str, float]:
    real_pairs = set(zip(real[customer_col], real[product_col]))
    syn_pairs = list(zip(synthetic[customer_col], synthetic[product_col]))
    real_events = set(zip(real[customer_col], real[product_col], real[time_col]))
    syn_events = list(zip(synthetic[customer_col], synthetic[product_col], synthetic[time_col]))
    pair_counts = pd.Series(syn_pairs).value_counts()
    duplicate_rows = int(pair_counts[pair_counts > 1].sum()) if len(pair_counts) else 0
    return {
        "duplicate_customer_product_rate": float(duplicate_rows / max(len(syn_pairs), 1)),
        "real_edge_overlap_rate": float(sum(pair in real_pairs for pair in syn_pairs) / max(len(syn_pairs), 1)),
        "exact_event_overlap_rate": float(sum(event in real_events for event in syn_events) / max(len(syn_events), 1)),
    }


def time_count_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, time_col: str) -> Dict[str, float]:
    real_daily = real[time_col].value_counts().sort_index()
    syn_daily = synthetic[time_col].value_counts().sort_index()
    days = sorted(set(real_daily.index).union(set(syn_daily.index)))
    real_arr = real_daily.reindex(days, fill_value=0).to_numpy(dtype=float)
    syn_arr = syn_daily.reindex(days, fill_value=0).to_numpy(dtype=float)
    real_month = month_counts(real, time_col)
    syn_month = month_counts(synthetic, time_col)
    months = sorted(set(real_month.index).union(set(syn_month.index)))
    return {
        "daily_count_corr": safe_corr(real_arr, syn_arr),
        "daily_count_l1": float(np.abs(real_arr - syn_arr).sum() / max(real_arr.sum(), 1.0)),
        "monthly_count_corr": safe_corr(
            real_month.reindex(months, fill_value=0).to_numpy(dtype=float),
            syn_month.reindex(months, fill_value=0).to_numpy(dtype=float),
        ),
    }


def block_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    time_col: str,
    customer_blocks: Mapping[Any, int],
    product_blocks: Mapping[Any, int],
) -> Dict[str, float]:
    if not customer_blocks or not product_blocks:
        return {
            "block_pair_count_exact_match_rate": None,
            "block_pair_time_count_l1": None,
        }
    real_keys = block_key_frame(real, customer_col, product_col, time_col, customer_blocks, product_blocks)
    syn_keys = block_key_frame(synthetic, customer_col, product_col, time_col, customer_blocks, product_blocks)
    real_pair = real_keys.groupby(["customer_block", "product_block"]).size()
    syn_pair = syn_keys.groupby(["customer_block", "product_block"]).size()
    pairs = sorted(set(real_pair.index).union(set(syn_pair.index)))
    exact = [real_pair.get(pair, 0) == syn_pair.get(pair, 0) for pair in pairs]
    real_bpt = real_keys.groupby(["customer_block", "product_block", time_col]).size()
    syn_bpt = syn_keys.groupby(["customer_block", "product_block", time_col]).size()
    cells = sorted(set(real_bpt.index).union(set(syn_bpt.index)))
    l1 = sum(abs(real_bpt.get(cell, 0) - syn_bpt.get(cell, 0)) for cell in cells) / max(len(real), 1)
    return {
        "block_pair_count_exact_match_rate": float(sum(exact) / max(len(exact), 1)),
        "block_pair_time_count_l1": float(l1),
    }


def lifecycle_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, entity_col: str, time_col: str, prefix: str) -> Dict[str, float]:
    real_life = lifecycle_table(real, entity_col, time_col)
    syn_life = lifecycle_table(synthetic, entity_col, time_col)
    entities = sorted(set(real_life.index).intersection(set(syn_life.index)))
    output = {}
    for col, metric_name in [
        ("first", "first_time_corr"),
        ("last", "last_time_corr"),
        ("peak", "peak_time_corr"),
    ]:
        output[f"{prefix}_{metric_name}"] = safe_corr(
            real_life.reindex(entities)[col].to_numpy(dtype=float),
            syn_life.reindex(entities)[col].to_numpy(dtype=float),
        )
    output[f"{prefix}_active_span_ks"] = ks_stat(
        real_life["span"].to_numpy(dtype=float),
        syn_life["span"].to_numpy(dtype=float),
    )
    output[f"{prefix}_activity_entropy_ks"] = ks_stat(
        real_life["entropy"].to_numpy(dtype=float),
        syn_life["entropy"].to_numpy(dtype=float),
    )
    output[f"{prefix}_time_activity_distribution_ks"] = ks_stat(
        entity_time_distribution_values(real, entity_col, time_col),
        entity_time_distribution_values(synthetic, entity_col, time_col),
    )
    return output


def coactivity_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, customer_col: str, product_col: str, time_col: str, margin_days: int) -> Dict[str, float]:
    customer_windows = windows(real, customer_col, time_col)
    product_windows = windows(real, product_col, time_col)
    customer_ok = []
    product_ok = []
    for _, row in synthetic.iterrows():
        t = day_index(row[time_col])
        cw = customer_windows.get(row[customer_col])
        pw = product_windows.get(row[product_col])
        customer_ok.append(bool(cw and cw[0] - margin_days <= t <= cw[1] + margin_days))
        product_ok.append(bool(pw and pw[0] - margin_days <= t <= pw[1] + margin_days))
    joint = [c and p for c, p in zip(customer_ok, product_ok)]
    return {
        "customer_active_window_rate": float(sum(customer_ok) / max(len(customer_ok), 1)),
        "product_active_window_rate": float(sum(product_ok) / max(len(product_ok), 1)),
        "joint_coactive_window_rate": float(sum(joint) / max(len(joint), 1)),
    }


def relative_age_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, entity_col: str, time_col: str, prefix: str) -> Dict[str, float]:
    real_age = relative_ages(real, entity_col, time_col, windows(real, entity_col, time_col))
    syn_age = relative_ages(synthetic, entity_col, time_col, windows(real, entity_col, time_col))
    return {
        f"{prefix}_relative_age_ks": ks_stat(real_age, syn_age),
        f"{prefix}_relative_age_mean_diff": float(np.mean(syn_age) - np.mean(real_age)) if len(real_age) and len(syn_age) else 0.0,
    }


def event_tuple_c2st(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    time_col: str,
    customer_blocks: Mapping[Any, int],
    product_blocks: Mapping[Any, int],
) -> Optional[float]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        data = pd.concat([real.assign(_label=0), synthetic.assign(_label=1)], ignore_index=True)
        customer_degree = real[customer_col].value_counts()
        product_degree = real[product_col].value_counts()
        customer_windows = windows(real, customer_col, time_col)
        product_windows = windows(real, product_col, time_col)
        features = pd.DataFrame(
            {
                "customer_degree": data[customer_col].map(customer_degree).fillna(0).astype(float),
                "product_degree": data[product_col].map(product_degree).fillna(0).astype(float),
                "customer_block": data[customer_col].map(customer_blocks).fillna(0).astype(float),
                "product_block": data[product_col].map(product_blocks).fillna(0).astype(float),
                "day_index": data[time_col].map(day_index).astype(float),
                "customer_relative_age": relative_ages(data, customer_col, time_col, customer_windows),
                "product_relative_age": relative_ages(data, product_col, time_col, product_windows),
            }
        )
        labels = data["_label"].to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(features.to_numpy(), labels, test_size=0.3, random_state=42, stratify=labels)
        clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        clf.fit(x_train, y_train)
        return float(clf.score(x_test, y_test))
    except Exception:
        return None


def lifecycle_table(frame: pd.DataFrame, entity_col: str, time_col: str) -> pd.DataFrame:
    rows = []
    for entity, group in frame.groupby(entity_col):
        counts = group[time_col].value_counts()
        first = min(counts.index)
        last = max(counts.index)
        peak = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        probs = counts.to_numpy(dtype=float) / max(float(counts.sum()), 1.0)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        rows.append(
            {
                "entity": entity,
                "first": day_index(first),
                "last": day_index(last),
                "peak": day_index(peak),
                "span": max(day_index(last) - day_index(first), 0),
                "entropy": entropy,
            }
        )
    return pd.DataFrame(rows).set_index("entity") if rows else pd.DataFrame(columns=["first", "last", "peak", "span", "entropy"])


def entity_time_distribution_values(frame: pd.DataFrame, entity_col: str, time_col: str) -> np.ndarray:
    values = []
    for _, group in frame.groupby(entity_col):
        counts = group[time_col].value_counts().to_numpy(dtype=float)
        probs = counts / max(float(counts.sum()), 1.0)
        values.extend(probs.tolist())
    return np.asarray(values, dtype=float)


def windows(frame: pd.DataFrame, entity_col: str, time_col: str) -> Dict[Any, tuple[int, int]]:
    output = {}
    for entity, group in frame.groupby(entity_col):
        days = group[time_col].map(day_index)
        output[entity] = (int(days.min()), int(days.max()))
    return output


def relative_ages(frame: pd.DataFrame, entity_col: str, time_col: str, real_windows: Mapping[Any, tuple[int, int]]) -> np.ndarray:
    ages = []
    for _, row in frame.iterrows():
        window = real_windows.get(row[entity_col])
        if not window:
            ages.append(0.0)
            continue
        first, last = window
        span = max(last - first, 1)
        ages.append((day_index(row[time_col]) - first) / span)
    return np.asarray(ages, dtype=float)


def month_counts(frame: pd.DataFrame, time_col: str) -> pd.Series:
    months = pd.to_datetime(frame[time_col], errors="coerce").dt.to_period("M").astype(str)
    return months.value_counts().sort_index()


def block_key_frame(frame: pd.DataFrame, customer_col: str, product_col: str, time_col: str, customer_blocks: Mapping[Any, int], product_blocks: Mapping[Any, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_block": frame[customer_col].map(customer_blocks).fillna(-1).astype(int),
            "product_block": frame[product_col].map(product_blocks).fillna(-1).astype(int),
            time_col: frame[time_col],
        }
    )


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def write_metrics(metrics: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
