#!/usr/bin/env python3
"""Evaluate generated non-text review attributes."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.generation.block_diagnostics import load_block_maps_from_debug_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal non-text attrs.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--cat-cols", nargs="+", default=["rating", "verified"])
    parser.add_argument("--num-cols", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = load_reviews(args.real_reviews, args.timestamp_col)
    synthetic = load_reviews(args.synthetic_reviews, args.timestamp_col)
    num_cols = [col for col in args.num_cols if col in real.columns and col in synthetic.columns]
    for col in args.num_cols:
        if col not in num_cols:
            print(f"Warning: numerical column {col!r} missing; skipping.")
    customer_blocks = product_blocks = None
    if args.structure_debug_dir:
        customer_blocks, product_blocks, _, _ = load_block_maps_from_debug_dir(
            args.structure_debug_dir, args.customer_id_col, args.product_id_col
        )
    metrics = evaluate_nontext_attrs(
        real,
        synthetic,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        cat_cols=args.cat_cols,
        num_cols=num_cols,
        customer_blocks=customer_blocks,
        product_blocks=product_blocks,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    pd.DataFrame([flatten(metrics)]).to_csv(output_path.with_suffix(".csv"), index=False)
    print(json.dumps(metrics, indent=2))


def load_reviews(path: str | Path, timestamp_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    return df.dropna(subset=[timestamp_col]).copy()


def evaluate_nontext_attrs(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    cat_cols: List[str],
    num_cols: List[str],
    customer_blocks: Optional[Dict[Any, int]] = None,
    product_blocks: Optional[Dict[Any, int]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "categorical": {},
        "temporal": {},
        "relational": {},
        "block": {},
        "entity_distribution": {},
        "numerical": {},
        "c2st": {},
    }
    for col in cat_cols:
        if col not in real.columns or col not in synthetic.columns:
            continue
        metrics["categorical"][f"{col}_distribution_tv"] = total_variation(real[col], synthetic[col])
        metrics["categorical"][f"{col}_distribution_js"] = js_divergence(real[col], synthetic[col])

    rating_col = "rating" if "rating" in cat_cols else cat_cols[0]
    verified_col = "verified" if "verified" in cat_cols else (cat_cols[1] if len(cat_cols) > 1 else cat_cols[0])
    if rating_col in real.columns and rating_col in synthetic.columns:
        metrics["temporal"]["rating_by_month_correlation"] = grouped_rate_corr(real, synthetic, timestamp_col, rating_col, "M")
        metrics["temporal"]["monthly_average_rating_correlation"] = grouped_rate_corr(real, synthetic, timestamp_col, rating_col, "M")
        metrics["temporal"]["daily_average_rating_correlation"] = grouped_rate_corr(real, synthetic, timestamp_col, rating_col, "D", min_periods=7)
        metrics["relational"]["product_average_rating_correlation"] = entity_mean_corr(real, synthetic, product_col, rating_col)
        metrics["relational"]["customer_average_rating_correlation"] = entity_mean_corr(real, synthetic, customer_col, rating_col)
        metrics["relational"]["product_rating_trajectory_correlation_top_products"] = trajectory_corr_top_entities(real, synthetic, product_col, timestamp_col, rating_col)
        metrics["relational"]["customer_rating_trajectory_correlation_active_customers"] = trajectory_corr_top_entities(real, synthetic, customer_col, timestamp_col, rating_col)
        metrics["relational"]["rating_vs_product_degree_correlation_real"] = rating_vs_degree_corr(real, product_col, rating_col)
        metrics["relational"]["rating_vs_product_degree_correlation_synthetic"] = rating_vs_degree_corr(synthetic, product_col, rating_col)
        metrics["relational"]["rating_vs_customer_degree_correlation_real"] = rating_vs_degree_corr(real, customer_col, rating_col)
        metrics["relational"]["rating_vs_customer_degree_correlation_synthetic"] = rating_vs_degree_corr(synthetic, customer_col, rating_col)
        metrics["entity_distribution"].update(
            entity_average_distribution_metrics(
                real,
                synthetic,
                product_col,
                customer_col,
                rating_col,
                prefix="avg_rating",
            )
        )
    if verified_col in real.columns and verified_col in synthetic.columns:
        real_v = normalize_binary(real[verified_col])
        syn_v = normalize_binary(synthetic[verified_col])
        metrics["temporal"]["verified_by_month_correlation"] = grouped_series_corr(real, synthetic, timestamp_col, real_v, syn_v, "M")
        metrics["temporal"]["monthly_verified_rate_correlation"] = grouped_series_corr(real, synthetic, timestamp_col, real_v, syn_v, "M")
        metrics["temporal"]["daily_verified_rate_correlation"] = grouped_series_corr(real, synthetic, timestamp_col, real_v, syn_v, "D", min_periods=7)
        metrics["relational"]["product_verified_rate_correlation"] = entity_series_corr(real, synthetic, product_col, real_v, syn_v)
        metrics["relational"]["customer_verified_rate_correlation"] = entity_series_corr(real, synthetic, customer_col, real_v, syn_v)
        metrics["entity_distribution"].update(
            entity_average_distribution_metrics(
                real.assign(_verified_value=real_v),
                synthetic.assign(_verified_value=syn_v),
                product_col,
                customer_col,
                "_verified_value",
                prefix="verified_rate",
            )
        )

    if customer_blocks is not None and product_blocks is not None:
        metrics["block"].update(block_metrics(real, synthetic, customer_col, product_col, rating_col, verified_col, customer_blocks, product_blocks))

    for col in num_cols:
        metrics["numerical"][col] = numerical_metrics(real, synthetic, customer_col, product_col, timestamp_col, col)
    metrics["c2st"]["c2st_accuracy"] = c2st_accuracy(real, synthetic, customer_col, product_col, timestamp_col, cat_cols, num_cols)
    return metrics


def total_variation(real: pd.Series, synthetic: pd.Series) -> float:
    real_p, syn_p = aligned_probs(real, synthetic)
    return float(0.5 * np.abs(real_p - syn_p).sum())


def js_divergence(real: pd.Series, synthetic: pd.Series) -> float:
    real_p, syn_p = aligned_probs(real, synthetic)
    m = 0.5 * (real_p + syn_p)
    return float(0.5 * kl(real_p, m) + 0.5 * kl(syn_p, m))


def aligned_probs(real: pd.Series, synthetic: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    real_counts = real.value_counts(normalize=True)
    syn_counts = synthetic.value_counts(normalize=True)
    index = real_counts.index.union(syn_counts.index)
    return (
        real_counts.reindex(index, fill_value=0).to_numpy(dtype=float),
        syn_counts.reindex(index, fill_value=0).to_numpy(dtype=float),
    )


def kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float((p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))).sum())


def corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2 or len(b) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def grouped_rate_corr(real, synthetic, timestamp_col, value_col, freq, min_periods=2):
    real_s = real.set_index(timestamp_col)[value_col].astype(float).resample(freq).mean()
    syn_s = synthetic.set_index(timestamp_col)[value_col].astype(float).resample(freq).mean()
    index = real_s.index.union(syn_s.index)
    real_v = real_s.reindex(index).dropna()
    syn_v = syn_s.reindex(index).dropna()
    index = real_v.index.intersection(syn_v.index)
    if len(index) < min_periods:
        return None
    return corr(real_v.loc[index].to_numpy(), syn_v.loc[index].to_numpy())


def grouped_series_corr(real, synthetic, timestamp_col, real_values, syn_values, freq, min_periods=2):
    real_s = pd.DataFrame({timestamp_col: real[timestamp_col], "value": real_values}).set_index(timestamp_col)["value"].resample(freq).mean()
    syn_s = pd.DataFrame({timestamp_col: synthetic[timestamp_col], "value": syn_values}).set_index(timestamp_col)["value"].resample(freq).mean()
    index = real_s.index.intersection(syn_s.index)
    if len(index) < min_periods:
        return None
    return corr(real_s.loc[index].to_numpy(), syn_s.loc[index].to_numpy())


def entity_mean_corr(real, synthetic, entity_col, value_col):
    real_s = real.groupby(entity_col)[value_col].mean()
    syn_s = synthetic.groupby(entity_col)[value_col].mean()
    index = real_s.index.intersection(syn_s.index)
    return corr(real_s.loc[index].to_numpy(), syn_s.loc[index].to_numpy())


def entity_series_corr(real, synthetic, entity_col, real_values, syn_values):
    real_s = pd.DataFrame({entity_col: real[entity_col], "value": real_values}).groupby(entity_col)["value"].mean()
    syn_s = pd.DataFrame({entity_col: synthetic[entity_col], "value": syn_values}).groupby(entity_col)["value"].mean()
    index = real_s.index.intersection(syn_s.index)
    return corr(real_s.loc[index].to_numpy(), syn_s.loc[index].to_numpy())


def trajectory_corr_top_entities(real, synthetic, entity_col, timestamp_col, value_col, k=20):
    entities = real[entity_col].value_counts().head(k).index
    values = []
    for entity in entities:
        real_e = real[real[entity_col] == entity]
        syn_e = synthetic[synthetic[entity_col] == entity]
        value = grouped_rate_corr(real_e, syn_e, timestamp_col, value_col, "M")
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else None


def rating_vs_degree_corr(df, entity_col, rating_col):
    grouped = df.groupby(entity_col).agg(degree=(rating_col, "size"), rating=(rating_col, "mean"))
    return corr(grouped["degree"].to_numpy(dtype=float), grouped["rating"].to_numpy(dtype=float))


def entity_average_distribution_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    product_col: str,
    customer_col: str,
    value_col: str,
    prefix: str,
) -> Dict[str, Optional[float]]:
    product_real = real.groupby(product_col)[value_col].mean().to_numpy(dtype=float)
    product_syn = synthetic.groupby(product_col)[value_col].mean().to_numpy(dtype=float)
    customer_real = real.groupby(customer_col)[value_col].mean().to_numpy(dtype=float)
    customer_syn = synthetic.groupby(customer_col)[value_col].mean().to_numpy(dtype=float)
    return {
        f"real_product_{prefix}_variance": variance_or_none(product_real),
        f"synthetic_product_{prefix}_variance": variance_or_none(product_syn),
        f"real_customer_{prefix}_variance": variance_or_none(customer_real),
        f"synthetic_customer_{prefix}_variance": variance_or_none(customer_syn),
        f"product_{prefix}_variance_ratio": variance_ratio(product_syn, product_real),
        f"customer_{prefix}_variance_ratio": variance_ratio(customer_syn, customer_real),
        f"product_{prefix}_distribution_ks": empirical_ks(product_real, product_syn),
        f"customer_{prefix}_distribution_ks": empirical_ks(customer_real, customer_syn),
    }


def variance_or_none(values: np.ndarray) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(np.var(values))


def variance_ratio(numerator: np.ndarray, denominator: np.ndarray) -> Optional[float]:
    denom = variance_or_none(denominator)
    num = variance_or_none(numerator)
    if denom is None or num is None or abs(denom) < 1e-12:
        return None
    return float(num / denom)


def normalize_binary(values: pd.Series) -> pd.Series:
    def convert(value):
        if isinstance(value, str):
            return 1.0 if value.strip().lower() in {"true", "t", "yes", "y", "1"} else 0.0
        return float(bool(value)) if not isinstance(value, (int, float, np.number)) else float(value)
    return values.map(convert).astype(float)


def block_metrics(real, synthetic, customer_col, product_col, rating_col, verified_col, customer_blocks, product_blocks):
    result = {}
    for frame in (real, synthetic):
        frame["customer_block"] = frame[customer_col].map(customer_blocks)
        frame["product_block"] = frame[product_col].map(product_blocks)
        frame["block_pair"] = list(zip(frame["customer_block"], frame["product_block"]))
    if rating_col in real.columns and rating_col in synthetic.columns:
        result["block_pair_average_rating_correlation"] = entity_mean_corr(real, synthetic, "block_pair", rating_col)
        result["customer_block_average_rating_correlation"] = entity_mean_corr(real, synthetic, "customer_block", rating_col)
        result["product_block_average_rating_correlation"] = entity_mean_corr(real, synthetic, "product_block", rating_col)
    if verified_col in real.columns and verified_col in synthetic.columns:
        result["block_pair_verified_rate_correlation"] = entity_series_corr(real, synthetic, "block_pair", normalize_binary(real[verified_col]), normalize_binary(synthetic[verified_col]))
    return result


def numerical_metrics(real, synthetic, customer_col, product_col, timestamp_col, col):
    real_v = pd.to_numeric(real[col], errors="coerce").dropna().to_numpy(dtype=float)
    syn_v = pd.to_numeric(synthetic[col], errors="coerce").dropna().to_numpy(dtype=float)
    result = {
        "mean_real": float(np.mean(real_v)) if len(real_v) else None,
        "mean_synthetic": float(np.mean(syn_v)) if len(syn_v) else None,
        "std_real": float(np.std(real_v)) if len(real_v) else None,
        "std_synthetic": float(np.std(syn_v)) if len(syn_v) else None,
        "distribution_ks": empirical_ks(real_v, syn_v),
        "wasserstein": wasserstein_1d(real_v, syn_v),
        "product_average_correlation": entity_mean_corr(real, synthetic, product_col, col),
        "customer_average_correlation": entity_mean_corr(real, synthetic, customer_col, col),
        "monthly_average_correlation": grouped_rate_corr(real, synthetic, timestamp_col, col, "M"),
    }
    return result


def empirical_ks(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    values = np.sort(np.unique(np.concatenate([a, b])))
    return float(np.max(np.abs(np.searchsorted(np.sort(a), values, side="right") / len(a) - np.searchsorted(np.sort(b), values, side="right") / len(b))))


def wasserstein_1d(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    q = np.linspace(0, 1, max(len(a), len(b)))
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def c2st_accuracy(real, synthetic, customer_col, product_col, timestamp_col, cat_cols, num_cols):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
    except Exception:
        return None
    columns = [col for col in cat_cols + num_cols if col in real.columns and col in synthetic.columns]
    if not columns:
        return None
    real_x = featurize_for_c2st(real, columns, timestamp_col)
    syn_x = featurize_for_c2st(synthetic, columns, timestamp_col)
    x = pd.concat([real_x, syn_x], ignore_index=True).fillna(0.0)
    y = np.asarray([0] * len(real_x) + [1] * len(syn_x))
    if len(np.unique(y)) < 2 or len(x) < 10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y
        )
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        clf.fit(x_train, y_train)
        return float(clf.score(x_test, y_test))


def featurize_for_c2st(df, columns, timestamp_col):
    frame = pd.DataFrame(index=df.index)
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            frame[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            frame[col] = pd.Categorical(df[col].astype(str)).codes
    frame["month"] = pd.to_datetime(df[timestamp_col]).dt.month
    return frame


def flatten(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        else:
            flat[name] = value
    return flat


if __name__ == "__main__":
    main()
