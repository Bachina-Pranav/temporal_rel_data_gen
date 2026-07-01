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
from reldiff.attributes.temporal_priors import temporal_bucket  # noqa: E402


DECOMPOSITION_DIAGNOSTIC_KEYS = [
    "average_norm_base_rating_logits",
    "average_norm_residual_rating_logits",
    "average_norm_final_rating_logits_pre_calibration",
    "average_norm_final_rating_logits_post_calibration",
    "residual_to_base_norm_ratio",
    "average_abs_base_verified_logit",
    "average_abs_residual_verified_logit",
    "verified_residual_to_base_abs_ratio",
    "sampled_product_rating_effect_variance",
    "sampled_customer_rating_effect_variance",
    "sampled_product_verified_effect_variance",
    "sampled_customer_verified_effect_variance",
    "sampled_product_effect_variance",
    "sampled_customer_effect_variance",
    "average_temporal_rating_prior_entropy",
    "average_model_rating_entropy_pre_calibration",
    "average_model_rating_entropy_post_calibration",
    "temporal_calibration_average_correction_norm",
    "temporal_calibration_max_correction_norm",
    "temporal_calibration_num_groups_calibrated",
    "average_precal_rating_target_js",
    "average_postcal_rating_target_js",
    "average_precal_verified_target_abs_error",
    "average_postcal_verified_target_abs_error",
]


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
    parser.add_argument("--diagnostics-dir", default=None)
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
        diagnostics_dir=args.diagnostics_dir,
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
    diagnostics_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "categorical": {},
        "temporal": {},
        "relational": {},
        "block": {},
        "entity_distribution": {},
        "temporal_diagnostics": {},
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
        metrics["temporal"]["monthly_rating_distribution_js_mean"] = grouped_distribution_js(
            real, synthetic, timestamp_col, rating_col, "M"
        )
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
        metrics["temporal"]["monthly_verified_rate_mae"] = grouped_series_mae(
            real, synthetic, timestamp_col, real_v, syn_v, "M"
        )
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

    monthly_table, monthly_summary = monthly_diagnostics(
        real, synthetic, timestamp_col, rating_col, verified_col
    )
    metrics["temporal_diagnostics"].update(monthly_summary)
    if diagnostics_dir is not None:
        diagnostics_path = Path(diagnostics_dir)
        diagnostics_path.mkdir(parents=True, exist_ok=True)
        monthly_table.to_csv(diagnostics_path / "monthly_real_vs_synthetic.csv", index=False)
        with (diagnostics_path / "monthly_summary.json").open("w") as handle:
            json.dump(monthly_summary, handle, indent=2)
            handle.write("\n")

    for col in num_cols:
        metrics["numerical"][col] = numerical_metrics(real, synthetic, customer_col, product_col, timestamp_col, col)
    metrics["c2st"]["c2st_accuracy"] = c2st_accuracy(real, synthetic, customer_col, product_col, timestamp_col, cat_cols, num_cols)
    metrics["decomposition"] = decomposition_diagnostics(synthetic, diagnostics_dir)
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
    real_s = grouped_mean_series(real, timestamp_col, pd.to_numeric(real[value_col], errors="coerce"), freq)
    syn_s = grouped_mean_series(synthetic, timestamp_col, pd.to_numeric(synthetic[value_col], errors="coerce"), freq)
    index = real_s.index.union(syn_s.index)
    real_v = real_s.reindex(index).dropna()
    syn_v = syn_s.reindex(index).dropna()
    index = real_v.index.intersection(syn_v.index)
    if len(index) < min_periods:
        return None
    return corr(real_v.loc[index].to_numpy(), syn_v.loc[index].to_numpy())


def grouped_series_corr(real, synthetic, timestamp_col, real_values, syn_values, freq, min_periods=2):
    real_s = grouped_mean_series(real, timestamp_col, real_values, freq)
    syn_s = grouped_mean_series(synthetic, timestamp_col, syn_values, freq)
    index = real_s.index.intersection(syn_s.index)
    if len(index) < min_periods:
        return None
    return corr(real_s.loc[index].to_numpy(), syn_s.loc[index].to_numpy())


def grouped_series_mae(real, synthetic, timestamp_col, real_values, syn_values, freq, min_periods=2):
    real_s = grouped_mean_series(real, timestamp_col, real_values, freq)
    syn_s = grouped_mean_series(synthetic, timestamp_col, syn_values, freq)
    index = real_s.index.intersection(syn_s.index)
    if len(index) < min_periods:
        return None
    return float(np.mean(np.abs(real_s.loc[index].to_numpy() - syn_s.loc[index].to_numpy())))


def grouped_distribution_js(real, synthetic, timestamp_col, value_col, freq, min_periods=2):
    if is_month_frequency(freq):
        real_frame = pd.DataFrame(
            {"bucket": temporal_bucket(real[timestamp_col], "month"), value_col: real[value_col]}
        )
        syn_frame = pd.DataFrame(
            {"bucket": temporal_bucket(synthetic[timestamp_col], "month"), value_col: synthetic[value_col]}
        )
        values = []
        for bucket in sorted(set(real_frame["bucket"].dropna()) & set(syn_frame["bucket"].dropna())):
            real_group = real_frame[real_frame["bucket"] == bucket]
            syn_group = syn_frame[syn_frame["bucket"] == bucket]
            if len(real_group) and len(syn_group):
                values.append(js_divergence(real_group[value_col], syn_group[value_col]))
        if len(values) < min_periods:
            return None
        return float(np.mean(values))
    real_groups = real.set_index(timestamp_col).groupby(pd.Grouper(freq=freq))
    syn_groups = synthetic.set_index(timestamp_col).groupby(pd.Grouper(freq=freq))
    values = []
    for key, real_group in real_groups:
        if key not in syn_groups.groups:
            continue
        syn_group = syn_groups.get_group(key)
        if len(real_group) == 0 or len(syn_group) == 0:
            continue
        values.append(js_divergence(real_group[value_col], syn_group[value_col]))
    if len(values) < min_periods:
        return None
    return float(np.mean(values))


def grouped_mean_series(frame: pd.DataFrame, timestamp_col: str, values: pd.Series, freq: str) -> pd.Series:
    values = pd.to_numeric(pd.Series(values, index=frame.index), errors="coerce")
    if is_month_frequency(freq):
        bucket = temporal_bucket(frame[timestamp_col], "month")
        return pd.DataFrame({"bucket": bucket, "value": values}).dropna().groupby("bucket")["value"].mean()
    return pd.DataFrame({timestamp_col: frame[timestamp_col], "value": values}).set_index(timestamp_col)["value"].resample(freq).mean()


def is_month_frequency(freq: str) -> bool:
    return str(freq).upper() in {"M", "ME"}


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


def monthly_diagnostics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    timestamp_col: str,
    rating_col: str,
    verified_col: str,
) -> tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    real_frame = real.copy()
    synthetic_frame = synthetic.copy()
    real_frame["_month"] = temporal_bucket(real_frame[timestamp_col], "month")
    synthetic_frame["_month"] = temporal_bucket(synthetic_frame[timestamp_col], "month")
    rating_values = sorted(
        set(real_frame[rating_col].dropna().tolist())
        | set(synthetic_frame[rating_col].dropna().tolist()),
        key=lambda value: str(value),
    )
    numeric_rating_values = pd.to_numeric(pd.Series(rating_values), errors="coerce")
    if numeric_rating_values.notna().all():
        numeric_array = numeric_rating_values.to_numpy(dtype=float)
        integer_like = np.allclose(numeric_array, np.round(numeric_array))
        existing = {int(value) for value in np.round(numeric_array).tolist()}
        if integer_like and existing.issubset({1, 2, 3, 4, 5}):
            rating_values = [1, 2, 3, 4, 5]
    rows = []
    months = sorted(set(real_frame["_month"].dropna()) | set(synthetic_frame["_month"].dropna()))
    for month in months:
        real_m = real_frame[real_frame["_month"] == month]
        syn_m = synthetic_frame[synthetic_frame["_month"] == month]
        real_rating = pd.to_numeric(real_m[rating_col], errors="coerce")
        syn_rating = pd.to_numeric(syn_m[rating_col], errors="coerce")
        real_verified = normalize_binary(real_m[verified_col]) if verified_col in real_m else pd.Series(dtype=float)
        syn_verified = normalize_binary(syn_m[verified_col]) if verified_col in syn_m else pd.Series(dtype=float)
        row = {
            "month": month,
            "real_count": int(len(real_m)),
            "synthetic_count": int(len(syn_m)),
            "real_avg_rating": float(real_rating.mean()) if len(real_rating) else None,
            "synthetic_avg_rating": float(syn_rating.mean()) if len(syn_rating) else None,
            "real_verified_rate": float(real_verified.mean()) if len(real_verified) else None,
            "synthetic_verified_rate": float(syn_verified.mean()) if len(syn_verified) else None,
        }
        row["rating_abs_error"] = abs_or_none(row["synthetic_avg_rating"], row["real_avg_rating"])
        row["verified_abs_error"] = abs_or_none(row["synthetic_verified_rate"], row["real_verified_rate"])
        for value in rating_values:
            suffix = str(value)
            row[f"real_rating_dist_{suffix}"] = float((real_m[rating_col] == value).mean()) if len(real_m) else 0.0
            row[f"synthetic_rating_dist_{suffix}"] = float((syn_m[rating_col] == value).mean()) if len(syn_m) else 0.0
        row["monthly_rating_distribution_js"] = js_divergence(real_m[rating_col], syn_m[rating_col]) if len(real_m) and len(syn_m) else None
        row["synthetic_minus_real_avg_rating"] = diff_or_none(row["synthetic_avg_rating"], row["real_avg_rating"])
        row["synthetic_minus_real_verified_rate"] = diff_or_none(row["synthetic_verified_rate"], row["real_verified_rate"])
        rows.append(row)
    table = pd.DataFrame(rows)
    summary = monthly_summary(table)
    return table, summary


def monthly_summary(table: pd.DataFrame) -> Dict[str, Optional[float]]:
    if table.empty:
        return {
            "monthly_avg_rating_corr": None,
            "monthly_avg_rating_mae": None,
            "monthly_avg_rating_rmse": None,
            "monthly_avg_rating_real_std": None,
            "monthly_avg_rating_synthetic_std": None,
            "monthly_avg_rating_variance_ratio": None,
            "monthly_verified_corr": None,
            "monthly_verified_mae": None,
            "monthly_verified_rmse": None,
            "monthly_verified_real_std": None,
            "monthly_verified_synthetic_std": None,
            "monthly_verified_variance_ratio": None,
            "monthly_rating_distribution_js_mean": None,
        }
    real_rating = pd.to_numeric(table["real_avg_rating"], errors="coerce")
    syn_rating = pd.to_numeric(table["synthetic_avg_rating"], errors="coerce")
    real_verified = pd.to_numeric(table["real_verified_rate"], errors="coerce")
    syn_verified = pd.to_numeric(table["synthetic_verified_rate"], errors="coerce")
    rating_mask = real_rating.notna() & syn_rating.notna()
    verified_mask = real_verified.notna() & syn_verified.notna()
    rating_errors = (syn_rating[rating_mask] - real_rating[rating_mask]).to_numpy(dtype=float)
    verified_errors = (syn_verified[verified_mask] - real_verified[verified_mask]).to_numpy(dtype=float)
    return {
        "monthly_avg_rating_corr": corr(real_rating[rating_mask].to_numpy(dtype=float), syn_rating[rating_mask].to_numpy(dtype=float)) if rating_mask.any() else None,
        "monthly_avg_rating_mae": float(np.mean(np.abs(rating_errors))) if len(rating_errors) else None,
        "monthly_avg_rating_rmse": float(np.sqrt(np.mean(rating_errors**2))) if len(rating_errors) else None,
        "monthly_avg_rating_real_std": finite_or_none(real_rating[rating_mask].std()) if rating_mask.any() else None,
        "monthly_avg_rating_synthetic_std": finite_or_none(syn_rating[rating_mask].std()) if rating_mask.any() else None,
        "monthly_avg_rating_variance_ratio": variance_ratio(syn_rating[rating_mask].to_numpy(dtype=float), real_rating[rating_mask].to_numpy(dtype=float)) if rating_mask.any() else None,
        "monthly_verified_corr": corr(real_verified[verified_mask].to_numpy(dtype=float), syn_verified[verified_mask].to_numpy(dtype=float)) if verified_mask.any() else None,
        "monthly_verified_mae": float(np.mean(np.abs(verified_errors))) if len(verified_errors) else None,
        "monthly_verified_rmse": float(np.sqrt(np.mean(verified_errors**2))) if len(verified_errors) else None,
        "monthly_verified_real_std": finite_or_none(real_verified[verified_mask].std()) if verified_mask.any() else None,
        "monthly_verified_synthetic_std": finite_or_none(syn_verified[verified_mask].std()) if verified_mask.any() else None,
        "monthly_verified_variance_ratio": variance_ratio(syn_verified[verified_mask].to_numpy(dtype=float), real_verified[verified_mask].to_numpy(dtype=float)) if verified_mask.any() else None,
        "monthly_rating_distribution_js_mean": finite_or_none(pd.to_numeric(table["monthly_rating_distribution_js"], errors="coerce").mean()),
    }


def abs_or_none(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return float(abs(a - b))


def diff_or_none(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return float(a - b)


def finite_or_none(value):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


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


def decomposition_diagnostics(
    synthetic: pd.DataFrame,
    diagnostics_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        key: None for key in DECOMPOSITION_DIAGNOSTIC_KEYS
    }
    if diagnostics_dir is not None:
        path = Path(diagnostics_dir) / "decomposition_diagnostics.json"
        if path.exists():
            with path.open() as handle:
                loaded = json.load(handle)
            result.update({key: finite_or_none(loaded.get(key)) for key in DECOMPOSITION_DIAGNOSTIC_KEYS})
            result["diagnostic_status"] = "loaded"
            result["diagnostic_source"] = str(path)
            return result
        result["diagnostic_status"] = "missing_diagnostics_file"
        result["diagnostic_reason"] = f"{path} does not exist"
        return result

    legacy_columns = {
        "base_rating_logit_norm",
        "residual_rating_logit_norm",
        "temporal_calibration_correction_norm",
    }
    if not legacy_columns.intersection(synthetic.columns):
        result["diagnostic_status"] = "not_v3_output"
        result["diagnostic_reason"] = (
            "Pass --diagnostics-dir from V3 sampling to include decomposition diagnostics."
        )
        return result

    result["diagnostic_status"] = "legacy_columns"
    if "base_rating_logit_norm" in synthetic.columns:
        result["average_norm_base_rating_logits"] = finite_or_none(
            pd.to_numeric(synthetic["base_rating_logit_norm"], errors="coerce").mean()
        )
    if "residual_rating_logit_norm" in synthetic.columns:
        result["average_norm_residual_rating_logits"] = finite_or_none(
            pd.to_numeric(synthetic["residual_rating_logit_norm"], errors="coerce").mean()
        )
    if (
        result["average_norm_base_rating_logits"] is not None
        and result["average_norm_residual_rating_logits"] is not None
    ):
        result["residual_to_base_norm_ratio"] = float(
            result["average_norm_residual_rating_logits"]
            / max(result["average_norm_base_rating_logits"], 1e-12)
        )
    if "temporal_calibration_correction_norm" in synthetic.columns:
        result["temporal_calibration_average_correction_norm"] = finite_or_none(
            pd.to_numeric(synthetic["temporal_calibration_correction_norm"], errors="coerce").mean()
        )
    product_cols = [col for col in synthetic.columns if col.startswith("sampled_product_rating_effect_")]
    customer_cols = [col for col in synthetic.columns if col.startswith("sampled_customer_rating_effect_")]
    if product_cols:
        result["sampled_product_rating_effect_variance"] = finite_or_none(
            synthetic[product_cols].to_numpy(dtype=float).var()
        )
        result["sampled_product_effect_variance"] = result["sampled_product_rating_effect_variance"]
    if customer_cols:
        result["sampled_customer_rating_effect_variance"] = finite_or_none(
            synthetic[customer_cols].to_numpy(dtype=float).var()
        )
        result["sampled_customer_effect_variance"] = result["sampled_customer_rating_effect_variance"]
    return result


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
