#!/usr/bin/env python3
"""Evaluate a time-biased block-stub matching event spine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.fast_event_spine_metrics import evaluate_fast_event_spine, load_metadata, write_metrics  # noqa: E402
from generators.fast_temporal_activity import canonical_time_bucket  # noqa: E402
from generators.lowrank_time_gated_affinity import LowRankTimeGatedAffinity  # noqa: E402
from generators.time_biased_block_stub_matching import ks_stat  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate time-biased block-stub matching event-spine metrics.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--time-granularity", default="day", choices=["day"])
    parser.add_argument("--time-gate-granularity", default=None, choices=["day", "month"])
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha-time-gate", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dynamic-affinity", action="store_true")
    parser.add_argument("--compute-c2st", action="store_true")
    parser.add_argument("--c2st-sample-size", type=int, default=200000)
    parser.add_argument("--c2st-model", choices=["hist_gradient_boosting", "logistic_regression"], default="hist_gradient_boosting")
    parser.add_argument("--c2st-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.real_reviews)
    synthetic = pd.read_csv(args.synthetic_reviews)
    metadata = load_metadata(args.metadata)
    metrics = evaluate_fast_event_spine(
        real,
        synthetic,
        structure_debug_dir=args.structure_debug_dir,
        customer_col=args.customer_id_col,
        product_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        compute_c2st=False,
        metadata=metadata,
    )
    metrics.update(pairing_instrumentation_metrics(metadata, Path(args.synthetic_reviews).parent / "debug"))
    if not args.skip_dynamic_affinity:
        metrics.update(
            dynamic_affinity_diagnostics(
                real,
                synthetic,
                args.customer_id_col,
                args.product_id_col,
                args.timestamp_col,
                args.time_granularity,
                args.time_gate_granularity or metadata_value(metadata, "time_gate_granularity", "month"),
                args.rank if args.rank is not None else int(metadata_value(metadata, "rank", 32)),
                args.alpha_time_gate if args.alpha_time_gate is not None else metadata_value(metadata, "alpha_time_gate", "auto"),
                args.seed,
            )
        )
    if args.compute_c2st:
        metrics.update(
            sampled_c2st_metrics(
                real,
                synthetic,
                args.customer_id_col,
                args.product_id_col,
                args.timestamp_col,
                args.structure_debug_dir,
                args.time_granularity,
                args.time_gate_granularity or metadata_value(metadata, "time_gate_granularity", "month"),
                args.rank if args.rank is not None else int(metadata_value(metadata, "rank", 32)),
                args.alpha_time_gate if args.alpha_time_gate is not None else metadata_value(metadata, "alpha_time_gate", "auto"),
                args.c2st_sample_size,
                args.c2st_model,
                args.c2st_seed,
            )
        )
    write_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2))


def dynamic_affinity_diagnostics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    time_granularity: str,
    time_gate_granularity: str,
    rank: int,
    alpha_time_gate: Any,
    seed: int,
) -> Dict[str, float]:
    real_frame = real[[customer_col, product_col, timestamp_col]].copy()
    synthetic_frame = synthetic[[customer_col, product_col, timestamp_col]].copy()
    real_frame[timestamp_col] = canonical_time_bucket(real_frame[timestamp_col], time_granularity)
    synthetic_frame[timestamp_col] = canonical_time_bucket(synthetic_frame[timestamp_col], time_granularity)
    affinity = LowRankTimeGatedAffinity(
        rank=rank,
        alpha_time_gate=alpha_time_gate,
        time_gate_granularity=time_gate_granularity,
        seed=seed,
    ).fit(real_frame, customer_col, product_col, timestamp_col)
    real_scores = score_pairs_by_time(affinity, real_frame, customer_col, product_col, timestamp_col)
    synthetic_scores = score_pairs_by_time(affinity, synthetic_frame, customer_col, product_col, timestamp_col)
    return {
        "mean_dynamic_affinity_real": float(np.mean(real_scores)) if len(real_scores) else 0.0,
        "mean_dynamic_affinity_synthetic": float(np.mean(synthetic_scores)) if len(synthetic_scores) else 0.0,
        "median_dynamic_affinity_real": float(np.median(real_scores)) if len(real_scores) else 0.0,
        "median_dynamic_affinity_synthetic": float(np.median(synthetic_scores)) if len(synthetic_scores) else 0.0,
        "dynamic_affinity_distribution_ks": ks_stat(real_scores, synthetic_scores),
    }


def score_pairs_by_time(
    affinity: LowRankTimeGatedAffinity,
    frame: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
) -> np.ndarray:
    scores = []
    for time_bucket, group in frame.groupby(timestamp_col, sort=False):
        scores.append(
            affinity.score_pairs(
                group[customer_col].to_numpy(dtype=object),
                group[product_col].to_numpy(dtype=object),
                time_bucket,
            )
        )
    return np.concatenate(scores) if scores else np.asarray([], dtype=float)


def metadata_value(metadata: Dict[str, Any] | None, key: str, default: Any) -> Any:
    return metadata.get(key, default) if metadata else default


PAIRING_COUNTER_KEYS = [
    "num_cells_processed",
    "num_exact_penalized_cells",
    "num_projection_fallback_cells",
    "num_random_cells",
    "num_events_exact_penalized",
    "num_events_projection_fallback",
    "num_events_random",
    "percent_cells_exact_penalized",
    "percent_cells_projection_fallback",
    "percent_cells_random",
    "percent_events_exact_penalized",
    "percent_events_projection_fallback",
    "percent_events_random",
    "percent_large_cells_projection_sort",
    "largest_cell_size",
    "max_cell_size",
    "average_cell_size",
    "p95_cell_size",
    "p99_cell_size",
    "max_exact_affinity_cell_size",
]


def pairing_instrumentation_metrics(metadata: Dict[str, Any] | None, debug_dir: Path) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    summary = load_json(debug_dir / "dynamic_pairing_summary.json")
    if metadata:
        for key in PAIRING_COUNTER_KEYS:
            if metadata.get(key) is not None:
                output[key] = metadata.get(key)
    if summary:
        normalized = normalize_pairing_summary(summary)
        for key, value in normalized.items():
            if value is not None:
                output[key] = value
    output.update(block_pair_time_cell_stats(debug_dir / "block_pair_time_counts.csv", output))
    return {key: output.get(key) for key in PAIRING_COUNTER_KEYS if key in output}


def normalize_pairing_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    num_cells = int(summary.get("num_cells_processed", summary.get("num_cells", 0)) or 0)
    exact_cells = int(summary.get("num_exact_penalized_cells", 0) or 0)
    fallback_cells = int(summary.get("num_projection_fallback_cells", 0) or 0)
    exact_events = int(summary.get("num_events_exact_penalized", 0) or 0)
    fallback_events = int(summary.get("num_events_projection_fallback", 0) or 0)
    if exact_events == 0 and summary.get("percent_events_exact_penalized") is not None:
        exact_events = None
    if fallback_events == 0 and summary.get("percent_events_projection_fallback") is not None:
        fallback_events = None
    return {
        "num_cells_processed": num_cells,
        "num_exact_penalized_cells": exact_cells,
        "num_projection_fallback_cells": fallback_cells,
        "num_random_cells": int(summary.get("num_random_cells", 0) or 0),
        "num_events_exact_penalized": exact_events,
        "num_events_projection_fallback": fallback_events,
        "num_events_random": summary.get("num_events_random"),
        "percent_cells_exact_penalized": float(summary.get("percent_cells_exact_penalized", exact_cells / max(num_cells, 1))),
        "percent_cells_projection_fallback": float(summary.get("percent_cells_projection_fallback", fallback_cells / max(num_cells, 1))),
        "percent_cells_random": summary.get("percent_cells_random"),
        "percent_events_exact_penalized": summary.get("percent_events_exact_penalized"),
        "percent_events_projection_fallback": summary.get("percent_events_projection_fallback"),
        "percent_events_random": summary.get("percent_events_random"),
        "percent_large_cells_projection_sort": summary.get(
            "percent_large_cells_projection_sort",
            summary.get("percent_events_projection_fallback"),
        ),
        "largest_cell_size": summary.get("largest_cell_size", summary.get("max_cell_size")),
        "average_cell_size": summary.get("average_cell_size"),
        "p95_cell_size": summary.get("p95_cell_size"),
        "p99_cell_size": summary.get("p99_cell_size"),
        "max_exact_affinity_cell_size": summary.get("max_exact_affinity_cell_size"),
    }


def block_pair_time_cell_stats(path: Path, existing: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, usecols=["count"])
    counts = frame["count"].to_numpy(dtype=float)
    if len(counts) == 0:
        return {}
    total_events = float(np.sum(counts))
    stats: Dict[str, Any] = {
        "num_cells_processed": int(existing.get("num_cells_processed", len(counts)) or len(counts)),
        "average_cell_size": float(np.mean(counts)),
        "largest_cell_size": int(np.max(counts)),
        "max_cell_size": int(np.max(counts)),
        "p95_cell_size": float(np.percentile(counts, 95.0)),
        "p99_cell_size": float(np.percentile(counts, 99.0)),
    }
    max_exact = existing.get("max_exact_affinity_cell_size")
    if max_exact is not None:
        max_exact_int = int(max_exact)
        exact_mask = counts <= max_exact_int
        fallback_mask = ~exact_mask
        exact_cells = int(exact_mask.sum())
        fallback_cells = int(fallback_mask.sum())
        exact_events = int(counts[exact_mask].sum())
        fallback_events = int(counts[fallback_mask].sum())
        stats.setdefault("num_exact_penalized_cells", exact_cells)
        stats.setdefault("num_projection_fallback_cells", fallback_cells)
        stats.setdefault("num_events_exact_penalized", exact_events)
        stats.setdefault("num_events_projection_fallback", fallback_events)
        stats.setdefault("percent_cells_exact_penalized", float(exact_cells / max(len(counts), 1)))
        stats.setdefault("percent_cells_projection_fallback", float(fallback_cells / max(len(counts), 1)))
        stats.setdefault("percent_events_exact_penalized", float(exact_events / max(total_events, 1.0)))
        stats.setdefault("percent_events_projection_fallback", float(fallback_events / max(total_events, 1.0)))
        stats.setdefault("percent_large_cells_projection_sort", float(fallback_events / max(total_events, 1.0)))
    for percent_key, count_key in [
        ("percent_events_exact_penalized", "num_events_exact_penalized"),
        ("percent_events_projection_fallback", "num_events_projection_fallback"),
        ("percent_events_random", "num_events_random"),
    ]:
        if existing.get(count_key) is None and existing.get(percent_key) is not None:
            stats[count_key] = int(round(float(existing[percent_key]) * total_events))
    return stats


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def score_pairs_by_time_aligned(
    affinity: LowRankTimeGatedAffinity,
    frame: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
) -> np.ndarray:
    scores = np.zeros(len(frame), dtype=float)
    if len(frame) == 0:
        return scores
    for time_bucket, group in frame.groupby(timestamp_col, sort=False):
        scores[group.index.to_numpy(dtype=int)] = affinity.score_pairs(
            group[customer_col].to_numpy(dtype=object),
            group[product_col].to_numpy(dtype=object),
            time_bucket,
        )
    return scores


def sampled_c2st_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    structure_debug_dir: str | None,
    time_granularity: str,
    time_gate_granularity: str,
    rank: int,
    alpha_time_gate: Any,
    sample_size: int,
    model_name: str,
    seed: int,
) -> Dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception:
        return {"event_tuple_c2st_accuracy": None, "event_tuple_c2st_auc": None, "c2st_sample_size": 0}
    try:
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception:
        HistGradientBoostingClassifier = None

    rng = np.random.default_rng(int(seed))
    n = min(int(sample_size), len(real), len(synthetic))
    if n < 10:
        return {"event_tuple_c2st_accuracy": None, "event_tuple_c2st_auc": None, "c2st_sample_size": int(n)}
    real_sample = real.iloc[rng.choice(len(real), size=n, replace=False)].copy()
    syn_sample = synthetic.iloc[rng.choice(len(synthetic), size=n, replace=False)].copy()
    data = pd.concat([real_sample.assign(_label=0), syn_sample.assign(_label=1)], ignore_index=True)
    real_c = real[[customer_col, product_col, timestamp_col]].copy()
    real_c[timestamp_col] = canonical_time_bucket(real_c[timestamp_col], time_granularity)
    data[timestamp_col] = canonical_time_bucket(data[timestamp_col], time_granularity)
    customer_degree = real_c[customer_col].value_counts()
    product_degree = real_c[product_col].value_counts()
    customer_windows = window_frame(real_c, customer_col, timestamp_col)
    product_windows = window_frame(real_c, product_col, timestamp_col)
    customer_blocks, product_blocks = load_blocks_for_c2st(structure_debug_dir, customer_col, product_col)
    affinity = LowRankTimeGatedAffinity(
        rank=rank,
        alpha_time_gate=alpha_time_gate,
        time_gate_granularity=time_gate_granularity,
        seed=seed,
    ).fit(real_c, customer_col, product_col, timestamp_col)
    day_values = pd.to_datetime(data[timestamp_col], errors="coerce").map(lambda value: value.toordinal() if pd.notna(value) else 0)
    month_values = pd.to_datetime(data[timestamp_col], errors="coerce").dt.to_period("M").astype(str).astype("category").cat.codes
    features = pd.DataFrame(
        {
            "customer_degree": data[customer_col].map(customer_degree).fillna(0).astype(float),
            "product_degree": data[product_col].map(product_degree).fillna(0).astype(float),
            "customer_block": data[customer_col].map(customer_blocks).fillna(0).astype(float),
            "product_block": data[product_col].map(product_blocks).fillna(0).astype(float),
            "day_index": day_values.astype(float),
            "month_index": month_values.astype(float),
            "customer_first_time": data[customer_col].map(customer_windows["first"]).fillna(0).astype(float),
            "customer_last_time": data[customer_col].map(customer_windows["last"]).fillna(0).astype(float),
            "product_first_time": data[product_col].map(product_windows["first"]).fillna(0).astype(float),
            "product_last_time": data[product_col].map(product_windows["last"]).fillna(0).astype(float),
            "customer_relative_age": relative_age_feature(data, customer_col, timestamp_col, customer_windows),
            "product_relative_age": relative_age_feature(data, product_col, timestamp_col, product_windows),
            "dynamic_affinity_score": score_pairs_by_time_aligned(affinity, data, customer_col, product_col, timestamp_col),
        }
    )
    labels = data["_label"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        features.to_numpy(dtype=float),
        labels,
        test_size=0.3,
        random_state=int(seed),
        stratify=labels,
    )
    actual_model = model_name
    if model_name == "logistic_regression" or HistGradientBoostingClassifier is None:
        clf = LogisticRegression(max_iter=500, random_state=int(seed))
        if model_name != "logistic_regression":
            actual_model = "logistic_regression_fallback"
    else:
        clf = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=int(seed))
    clf.fit(x_train, y_train)
    accuracy = float(clf.score(x_test, y_test))
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(x_test)[:, 1]
    else:
        scores = clf.decision_function(x_test)
    try:
        auc = float(roc_auc_score(y_test, scores))
    except Exception:
        auc = None
    return {
        "event_tuple_c2st_accuracy": accuracy,
        "event_tuple_c2st_auc": auc,
        "c2st_sample_size": int(n),
        "c2st_model": actual_model,
    }


def load_blocks_for_c2st(structure_debug_dir: str | None, customer_col: str, product_col: str) -> tuple[Dict[Any, int], Dict[Any, int]]:
    if not structure_debug_dir:
        return {}, {}
    root = Path(structure_debug_dir)
    return load_block_csv(root / "customer_blocks.csv", customer_col, "customer_block"), load_block_csv(root / "product_blocks.csv", product_col, "product_block")


def load_block_csv(path: Path, entity_col: str, block_col: str) -> Dict[Any, int]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if entity_col not in frame.columns or block_col not in frame.columns:
        return {}
    return dict(zip(frame[entity_col], frame[block_col].astype(int)))


def window_frame(frame: pd.DataFrame, entity_col: str, timestamp_col: str) -> Dict[str, pd.Series]:
    days = pd.to_datetime(frame[timestamp_col], errors="coerce").map(lambda value: value.toordinal() if pd.notna(value) else 0)
    tmp = pd.DataFrame({"entity": frame[entity_col].to_numpy(dtype=object), "day": days.to_numpy(dtype=float)})
    grouped = tmp.groupby("entity", sort=False)["day"].agg(["min", "max"])
    return {"first": grouped["min"], "last": grouped["max"]}


def relative_age_feature(data: pd.DataFrame, entity_col: str, timestamp_col: str, windows: Dict[str, pd.Series]) -> np.ndarray:
    first = data[entity_col].map(windows["first"]).to_numpy(dtype=float)
    last = data[entity_col].map(windows["last"]).to_numpy(dtype=float)
    day = pd.to_datetime(data[timestamp_col], errors="coerce").map(lambda value: value.toordinal() if pd.notna(value) else 0).to_numpy(dtype=float)
    output = np.zeros(len(data), dtype=float)
    valid = np.isfinite(first) & np.isfinite(last)
    span = np.maximum(last[valid] - first[valid], 1.0)
    output[valid] = (day[valid] - first[valid]) / span
    return output


if __name__ == "__main__":
    main()
