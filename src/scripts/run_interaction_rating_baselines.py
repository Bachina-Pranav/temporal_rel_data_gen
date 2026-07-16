#!/usr/bin/env python3
"""Train-only empirical rating baselines for one interaction table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


BASELINES = ("global_empirical", "user_empirical", "movie_empirical", "user_movie_mixture")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate empirical rating baseline samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-table", default=None)
    parser.add_argument("--num-rows", default="all")
    parser.add_argument("--smoothing", type=float, default=5.0)
    parser.add_argument("--min-group-count", type=int, default=2)
    parser.add_argument("--include-time-baseline", action="store_true")
    parser.add_argument("--time-bin", choices=["year", "quarter"], default="year")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rng = np.random.default_rng(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_path = Path(args.fit_table) if args.fit_table else config.train_data_path
    fit = pd.read_csv(fit_path)
    if "split" in fit.columns:
        fit = fit.loc[fit["split"].astype(str).str.lower().isin(["train", "training"])].copy()
    spine = pd.read_csv(args.spine)
    if args.num_rows not in (None, "all"):
        spine = spine.head(int(args.num_rows)).copy()
    rating_col = config.schema.categorical_targets[0]
    user_col = config.schema.foreign_key_columns[0]
    movie_col = config.schema.foreign_key_columns[1] if len(config.schema.foreign_key_columns) > 1 else config.schema.foreign_key_columns[0]
    fit = fit.dropna(subset=[rating_col]).copy()
    global_dist = empirical_distribution(fit[rating_col])
    grouped = {
        "user_empirical": grouped_distributions(fit, user_col, rating_col),
        "movie_empirical": grouped_distributions(fit, movie_col, rating_col),
    }
    group_counts = {
        user_col: fit.groupby(fit[user_col].astype(str), sort=False).size().to_dict(),
        movie_col: fit.groupby(fit[movie_col].astype(str), sort=False).size().to_dict(),
    }
    time_grouped = grouped_distributions(add_time_bin(fit, config.schema.datetime_columns[0], args.time_bin), "_time_bin", rating_col)
    baseline_names = list(BASELINES)
    if args.include_time_baseline:
        baseline_names.append("time_empirical")
    manifest: dict[str, Any] = {
        "fit_table": str(fit_path),
        "spine": str(args.spine),
        "fit_rows": int(len(fit)),
        "rating_col": rating_col,
        "source_fk": user_col,
        "destination_fk": movie_col,
        "smoothing": float(args.smoothing),
        "min_group_count": int(args.min_group_count),
        "time_bin": str(args.time_bin),
        "uses_train_split_only": True,
        "outputs": {},
    }
    for name in baseline_names:
        output = spine.loc[:, [column for column in ["event_id", *config.schema.condition_columns] if column in spine.columns]].copy()
        if name == "global_empirical":
            output[rating_col] = sample_distribution(global_dist, len(output), rng)
        elif name in {"user_empirical", "movie_empirical"}:
            key_col = user_col if name == "user_empirical" else movie_col
            output[rating_col] = sample_grouped(
                output[key_col],
                grouped[name],
                global_dist,
                float(args.smoothing),
                rng,
                group_counts=group_counts[key_col],
                min_group_count=int(args.min_group_count),
            )
        elif name == "user_movie_mixture":
            output[rating_col] = sample_user_movie_mixture(
                output,
                user_col=user_col,
                movie_col=movie_col,
                user_dist=grouped["user_empirical"],
                movie_dist=grouped["movie_empirical"],
                global_dist=global_dist,
                user_counts=group_counts[user_col],
                movie_counts=group_counts[movie_col],
                smoothing=float(args.smoothing),
                min_group_count=int(args.min_group_count),
                rng=rng,
            )
        elif name == "time_empirical":
            output["_time_bin"] = add_time_bin(output, config.schema.datetime_columns[0], args.time_bin)["_time_bin"]
            output[rating_col] = sample_grouped(output["_time_bin"], time_grouped, global_dist, float(args.smoothing), rng)
            output = output.drop(columns=["_time_bin"])
        else:
            raise ValueError(f"Unsupported baseline: {name}")
        if "event_id" not in output.columns:
            output.insert(0, "event_id", [f"{name}_{idx}" for idx in range(len(output))])
        path = output_dir / name / "synthetic_interactions.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
        manifest["outputs"][name] = str(path)
        print(f"Wrote {path}")
    manifest_path = output_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


def empirical_distribution(values: pd.Series) -> dict[str, float]:
    counts = values.astype(str).value_counts(normalize=True).sort_index()
    if counts.empty:
        raise ValueError("Cannot fit empirical distribution from empty rating column")
    return {str(key): float(value) for key, value in counts.items()}


def grouped_distributions(frame: pd.DataFrame, key_col: str, rating_col: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, group in frame.groupby(frame[key_col].astype(str), sort=False):
        out[str(key)] = empirical_distribution(group[rating_col])
    return out


def sample_distribution(dist: dict[str, float], size: int, rng: np.random.Generator) -> list[Any]:
    values = np.asarray(list(dist), dtype=object)
    probs = np.asarray(list(dist.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(values, size=int(size), p=probs, replace=True).tolist()


def sample_grouped(
    keys: pd.Series,
    grouped: dict[str, dict[str, float]],
    fallback: dict[str, float],
    smoothing: float,
    rng: np.random.Generator,
    *,
    group_counts: dict[Any, int] | None = None,
    min_group_count: int = 1,
) -> list[Any]:
    fallback_counts = probabilities_to_pseudo_counts(fallback, smoothing)
    values = sorted(fallback_counts)
    samples: list[Any] = []
    for key in keys.astype(str):
        counts = dict(fallback_counts)
        local = grouped.get(str(key))
        observed_count = int((group_counts or {}).get(str(key), 0))
        if local is not None and observed_count >= int(min_group_count):
            for value, prob in local.items():
                counts[str(value)] = counts.get(str(value), 0.0) + float(prob)
        probs = np.asarray([counts.get(value, 0.0) for value in values], dtype=float)
        probs = probs / probs.sum()
        samples.append(rng.choice(np.asarray(values, dtype=object), p=probs))
    return samples


def sample_user_movie_mixture(
    frame: pd.DataFrame,
    *,
    user_col: str,
    movie_col: str,
    user_dist: dict[str, dict[str, float]],
    movie_dist: dict[str, dict[str, float]],
    global_dist: dict[str, float],
    user_counts: dict[Any, int],
    movie_counts: dict[Any, int],
    smoothing: float,
    min_group_count: int,
    rng: np.random.Generator,
) -> list[Any]:
    values = sorted(global_dist)
    samples: list[Any] = []
    for _, row in frame.iterrows():
        user_key = str(row[user_col])
        movie_key = str(row[movie_col])
        u_count = int(user_counts.get(user_key, 0))
        m_count = int(movie_counts.get(movie_key, 0))
        u_weight = float(u_count / (u_count + smoothing)) if u_count >= int(min_group_count) else 0.0
        m_weight = float(m_count / (m_count + smoothing)) if m_count >= int(min_group_count) else 0.0
        g_weight = 1.0
        total_weight = u_weight + m_weight + g_weight
        probs = np.zeros(len(values), dtype=float)
        for idx, value in enumerate(values):
            probs[idx] += g_weight * float(global_dist.get(value, 0.0))
            probs[idx] += u_weight * float(user_dist.get(user_key, {}).get(value, 0.0))
            probs[idx] += m_weight * float(movie_dist.get(movie_key, {}).get(value, 0.0))
        probs = probs / max(total_weight, 1e-12)
        probs = probs / probs.sum()
        samples.append(rng.choice(np.asarray(values, dtype=object), p=probs))
    return samples


def add_time_bin(frame: pd.DataFrame, timestamp_col: str, mode: str) -> pd.DataFrame:
    output = frame.copy()
    timestamps = pd.to_datetime(output[timestamp_col], errors="coerce")
    if mode == "quarter":
        output["_time_bin"] = timestamps.dt.to_period("Q").astype(str)
    else:
        output["_time_bin"] = timestamps.dt.year.astype("Int64").astype(str)
    output["_time_bin"] = output["_time_bin"].replace("<NA>", "unknown")
    return output


def probabilities_to_pseudo_counts(dist: dict[str, float], smoothing: float) -> dict[str, float]:
    alpha = max(float(smoothing), 0.0)
    return {str(value): max(float(prob) * alpha, 1e-12) for value, prob in dist.items()}


if __name__ == "__main__":
    main()
