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


BASELINES = ("global_empirical", "user_empirical", "movie_empirical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate empirical rating baseline samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-table", default=None)
    parser.add_argument("--num-rows", default="all")
    parser.add_argument("--smoothing", type=float, default=5.0)
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
    manifest: dict[str, Any] = {
        "fit_table": str(fit_path),
        "spine": str(args.spine),
        "fit_rows": int(len(fit)),
        "rating_col": rating_col,
        "source_fk": user_col,
        "destination_fk": movie_col,
        "smoothing": float(args.smoothing),
        "outputs": {},
    }
    for name in BASELINES:
        output = spine.loc[:, [column for column in ["event_id", *config.schema.condition_columns] if column in spine.columns]].copy()
        if name == "global_empirical":
            output[rating_col] = sample_distribution(global_dist, len(output), rng)
        else:
            key_col = user_col if name == "user_empirical" else movie_col
            output[rating_col] = sample_grouped(output[key_col], grouped[name], global_dist, float(args.smoothing), rng)
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
) -> list[Any]:
    fallback_counts = probabilities_to_pseudo_counts(fallback, smoothing)
    values = sorted(fallback_counts)
    samples: list[Any] = []
    for key in keys.astype(str):
        counts = dict(fallback_counts)
        local = grouped.get(str(key))
        if local is not None:
            for value, prob in local.items():
                counts[str(value)] = counts.get(str(value), 0.0) + float(prob)
        probs = np.asarray([counts.get(value, 0.0) for value in values], dtype=float)
        probs = probs / probs.sum()
        samples.append(rng.choice(np.asarray(values, dtype=object), p=probs))
    return samples


def probabilities_to_pseudo_counts(dist: dict[str, float], smoothing: float) -> dict[str, float]:
    alpha = max(float(smoothing), 0.0)
    return {str(value): max(float(prob) * alpha, 1e-12) for value, prob in dist.items()}


if __name__ == "__main__":
    main()
