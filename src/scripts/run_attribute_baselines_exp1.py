#!/usr/bin/env python3
"""Run simple Exp1 attribute baselines for Conditional TABDLM comparison."""

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

from attribute_generation.conditional_tabdlm.evaluate import evaluate_frames  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import normalize_text  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import ensure_dir, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global/monthly/product marginal Exp1 baselines.")
    parser.add_argument("--config", default="configs/attribute_generation/conditional_tabdlm_rel_amazon_exp1.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-rows", default=None)
    parser.add_argument("--max-fit-rows", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    output_dir = ensure_dir(args.output_dir or config.output_dir / "baselines")
    real = pd.read_csv(config.train_data_path)
    spine = pd.read_csv(config.synthetic_spine_path)
    if args.num_rows not in (None, "all"):
        spine = spine.head(int(args.num_rows))
    elif config.raw.get("sampling", {}).get("num_rows") not in (None, "all"):
        spine = spine.head(int(config.raw["sampling"]["num_rows"]))
    fit = real.dropna(subset=list(config.schema.target_columns)).copy()
    for column in config.schema.datetime_columns:
        fit[column] = pd.to_datetime(fit[column], errors="coerce")
        spine[column] = pd.to_datetime(spine[column], errors="coerce")
    for column in config.schema.text_targets:
        fit[column] = fit[column].map(normalize_text)
    fit = fit.dropna(subset=list(config.schema.datetime_columns))
    if len(fit) > int(args.max_fit_rows):
        fit = fit.sample(int(args.max_fit_rows), random_state=args.seed)

    metrics_by_name: dict[str, Any] = {}
    rows = []
    for name in ["global_marginal", "monthly_marginal", "product_marginal"]:
        synthetic = make_baseline(name, fit, spine, config, rng)
        out_path = output_dir / f"{name}.csv"
        synthetic.to_csv(out_path, index=False)
        metrics = evaluate_frames(real, synthetic, config)
        save_json(metrics, output_dir / f"{name}_eval_metrics.json")
        metrics_by_name[name] = metrics
        rows.append(flatten_metrics({"baseline": name, **metrics}))
        print(f"Wrote {out_path}")
    pd.DataFrame(rows).to_csv(output_dir / "baseline_comparison.csv", index=False)
    with (output_dir / "baseline_comparison.json").open("w") as handle:
        json.dump(metrics_by_name, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {output_dir / 'baseline_comparison.csv'}")


def make_baseline(name: str, fit: pd.DataFrame, spine: pd.DataFrame, config, rng: np.random.Generator) -> pd.DataFrame:
    schema = config.schema
    output = spine.loc[:, list(schema.condition_columns)].copy()
    month_col = "_month"
    fit = fit.copy()
    spine_month = pd.to_datetime(spine[schema.datetime_columns[0]], errors="coerce").dt.to_period("M").astype(str)
    fit[month_col] = pd.to_datetime(fit[schema.datetime_columns[0]], errors="coerce").dt.to_period("M").astype(str)
    product_col = schema.foreign_key_columns[1] if len(schema.foreign_key_columns) > 1 else schema.foreign_key_columns[0]
    group_col = None
    group_values = pd.Series(["__global__"] * len(spine), index=spine.index)
    if name == "monthly_marginal":
        group_col = month_col
        group_values = spine_month
    elif name == "product_marginal":
        group_col = product_col
        group_values = spine[product_col].astype(str)
        fit[product_col] = fit[product_col].astype(str)

    for column in schema.categorical_targets:
        output[column] = sample_grouped_column(fit, column, group_col, group_values, rng)
    for column in schema.text_targets:
        output[column] = sample_grouped_column(fit, column, group_col, group_values, rng)
    return output


def sample_grouped_column(
    fit: pd.DataFrame,
    column: str,
    group_col: str | None,
    group_values: pd.Series | None,
    rng: np.random.Generator,
) -> list[Any]:
    global_values = fit[column].dropna().to_numpy()
    if len(global_values) == 0:
        return [None] * (len(group_values) if group_values is not None else len(fit))
    if group_col is None or group_values is None:
        size = len(group_values) if group_values is not None else len(fit)
        return rng.choice(global_values, size=size, replace=True).tolist()
    grouped = {str(key): group[column].dropna().to_numpy() for key, group in fit.groupby(group_col)}
    sampled = []
    for value in group_values.astype(str):
        choices = grouped.get(str(value), global_values)
        if len(choices) == 0:
            choices = global_values
        sampled.append(rng.choice(choices))
    return sampled


def flatten_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_metrics(value, full))
        else:
            out[full] = value
    return out


if __name__ == "__main__":
    main()
