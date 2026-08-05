#!/usr/bin/env python3
"""Compare structured attribute generators on one fixed event spine."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.diffusion_diagnostics import (  # noqa: E402
    unique_run_root,
    write_json,
)
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from evaluation.paper_metrics.reporting import write_markdown_report  # noqa: E402
from evaluation.paper_metrics.utils import write_json as write_paper_json  # noqa: E402
from scripts.evaluate_single_event_table_paper_metrics import evaluate_paper_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument(
        "--model-output",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="Existing diffusion or LSTM output; may be repeated.",
    )
    parser.add_argument(
        "--runtime-metadata",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help=(
            "Optional training/sampling runtime metadata matching a model-output "
            "name; may be repeated."
        ),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_config(args.model_config)
    with Path(args.benchmark_manifest).open(encoding="utf-8") as handle:
        benchmark = json.load(handle)
    paths = benchmark["files"]
    train = pd.read_csv(paths["train_real"]["path"])
    real = pd.read_csv(paths["evaluation_real"]["path"])
    spine = pd.read_csv(paths["evaluation_spine"]["path"])
    with Path(args.evaluation_config).open(encoding="utf-8") as handle:
        evaluation_config = yaml.safe_load(handle)
    output_root = unique_run_root(
        args.output_root, "structured_attribute_comparison"
    )
    rng = np.random.RandomState(args.seed)
    condition_columns = list(model_config.schema.condition_columns)
    target_columns = list(model_config.schema.categorical_targets) + list(
        model_config.schema.numerical_targets
    )
    if not target_columns:
        raise ValueError("Model schema has no structured targets")

    candidates: dict[str, pd.DataFrame] = {}
    runtime_records: dict[str, dict[str, float | None]] = {}
    started = time.perf_counter()
    candidates["independent_column"] = independent_column_baseline(
        train, spine, condition_columns, target_columns, rng
    )
    runtime_records["independent_column"] = {
        "training_seconds": None,
        "sampling_seconds": time.perf_counter() - started,
    }
    started = time.perf_counter()
    candidates["empirical_conditional"] = empirical_conditional_baseline(
        train,
        spine,
        model_config.schema.foreign_key_columns,
        model_config.schema.datetime_columns,
        condition_columns,
        target_columns,
        rng,
    )
    runtime_records["empirical_conditional"] = {
        "training_seconds": None,
        "sampling_seconds": time.perf_counter() - started,
    }
    for item in args.model_output:
        name, path = parse_named_path(item)
        candidates[name] = normalize_model_output(
            pd.read_csv(path), spine, condition_columns, target_columns
        )
    for item in args.runtime_metadata:
        name, path = parse_named_path(item, option="--runtime-metadata")
        with path.open(encoding="utf-8") as handle:
            runtime_records[name] = extract_runtime_fields(json.load(handle))

    structured_evaluation_config = restrict_to_structured(
        evaluation_config,
        condition_columns=condition_columns,
        target_columns=target_columns,
    )
    rows: list[dict[str, Any]] = []
    for name, synthetic in candidates.items():
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=False)
        synthetic_path = run_dir / "synthetic_structured.csv"
        synthetic.to_csv(synthetic_path, index=False)
        evaluation_dir = run_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=False)
        config = copy.deepcopy(structured_evaluation_config)
        config["real_table_path"] = str(paths["evaluation_real"]["path"])
        config["synthetic_table_path"] = str(synthetic_path)
        config.setdefault("evaluation", {})["random_seed"] = int(args.seed)
        metrics = evaluate_paper_metrics(config, evaluation_dir)
        write_paper_json(metrics, evaluation_dir / "metrics.json")
        write_markdown_report(metrics, evaluation_dir / "metrics.md")
        summary = metrics.get("paper_metrics_summary") or {}
        row = {
            "generator": name,
            "constraint_violation_rate": summary.get(
                "constraint_violation_rate"
            ),
            "shape_error": summary.get("shape_error"),
            "pairwise_association_error": summary.get("trend_error"),
            "structured_c2st_error": summary.get(
                "single_table_c2st_error"
            ),
            "temporal_trend_error": summary.get("trend_error"),
            "training_seconds": runtime_records.get(name, {}).get(
                "training_seconds"
            ),
            "sampling_seconds": runtime_records.get(name, {}).get(
                "sampling_seconds"
            ),
        }
        for column in model_config.schema.categorical_targets:
            row[f"{column}_tv_distance"] = (
                (metrics.get("shape") or {})
                .get("per_column", {})
                .get(column, {})
                .get("shape_error")
            )
        for foreign_key in model_config.schema.foreign_key_columns:
            for column in target_columns:
                row[f"{foreign_key}_{column}_conditioned_error"] = (
                    foreign_key_conditioned_target_error(
                        real,
                        synthetic,
                        foreign_key=foreign_key,
                        target=column,
                        categorical=(
                            column
                            in model_config.schema.categorical_targets
                        ),
                    )
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "structured_comparison.csv", index=False)
    write_json(
        output_root / "structured_comparison.json",
        frame.where(pd.notna(frame), None).to_dict(orient="records"),
    )
    print(frame.to_string(index=False))
    print(output_root)


def independent_column_baseline(
    train: pd.DataFrame,
    spine: pd.DataFrame,
    condition_columns: list[str],
    target_columns: list[str],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    output = spine.loc[:, condition_columns].copy()
    for column in target_columns:
        values = train[column].dropna().to_numpy(dtype=object)
        if not len(values):
            output[column] = None
        else:
            output[column] = values[
                rng.randint(0, len(values), size=len(output))
            ]
    return output


def empirical_conditional_baseline(
    train: pd.DataFrame,
    spine: pd.DataFrame,
    foreign_keys: tuple[str, ...],
    datetimes: tuple[str, ...],
    condition_columns: list[str],
    target_columns: list[str],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Sample a joint target row through generic FK/time backoff levels."""

    train_keys = train.copy()
    spine_keys = spine.copy()
    time_key = "__temporal_bucket"
    if datetimes:
        column = datetimes[0]
        train_keys[time_key] = pd.to_datetime(
            train_keys[column], errors="coerce"
        ).dt.to_period("M").astype(str)
        spine_keys[time_key] = pd.to_datetime(
            spine_keys[column], errors="coerce"
        ).dt.to_period("M").astype(str)
    levels: list[list[str]] = []
    if datetimes:
        levels.append(list(foreign_keys) + [time_key])
    levels.append(list(foreign_keys))
    for column in foreign_keys:
        levels.append([column, time_key] if datetimes else [column])
    if datetimes:
        levels.append([time_key])
    levels = [level for index, level in enumerate(levels) if level and level not in levels[:index]]
    lookup: list[dict[tuple[str, ...], np.ndarray]] = []
    for level in levels:
        groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for row_index, row in train_keys[level].astype(str).iterrows():
            groups[tuple(row.tolist())].append(int(row_index))
        lookup.append(
            {
                key: np.asarray(indices, dtype=np.int64)
                for key, indices in groups.items()
            }
        )
    all_indices = np.arange(len(train), dtype=np.int64)
    sampled_indices: list[int] = []
    for _, row in spine_keys.iterrows():
        candidates = None
        for level, level_lookup in zip(levels, lookup):
            key = tuple(str(row[column]) for column in level)
            candidates = level_lookup.get(key)
            if candidates is not None and len(candidates):
                break
        if candidates is None or not len(candidates):
            candidates = all_indices
        sampled_indices.append(
            int(candidates[rng.randint(0, len(candidates))])
        )
    output = spine.loc[:, condition_columns].copy()
    sampled = train.iloc[sampled_indices].reset_index(drop=True)
    for column in target_columns:
        output[column] = sampled[column].to_numpy()
    return output


def normalize_model_output(
    output: pd.DataFrame,
    spine: pd.DataFrame,
    condition_columns: list[str],
    target_columns: list[str],
) -> pd.DataFrame:
    missing = [
        column
        for column in condition_columns + target_columns
        if column not in output
    ]
    if missing:
        raise ValueError(f"Model output is missing structured columns: {missing}")
    if len(output) != len(spine):
        raise ValueError(
            f"Model output has {len(output)} rows; expected {len(spine)}"
        )
    normalized = output.loc[:, condition_columns + target_columns].copy()
    for column in condition_columns:
        expected = spine[column].fillna("<missing>").astype(str)
        actual = normalized[column].fillna("<missing>").astype(str)
        if not actual.reset_index(drop=True).equals(
            expected.reset_index(drop=True)
        ):
            raise ValueError(
                f"Model output does not use the fixed evaluation spine: {column}"
            )
    return normalized


def restrict_to_structured(
    config: dict[str, Any],
    *,
    condition_columns: list[str],
    target_columns: list[str],
) -> dict[str, Any]:
    restricted = copy.deepcopy(config)
    allowed = set(condition_columns + target_columns)
    columns = (restricted.setdefault("table", {}).get("columns") or {})
    restricted["table"]["columns"] = {
        column: value
        for column, value in columns.items()
        if column in allowed
    }
    restricted.setdefault("evaluation", {}).setdefault("text", {})[
        "enabled"
    ] = False
    restricted["evaluation"]["text"]["text_columns"] = []
    restricted["legacy_evaluator"] = {"enabled": False}
    return restricted


def foreign_key_conditioned_target_error(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    foreign_key: str,
    target: str,
    categorical: bool,
    min_group_size: int = 2,
) -> float | None:
    """Weighted target error within each entity's fixed-spine rows."""

    required = {foreign_key, target}
    if not required.issubset(real.columns) or not required.issubset(
        synthetic.columns
    ):
        return None
    real_groups = real.groupby(foreign_key, dropna=False, sort=False)
    synthetic_groups = synthetic.groupby(foreign_key, dropna=False, sort=False)
    weighted_errors: list[float] = []
    weights: list[int] = []
    scale = None
    if not categorical:
        numeric = pd.to_numeric(real[target], errors="coerce")
        scale = float(numeric.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
    for entity, real_group in real_groups:
        if len(real_group) < int(min_group_size):
            continue
        try:
            synthetic_group = synthetic_groups.get_group(entity)
        except KeyError:
            continue
        if categorical:
            real_dist = (
                real_group[target]
                .fillna("<missing>")
                .astype(str)
                .value_counts(normalize=True)
            )
            synthetic_dist = (
                synthetic_group[target]
                .fillna("<missing>")
                .astype(str)
                .value_counts(normalize=True)
            )
            support = real_dist.index.union(synthetic_dist.index)
            error = 0.5 * float(
                np.abs(
                    real_dist.reindex(support, fill_value=0.0).to_numpy()
                    - synthetic_dist.reindex(
                        support, fill_value=0.0
                    ).to_numpy()
                ).sum()
            )
        else:
            real_values = pd.to_numeric(real_group[target], errors="coerce")
            synthetic_values = pd.to_numeric(
                synthetic_group[target], errors="coerce"
            )
            if not real_values.notna().any() or not synthetic_values.notna().any():
                continue
            error = abs(
                float(real_values.mean()) - float(synthetic_values.mean())
            ) / float(scale)
        weighted_errors.append(float(error))
        weights.append(int(len(real_group)))
    if not weights:
        return None
    return float(np.average(weighted_errors, weights=weights))


def extract_runtime_fields(metadata: dict[str, Any]) -> dict[str, float | None]:
    return {
        "training_seconds": find_numeric_value(
            metadata,
            (
                "total_training_seconds",
                "training_seconds",
                "wall_clock_seconds",
            ),
        ),
        "sampling_seconds": find_numeric_value(
            metadata,
            (
                "total_sampling_seconds",
                "sampling_seconds",
            ),
        ),
    }


def find_numeric_value(
    value: Any, candidate_keys: tuple[str, ...]
) -> float | None:
    if isinstance(value, dict):
        for key in candidate_keys:
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and np.isfinite(candidate):
                return float(candidate)
        for nested in value.values():
            candidate = find_numeric_value(nested, candidate_keys)
            if candidate is not None:
                return candidate
    elif isinstance(value, list):
        for nested in value:
            candidate = find_numeric_value(nested, candidate_keys)
            if candidate is not None:
                return candidate
    return None


def parse_named_path(
    value: str, *, option: str = "--model-output"
) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"{option} must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name.strip() or not path.exists():
        raise FileNotFoundError(f"Invalid {option} value {value!r}")
    return name.strip(), path


if __name__ == "__main__":
    main()
