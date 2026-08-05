#!/usr/bin/env python3
"""Materialize a fixed benchmark for hierarchical diffusion diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.dataset import load_prepared_tables  # noqa: E402
from attribute_generation.conditional_tabdlm.diffusion_diagnostics import (  # noqa: E402
    current_git_commit,
    dataframe_fingerprint,
    file_sha256,
    write_json,
)
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_dataset import (  # noqa: E402
    build_temporal_history_index,
)
from attribute_generation.conditional_tabdlm.graph_schema import (  # noqa: E402
    graph_conditioning_enabled,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze train/validation/test rows for diffusion diagnostics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--num-evaluation-rows",
        default="all",
        help="Use all test rows or a positive integer prefix of the time-sorted test split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace benchmark files in an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    prepare_fixed_benchmark(
        config_path=config_path,
        output_dir=output_dir,
        num_evaluation_rows=args.num_evaluation_rows,
        seed=int(args.seed),
        force=bool(args.force),
    )


def prepare_fixed_benchmark(
    *,
    config_path: Path,
    output_dir: Path,
    num_evaluation_rows: str | int,
    seed: int,
    force: bool = False,
) -> dict[str, Any]:
    manifest_path = output_dir / "benchmark_manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"Benchmark already exists at {manifest_path}. "
            "Use --force only when intentionally replacing it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    train, validation, test = load_prepared_tables(config)
    evaluation_real = select_evaluation_rows(test, num_evaluation_rows)
    condition_columns = list(config.schema.condition_columns)
    evaluation_spine = evaluation_real.loc[:, condition_columns].copy()
    history_prefix = pd.concat(
        [
            train.loc[:, condition_columns],
            validation.loc[:, condition_columns],
        ],
        ignore_index=True,
    )

    frames = {
        "train_real": train,
        "validation_real": validation,
        "test_real": test,
        "evaluation_real": evaluation_real,
        "evaluation_spine": evaluation_spine,
        "graph_history_prefix": history_prefix,
    }
    if graph_conditioning_enabled(config.raw):
        graph_frame = pd.concat(
            [history_prefix, evaluation_spine], ignore_index=True
        )
        graph_index = build_temporal_history_index(
            graph_frame, config, seed=int(seed)
        )
        query_start = int(len(history_prefix))
        coverage = graph_index.coverage_frame_for_rows(
            list(range(query_start, query_start + len(evaluation_spine)))
        )
        coverage.insert(
            0,
            "evaluation_row_index",
            range(len(evaluation_spine)),
        )
        frames["evaluation_history_coverage"] = coverage
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    prepared_files = sorted(config.data_dir.glob("*"))
    tokenizer_path = config.data_dir / "tokenizer_metadata.json"
    manifest = {
        "benchmark_version": 2,
        "dataset_name": config.raw.get("dataset_name"),
        "model_config_path": str(config_path),
        "model_config_sha256": file_sha256(config_path),
        "source_table_path": str(config.train_data_path),
        "source_table_sha256": (
            file_sha256(config.train_data_path)
            if config.train_data_path.exists()
            else None
        ),
        "git_commit": current_git_commit(),
        "seed": int(seed),
        "evaluation_row_selection": {
            "method": "time_sorted_test_prefix",
            "requested": str(num_evaluation_rows),
            "selected": int(len(evaluation_real)),
        },
        "condition_columns": condition_columns,
        "target_columns": list(config.schema.target_columns),
        "text_max_lengths": dict(config.schema.text_max_lengths),
        "split_integrity": split_integrity_metadata(
            train,
            validation,
            evaluation_real,
            datetime_columns=list(config.schema.datetime_columns),
        ),
        "row_counts": {
            name: int(len(frame)) for name, frame in frames.items()
        },
        "dataframe_fingerprints": {
            name: dataframe_fingerprint(frame)
            for name, frame in frames.items()
        },
        "files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": file_sha256(tokenizer_path),
        },
        "prepared_artifacts": {
            str(path): file_sha256(path)
            for path in prepared_files
            if path.is_file()
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def select_evaluation_rows(
    test: pd.DataFrame, requested: str | int
) -> pd.DataFrame:
    if requested is None or str(requested).strip().lower() == "all":
        return test.reset_index(drop=True)
    count = int(requested)
    if count <= 0:
        raise ValueError("--num-evaluation-rows must be 'all' or a positive integer")
    if count > len(test):
        raise ValueError(
            f"Requested {count} evaluation rows but test split has only {len(test)}"
        )
    return test.iloc[:count].reset_index(drop=True)


def split_integrity_metadata(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    datetime_columns: list[str],
) -> dict[str, Any]:
    common = [
        column
        for column in train.columns
        if column in validation.columns and column in evaluation.columns
    ]
    train_hashes = set(
        pd.util.hash_pandas_object(
            train.loc[:, common], index=False
        ).tolist()
    )
    validation_hashes = set(
        pd.util.hash_pandas_object(
            validation.loc[:, common], index=False
        ).tolist()
    )
    evaluation_hashes = set(
        pd.util.hash_pandas_object(
            evaluation.loc[:, common], index=False
        ).tolist()
    )
    result: dict[str, Any] = {
        "full_row_content_overlap": {
            "train_evaluation_unique_hashes": int(
                len(train_hashes.intersection(evaluation_hashes))
            ),
            "validation_evaluation_unique_hashes": int(
                len(validation_hashes.intersection(evaluation_hashes))
            ),
            "interpretation": (
                "Content-hash overlap is reported as an audit signal; repeated "
                "events may be legitimate and are not treated as row identity."
            ),
        }
    }
    if not datetime_columns:
        result["temporal_separation"] = {
            "status": "not_applicable",
            "reason": "schema_has_no_datetime_condition",
        }
        return result
    timestamp_column = datetime_columns[0]
    ranges: dict[str, dict[str, str | None]] = {}
    parsed_by_split: dict[str, pd.Series] = {}
    for name, frame in (
        ("train", train),
        ("validation", validation),
        ("evaluation", evaluation),
    ):
        parsed = pd.to_datetime(
            frame[timestamp_column], errors="coerce", utc=True
        )
        if parsed.isna().any():
            raise ValueError(
                f"Invalid {timestamp_column!r} values in benchmark {name} split"
            )
        parsed_by_split[name] = parsed
        ranges[name] = {
            "min": parsed.min().isoformat() if len(parsed) else None,
            "max": parsed.max().isoformat() if len(parsed) else None,
        }
    ordered = True
    if len(parsed_by_split["train"]) and len(parsed_by_split["validation"]):
        ordered = ordered and (
            parsed_by_split["train"].max()
            <= parsed_by_split["validation"].min()
        )
    if len(parsed_by_split["validation"]) and len(
        parsed_by_split["evaluation"]
    ):
        ordered = ordered and (
            parsed_by_split["validation"].max()
            <= parsed_by_split["evaluation"].min()
        )
    result["temporal_separation"] = {
        "status": "passed" if ordered else "failed",
        "timestamp_column": timestamp_column,
        "ranges": ranges,
        "nondecreasing_split_boundaries": bool(ordered),
    }
    return result


if __name__ == "__main__":
    main()
