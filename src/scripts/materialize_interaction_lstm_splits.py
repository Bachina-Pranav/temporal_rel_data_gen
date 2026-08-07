#!/usr/bin/env python3
"""Write split-specific real tables and event spines for schema-driven LSTM runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.dataset import normalize_frame  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import normalize_text  # noqa: E402


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "validation",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize train/validation/test LSTM spines from an interaction table.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--table", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir / "spines"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = Path(args.table) if args.table else config.train_data_path
    frame = pd.read_csv(table_path)
    frame = normalize_frame(frame, config.schema)
    for column in config.schema.text_targets:
        frame = frame[frame[column].map(normalize_text).str.len() > 0]
    frame = frame.dropna(
        subset=list(config.schema.condition_columns)
    ).reset_index(drop=True)
    timestamp_col = config.schema.datetime_columns[0]
    labels, split_source = resolve_split_labels(frame, timestamp_col)
    frame = frame.assign(split=labels)
    sort_cols = [timestamp_col]
    if "event_id" in frame.columns:
        sort_cols.append("event_id")
    summary: dict[str, Any] = {
        "source_table": str(table_path),
        "split_source": split_source,
        "splits": {},
    }
    spine_frames: dict[str, pd.DataFrame] = {}
    for split_name in ["train", "validation", "test"]:
        split = frame.loc[frame["split"] == split_name].copy()
        split[timestamp_col] = pd.to_datetime(split[timestamp_col], errors="coerce")
        split = split.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        real_cols = [column for column in ["event_id", *config.schema.condition_columns, *config.schema.target_columns] if column in split.columns]
        spine_cols = [column for column in ["event_id", *config.schema.condition_columns] if column in split.columns]
        real_path = output_dir / f"{split_name}_real.csv"
        spine_path = output_dir / f"{split_name}_spine.csv"
        split.loc[:, real_cols].to_csv(real_path, index=False)
        split.loc[:, spine_cols].to_csv(spine_path, index=False)
        spine_frames[split_name] = split.loc[:, spine_cols].copy()
        timestamps = pd.to_datetime(split[timestamp_col], errors="coerce")
        summary["splits"][split_name] = {
            "rows": int(len(split)),
            "real_path": str(real_path),
            "spine_path": str(spine_path),
            "timestamp_min": timestamps.min().isoformat() if len(timestamps) else None,
            "timestamp_max": timestamps.max().isoformat() if len(timestamps) else None,
        }
        print(f"Wrote {real_path}")
        print(f"Wrote {spine_path}")
    history_prefix_path = output_dir / "history_prefix_spine.csv"
    history_prefix = pd.concat(
        [spine_frames["train"], spine_frames["validation"]],
        ignore_index=True,
    )
    history_prefix.to_csv(history_prefix_path, index=False)
    summary["history_prefix_spine_path"] = str(history_prefix_path)
    summary["history_prefix_rows"] = int(len(history_prefix))
    print(f"Wrote {history_prefix_path}")
    summary_path = output_dir / "split_spines_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


def resolve_split_labels(
    frame: pd.DataFrame,
    timestamp_col: str,
) -> tuple[pd.Series, str]:
    """Use explicit labels or the legacy chronological 90/5/5 policy."""

    if "split" in frame.columns:
        labels = (
            frame["split"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(SPLIT_ALIASES)
        )
        unknown = sorted(
            set(frame.loc[labels.isna(), "split"].astype(str))
        )
        if unknown:
            raise ValueError(f"Unknown split labels: {unknown}")
        return labels, "explicit_split_column"

    timestamps = pd.to_datetime(
        frame[timestamp_col],
        errors="coerce",
        utc=True,
    )
    if timestamps.isna().any():
        raise ValueError(
            f"Cannot derive chronological splits: {timestamp_col!r} "
            f"has {int(timestamps.isna().sum())} invalid values"
        )
    order = timestamps.sort_values(kind="mergesort").index.to_numpy()
    train_end = int(len(frame) * 0.90)
    validation_end = int(len(frame) * 0.95)
    labels = pd.Series(index=frame.index, dtype="object")
    labels.loc[order[:train_end]] = "train"
    labels.loc[order[train_end:validation_end]] = "validation"
    labels.loc[order[validation_end:]] = "test"
    return labels, "legacy_time_aware_90_5_5"


if __name__ == "__main__":
    main()
