#!/usr/bin/env python3
"""Train the joint LSTM full-review-text generator."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.lstm_joint import train_lstm_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMConfig,
    ConditionalTABDLMSchema,
    resolve_auto_review_text_config,
)
from attribute_generation.conditional_tabdlm.utils import load_yaml, save_json  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp5_lstm_joint_full_review_text.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train joint LSTM full-review-text generator.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--real-table", default=None)
    parser.add_argument("--synthetic-spine", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mixed-precision", action="store_true", default=None)
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--auto-batch-size", action="store_true", default=None)
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--train-row-sampling", choices=["full", "uniform", "temporal_stratified"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_with_overrides(args)
    if config.raw.get("training", {}).get("train_row_sampling") == "temporal_stratified":
        raise SystemExit("--train-row-sampling temporal_stratified is not implemented for this trainer yet; use full or uniform.")
    start = time.perf_counter()
    best = train_lstm_from_config(config, device=args.device)
    elapsed = time.perf_counter() - start
    write_training_metadata(config, best, elapsed)
    print(best)


def load_config_with_overrides(args: argparse.Namespace) -> ConditionalTABDLMConfig:
    raw = load_yaml(args.config)
    paths = raw.setdefault("paths", {})
    training = raw.setdefault("training", {})
    if args.real_table:
        paths["train_data_path"] = args.real_table
    if args.synthetic_spine:
        paths["synthetic_spine_path"] = args.synthetic_spine
    if args.output_dir:
        paths["output_dir"] = args.output_dir
    if args.mixed_precision is not None:
        training["mixed_precision"] = bool(args.mixed_precision)
    if args.auto_batch_size is not None:
        training["auto_reduce_batch_size"] = bool(args.auto_batch_size)
    if args.num_workers is not None:
        training["num_workers"] = int(args.num_workers)
    if args.max_train_rows is not None:
        training["max_rows"] = int(args.max_train_rows)
    if args.train_row_sampling is not None:
        training["train_row_sampling"] = args.train_row_sampling
    raw = resolve_auto_review_text_config(raw)
    schema = ConditionalTABDLMSchema.from_config_dict(raw)
    return ConditionalTABDLMConfig(raw=raw, schema=schema, config_path=Path(args.config))


def write_training_metadata(config: ConditionalTABDLMConfig, best_path: Path, elapsed: float) -> None:
    training = config.raw.get("training", {})
    train_rows_used = prepared_row_count(config.data_dir / "train.parquet")
    max_train_rows = training.get("max_rows")
    metadata = {
        "dataset_name": config.raw.get("dataset_name"),
        "architecture": config.raw.get("model_type", "conditional_tabdlm_lstm_joint_full_text"),
        "architecture_changed_from_amazon_toy": False,
        "train_rows_used": train_rows_used,
        "full_training_used": max_train_rows in (None, "null"),
        "train_subset_used": max_train_rows not in (None, "null"),
        "max_train_rows": max_train_rows,
        "sampling_strategy": training.get("train_row_sampling", "full"),
        "train_time_seconds": float(elapsed),
        "best_checkpoint_path": str(best_path),
        "validation_metrics": best_validation_metrics(config.output_dir / "train_log.jsonl"),
    }
    save_json(metadata, config.output_dir / "training_metadata.json")


def prepared_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(__import__("pandas").read_parquet(path, columns=[])))
    except Exception:
        try:
            return int(len(__import__("pandas").read_pickle(path)))
        except Exception:
            return None


def best_validation_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return {}
    return min(rows, key=lambda row: float(row.get("valid_total_loss", row.get("best_valid_total_loss", float("inf")))))


if __name__ == "__main__":
    main()
