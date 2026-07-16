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

from tempdir_bootstrap import configure_tempdir  # noqa: E402

configure_tempdir(Path(__file__).resolve().parents[2])

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
    parser.add_argument("--train-row-sampling", choices=["full", "uniform", "temporal_stratified", "temporal_weighted", "hybrid"], default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--steps-per-eval", type=int, default=None)
    parser.add_argument("--steps-per-checkpoint", type=int, default=None)
    parser.add_argument("--validation-max-batches", type=int, default=None)
    parser.add_argument("--epoch-mode", choices=["true", "false"], default=None)
    parser.add_argument("--sampling-mode", choices=["uniform", "temporal_stratified", "temporal_weighted", "hybrid"], default=None)
    parser.add_argument("--effective-batch-size", type=int, default=None)
    parser.add_argument("--target-effective-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--physical-batch-size", type=int, default=None)
    parser.add_argument("--profile-steps", type=int, default=None)
    parser.add_argument("--warmup-profile-steps", type=int, default=None)
    parser.add_argument("--pretokenized-dir", default=None)
    parser.add_argument("--neighbor-cache-dir", default=None)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default=None)
    parser.add_argument("--resume-from", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_with_overrides(args)
    start = time.perf_counter()
    best = train_lstm_from_config(config, device=args.device)
    elapsed = time.perf_counter() - start
    write_training_metadata(config, best, elapsed)
    print(best)


def load_config_with_overrides(args: argparse.Namespace) -> ConditionalTABDLMConfig:
    raw = load_yaml(args.config)
    paths = raw.setdefault("paths", {})
    if "train" in raw:
        raw.setdefault("training", {}).update(raw.get("train", {}) or {})
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
    if args.max_steps is not None:
        training["max_steps"] = int(args.max_steps)
    if args.steps_per_eval is not None:
        training["steps_per_eval"] = int(args.steps_per_eval)
    if args.steps_per_checkpoint is not None:
        training["steps_per_checkpoint"] = int(args.steps_per_checkpoint)
    if args.validation_max_batches is not None:
        training["validation_max_batches"] = int(args.validation_max_batches)
    if args.epoch_mode is not None:
        training["epoch_mode"] = args.epoch_mode == "true"
    if args.sampling_mode is not None:
        training["sampling_mode"] = args.sampling_mode
        training["train_row_sampling"] = args.sampling_mode
    if args.effective_batch_size is not None:
        training["effective_batch_size"] = int(args.effective_batch_size)
    if args.target_effective_batch_size is not None:
        training["target_effective_batch_size"] = int(args.target_effective_batch_size)
    if args.gradient_accumulation_steps is not None:
        training["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if args.physical_batch_size is not None:
        training["physical_batch_size"] = int(args.physical_batch_size)
        training["batch_size"] = int(args.physical_batch_size)
    if args.profile_steps is not None:
        training["profile_steps"] = int(args.profile_steps)
    if args.warmup_profile_steps is not None:
        training["warmup_profile_steps"] = int(args.warmup_profile_steps)
    if args.pretokenized_dir is not None:
        training["pretokenized_dir"] = args.pretokenized_dir
        paths["pretokenized_dir"] = args.pretokenized_dir
    if args.neighbor_cache_dir is not None:
        training["neighbor_cache_dir"] = args.neighbor_cache_dir
        paths["neighbor_cache_dir"] = args.neighbor_cache_dir
    if args.amp_dtype is not None:
        training["amp_dtype"] = args.amp_dtype
    if args.resume_from is not None:
        training["resume_from"] = args.resume_from
    if args.profile:
        training["profile"] = True
    raw = resolve_auto_review_text_config(raw)
    schema = ConditionalTABDLMSchema.from_config_dict(raw)
    return ConditionalTABDLMConfig(raw=raw, schema=schema, config_path=Path(args.config))


def write_training_metadata(config: ConditionalTABDLMConfig, best_path: Path, elapsed: float) -> None:
    training = config.raw.get("training", {})
    runtime_path = config.output_dir / "metadata" / "training_runtime.json"
    runtime = {}
    if runtime_path.exists():
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            runtime = {}
    train_rows_used = prepared_row_count(config.data_dir / "train.parquet")
    if train_rows_used is None:
        train_rows_used = runtime.get("train_rows_available")
    max_train_rows = training.get("max_rows")
    metadata = {
        "dataset_name": config.raw.get("dataset_name"),
        "real_table_path": str(config.train_data_path),
        "synthetic_spine_path": str(config.synthetic_spine_path),
        "architecture": config.raw.get("model_type", "conditional_tabdlm_lstm_joint_full_text"),
        "model_family": config.raw.get("model_family", "conditional_tabdlm_lstm_joint_full_text"),
        "architecture_changed_from_amazon_toy": False,
        "architecture_changed": False,
        "train_mode": runtime.get("train_mode", "epoch"),
        "epoch_mode": bool(training.get("epoch_mode", True)),
        "max_steps": training.get("max_steps"),
        "physical_batch_size": runtime.get("physical_batch_size", training.get("physical_batch_size", training.get("batch_size"))),
        "gradient_accumulation_steps": runtime.get("gradient_accumulation_steps", training.get("gradient_accumulation_steps", 1)),
        "effective_batch_size": runtime.get("effective_batch_size", training.get("effective_batch_size")),
        "sampling_mode": runtime.get("sampling_mode", training.get("sampling_mode", training.get("train_row_sampling", "full"))),
        "train_rows_used": train_rows_used,
        "train_rows_available": runtime.get("train_rows_available", train_rows_used),
        "train_rows_seen_approx": runtime.get("train_rows_seen_approx"),
        "full_epoch_equivalent_fraction": runtime.get("full_epoch_equivalent_fraction"),
        "full_training_used": max_train_rows in (None, "null"),
        "train_subset_used": runtime.get("train_subset_used", max_train_rows not in (None, "null")),
        "max_train_rows": max_train_rows,
        "sampling_strategy": training.get("train_row_sampling", "full"),
        "train_time_seconds": float(elapsed),
        "best_checkpoint_path": str(best_path),
        "mixed_precision_used": runtime.get("mixed_precision_used", training.get("mixed_precision")),
        "amp_dtype": runtime.get("amp_dtype", training.get("amp_dtype", "fp16")),
        "total_training_seconds": runtime.get("total_training_seconds", float(elapsed)),
        "validation_metrics": best_validation_metrics(config.output_dir / "train_log.jsonl"),
    }
    metadata.update({key: value for key, value in runtime.items() if key not in metadata})
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
