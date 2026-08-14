#!/usr/bin/env python3
"""Train the joint LSTM full-review-text generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tempdir_bootstrap import configure_tempdir  # noqa: E402

configure_tempdir(Path(__file__).resolve().parents[2])

from attribute_generation.conditional_tabdlm.lstm_joint import train_lstm_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.numerical_head import numerical_head_config  # noqa: E402
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
    parser.add_argument("--seed", type=int, default=None)
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
    validation_max_batches = getattr(args, "validation_max_batches", None)
    if validation_max_batches is not None:
        training["validation_max_batches"] = int(validation_max_batches)
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
    seed = getattr(args, "seed", None)
    if seed is not None:
        training["seed"] = int(seed)
    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        training["resume_from"] = resume_from
    if args.profile:
        training["profile"] = True
    raw = resolve_auto_review_text_config(raw)
    schema = ConditionalTABDLMSchema.from_config_dict(raw)
    return ConditionalTABDLMConfig(raw=raw, schema=schema, config_path=Path(args.config))


def write_training_metadata(config: ConditionalTABDLMConfig, best_path: Path, elapsed: float) -> None:
    training = config.raw.get("training", {})
    architecture_changed = bool(
        (config.raw.get("experiment_metadata") or {}).get(
            "baseline_architecture_changed",
            False,
        )
    )
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
        "architecture_changed_from_amazon_toy": (
            architecture_changed
        ),
        "architecture_changed": architecture_changed,
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
        "validation_metrics": best_validation_metrics(
            config.output_dir / "train_log.jsonl",
            config,
        ),
        "dataset_fingerprint_sha256": sha256_file(config.train_data_path),
        "configuration_fingerprint_sha256": sha256_json(config.to_dict()),
        "git_commit": git_revision(),
        "hardware": hardware_metadata(),
    }
    metadata.update({key: value for key, value in runtime.items() if key not in metadata})
    save_json(metadata, config.output_dir / "training_metadata.json")


def hardware_metadata() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "cuda_device_count": (
            int(torch.cuda.device_count()) if cuda_available else 0
        ),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if cuda_available
            else None
        ),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
    }


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


def best_validation_metrics(
    path: Path,
    config: ConditionalTABDLMConfig,
) -> dict[str, Any]:
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
    selection = (
        numerical_head_config(config.raw).get(
            "validation_selection"
        )
        or {}
    )
    metric = str(selection.get("metric", "total_loss"))
    if metric == "numerical_composite":
        metric = "numerical_validation_composite"
    key = f"valid_{metric}"
    return min(
        rows,
        key=lambda row: float(
            row.get(
                key,
                row.get(
                    "validation_selection_value",
                    row.get(
                        "valid_total_loss",
                        row.get(
                            "best_valid_total_loss",
                            float("inf"),
                        ),
                    ),
                ),
            )
        ),
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
