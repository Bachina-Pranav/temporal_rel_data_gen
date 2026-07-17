#!/usr/bin/env python3
"""Train hierarchical v4.1 Conditional TABDLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.hierarchical_train import train_hierarchical_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_hierarchical_v41.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical v4.1 Conditional TABDLM.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pretokenized-dir", default=None)
    parser.add_argument("--neighbor-cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config.raw.setdefault("paths", {})
    training = config.raw.setdefault("training", {})
    if args.output_dir is not None:
        paths["output_dir"] = args.output_dir
    if args.pretokenized_dir is not None:
        paths["pretokenized_dir"] = args.pretokenized_dir
        training["pretokenized_dir"] = args.pretokenized_dir
    if args.neighbor_cache_dir is not None:
        paths["neighbor_cache_dir"] = args.neighbor_cache_dir
        training["neighbor_cache_dir"] = args.neighbor_cache_dir
    if args.epochs is not None:
        training["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        training["num_workers"] = int(args.num_workers)
    if args.prefetch_factor is not None:
        training["prefetch_factor"] = int(args.prefetch_factor)
    if args.persistent_workers is not None:
        training["persistent_workers"] = bool(args.persistent_workers)
    if args.pin_memory is not None:
        training["pin_memory"] = bool(args.pin_memory)
    if args.fused_adamw:
        training["fused_adamw"] = True
    if args.compile_model:
        training["compile_model"] = True
    if args.profile:
        training["profile"] = True
    if args.skip_checkpoints:
        training["skip_checkpoints"] = True
    if args.max_train_batches is not None:
        training["max_train_batches"] = int(args.max_train_batches)
    if args.max_valid_batches is not None:
        training["max_valid_batches"] = int(args.max_valid_batches)
    train_hierarchical_from_config(config, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
