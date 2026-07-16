#!/usr/bin/env python3
"""Build source-entity-induced interaction benchmark subsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preprocessing.interaction_datasets.registry import get_adapter, list_datasets  # noqa: E402
from data_preprocessing.interaction_datasets.subset import build_interaction_subset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build induced 100K interaction subsets.")
    parser.add_argument("--datasets", nargs="+", default=None, choices=list_datasets())
    parser.add_argument("--dataset", default=None, choices=list_datasets())
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--processed-root", default="data/processed/interaction_benchmarks")
    parser.add_argument("--target-interactions", type=int, default=100_000)
    parser.add_argument("--allowed-relative-error", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--memory-budget-mb", type=int, default=None)
    parser.add_argument("--temp-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = args.datasets or ([args.dataset] if args.dataset else list_datasets())
    for name in names:
        adapter = get_adapter(name)
        manifest = build_interaction_subset(
            adapter,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            target_interactions=args.target_interactions,
            allowed_relative_error=args.allowed_relative_error,
            seed=args.seed,
            chunk_size=args.chunk_size,
            memory_budget_mb=args.memory_budget_mb,
            temp_dir=args.temp_dir,
        )
        print(json.dumps({"dataset_name": adapter.benchmark_name, "output": str(Path(args.processed_root) / adapter.benchmark_name), **manifest}, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
