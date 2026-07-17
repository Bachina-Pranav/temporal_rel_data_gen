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
try:  # noqa: E402
    from scripts.build_hm_induced_subset import build_hm_induced_subset
except ModuleNotFoundError:  # pragma: no cover - script-file execution fallback
    from build_hm_induced_subset import build_hm_induced_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build induced 100K interaction subsets.")
    parser.add_argument("--datasets", nargs="+", default=None, choices=list_datasets())
    parser.add_argument("--dataset", default=None, choices=list_datasets())
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--processed-root", default="data/processed/interaction_benchmarks")
    parser.add_argument("--target-interactions", type=int, default=100_000)
    parser.add_argument("--num-source-entities", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--allowed-relative-error", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--memory-budget-mb", type=int, default=None)
    parser.add_argument("--temp-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = args.datasets or ([args.dataset] if args.dataset else list_datasets())
    failed = False
    for name in names:
        adapter = get_adapter(name)
        try:
            if args.num_source_entities is not None:
                if adapter.dataset_name != "hm":
                    raise ValueError("--num-source-entities is currently implemented for the H&M complete-history subset")
                manifest = build_hm_induced_subset(
                    raw_root=args.raw_root,
                    processed_root=args.processed_root,
                    output_name=args.output_name or "hm_10k_customers",
                    num_customers=int(args.num_source_entities),
                    seed=args.seed,
                    chunk_size=args.chunk_size,
                )
            else:
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
            dataset_name = str(manifest.get("dataset_name", adapter.benchmark_name))
            print(json.dumps({"dataset_name": dataset_name, "output": str(Path(args.processed_root) / dataset_name), **manifest}, sort_keys=True, default=str))
        except FileNotFoundError as exc:
            failed = True
            print(
                json.dumps(
                    {
                        "dataset_name": adapter.benchmark_name,
                        "status": "missing_raw_data",
                        "message": str(exc),
                        "output": str(Path(args.processed_root) / adapter.benchmark_name),
                    },
                    sort_keys=True,
                )
            )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
