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
    parser.add_argument("--relbench-root", default="data/original")
    parser.add_argument("--processed-root", default="data/processed/interaction_benchmarks")
    parser.add_argument("--target-interactions", type=int, default=100_000)
    parser.add_argument("--num-source-entities", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--allowed-relative-error", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--memory-budget-mb", type=int, default=None)
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--archive", default=None, help="Optional local archive to extract if raw files are missing. Use with one dataset.")
    parser.add_argument("--force-download", action="store_true", help="Force adapter download when raw files are missing.")
    parser.add_argument("--no-download", action="store_true", help="Fail immediately if raw files are missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = args.datasets or ([args.dataset] if args.dataset else list_datasets())
    if args.archive and len(names) != 1:
        raise SystemExit("--archive can only be used with exactly one dataset")
    failed = False
    for name in names:
        adapter = get_adapter(name)
        try:
            if args.num_source_entities is not None:
                if adapter.dataset_name != "hm":
                    raise ValueError("--num-source-entities is currently implemented for the H&M complete-history subset")
                manifest = build_hm_induced_subset(
                    raw_root=args.raw_root,
                    relbench_root=args.relbench_root,
                    processed_root=args.processed_root,
                    output_name=args.output_name or "hm_10k_customers",
                    num_customers=int(args.num_source_entities),
                    seed=args.seed,
                    chunk_size=args.chunk_size,
                    archive=args.archive,
                    force_download=bool(args.force_download),
                    download_if_missing=not bool(args.no_download),
                )
            else:
                ensure_raw_files(
                    adapter,
                    raw_root=args.raw_root,
                    archive=args.archive,
                    force_download=bool(args.force_download),
                    download_if_missing=not bool(args.no_download),
                )
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
            dataset_name = args.output_name or ("hm_10k_customers" if args.num_source_entities is not None and adapter.dataset_name == "hm" else adapter.benchmark_name)
            print(
                json.dumps(
                    {
                        "dataset_name": dataset_name,
                        "status": "missing_raw_data",
                        "message": str(exc),
                        "output": str(Path(args.processed_root) / dataset_name),
                    },
                    sort_keys=True,
                )
            )
    if failed:
        raise SystemExit(1)


def ensure_raw_files(
    adapter,
    *,
    raw_root: str | Path,
    archive: str | Path | None,
    force_download: bool,
    download_if_missing: bool,
) -> None:
    try:
        adapter.locate_raw_files(raw_root)
        return
    except FileNotFoundError as first_error:
        if not download_if_missing:
            raise
        print(
            json.dumps(
                {
                    "dataset_name": adapter.dataset_name,
                    "status": "raw_missing_attempting_download",
                    "message": str(first_error),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        result = adapter.download(raw_root, force=force_download, archive=archive)
        print(
            json.dumps(
                {
                    "dataset_name": result.dataset_name,
                    "download_status": result.status,
                    "raw_dir": str(result.raw_dir),
                    "message": result.message,
                    "metadata": result.metadata,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            adapter.locate_raw_files(raw_root)
        except FileNotFoundError as second_error:
            message = (
                f"{second_error}. Attempted automatic download/extract after: {first_error}. "
                f"Download status={result.status!r}. Message={result.message!r}. "
                "Configure dataset credentials/license access or pass --archive /path/to/local_archive."
            )
            raise FileNotFoundError(message) from second_error


if __name__ == "__main__":
    main()
