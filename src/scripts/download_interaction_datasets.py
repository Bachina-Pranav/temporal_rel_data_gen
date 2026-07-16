#!/usr/bin/env python3
"""Download or verify raw interaction benchmark datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preprocessing.interaction_datasets.registry import get_adapter, list_datasets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/verify interaction benchmark raw datasets.")
    parser.add_argument("--datasets", nargs="+", default=None, choices=list_datasets())
    parser.add_argument("--dataset", default=None, choices=list_datasets())
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--archive", default=None, help="Local archive for one dataset, especially Yelp.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = args.datasets or ([args.dataset] if args.dataset else list_datasets())
    if args.archive and len(names) != 1:
        raise SystemExit("--archive can only be used with exactly one dataset")
    results = []
    for name in names:
        adapter = get_adapter(name)
        result = adapter.download(
            args.raw_root,
            force=args.force,
            verify_only=args.verify_only,
            archive=args.archive,
        )
        payload = {
            "dataset_name": result.dataset_name,
            "status": result.status,
            "raw_dir": str(result.raw_dir),
            "message": result.message,
            "metadata": result.metadata,
        }
        print(json.dumps(payload, sort_keys=True))
        results.append(payload)
    blocked = [row for row in results if str(row["status"]).startswith("blocked")]
    if blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
