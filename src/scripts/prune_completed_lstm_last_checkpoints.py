#!/usr/bin/env python3
"""Prune resumable LSTM checkpoints after complete experiment runs."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


COMPLETION_MARKERS = (
    "checkpoints/best.pt",
    "training_metadata.json",
    "samples/synthetic_interactions.csv",
    "sampling_validation.json",
    "evaluation/paper_grade/metrics.json",
    "evaluation/attribute_diagnostics.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--watch-pid",
        type=int,
        default=None,
        help="Repeat until this process exits, then perform one final pass.",
    )
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    while True:
        summary = prune_completed_runs(root, dry_run=bool(args.dry_run))
        print(
            f"[checkpoint-pruner] removed={summary['removed']} "
            f"freed_gib={summary['freed_bytes'] / 1024**3:.2f} "
            f"incomplete_retained={summary['incomplete_retained']}",
            flush=True,
        )
        if args.watch_pid is None or not process_exists(int(args.watch_pid)):
            break
        time.sleep(max(float(args.interval_seconds), 1.0))


def prune_completed_runs(root: Path, *, dry_run: bool) -> dict[str, int]:
    summary = {
        "removed": 0,
        "freed_bytes": 0,
        "incomplete_retained": 0,
    }
    if not root.exists():
        return summary
    for checkpoint in sorted(root.rglob("checkpoints/last.pt")):
        run_dir = checkpoint.parent.parent
        if not run_is_complete(run_dir):
            summary["incomplete_retained"] += 1
            continue
        size = checkpoint.stat().st_size
        print(
            f"[checkpoint-pruner] {'would remove' if dry_run else 'removing'} "
            f"{checkpoint} ({size / 1024**3:.2f} GiB)",
            flush=True,
        )
        if not dry_run:
            checkpoint.unlink()
        summary["removed"] += 1
        summary["freed_bytes"] += size
    return summary


def run_is_complete(run_dir: Path) -> bool:
    return all((run_dir / relative).is_file() for relative in COMPLETION_MARKERS)


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


if __name__ == "__main__":
    main()
