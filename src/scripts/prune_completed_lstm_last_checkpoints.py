#!/usr/bin/env python3
"""Prune resumable LSTM checkpoints after complete experiment runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import yaml


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
    parser.add_argument(
        "--normalize-reuse-config",
        action="store_true",
        help=(
            "Archive and remove training-added runtime fields from completed "
            "config_resolved.yaml files, but only when the normalized hash "
            "matches the immutable model_request_sha256."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    while True:
        summary = maintain_completed_runs(
            root,
            dry_run=bool(args.dry_run),
            normalize_reuse_config=bool(args.normalize_reuse_config),
        )
        print(
            f"[checkpoint-pruner] removed={summary['removed']} "
            f"freed_gib={summary['freed_bytes'] / 1024**3:.2f} "
            f"configs_normalized={summary['configs_normalized']} "
            f"normalization_refused={summary['normalization_refused']} "
            f"incomplete_retained={summary['incomplete_retained']}",
            flush=True,
        )
        if args.watch_pid is None or not process_exists(int(args.watch_pid)):
            break
        time.sleep(max(float(args.interval_seconds), 1.0))


def maintain_completed_runs(
    root: Path,
    *,
    dry_run: bool,
    normalize_reuse_config: bool,
) -> dict[str, int]:
    summary = {
        "removed": 0,
        "freed_bytes": 0,
        "configs_normalized": 0,
        "normalization_refused": 0,
        "incomplete_retained": 0,
    }
    if not root.exists():
        return summary
    run_dirs = sorted(
        path.parent.parent
        for path in root.rglob("checkpoints/best.pt")
    )
    for run_dir in run_dirs:
        if not run_is_complete(run_dir):
            summary["incomplete_retained"] += 1
            continue
        if normalize_reuse_config:
            status = normalize_completed_run_config(run_dir, dry_run=dry_run)
            if status == "normalized":
                summary["configs_normalized"] += 1
            elif status == "refused":
                summary["normalization_refused"] += 1
        checkpoint = run_dir / "checkpoints/last.pt"
        if checkpoint.is_file():
            size = checkpoint.stat().st_size
            print(
                f"[checkpoint-pruner] "
                f"{'would remove' if dry_run else 'removing'} "
                f"{checkpoint} ({size / 1024**3:.2f} GiB)",
                flush=True,
            )
            if not dry_run:
                checkpoint.unlink()
            summary["removed"] += 1
            summary["freed_bytes"] += size
    return summary


def prune_completed_runs(root: Path, *, dry_run: bool) -> dict[str, int]:
    """Retain the original one-shot checkpoint-pruning API."""

    return maintain_completed_runs(
        root,
        dry_run=dry_run,
        normalize_reuse_config=False,
    )


def normalize_completed_run_config(run_dir: Path, *, dry_run: bool) -> str:
    config_path = run_dir / "config_resolved.yaml"
    request_path = run_dir / "run_request.json"
    if not config_path.is_file() or not request_path.is_file():
        return "unchanged"
    raw = load_yaml(config_path)
    normalized, removed = strip_training_runtime_fields(raw)
    if not removed:
        return "unchanged"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    comparable = copy.deepcopy(normalized)
    comparable.pop("_numerical_head_metadata", None)
    comparable.pop("_categorical_head_metadata", None)
    actual_hash = object_sha256(comparable)
    expected_hash = str(request.get("model_request_sha256", ""))
    if actual_hash != expected_hash:
        print(
            f"[checkpoint-pruner] refusing config normalization for "
            f"{run_dir}: normalized hash {actual_hash} does not match "
            f"saved request {expected_hash}",
            flush=True,
        )
        return "refused"

    archive = run_dir / "metadata/config_resolved_with_training_runtime.yaml"
    audit = run_dir / "metadata/reuse_config_normalization.json"
    print(
        f"[checkpoint-pruner] "
        f"{'would normalize' if dry_run else 'normalizing'} "
        f"{config_path}; archived runtime fields={removed}",
        flush=True,
    )
    if not dry_run:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_text(
                yaml.safe_dump(raw, sort_keys=False),
                encoding="utf-8",
            )
        temporary = config_path.with_suffix(".yaml.tmp")
        temporary.write_text(
            yaml.safe_dump(normalized, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(config_path)
        audit.write_text(
            json.dumps(
                {
                    "archive": str(archive),
                    "model_request_sha256": expected_hash,
                    "normalized_model_sha256": actual_hash,
                    "removed_runtime_fields": removed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return "normalized"


def strip_training_runtime_fields(
    raw: dict,
) -> tuple[dict, list[str]]:
    normalized = copy.deepcopy(raw)
    removed = []
    for key in (
        "_numerical_metadata",
        "config_path",
        "numerical_columns",
        "schema_resolved",
    ):
        if key in normalized:
            normalized.pop(key)
            removed.append(key)
    training = normalized.get("training")
    if isinstance(training, dict):
        for key in ("neighbor_cache_dir", "pretokenized_dir"):
            if key in training:
                training.pop(key)
                removed.append(f"training.{key}")
    return normalized, removed


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def object_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
