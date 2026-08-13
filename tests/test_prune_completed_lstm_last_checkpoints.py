from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.prune_completed_lstm_last_checkpoints import (  # noqa: E402
    COMPLETION_MARKERS,
    prune_completed_runs,
)


def create_run(root: Path, *, complete: bool) -> tuple[Path, Path]:
    run = root / "runs/seed_42"
    last = run / "checkpoints/last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"resumable")
    (run / "checkpoints/best.pt").write_bytes(b"best")
    if complete:
        for relative in COMPLETION_MARKERS:
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("{}\n", encoding="utf-8")
    return run, last


def test_pruner_removes_last_only_from_complete_runs(tmp_path: Path):
    complete_run, complete_last = create_run(tmp_path / "complete", complete=True)
    _, incomplete_last = create_run(tmp_path / "incomplete", complete=False)

    summary = prune_completed_runs(tmp_path, dry_run=False)

    assert summary["removed"] == 1
    assert not complete_last.exists()
    assert (complete_run / "checkpoints/best.pt").read_bytes() == b"best"
    assert incomplete_last.is_file()


def test_pruner_dry_run_does_not_remove_checkpoint(tmp_path: Path):
    _, last = create_run(tmp_path, complete=True)

    summary = prune_completed_runs(tmp_path, dry_run=True)

    assert summary["removed"] == 1
    assert last.is_file()
