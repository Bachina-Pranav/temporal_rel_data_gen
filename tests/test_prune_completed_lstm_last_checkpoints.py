from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.prune_completed_lstm_last_checkpoints import (  # noqa: E402
    COMPLETION_MARKERS,
    normalize_completed_run_config,
    object_sha256,
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


def test_completed_config_normalization_is_hash_guarded(tmp_path: Path):
    run, _ = create_run(tmp_path, complete=True)
    requested = {
        "training": {"seed": 42},
        "paths": {"output_dir": str(run)},
    }
    fitted = {
        **requested,
        "_numerical_metadata": {"price": {"mean": 1.0}},
        "config_path": str(run / "config_resolved.yaml"),
        "numerical_columns": {"price": {"selected_head": "continuous"}},
        "schema_resolved": {"target_columns": {"numerical": ["price"]}},
    }
    fitted["training"] = {
        **requested["training"],
        "pretokenized_dir": "cache/pretokenized",
        "neighbor_cache_dir": "cache/neighbors",
    }
    config = run / "config_resolved.yaml"
    config.write_text(yaml.safe_dump(fitted), encoding="utf-8")
    (run / "run_request.json").write_text(
        json.dumps({"model_request_sha256": object_sha256(requested)}),
        encoding="utf-8",
    )

    assert normalize_completed_run_config(run, dry_run=False) == "normalized"
    assert yaml.safe_load(config.read_text()) == requested
    assert (
        run / "metadata/config_resolved_with_training_runtime.yaml"
    ).is_file()


def test_completed_config_normalization_refuses_hash_mismatch(tmp_path: Path):
    run, _ = create_run(tmp_path, complete=True)
    config = run / "config_resolved.yaml"
    fitted = {"training": {"seed": 42}, "config_path": "runtime.yaml"}
    config.write_text(yaml.safe_dump(fitted), encoding="utf-8")
    original = config.read_text()
    (run / "run_request.json").write_text(
        json.dumps({"model_request_sha256": object_sha256({"different": True})}),
        encoding="utf-8",
    )

    assert normalize_completed_run_config(run, dry_run=False) == "refused"
    assert config.read_text() == original
