from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "scripts" / "pretokenize_single_event_table_text_fields.py"
SPEC = importlib.util.spec_from_file_location("pretokenize_single_event_table_text_fields", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_pretokenization_preserves_explicit_split_labels():
    timestamps = np.arange(8, dtype=np.int64)
    labels = np.asarray(
        ["train", "training", "train", "validation", "val", "test", "test", "test"]
    )

    split = MODULE.split_indices_from_timestamps(timestamps, labels)

    assert split["train"].tolist() == [0, 1, 2]
    assert split["valid"].tolist() == [3, 4]
    assert split["test"].tolist() == [5, 6, 7]


def test_pretokenization_keeps_legacy_time_split_without_labels():
    timestamps = np.arange(100, dtype=np.int64)[::-1]

    split = MODULE.split_indices_from_timestamps(timestamps)

    assert len(split["train"]) == 90
    assert len(split["valid"]) == 5
    assert len(split["test"]) == 5
