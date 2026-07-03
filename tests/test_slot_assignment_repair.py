from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.fast_temporal_activity import FastTemporalActivityModel  # noqa: E402
from generators.slot_assignment_repair import repair_entity_degrees_by_replacement  # noqa: E402


def test_degree_repair_exact_preserves_blocks():
    frame = pd.DataFrame(
        {
            "entity": ["a", "a", "b", "c"],
            "time": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
        }
    )
    blocks = {"a": 0, "b": 0, "c": 1}
    activity = FastTemporalActivityModel(alpha="auto").fit(frame, "entity", "time", blocks)
    slot_ids = np.asarray(["a", "a", "a", "c"], dtype=object)
    slot_blocks = np.asarray([0, 0, 0, 1], dtype=int)
    slot_times = np.asarray(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"], dtype=object)
    target = {"a": 2, "b": 1, "c": 1}

    repaired, summary = repair_entity_degrees_by_replacement(
        slot_ids,
        slot_blocks,
        slot_times,
        target,
        blocks,
        activity,
        np.random.default_rng(1),
    )

    counts = pd.Series(repaired).value_counts().to_dict()
    assert counts == target
    assert all(blocks[entity] == block for entity, block in zip(repaired, slot_blocks))
    assert summary["l1_error_after"] == 0
    assert summary["num_replacements"] == 1


def test_product_degree_repair_exact_preserves_blocks():
    frame = pd.DataFrame(
        {
            "entity": ["p0", "p0", "p1", "p2"],
            "time": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
        }
    )
    blocks = {"p0": 0, "p1": 0, "p2": 1}
    activity = FastTemporalActivityModel(alpha="auto", entity_kind="product").fit(frame, "entity", "time", blocks)
    slot_ids = np.asarray(["p0", "p0", "p0", "p2"], dtype=object)
    slot_blocks = np.asarray([0, 0, 0, 1], dtype=int)
    slot_times = np.asarray(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"], dtype=object)
    target = {"p0": 2, "p1": 1, "p2": 1}

    repaired, summary = repair_entity_degrees_by_replacement(
        slot_ids,
        slot_blocks,
        slot_times,
        target,
        blocks,
        activity,
        np.random.default_rng(3),
    )

    counts = pd.Series(repaired).value_counts().to_dict()
    assert counts == target
    assert all(blocks[entity] == block for entity, block in zip(repaired, slot_blocks))
    assert summary["l1_error_after"] == 0
