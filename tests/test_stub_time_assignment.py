from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.fast_temporal_activity import FastTemporalActivityModel  # noqa: E402
from generators.stub_time_assignment import assign_stubs_to_slots_by_time  # noqa: E402


def activity_model() -> FastTemporalActivityModel:
    frame = pd.DataFrame(
        {
            "customer_id": ["early"] * 80 + ["early"] * 3 + ["late"] * 3 + ["late"] * 80,
            "review_time": ["2020-01-01"] * 80
            + ["2020-01-10"] * 3
            + ["2020-01-01"] * 3
            + ["2020-01-10"] * 80,
        }
    )
    return FastTemporalActivityModel(alpha=0.1).fit(
        frame,
        "customer_id",
        "review_time",
        {"early": 0, "late": 0},
    )


def test_assign_stubs_to_slots_by_time_preserves_degrees_and_blocks():
    entity_ids = ["c0", "c1", "c2"]
    degrees = {"c0": 2, "c1": 1, "c2": 3}
    blocks = {"c0": 0, "c1": 0, "c2": 1}
    slots = np.asarray([0, 0, 0, 1, 1, 1])
    slot_times = np.asarray([0, 1, 2, 0, 1, 2])
    model = FastTemporalActivityModel(alpha=1.0).fit(
        pd.DataFrame(
            {
                "entity": ["c0", "c0", "c1", "c2", "c2", "c2"],
                "review_time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01", "2020-01-02", "2020-01-03"],
            }
        ),
        "entity",
        "review_time",
        blocks,
    )

    assigned, summary = assign_stubs_to_slots_by_time(
        entity_ids,
        degrees,
        blocks,
        slots,
        slot_times,
        model,
        np.random.default_rng(3),
    )

    assert len(assigned) == len(slots)
    assert Counter(assigned.tolist()) == Counter(degrees)
    assert all(blocks[entity] == block for entity, block in zip(assigned, slots))
    assert summary["exact_degree_preserved"] is True


def test_assign_stubs_to_slots_by_time_raises_on_block_mismatch():
    frame = pd.DataFrame({"entity": ["c0", "c0"], "review_time": ["2020-01-01", "2020-01-02"]})
    model = FastTemporalActivityModel(alpha=1.0).fit(frame, "entity", "review_time", {"c0": 0})
    with pytest.raises(ValueError, match="Stub/slot mismatch"):
        assign_stubs_to_slots_by_time(
            ["c0"],
            {"c0": 2},
            {"c0": 0},
            np.asarray([0]),
            np.asarray([0]),
            model,
            np.random.default_rng(0),
        )


def test_assign_stubs_to_slots_by_time_is_time_biased():
    entity_ids = ["early", "late"]
    degrees = {"early": 100, "late": 100}
    blocks = {"early": 0, "late": 0}
    slots = np.zeros(200, dtype=int)
    slot_times = np.asarray([0] * 100 + [1] * 100)

    assigned, _ = assign_stubs_to_slots_by_time(
        entity_ids,
        degrees,
        blocks,
        slots,
        slot_times,
        activity_model(),
        np.random.default_rng(12),
    )
    early_mean_time = float(slot_times[assigned == "early"].mean())
    late_mean_time = float(slot_times[assigned == "late"].mean())

    assert early_mean_time < late_mean_time


def test_assign_stubs_to_slots_by_time_can_return_integer_indices_with_timings():
    entity_ids = [f"e{i}" for i in range(20)]
    blocks = {entity: idx % 4 for idx, entity in enumerate(entity_ids)}
    rows = []
    degrees = {}
    for idx, entity in enumerate(entity_ids):
        degree = (idx % 5) + 1
        degrees[entity] = degree
        for event_idx in range(degree):
            rows.append((entity, f"2020-01-{(event_idx % 5) + 1:02d}"))
    frame = pd.DataFrame(rows, columns=["entity", "review_time"])
    model = FastTemporalActivityModel(alpha=1.0).fit(frame, "entity", "review_time", blocks)
    slot_blocks = []
    for entity, degree in degrees.items():
        slot_blocks.extend([blocks[entity]] * degree)
    slot_blocks = np.asarray(slot_blocks, dtype=int)
    slot_times = np.arange(len(slot_blocks), dtype=int) % 5

    assigned_idx, summary = assign_stubs_to_slots_by_time(
        entity_ids,
        degrees,
        blocks,
        slot_blocks,
        slot_times,
        model,
        np.random.default_rng(19),
        log_label="test-stubs",
        return_entity_indices=True,
    )

    assert assigned_idx.dtype.kind in {"i", "u"}
    assert len(assigned_idx) == len(slot_blocks)
    assert summary["stub_construction_seconds"] >= 0.0
    assert summary["desired_time_sampling_seconds"] >= 0.0
    assert summary["sorting_seconds"] >= 0.0
    assert summary["slot_assignment_seconds"] >= 0.0
