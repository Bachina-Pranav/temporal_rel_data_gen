from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.lowrank_time_gated_affinity import LowRankTimeGatedAffinity  # noqa: E402
from generators.ultrafast_lowrank_temporal_event import (  # noqa: E402
    UltraFastLowRankTemporalEventGenerator,
    assign_entities_to_slots_vectorized,
)


def tiny_events():
    rows = []
    days = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d").tolist()
    customer_by_block = {0: ["c0", "c1"], 1: ["c2", "c3"]}
    product_by_block = {0: ["p0", "p1"], 1: ["p2", "p3", "p4"]}
    pattern = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)]
    for day_index, day in enumerate(days):
        for event_index, (cblock, pblock) in enumerate(pattern):
            rows.append(
                (
                    customer_by_block[cblock][(day_index + event_index) % len(customer_by_block[cblock])],
                    product_by_block[pblock][(day_index + event_index) % len(product_by_block[pblock])],
                    day,
                )
            )
    return pd.DataFrame(rows, columns=["customer_id", "product_id", "review_time"])


def write_blocks(debug: Path):
    debug.mkdir(parents=True)
    pd.DataFrame({"customer_id": ["c0", "c1", "c2", "c3"], "customer_block": [0, 0, 1, 1]}).to_csv(
        debug / "customer_blocks.csv",
        index=False,
    )
    pd.DataFrame({"product_id": ["p0", "p1", "p2", "p3", "p4"], "product_block": [0, 0, 1, 1, 1]}).to_csv(
        debug / "product_blocks.csv",
        index=False,
    )


def bpt_counts(frame):
    cblocks = {"c0": 0, "c1": 0, "c2": 1, "c3": 1}
    pblocks = {"p0": 0, "p1": 0, "p2": 1, "p3": 1, "p4": 1}
    tmp = frame.copy()
    tmp["customer_block"] = tmp["customer_id"].map(cblocks)
    tmp["product_block"] = tmp["product_id"].map(pblocks)
    tmp["review_time"] = pd.to_datetime(tmp["review_time"]).dt.strftime("%Y-%m-%d")
    return tmp.groupby(["customer_block", "product_block", "review_time"]).size().sort_index()


def test_slot_build_exact_counts(tmp_path):
    real = tiny_events()
    debug = tmp_path / "debug"
    write_blocks(debug)
    generator = UltraFastLowRankTemporalEventGenerator(structure_debug_dir=debug, rank=2, seed=1).fit(real)

    generator._build_slots()
    slot_frame = pd.DataFrame(
        {
            "customer_block": generator.slot_customer_block,
            "product_block": generator.slot_product_block,
            "review_time": generator.slot_time_bucket,
        }
    )
    slot_counts = slot_frame.groupby(["customer_block", "product_block", "review_time"]).size().sort_index()

    assert len(generator.slot_time_bucket) == len(real)
    assert slot_counts.equals(bpt_counts(real))


def test_vectorized_assignment_fills_slots_with_correct_block(tmp_path):
    real = tiny_events()
    debug = tmp_path / "debug"
    write_blocks(debug)
    generator = UltraFastLowRankTemporalEventGenerator(structure_debug_dir=debug, rank=2, seed=2).fit(real)
    generator._build_slots()

    assigned, summary = assign_entities_to_slots_vectorized(
        generator.slot_customer_block,
        generator.slot_time_bucket,
        generator.customer_degrees,
        generator.customer_activity,
        np.random.default_rng(2),
        "customer",
    )

    assert summary["total_assigned"] == len(real)
    assert all(entity is not None for entity in assigned)
    assert all(generator.customer_blocks[entity] == block for entity, block in zip(assigned, generator.slot_customer_block))


def test_dynamic_affinity_changes_with_time():
    model = LowRankTimeGatedAffinity(rank=2)
    model.customer_index = {"u": 0}
    model.product_index = {"i": 0}
    model.customer_embeddings = np.asarray([[1.0, 2.0]])
    model.product_embeddings = np.asarray([[3.0, 5.0]])
    model.global_gate = np.ones(2)
    model.time_gates = {"2020-01": np.asarray([1.0, 0.0]), "2020-02": np.asarray([0.0, 1.0])}

    assert model.score_pairs(["u"], ["i"], "2020-01-01")[0] != model.score_pairs(["u"], ["i"], "2020-02-01")[0]


def test_full_ultrafast_generator_tiny_exact(tmp_path):
    real = tiny_events()
    debug_in = tmp_path / "debug_in"
    write_blocks(debug_in)
    generator = UltraFastLowRankTemporalEventGenerator(
        structure_debug_dir=debug_in,
        rank=2,
        block_pair_time_mode="exact",
        pairing_mode="dynamic_projection",
        seed=4,
    )

    synthetic = generator.fit(real).sample(seed=4)
    debug_out = tmp_path / "debug_out"
    generator.save_debug(debug_out)
    metadata = generator.metadata()

    assert len(synthetic) == len(real)
    assert real["customer_id"].value_counts().sort_index().equals(synthetic["customer_id"].value_counts().sort_index())
    assert real["product_id"].value_counts().sort_index().equals(synthetic["product_id"].value_counts().sort_index())
    assert bpt_counts(real).equals(bpt_counts(synthetic))
    assert metadata["method"] == "ultrafast_lowrank_temporal_event"
    assert metadata["cell_level_quota_rejection_sampling"] is False
    assert metadata["per_event_candidate_pool_scoring"] is False
    assert metadata["uses_time_dependent_pairing_score"] is True
    assert (debug_out / "slot_summary.json").exists()
