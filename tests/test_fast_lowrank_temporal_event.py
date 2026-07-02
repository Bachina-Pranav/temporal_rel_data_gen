from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.fast_lowrank_temporal_event import FastLowRankTemporalEventGenerator  # noqa: E402


def tiny_events():
    rows = []
    days = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d").tolist()
    customer_by_block = {0: ["c0", "c1"], 1: ["c2", "c3"]}
    product_by_block = {0: ["p0", "p1"], 1: ["p2", "p3", "p4"]}
    pattern = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)]
    for day_index, day in enumerate(days):
        for event_index, (cblock, pblock) in enumerate(pattern):
            customer_pool = customer_by_block[cblock]
            product_pool = product_by_block[pblock]
            rows.append(
                (
                    customer_pool[(day_index + event_index) % len(customer_pool)],
                    product_pool[(day_index + event_index) % len(product_pool)],
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


def test_full_fast_lowrank_temporal_event_generator_exact_tiny(tmp_path):
    real = tiny_events()
    debug_in = tmp_path / "debug_in"
    write_blocks(debug_in)
    generator = FastLowRankTemporalEventGenerator(
        structure_debug_dir=debug_in,
        rank=2,
        time_gate_granularity="month",
        block_pair_time_mode="exact",
        max_exact_affinity_cell_size=4,
        seed=5,
    )

    synthetic = generator.fit(real).sample(seed=5)
    debug_out = tmp_path / "debug_out"
    generator.save_debug(debug_out)
    metadata = generator.metadata()

    assert len(synthetic) == len(real)
    assert real["customer_id"].value_counts().sort_index().equals(synthetic["customer_id"].value_counts().sort_index())
    assert real["product_id"].value_counts().sort_index().equals(synthetic["product_id"].value_counts().sort_index())
    assert real["review_time"].value_counts().sort_index().equals(synthetic["review_time"].value_counts().sort_index())
    assert bpt_counts(real).equals(bpt_counts(synthetic))
    assert metadata["method"] == "fast_lowrank_temporal_event"
    assert metadata["per_event_candidate_pool_scoring"] is False
    assert metadata["uses_time_dependent_pairing_score"] is True
    assert (debug_out / "lowrank_time_gated_affinity_summary.json").exists()
