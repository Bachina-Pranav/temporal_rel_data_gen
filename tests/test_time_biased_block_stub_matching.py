from __future__ import annotations

from collections import Counter
from pathlib import Path
import inspect

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.time_biased_block_stub_matching import TimeBiasedBlockStubMatchingGenerator  # noqa: E402


def tiny_events() -> pd.DataFrame:
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


def write_blocks(debug: Path) -> None:
    debug.mkdir(parents=True)
    pd.DataFrame({"customer_id": ["c0", "c1", "c2", "c3"], "customer_block": [0, 0, 1, 1]}).to_csv(
        debug / "customer_blocks.csv",
        index=False,
    )
    pd.DataFrame({"product_id": ["p0", "p1", "p2", "p3", "p4"], "product_block": [0, 0, 1, 1, 1]}).to_csv(
        debug / "product_blocks.csv",
        index=False,
    )


def bpt_counts(frame: pd.DataFrame) -> pd.Series:
    cblocks = {"c0": 0, "c1": 0, "c2": 1, "c3": 1}
    pblocks = {"p0": 0, "p1": 0, "p2": 1, "p3": 1, "p4": 1}
    tmp = frame.copy()
    tmp["customer_block"] = tmp["customer_id"].map(cblocks)
    tmp["product_block"] = tmp["product_id"].map(pblocks)
    tmp["review_time"] = pd.to_datetime(tmp["review_time"]).dt.strftime("%Y-%m-%d")
    return tmp.groupby(["customer_block", "product_block", "review_time"]).size().sort_index()


def test_exact_stubs_equal_slots_per_block(tmp_path):
    real = tiny_events()
    debug = tmp_path / "debug"
    write_blocks(debug)
    generator = TimeBiasedBlockStubMatchingGenerator(structure_debug_dir=debug, rank=2, seed=1).fit(real)

    generator._build_slots()

    for block in set(generator.customer_blocks.values()):
        stub_count = sum(
            degree for entity, degree in generator.customer_degrees.items() if generator.customer_blocks[entity] == block
        )
        assert stub_count == int((generator.slot_customer_block == block).sum())
    for block in set(generator.product_blocks.values()):
        stub_count = sum(
            degree for entity, degree in generator.product_degrees.items() if generator.product_blocks[entity] == block
        )
        assert stub_count == int((generator.slot_product_block == block).sum())


def test_full_time_biased_block_stub_matching_exact_constraints(tmp_path):
    real = tiny_events()
    debug_in = tmp_path / "debug_in"
    write_blocks(debug_in)
    generator = TimeBiasedBlockStubMatchingGenerator(
        structure_debug_dir=debug_in,
        rank=2,
        pairing_mode="dynamic_projection",
        seed=4,
    )

    synthetic = generator.fit(real).sample(seed=4)
    debug_out = tmp_path / "debug_out"
    generator.save_debug(debug_out)
    metadata = generator.metadata()
    metrics = generator.evaluate(real, synthetic)

    assert len(synthetic) == len(real)
    assert Counter(synthetic["customer_id"]) == Counter(real["customer_id"])
    assert Counter(synthetic["product_id"]) == Counter(real["product_id"])
    assert real["review_time"].value_counts().sort_index().equals(synthetic["review_time"].value_counts().sort_index())
    assert bpt_counts(real).equals(bpt_counts(synthetic))
    assert metadata["method"] == "time_biased_block_stub_matching"
    assert metadata["alias"] == "temporal_stub_matching_event"
    assert metadata["no_degree_repair"] is True
    assert metadata["no_quota_rejection_sampling"] is True
    assert metadata["uses_time_dependent_pairing_score"] is True
    assert metadata["preserves_block_pair_time_counts_exactly"] is True
    assert metrics["block_pair_time_exact_match"] is True
    assert "mean_dynamic_affinity_synthetic" in metrics
    assert (debug_out / "slot_summary.json").exists()
    assert (debug_out / "customer_stub_assignment_summary.json").exists()
    assert (debug_out / "dynamic_pairing_summary.json").exists()


def test_time_biased_generator_does_not_call_quota_or_degree_repair():
    source = inspect.getsource(TimeBiasedBlockStubMatchingGenerator)

    assert "repair_entity_degrees_by_replacement" not in source
    assert "sample_entities_with_quotas" not in source
    assert "customer_candidate_pool_size" not in source
    assert "product_candidate_pool_size" not in source
