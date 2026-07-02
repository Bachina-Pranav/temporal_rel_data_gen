from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.joint_temporal_2k_sbm_event import (  # noqa: E402
    JointTemporal2KSBMEventGenerator,
    sample_entities_with_quotas,
)


def tiny_events():
    rows = []
    customers = ["c0", "c1", "c2"]
    products = ["p0", "p1", "p2", "p3"]
    days = pd.date_range("2020-01-01", periods=5, freq="D").strftime("%Y-%m-%d").tolist()
    pairs = [
        ("c0", "p0", days[0]),
        ("c0", "p1", days[0]),
        ("c1", "p0", days[1]),
        ("c1", "p2", days[1]),
        ("c2", "p3", days[2]),
        ("c2", "p1", days[2]),
        ("c0", "p2", days[3]),
        ("c1", "p3", days[3]),
        ("c2", "p0", days[4]),
        ("c0", "p3", days[4]),
        ("c1", "p1", days[0]),
        ("c2", "p2", days[1]),
        ("c0", "p0", days[2]),
        ("c1", "p2", days[3]),
        ("c2", "p3", days[4]),
        ("c0", "p1", days[1]),
        ("c1", "p0", days[2]),
        ("c2", "p2", days[3]),
        ("c0", "p3", days[4]),
        ("c1", "p1", days[4]),
    ]
    rows.extend(pairs)
    return pd.DataFrame(rows, columns=["customer_id", "product_id", "review_time"])


def write_blocks(debug: Path):
    debug.mkdir(parents=True)
    pd.DataFrame({"customer_id": ["c0", "c1", "c2"], "customer_block": [0, 0, 1]}).to_csv(debug / "customer_blocks.csv", index=False)
    pd.DataFrame({"product_id": ["p0", "p1", "p2", "p3"], "product_block": [0, 0, 1, 1]}).to_csv(debug / "product_blocks.csv", index=False)


def bpt_counts(frame, cblocks, pblocks):
    tmp = frame.copy()
    tmp["customer_block"] = tmp["customer_id"].map(cblocks)
    tmp["product_block"] = tmp["product_id"].map(pblocks)
    tmp["review_time"] = pd.to_datetime(tmp["review_time"]).dt.strftime("%Y-%m-%d")
    return tmp.groupby(["customer_block", "product_block", "review_time"]).size().sort_index()


def test_joint_temporal_event_generator_preserves_exact_counts(tmp_path):
    real = tiny_events()
    debug = tmp_path / "debug_in"
    write_blocks(debug)
    generator = JointTemporal2KSBMEventGenerator(
        structure_debug_dir=debug,
        mf_rank=2,
        seed=9,
        product_candidate_pool_size=4,
    )

    synthetic = generator.fit(real).sample(seed=9)
    out_debug = tmp_path / "debug_out"
    generator.save_debug(out_debug)
    metadata = generator.metadata()

    assert len(synthetic) == len(real)
    assert real["customer_id"].value_counts().sort_index().equals(synthetic["customer_id"].value_counts().sort_index())
    assert real["product_id"].value_counts().sort_index().equals(synthetic["product_id"].value_counts().sort_index())
    assert real["review_time"].value_counts().sort_index().equals(synthetic["review_time"].value_counts().sort_index())
    cblocks = {"c0": 0, "c1": 0, "c2": 1}
    pblocks = {"p0": 0, "p1": 0, "p2": 1, "p3": 1}
    assert bpt_counts(real, cblocks, pblocks).equals(bpt_counts(synthetic, cblocks, pblocks))
    assert metadata["uses_dense_F_u_i_t"] is False
    assert metadata["uses_time_dependent_pairing_score"] is True
    assert metadata["sampling_mode"] == "fast_time_conditioned"
    assert metadata["uses_candidate_pool_scoring"] is False
    assert (out_debug / "block_pair_time_counts.csv").exists()


def test_fast_sampler_skips_static_affinity_fit_by_default(tmp_path, monkeypatch):
    real = tiny_events()
    debug = tmp_path / "debug_in"
    write_blocks(debug)

    def fail_fit(*args, **kwargs):
        raise AssertionError("Static affinity should not be fit in default fast mode")

    monkeypatch.setattr("generators.joint_temporal_2k_sbm_event.StaticCustomerProductAffinity.fit", fail_fit)
    generator = JointTemporal2KSBMEventGenerator(
        structure_debug_dir=debug,
        lambda_static=1.0,
        use_static_affinity=False,
        sampling_mode="fast_time_conditioned",
        seed=10,
    )

    synthetic = generator.fit(real).sample(seed=10)

    assert len(synthetic) == len(real)
    assert generator.metadata()["uses_static_affinity_in_default_sampler"] is False


def test_sample_entities_with_quotas_respects_remaining_degrees():
    rng = np.random.default_rng(4)
    entity_ids = np.asarray(["a", "b", "c"], dtype=object)
    remaining = {"a": 1, "b": 2, "c": 0}
    probs = np.asarray([0.9, 0.1, 1.0])

    sampled, status = sample_entities_with_quotas(entity_ids, remaining, probs, 3, rng)

    counts = pd.Series(sampled).value_counts().to_dict()
    assert status["success"] is True
    assert len(sampled) == 3
    assert counts.get("a", 0) <= 1
    assert counts.get("b", 0) <= 2
    assert counts.get("c", 0) == 0


def test_sample_entities_with_quotas_falls_back_to_degree_weights():
    rng = np.random.default_rng(5)
    entity_ids = np.asarray(["a", "b"], dtype=object)
    remaining = {"a": 2, "b": 1}
    probs = np.zeros(2, dtype=float)

    sampled, status = sample_entities_with_quotas(entity_ids, remaining, probs, 3, rng)

    counts = pd.Series(sampled).value_counts().to_dict()
    assert status["success"] is True
    assert len(sampled) == 3
    assert counts.get("a", 0) <= 2
    assert counts.get("b", 0) <= 1
