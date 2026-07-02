from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.joint_temporal_2k_sbm_event import JointTemporal2KSBMEventGenerator  # noqa: E402


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
    assert (out_debug / "block_pair_time_counts.csv").exists()
