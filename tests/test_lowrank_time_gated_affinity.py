from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.lowrank_time_gated_affinity import LowRankTimeGatedAffinity  # noqa: E402


def test_dynamic_affinity_is_time_dependent_for_same_pair():
    model = LowRankTimeGatedAffinity(rank=2)
    model.customer_index = {"u": 0}
    model.product_index = {"i": 0}
    model.customer_embeddings = np.asarray([[1.0, 2.0]])
    model.product_embeddings = np.asarray([[3.0, 5.0]])
    model.global_gate = np.ones(2)
    model.time_gates = {
        "2020-01": np.asarray([1.0, 0.0]),
        "2020-02": np.asarray([0.0, 1.0]),
    }

    score_1 = model.score_pairs(["u"], ["i"], "2020-01-10")[0]
    score_2 = model.score_pairs(["u"], ["i"], "2020-02-10")[0]

    assert score_1 != score_2


def test_lowrank_fit_does_not_create_dense_tensor():
    frame = pd.DataFrame(
        {
            "customer_id": ["u0", "u0", "u1", "u2"],
            "product_id": ["i0", "i1", "i1", "i0"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-02-01", "2020-02-02"],
        }
    )

    model = LowRankTimeGatedAffinity(rank=2, seed=3).fit(frame, "customer_id", "product_id", "review_time")
    summary = model.summary()

    assert not hasattr(model, "dense_F")
    assert summary["uses_dense_F_u_i_t"] is False
    assert summary["formula"] == "(z_u * g_t)^T z_i"
