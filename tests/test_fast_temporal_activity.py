from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.fast_temporal_activity import FastTemporalActivityModel, canonical_time_bucket  # noqa: E402


def test_canonical_time_bucket_day_and_month():
    values = pd.Series(["2020-01-01 12:34:00", "2020-02-03"])

    assert canonical_time_bucket(values, "day").tolist() == ["2020-01-01", "2020-02-03"]
    assert canonical_time_bucket(values, "month").tolist() == ["2020-01", "2020-02"]


def test_fast_temporal_activity_cache_alignment():
    frame = pd.DataFrame(
        {
            "customer_id": ["c0", "c0", "c1", "c2"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"],
        }
    )
    blocks = {"c0": 0, "c1": 0, "c2": 1}
    model = FastTemporalActivityModel(alpha="auto", entity_kind="customer").fit(frame, "customer_id", "review_time", blocks)

    ids, probs = model.probabilities_for_block_time(0, "2020-01-02")
    ids_again, probs_again = model.probabilities_for_block_time(0, "2020-01-02")

    assert list(ids) == ["c0", "c1"]
    assert len(ids) == len(probs)
    assert (probs >= 0).all()
    assert np.allclose(probs, probs_again)
    assert list(ids_again) == list(ids)
