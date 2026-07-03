from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.fast_temporal_activity import FastTemporalActivityModel  # noqa: E402
from generators.time_biased_stub_sampler import sample_desired_times_for_stubs  # noqa: E402


def test_sample_desired_times_for_stubs_respects_entity_activity():
    frame = pd.DataFrame(
        {
            "customer_id": ["early"] * 50 + ["early"] * 2 + ["late"] * 2 + ["late"] * 50,
            "review_time": ["2020-01-01"] * 50
            + ["2020-01-10"] * 2
            + ["2020-01-01"] * 2
            + ["2020-01-10"] * 50,
        }
    )
    model = FastTemporalActivityModel(alpha=0.1).fit(
        frame,
        "customer_id",
        "review_time",
        {"early": 0, "late": 0},
    )
    stubs = np.asarray(["early"] * 500 + ["late"] * 500, dtype=object)

    desired = sample_desired_times_for_stubs(stubs, model, np.random.default_rng(7))

    assert desired[:500].mean() < desired[500:].mean()
    assert desired.min() >= 0
    assert desired.max() < len(model.time_buckets)


def test_sample_desired_times_for_stubs_does_not_call_probability():
    class NoDenseProbabilityModel(FastTemporalActivityModel):
        def probability(self, entity_id, time_bucket):
            raise AssertionError("dense probability path should not be used")

    frame = pd.DataFrame(
        {
            "customer_id": ["c0", "c0", "c1", "c1"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"],
        }
    )
    model = NoDenseProbabilityModel(alpha=1.0).fit(
        frame,
        "customer_id",
        "review_time",
        {"c0": 0, "c1": 0},
    )

    desired = sample_desired_times_for_stubs(["c0", "c0", "c1", "c1"], model, np.random.default_rng(11))

    assert len(desired) == 4
    assert desired.dtype.kind in {"i", "u"}
