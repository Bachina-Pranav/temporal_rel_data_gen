from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.temporal_activity_models import TemporalActivityModel  # noqa: E402


def test_temporal_activity_shrinkage_weights_and_probabilities():
    high = pd.DataFrame(
        {
            "customer_id": ["hi"] * 20 + ["lo"],
            "review_time": ["2020-01-10"] * 20 + ["2020-01-01"],
        }
    )
    blocks = {"hi": 0, "lo": 0}

    model = TemporalActivityModel.fit_customer_activity(
        high,
        "customer_id",
        "review_time",
        blocks,
        alpha_customer_time=2.0,
    )

    p_hi_late = model.probability("hi", "2020-01-10")
    p_hi_early = model.probability("hi", "2020-01-01")
    p_lo_late = model.probability("lo", "2020-01-10")
    p_lo_early = model.probability("lo", "2020-01-01")
    hi_probs = np.asarray([model.probability("hi", t) for t in model.time_buckets])
    lo_probs = np.asarray([model.probability("lo", t) for t in model.time_buckets])

    assert model.weights["hi"] > model.weights["lo"]
    assert p_hi_late > p_hi_early
    assert p_lo_early > 0
    assert p_lo_late > 0
    assert np.isclose(hi_probs.sum(), 1.0)
    assert np.isclose(lo_probs.sum(), 1.0)
