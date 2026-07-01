from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from reldiff.attributes.temporal_priors import TemporalAttributePrior  # noqa: E402


def test_temporal_prior_smoothing_has_no_zeroes_and_differs_by_month():
    reviews = pd.DataFrame(
        {
            "review_time": ["2020-01-01"] * 20 + ["2020-02-01"] * 20,
            "rating": [5] * 18 + [1] * 2 + [1] * 18 + [5] * 2,
            "verified": [True] * 20 + [False] * 20,
        }
    )
    prior = TemporalAttributePrior([1, 5], temporal_prior_level="year_month").fit(reviews)
    jan = np.asarray(prior.per_bucket_rating_distribution["2020-01"])
    feb = np.asarray(prior.per_bucket_rating_distribution["2020-02"])
    assert prior.bucket_format == "YYYY-MM"
    assert prior.num_buckets == 2
    assert np.isclose(jan.sum(), 1.0)
    assert np.isclose(feb.sum(), 1.0)
    assert (jan > 0).all()
    assert (feb > 0).all()
    assert not np.allclose(jan, feb)
