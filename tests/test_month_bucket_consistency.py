from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_temporal_nontext_attrs import monthly_diagnostics  # noqa: E402
from reldiff.attributes.temporal_priors import (  # noqa: E402
    TemporalAttributePrior,
    check_temporal_bucket_consistency,
    temporal_bucket,
)


def test_month_buckets_are_canonical_year_month_strings():
    reviews = pd.DataFrame(
        {
            "customer_id": ["c0", "c1", "c2", "c3"],
            "product_id": ["p0", "p0", "p1", "p1"],
            "review_time": ["2020-01-01", "2020-01-31", "2020-02-01", "2020-02-29"],
            "rating": [1, 5, 2, 4],
            "verified": [True, False, True, False],
        }
    )
    buckets = temporal_bucket(pd.to_datetime(reviews["review_time"]), "month")
    assert buckets.tolist() == ["2020-01", "2020-01", "2020-02", "2020-02"]

    prior = TemporalAttributePrior([1, 2, 4, 5], temporal_prior_level="month").fit(reviews)
    assert sorted(prior.per_bucket_rating_distribution) == ["2020-01", "2020-02"]

    consistency = check_temporal_bucket_consistency(
        prior,
        pd.to_datetime(reviews["review_time"]),
        pd.to_datetime(reviews["review_time"]),
    )
    assert consistency["bucket_format"] == "YYYY-MM"
    assert consistency["is_consistent"] is True

    monthly_table, _ = monthly_diagnostics(
        reviews,
        reviews,
        timestamp_col="review_time",
        rating_col="rating",
        verified_col="verified",
    )
    assert monthly_table["month"].tolist() == ["2020-01", "2020-02"]
