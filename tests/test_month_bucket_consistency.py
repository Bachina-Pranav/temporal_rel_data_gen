from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from diagnose_temporal_nontext_v3 import enrich_temporal_prior_diagnostics  # noqa: E402
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


def test_legacy_month_number_prior_is_reported_as_bucket_mismatch(tmp_path):
    prior = TemporalAttributePrior([1, 5], temporal_prior_level="month")
    prior.rating_global_distribution = [0.5, 0.5]
    prior.verified_global_rate = 0.5
    prior.per_bucket_rating_distribution = {"1": [0.1, 0.9], "2": [0.9, 0.1]}
    prior.per_bucket_verified_rate = {"1": 0.8, "2": 0.2}
    prior.bucket_counts = {"1": 10, "2": 10}

    assert np.allclose(prior.target_rating_distribution("2020-01"), [0.1, 0.9])
    consistency = check_temporal_bucket_consistency(
        prior,
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
    )
    assert consistency["train_prior_bucket_format"] == "legacy-month-number"
    assert consistency["is_consistent"] is False

    pd.DataFrame(
        {
            "month": [1, 2],
            "prior_avg_rating": [4.6, 1.4],
            "prior_verified_rate": [0.8, 0.2],
        }
    ).to_csv(tmp_path / "temporal_rating_prior_monthly_avg_curve.csv", index=False)
    pd.DataFrame(
        {
            "month": ["2020-01", "2020-02"],
            "real_avg_rating": [4.5, 1.5],
            "real_verified_rate": [0.75, 0.25],
        }
    ).to_csv(tmp_path / "monthly_real_vs_synthetic.csv", index=False)

    summary = enrich_temporal_prior_diagnostics(tmp_path)
    assert summary["diagnostic_status"] == "bucket_mismatch"
    assert summary["prior_curve_month_format"] == "legacy-month-number"
    assert summary["monthly_table_month_format"] == "YYYY-MM"
