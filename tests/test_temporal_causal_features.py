from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reldiff.attributes.temporal_causal_features import TemporalCausalFeatureBuilder  # noqa: E402


def make_leakage_reviews():
    return pd.DataFrame(
        [
            ("c0", "p0", "2020-01-01", 5, True),
            ("c0", "p1", "2020-01-01", 1, False),
            ("c0", "p0", "2020-01-02", 3, True),
            ("c1", "p0", "2020-01-02", 4, True),
            ("c1", "p0", "2020-01-03", 2, False),
        ],
        columns=["customer_id", "product_id", "review_time", "rating", "verified"],
    )


def test_causal_features_exclude_current_and_same_date_rows():
    reviews = make_leakage_reviews()
    builder = TemporalCausalFeatureBuilder(date_only=True)
    features = builder.transform_training(reviews)

    assert features.loc[0, "customer_past_review_count"] == 0
    assert features.loc[1, "customer_past_review_count"] == 0
    assert features.loc[0, "product_past_review_count"] == 0

    assert features.loc[2, "customer_past_review_count"] == 2
    assert features.loc[2, "customer_past_avg_rating"] == 3.0
    assert features.loc[2, "product_past_review_count"] == 1
    assert features.loc[3, "product_past_review_count"] == 1


def test_sampling_chronology_uses_generated_past_only():
    reviews = make_leakage_reviews()
    spine = reviews[["customer_id", "product_id", "review_time"]].copy()
    builder = TemporalCausalFeatureBuilder(date_only=True, marginal_rating=3.0)
    builder.fit_metadata(reviews)
    builder.prepare_sampling(spine)

    _, first_day = next(builder.iter_time_groups(spine))
    first_features = builder.transform_current_group(first_day)
    assert first_features["global_past_review_count"].eq(0).all()
    generated_first = first_day.copy()
    generated_first["rating"] = [5, 1]
    generated_first["verified"] = [True, False]
    builder.update_history(generated_first)

    groups = list(builder.iter_time_groups(spine))
    second_day = groups[1][1]
    second_features = builder.transform_current_group(second_day)
    assert second_features["global_past_review_count"].eq(2).all()
    assert second_features.loc[2, "customer_past_review_count"] == 2
