from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.event_affinity import (  # noqa: E402
    EventAffinityScorer,
    EventScoreWeights,
    ProductAgeAffinity,
    StaticCustomerProductAffinity,
    product_age_bin,
    product_lifecycle_table,
)
from generators.temporal_activity_models import TemporalActivityModel  # noqa: E402


def make_events():
    return pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2", "c3", "c3", "c3"],
            "product_id": ["p1", "p1", "p2", "p1", "p2", "p2"],
            "review_time": ["2020-01-10", "2020-01-11", "2020-01-20", "2020-01-10", "2020-01-20", "2020-01-20"],
        }
    )


def test_product_lifecycle_age_bins():
    assert product_age_bin("2020-01-10", "2020-01-20", "2020-01-01") == "pre_active"
    assert product_age_bin("2020-01-10", "2020-01-20", "2020-01-10") == "early"
    assert product_age_bin("2020-01-10", "2020-01-20", "2020-01-15") == "mid"
    assert product_age_bin("2020-01-10", "2020-01-20", "2020-01-20") == "late"
    assert product_age_bin("2020-01-10", "2020-01-20", "2020-01-30") == "post_active"
    assert product_age_bin("2020-01-10", "2020-01-10", "2020-01-10") == "single_day"


def test_event_score_is_time_dependent_for_same_pair():
    events = make_events()
    customer_blocks = {"c1": 0, "c2": 0, "c3": 1}
    product_blocks = {"p1": 0, "p2": 0}
    customer_activity = TemporalActivityModel.fit_customer_activity(events, "customer_id", "review_time", customer_blocks)
    product_activity = TemporalActivityModel.fit_product_activity(events, "product_id", "review_time", product_blocks)
    lifecycle = product_lifecycle_table(events, "product_id", "review_time", product_blocks)
    age = ProductAgeAffinity().fit(events, "customer_id", "product_id", "review_time", customer_blocks, lifecycle)
    static = StaticCustomerProductAffinity(rank=2).fit(events, "customer_id", "product_id")
    scorer = EventAffinityScorer(
        static,
        customer_activity,
        product_activity,
        age,
        customer_blocks,
        set(),
        EventScoreWeights(lambda_static=0.0, lambda_ut=1.0, lambda_it=1.0, lambda_age=0.0),
    )

    score_t1 = scorer.event_score("c1", ["p2"], "2020-01-10", [2], defaultdict(int))[0]
    score_t2 = scorer.event_score("c1", ["p2"], "2020-01-20", [2], defaultdict(int))[0]

    assert score_t1 != score_t2
