from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402


def test_no_future_neighbors_for_customer_history():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "product_id": ["i1", "i2", "i3"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    index = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64)
    assert index.history_for_row(1, kind="customer") == [0]


def test_no_self_in_product_history():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3"],
            "product_id": ["i1", "i1", "i1"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    index = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64)
    assert index.history_for_row(2, kind="product") == [0, 1]


def test_same_timestamp_temporal_filter_modes():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "product_id": ["i1", "i2", "i3"],
            "review_time": ["2020-01-01", "2020-01-01", "2020-01-01"],
        }
    )
    strict = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64)
    assert strict.history_for_row(2, kind="customer") == []

    with_tiebreak = TemporalHistoryIndex(
        frame,
        "customer_id",
        "product_id",
        "review_time",
        64,
        allow_same_timestamp_events=True,
    )
    assert with_tiebreak.history_for_row(2, kind="customer") == [0, 1]
