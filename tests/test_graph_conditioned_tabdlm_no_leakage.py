from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402


def test_customer_history_excludes_future_and_self():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "product_id": ["i1", "i2", "i3"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    index = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64)
    assert index.history_for_row(1, kind="customer") == [0]


def test_product_history_excludes_self():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u2", "u3"],
            "product_id": ["i1", "i1", "i1"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    index = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64)
    assert index.history_for_row(2, kind="product") == [0, 1]


def test_same_timestamp_default_is_excluded():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "product_id": ["i1", "i2", "i3"],
            "review_time": ["2020-01-01", "2020-01-01", "2020-01-02"],
        }
    )
    index = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64)
    assert index.history_for_row(1, kind="customer") == []
    assert index.history_for_row(2, kind="customer") == [0, 1]


def test_same_timestamp_tiebreak_only_allows_earlier_row_index():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "product_id": ["i1", "i2", "i3"],
            "review_time": ["2020-01-01", "2020-01-01", "2020-01-01"],
        }
    )
    index = TemporalHistoryIndex(
        frame,
        "customer_id",
        "product_id",
        "review_time",
        64,
        allow_same_timestamp_events=True,
    )
    assert index.history_for_row(0, kind="customer") == []
    assert index.history_for_row(1, kind="customer") == [0]
    assert index.history_for_row(2, kind="customer") == [0, 1]
