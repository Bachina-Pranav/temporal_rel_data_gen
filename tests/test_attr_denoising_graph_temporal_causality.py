from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402


def test_attr_denoising_graph_keeps_v2_past_only_temporal_causality():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u1"],
            "product_id": ["i1", "i2", "i3"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    history = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64)
    assert history.history_for_row(1, kind="customer") == [0]


def test_attr_denoising_graph_excludes_same_timestamp_by_default():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u1"],
            "product_id": ["i1", "i2"],
            "review_time": ["2020-01-01", "2020-01-01"],
        }
    )
    history = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64)
    assert history.history_for_row(1, kind="customer") == []
