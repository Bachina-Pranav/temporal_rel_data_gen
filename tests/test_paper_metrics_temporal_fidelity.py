from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.temporal_fidelity import temporal_metrics  # noqa: E402


def test_paper_metrics_temporal_distance_increases_when_timestamps_shift():
    real = pd.DataFrame({"event_ts": pd.date_range("2020-01-01", periods=20, freq="D"), "entity": ["a"] * 20})
    same = real.copy()
    shifted = pd.DataFrame({"event_ts": pd.date_range("2021-01-01", periods=20, freq="D"), "entity": ["a"] * 20})
    config = {
        "table": {"columns": {"event_ts": {"type": "datetime"}}},
        "evaluation": {"temporal": {"timestamp_columns": ["event_ts"], "binning": {"modes": ["adaptive"], "adaptive_target_bins": 5}}},
    }

    same_metrics, _ = temporal_metrics(real, same, config)
    shifted_metrics, _ = temporal_metrics(real, shifted, config)

    assert same_metrics["macro_temporal_event_distance"] == 0.0
    assert shifted_metrics["macro_temporal_event_distance"] > same_metrics["macro_temporal_event_distance"]

