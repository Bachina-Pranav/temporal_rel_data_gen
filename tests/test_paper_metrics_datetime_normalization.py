from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.shape_trend import shape_metrics  # noqa: E402
from evaluation.paper_metrics.temporal_fidelity import temporal_metrics  # noqa: E402


def test_datetime_shape_and_temporal_wasserstein_are_normalized():
    real = pd.DataFrame({"event_ts": pd.date_range("2020-01-01", periods=10, freq="D")})
    same = real.copy()
    shifted = pd.DataFrame({"event_ts": pd.date_range("2020-01-06", periods=10, freq="D")})
    table_config = {"columns": {"event_ts": {"type": "datetime"}}}
    config = {
        "table": table_config,
        "evaluation": {"temporal": {"timestamp_columns": ["event_ts"], "binning": {"modes": ["adaptive"], "adaptive_target_bins": 5}}},
    }

    same_shape, _ = shape_metrics(real, same, table_config)
    shifted_temporal, _ = temporal_metrics(real, shifted, config)

    adaptive = shifted_temporal["per_timestamp"]["event_ts"]["adaptive"]
    assert same_shape["per_column"]["event_ts"]["shape_error"] == 0.0
    assert adaptive["normalized_wasserstein"] > 0
    assert adaptive["wasserstein_days"] > 0
    assert adaptive["wasserstein_days"] < 100
