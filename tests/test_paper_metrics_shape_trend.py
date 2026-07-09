from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.shape_trend import shape_metrics, trend_metrics  # noqa: E402


def test_paper_metrics_shape_and_trend_increase_when_perturbed():
    real = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "y": [0, 1, 2, 3, 4, 5], "label": ["a", "a", "b", "b", "c", "c"]})
    same = real.copy()
    perturbed = pd.DataFrame({"x": [10, 10, 10, 10, 10, 10], "y": [5, 4, 3, 2, 1, 0], "label": ["z"] * 6})
    table_config = {
        "columns": {
            "x": {"type": "numerical"},
            "y": {"type": "numerical"},
            "label": {"type": "categorical"},
        }
    }

    same_shape, _ = shape_metrics(real, same, table_config)
    bad_shape, _ = shape_metrics(real, perturbed, table_config)
    same_trend, _ = trend_metrics(real, same, table_config)
    bad_trend, _ = trend_metrics(real, perturbed, table_config)

    assert same_shape["macro_shape_error"] == 0.0
    assert bad_shape["macro_shape_error"] > same_shape["macro_shape_error"]
    assert same_trend["macro_trend_error"] == 0.0
    assert bad_trend["macro_trend_error"] > same_trend["macro_trend_error"]
