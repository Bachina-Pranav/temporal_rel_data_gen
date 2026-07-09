from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.c2st import single_table_c2st_metrics  # noqa: E402


def test_paper_metrics_single_table_c2st_detects_obvious_shift():
    real = pd.DataFrame({"value": list(range(40)), "label": ["a", "b"] * 20})
    same = real.copy()
    shifted = pd.DataFrame({"value": list(range(100, 140)), "label": ["z"] * 40})
    config = {
        "table": {"columns": {"value": {"type": "numerical"}, "label": {"type": "categorical"}}},
        "evaluation": {"random_seed": 42, "c2st": {"enabled": True, "classifiers": ["logistic_regression"], "max_rows": 40}},
    }

    same_metrics, _ = single_table_c2st_metrics(real, same, config)
    shifted_metrics, _ = single_table_c2st_metrics(real, shifted, config)

    assert same_metrics["error"] is not None
    assert shifted_metrics["error"] >= same_metrics["error"]

