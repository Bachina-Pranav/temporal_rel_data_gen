from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics  # noqa: E402


def test_c2st_feature_importance_has_names_and_top_features(tmp_path):
    real = pd.DataFrame(
        {
            "value": list(range(40)),
            "label": ["a", "b"] * 20,
            "event_ts": pd.date_range("2020-01-01", periods=40, freq="D"),
        }
    )
    synthetic = pd.DataFrame(
        {
            "value": list(range(100, 140)),
            "label": ["z"] * 40,
            "event_ts": pd.date_range("2020-03-01", periods=40, freq="D"),
        }
    )
    real_path = tmp_path / "real.csv"
    syn_path = tmp_path / "synthetic.csv"
    real.to_csv(real_path, index=False)
    synthetic.to_csv(syn_path, index=False)
    config = {
        "real_table_path": str(real_path),
        "synthetic_table_path": str(syn_path),
        "table": {
            "columns": {
                "value": {"type": "numerical"},
                "label": {"type": "categorical"},
                "event_ts": {"type": "datetime"},
            }
        },
        "evaluation": {
            "temporal": {"timestamp_columns": ["event_ts"], "binning": {"modes": ["adaptive"]}},
            "c2st": {"enabled": True, "classifiers": ["logistic_regression"], "max_rows": 40},
        },
    }

    metrics = evaluate_paper_metrics(config, tmp_path / "out")
    feature_importance = pd.read_csv(tmp_path / "out" / "c2st_feature_importance.csv")

    assert len(metrics["single_table_c2st"]["feature_names"]) == metrics["single_table_c2st"]["num_features"]
    assert {"classifier", "feature_name", "importance", "abs_importance", "rank"}.issubset(feature_importance.columns)
    assert metrics["single_table_c2st"]["top_features"]
    assert "feature_name" in metrics["single_table_c2st"]["top_features"][0]
