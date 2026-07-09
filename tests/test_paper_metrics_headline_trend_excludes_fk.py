from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics  # noqa: E402


def test_headline_trend_excludes_raw_fk_pairs(tmp_path):
    real = pd.DataFrame(
        {
            "entity_id": ["u1", "u1", "u2", "u2"],
            "label": ["a", "a", "b", "b"],
            "value": [1, 1, 2, 2],
        }
    )
    synthetic = pd.DataFrame(
        {
            "entity_id": ["u1", "u2", "u1", "u2"],
            "label": ["a", "a", "b", "b"],
            "value": [1, 1, 2, 2],
        }
    )
    real_path = tmp_path / "real.csv"
    syn_path = tmp_path / "synthetic.csv"
    real.to_csv(real_path, index=False)
    synthetic.to_csv(syn_path, index=False)
    config = {
        "real_table_path": str(real_path),
        "synthetic_table_path": str(syn_path),
        "trend": {"exclude_foreign_keys_from_headline": True},
        "table": {
            "columns": {
                "entity_id": {"type": "foreign_key"},
                "label": {"type": "categorical"},
                "value": {"type": "numerical"},
            }
        },
        "evaluation": {"c2st": {"enabled": False}},
    }

    metrics = evaluate_paper_metrics(config, tmp_path / "out")

    assert metrics["trend"]["macro_trend_error_all_pairs"] > metrics["trend"]["macro_headline_trend_error"]
    assert metrics["paper_metrics_summary"]["trend_error"] == metrics["trend"]["macro_headline_trend_error"]
