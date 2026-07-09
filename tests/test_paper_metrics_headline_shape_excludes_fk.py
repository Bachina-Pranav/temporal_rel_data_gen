from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics  # noqa: E402


def test_headline_shape_excludes_raw_fk_columns(tmp_path):
    parent = pd.DataFrame({"entity_id": ["a", "b", "c"]})
    parent_path = tmp_path / "parent.csv"
    parent.to_csv(parent_path, index=False)
    real = pd.DataFrame({"entity_id": ["a", "b", "c", "a"], "label": ["x", "y", "x", "y"]})
    synthetic = pd.DataFrame({"entity_id": ["a", "a", "a", "a"], "label": ["x", "y", "x", "y"]})
    real_path = tmp_path / "real.csv"
    syn_path = tmp_path / "synthetic.csv"
    real.to_csv(real_path, index=False)
    synthetic.to_csv(syn_path, index=False)
    config = {
        "real_table_path": str(real_path),
        "synthetic_table_path": str(syn_path),
        "table": {
            "columns": {
                "entity_id": {
                    "type": "foreign_key",
                    "references": {"table": "entity", "column": "entity_id"},
                    "parent_table_path": str(parent_path),
                },
                "label": {"type": "categorical"},
            }
        },
        "evaluation": {"c2st": {"enabled": False}},
    }

    metrics = evaluate_paper_metrics(config, tmp_path / "out")

    assert metrics["shape"]["macro_shape_error_all_columns"] > metrics["shape"]["macro_non_id_shape_error"]
    assert metrics["paper_metrics_summary"]["shape_error"] == metrics["shape"]["macro_non_id_shape_error"]
