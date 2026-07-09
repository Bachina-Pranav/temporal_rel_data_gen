from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.fk_cardinality import fk_cardinality_metrics  # noqa: E402


def test_fk_normalized_cardinality_handles_proportional_row_count_mismatch(tmp_path):
    parent = pd.DataFrame({"entity_id": ["a", "b"]})
    parent_path = tmp_path / "parent.csv"
    parent.to_csv(parent_path, index=False)
    real = pd.DataFrame({"entity_id": ["a"] * 50 + ["b"] * 50})
    synthetic = pd.DataFrame({"entity_id": ["a"] * 25 + ["b"] * 25})
    table_config = {
        "columns": {
            "entity_id": {
                "type": "foreign_key",
                "references": {"table": "entity", "column": "entity_id"},
                "parent_table_path": str(parent_path),
            }
        }
    }

    metrics, _ = fk_cardinality_metrics(real, synthetic, table_config, row_count_match=False)
    fk = metrics["per_fk"]["entity_id"]

    assert fk["absolute_similarity"] < fk["normalized_similarity"]
    assert fk["normalized_similarity"] == 1.0
    assert any("row_count_confounded" in warning for warning in metrics["warnings"])
