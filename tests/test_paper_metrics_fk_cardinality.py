from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.fk_cardinality import fk_cardinality_metrics  # noqa: E402


def test_paper_metrics_fk_cardinality_similarity_decreases_when_perturbed(tmp_path):
    parent = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    parent_path = tmp_path / "parent.csv"
    parent.to_csv(parent_path, index=False)
    real = pd.DataFrame({"fk": [1] * 5 + [2] * 4 + [3] * 3 + [4] * 2 + [5]})
    same = real.copy()
    perturbed = pd.DataFrame({"fk": [1] * 15})
    table_config = {
        "columns": {
            "fk": {
                "type": "foreign_key",
                "references": {"table": "parent", "column": "id"},
                "parent_table_path": str(parent_path),
            }
        }
    }

    same_metrics, _ = fk_cardinality_metrics(real, same, table_config)
    perturbed_metrics, _ = fk_cardinality_metrics(real, perturbed, table_config)

    assert same_metrics["macro_similarity"] == 1.0
    assert perturbed_metrics["macro_similarity"] < same_metrics["macro_similarity"]

