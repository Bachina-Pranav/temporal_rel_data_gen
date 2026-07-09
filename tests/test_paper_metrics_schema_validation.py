from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.schema_validation import constraint_violation_metrics  # noqa: E402


def test_paper_metrics_schema_validation_counts_violations(tmp_path):
    parent = pd.DataFrame({"entity_id": ["a", "b"]})
    parent_path = tmp_path / "parent.csv"
    parent.to_csv(parent_path, index=False)
    synthetic = pd.DataFrame(
        {
            "entity_id": ["a", "z", ""],
            "event_time": ["2020-01-01", "not-a-date", "2020-01-03"],
            "label": ["x", "bad", None],
        }
    )
    table_config = {
        "columns": {
            "entity_id": {
                "type": "foreign_key",
                "references": {"table": "entity", "column": "entity_id"},
                "parent_table_path": str(parent_path),
                "nullable": False,
            },
            "event_time": {"type": "datetime", "nullable": False},
            "label": {"type": "categorical", "valid_values": ["x", "y"], "nullable": False},
        }
    }

    metrics = constraint_violation_metrics(synthetic, table_config)

    assert metrics["constraint_violation_rate"] > 0
    assert metrics["num_violating_rows"] == 2
    assert metrics["per_constraint"]["counts"]["foreign_key"] == 1
    assert metrics["per_constraint"]["counts"]["datetime_parse"] == 1
    assert metrics["per_constraint"]["counts"]["categorical_domain"] == 1
    assert metrics["per_constraint"]["counts"]["null"] == 2
