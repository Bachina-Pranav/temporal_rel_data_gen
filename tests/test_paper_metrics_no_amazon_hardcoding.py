from __future__ import annotations

import sys
from pathlib import Path

from test_paper_metrics_report_outputs import write_tiny_tables


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics  # noqa: E402


def test_paper_metrics_runs_without_amazon_column_names(tmp_path):
    _, _, config = write_tiny_tables(tmp_path)
    metrics = evaluate_paper_metrics(config, tmp_path / "out")

    assert metrics["paper_metrics_summary"]["constraint_violation_rate"] == 0.0
    assert "customer_id" not in str(metrics)
    assert "review_text" not in str(metrics)

