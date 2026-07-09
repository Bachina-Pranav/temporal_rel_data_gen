from __future__ import annotations

import sys
from pathlib import Path

from test_paper_metrics_report_outputs import write_tiny_tables


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics  # noqa: E402


def test_paper_metrics_skips_full_relational_metrics(tmp_path):
    _, _, config = write_tiny_tables(tmp_path)
    metrics = evaluate_paper_metrics(config, tmp_path / "out")

    assert metrics["skipped_metrics"]["k_hop_relational_correlation"]["status"] == "skipped"
    assert metrics["skipped_metrics"]["c2st_agg"]["status"] == "skipped"
    assert "requires full multi-table relational generation" in metrics["skipped_metrics"]["c2st_agg"]["reason"]
