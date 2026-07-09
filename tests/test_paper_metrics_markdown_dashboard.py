from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from evaluation.paper_metrics.reporting import write_markdown_report  # noqa: E402
from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics  # noqa: E402
from test_paper_metrics_report_outputs import write_tiny_tables  # noqa: E402


def test_metrics_markdown_starts_with_dashboard_and_stays_compact(tmp_path):
    _, _, config = write_tiny_tables(tmp_path)
    output_dir = tmp_path / "out"
    metrics = evaluate_paper_metrics(config, output_dir)
    write_markdown_report(metrics, output_dir / "metrics.md")

    text = (output_dir / "metrics.md").read_text(encoding="utf-8")

    assert text.startswith("# Main Dashboard")
    assert "feature_importances" not in text
    assert "# Skipped Full-Relational Metrics" in text
    assert "requires full multi-table relational generation" in text
