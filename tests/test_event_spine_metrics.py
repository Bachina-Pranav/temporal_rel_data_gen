from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.event_spine_metrics import evaluate_event_spine  # noqa: E402


def test_event_spine_metrics_run_on_tiny_data(tmp_path):
    real = pd.DataFrame(
        {
            "customer_id": ["c0", "c0", "c1", "c2"],
            "product_id": ["p0", "p1", "p0", "p1"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "customer_id": ["c0", "c1", "c0", "c2"],
            "product_id": ["p0", "p0", "p1", "p1"],
            "review_time": ["2020-01-01", "2020-01-03", "2020-01-02", "2020-01-04"],
        }
    )
    debug = tmp_path / "debug"
    debug.mkdir()
    pd.DataFrame({"customer_id": ["c0", "c1", "c2"], "customer_block": [0, 0, 1]}).to_csv(debug / "customer_blocks.csv", index=False)
    pd.DataFrame({"product_id": ["p0", "p1"], "product_block": [0, 1]}).to_csv(debug / "product_blocks.csv", index=False)

    metrics = evaluate_event_spine(real, synthetic, structure_debug_dir=debug, compute_c2st=True)

    assert "product_first_time_corr" in metrics
    assert "joint_coactive_window_rate" in metrics
    assert "event_tuple_c2st_accuracy" in metrics
    assert metrics["num_reviews_real"] == 4
    assert metrics["num_reviews_synthetic"] == 4


def test_event_spine_metrics_report_real_and_synthetic_duplicate_rates(tmp_path):
    real = pd.DataFrame(
        {
            "customer_id": ["c0", "c0", "c1", "c2"],
            "product_id": ["p0", "p0", "p1", "p2"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "customer_id": ["c0", "c0", "c1", "c1"],
            "product_id": ["p0", "p0", "p1", "p1"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        }
    )

    metrics = evaluate_event_spine(real, synthetic, structure_debug_dir=tmp_path)

    assert metrics["real_duplicate_customer_product_rate"] == 0.5
    assert metrics["synthetic_duplicate_customer_product_rate"] == 1.0
    assert metrics["duplicate_customer_product_rate"] == 1.0
    assert metrics["duplicate_rate_ratio"] == 2.0
