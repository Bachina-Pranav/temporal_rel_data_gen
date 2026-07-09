from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics  # noqa: E402


def test_paper_metrics_row_count_mismatch_warns_and_c2st_balances(tmp_path):
    real = pd.DataFrame({"value": range(100), "label": ["a", "b"] * 50})
    synthetic = pd.DataFrame({"value": range(50), "label": ["a", "b"] * 25})
    real_path = tmp_path / "real.csv"
    syn_path = tmp_path / "synthetic.csv"
    real.to_csv(real_path, index=False)
    synthetic.to_csv(syn_path, index=False)
    config = {
        "dataset_name": "toy",
        "real_table_path": str(real_path),
        "synthetic_table_path": str(syn_path),
        "table": {"columns": {"value": {"type": "numerical"}, "label": {"type": "categorical"}}},
        "evaluation": {"c2st": {"enabled": True, "classifiers": ["logistic_regression"], "max_rows": 100}},
    }

    metrics = evaluate_paper_metrics(config, tmp_path / "out")

    assert metrics["dataset"]["row_count_match"] is False
    assert metrics["dataset"]["row_count_ratio"] == 0.5
    assert any(item["code"] == "ROW_COUNT_MISMATCH" for item in metrics["evaluator_warnings"])
    assert metrics["single_table_c2st"]["balanced_eval_n_real"] == 50
    assert metrics["single_table_c2st"]["balanced_eval_n_synthetic"] == 50


def test_paper_metrics_row_count_match_has_no_mismatch_warning(tmp_path):
    real = pd.DataFrame({"value": range(20), "label": ["a", "b"] * 10})
    synthetic = real.copy()
    real_path = tmp_path / "real.csv"
    syn_path = tmp_path / "synthetic.csv"
    real.to_csv(real_path, index=False)
    synthetic.to_csv(syn_path, index=False)
    config = {
        "dataset_name": "toy",
        "real_table_path": str(real_path),
        "synthetic_table_path": str(syn_path),
        "table": {"columns": {"value": {"type": "numerical"}, "label": {"type": "categorical"}}},
        "evaluation": {"c2st": {"enabled": True, "classifiers": ["logistic_regression"], "max_rows": 20}},
    }

    metrics = evaluate_paper_metrics(config, tmp_path / "out")

    assert metrics["dataset"]["row_count_match"] is True
    assert not any(item["code"] == "ROW_COUNT_MISMATCH" for item in metrics["evaluator_warnings"])
