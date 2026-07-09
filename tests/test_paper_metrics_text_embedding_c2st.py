from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.text_embedding import text_embedding_c2st_metrics  # noqa: E402


def test_paper_metrics_text_embedding_c2st_runs_with_dummy_backend(tmp_path):
    real = pd.DataFrame({"description": ["good item", "nice product", "works well", "solid"] * 4})
    synthetic = pd.DataFrame({"description": ["bad item", "poor product", "broken soon", "weak"] * 4})
    config = {
        "table": {"columns": {"description": {"type": "text"}}},
        "evaluation": {
            "random_seed": 42,
            "text": {"embedding_model": "dummy", "text_columns": ["description"], "max_text_rows": 16, "cache_embeddings": True},
            "c2st": {"classifiers": ["logistic_regression"]},
        },
    }

    metrics = text_embedding_c2st_metrics(real, synthetic, config, tmp_path)

    assert metrics["macro_auc"] is not None
    assert metrics["macro_error"] is not None
    assert "description" in metrics["per_text_column"]

