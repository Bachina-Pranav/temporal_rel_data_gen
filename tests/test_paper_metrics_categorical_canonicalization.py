from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.shape_trend import shape_metrics  # noqa: E402


def test_categorical_canonicalization_prevents_numeric_string_shape_error():
    real = pd.DataFrame({"rating": [1, 2, 3, 4, 5]})
    synthetic = pd.DataFrame({"rating": ["1.0", "2", "3", "4", "5"]})
    table_config = {
        "columns": {
            "rating": {"type": "categorical", "dtype": "int", "valid_values": [1, 2, 3, 4, 5]},
        }
    }

    shape, _ = shape_metrics(real, synthetic, table_config)

    assert shape["per_column"]["rating"]["shape_error"] == 0.0
