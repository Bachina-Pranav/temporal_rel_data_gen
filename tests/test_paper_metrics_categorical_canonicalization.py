from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.shape_trend import shape_metrics  # noqa: E402
from scripts.evaluate_lstm_attribute_diagnostics import (  # noqa: E402
    categorical_metrics,
)
from scripts.audit_lstm_categorical_validity import audit_field  # noqa: E402


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


def test_attribute_diagnostics_canonicalize_integer_categories():
    metrics = categorical_metrics(
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1, 2, 3]),
        column_config={
            "type": "categorical",
            "dtype": "int",
            "valid_values": [1, 2, 3],
        },
    )

    assert metrics["invalid_category_rate"] == 0.0
    assert metrics["total_variation_distance"] == 0.0
    assert metrics["canonicalization_applied"] is True


def test_attribute_diagnostics_empty_category_metric_is_not_failure():
    metrics = categorical_metrics(
        pd.Series(["a", "b"]),
        pd.Series(["a", "b"]),
        pd.Series([None, None]),
    )

    assert metrics["invalid_category_rate"] is None


def test_validity_audit_identifies_numeric_representation_bug():
    result = audit_field(
        pd.Series([1, 2, 3]),
        ["1.0", "2.0", "3.0"],
        {
            "type": "categorical",
            "dtype": "int",
            "valid_values": [1, 2, 3],
        },
        dataset="amazon_toy",
        model="example",
        seed=42,
        column="rating",
        stored_invalid_rate=1.0,
    )

    assert result["raw_invalid_rate"] == 1.0
    assert result["canonical_invalid_rate"] == 0.0
    assert result["status"] == "evaluator_representation_bug"
