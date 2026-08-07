from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.numerical_type import (  # noqa: E402
    NumericalTypeThresholds,
    infer_numerical_column_type,
    infer_numerical_types,
)


def test_infers_low_cardinality_discrete_numerical():
    report = infer_numerical_column_type(
        pd.Series(np.tile([0.0, 1.0, 2.0], 100))
    )

    assert report["label"] == "low_cardinality_discrete_numerical"
    assert report["recommended_head"] == "discrete_support"
    assert report["training_only"] is True


def test_infers_repeated_quantized_direct_support():
    values = np.repeat(np.linspace(0.0, 1.0, 100), 20)

    report = infer_numerical_column_type(pd.Series(values))

    assert report["label"] == "repeated_or_quantized"
    assert report["recommended_head"] == "discrete_support"
    assert report["structured_signal_count"] >= 3


def test_infers_high_cardinality_structured_support():
    support = np.linspace(0.0, 1.0, 500)
    values = np.repeat(support, 4)
    thresholds = NumericalTypeThresholds(
        direct_support_max_values=100,
        minimum_structured_signals=3,
    )

    report = infer_numerical_column_type(
        pd.Series(values),
        thresholds=thresholds,
    )

    assert report["label"] == "high_cardinality_structured_support"
    assert report["recommended_head"] == "hierarchical_support"


def test_infers_continuous_for_nearly_unique_irregular_values():
    rng = np.random.default_rng(42)
    values = rng.normal(size=2000)

    report = infer_numerical_column_type(pd.Series(values))

    assert report["label"] == "continuous"
    assert report["recommended_head"] == "continuous_baseline"


def test_multi_column_report_is_schema_driven():
    train = pd.DataFrame(
        {
            "small": np.tile([1.0, 2.0], 100),
            "continuous": np.linspace(0.0, 1.0, 200)
            + np.arange(200) ** 2 * 1e-9,
        }
    )

    report = infer_numerical_types(
        train,
        ("small", "continuous"),
        config={"low_cardinality_max_values": 10},
    )

    assert set(report) == {"small", "continuous"}
    assert report["small"]["recommended_head"] == "discrete_support"
