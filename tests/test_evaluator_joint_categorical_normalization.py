from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.evaluate import evaluate_frames  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402


def test_rating_verified_joint_metrics_use_normalized_fixed_support():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
    )
    config = ConditionalTABDLMConfig(
        raw={
            "paths": {
                "train_data_path": "unused.csv",
                "synthetic_spine_path": "unused.csv",
                "output_dir": "unused",
            }
        },
        schema=schema,
    )
    real = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "product_id": ["p1", "p2"],
            "review_time": ["2020-01-01", "2020-01-02"],
            "rating": ["5.0", "4.0"],
            "verified": ["true", "false"],
        }
    )
    synthetic = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "product_id": ["p1", "p2"],
            "review_time": ["2020-01-01", "2020-01-02"],
            "rating": [5, 4],
            "verified": [1, 0],
        }
    )

    metrics = evaluate_frames(real, synthetic, config)

    assert metrics["validity"]["invalid_rating_rate"] == pytest.approx(0.0)
    assert metrics["validity"]["invalid_verified_rate"] == pytest.approx(0.0)
    assert metrics["joint"]["rating_verified_joint_l1"] == pytest.approx(0.0)
    assert metrics["joint"]["rating_distribution_given_verified_l1"] == pytest.approx(0.0)
