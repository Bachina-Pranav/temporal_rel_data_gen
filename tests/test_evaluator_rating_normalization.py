from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.evaluate import evaluate_frames, normalize_rating_series  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402


def make_config(categorical_targets: tuple[str, ...] = ("rating",)) -> ConditionalTABDLMConfig:
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=categorical_targets,
    )
    raw = {
        "paths": {
            "train_data_path": "unused.csv",
            "synthetic_spine_path": "unused.csv",
            "output_dir": "unused",
        }
    }
    return ConditionalTABDLMConfig(raw=raw, schema=schema)


def test_rating_distribution_normalizes_int_and_float_string_labels():
    real = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(5)],
            "product_id": [f"p{i}" for i in range(5)],
            "review_time": ["2020-01-01"] * 5,
            "rating": ["1.0", "2.0", "3.0", "4.0", "5.0"],
        }
    )
    synthetic = real.copy()
    synthetic["rating"] = [1, 2, 3, 4, 5]

    metrics = evaluate_frames(real, synthetic, make_config())
    marginal = metrics["marginal_categorical"]

    assert marginal["rating_distribution_l1"] == pytest.approx(0.0)
    assert marginal["rating_distribution_js"] == pytest.approx(0.0)
    assert marginal["rating_ks"] == pytest.approx(0.0)
    assert metrics["validity"]["invalid_rating_rate"] == pytest.approx(0.0)


def test_invalid_rating_normalization_mask_catches_bad_values():
    _, invalid_mask = normalize_rating_series(pd.Series([0, 6, "bad", np.nan, "", 4.5]))

    assert invalid_mask.tolist() == [True, True, True, True, True, True]
