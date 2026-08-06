from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.numerical_support import (  # noqa: E402
    decimal_precision,
    infer_support_kind,
    numerical_support_profile,
    project_numerical_support,
)


def test_support_profile_detects_repeated_quantized_values():
    train = pd.Series([0.01] * 60 + [0.02] * 30 + [0.03] * 10)
    test = pd.Series([0.01, 0.02, 0.03, 0.01])
    synthetic = pd.Series([0.010001, 0.019999, 0.025001, 0.03])

    report = numerical_support_profile(train, test, synthetic)

    inferred = report["training_support"]["inferred_support_kind"]
    assert inferred["quantized"] is True
    assert report["test"]["exact_training_support_overlap_rate"] == 1.0
    assert report["synthetic"]["exact_training_support_overlap_rate"] == 0.25
    assert report["synthetic_nearest_training_support"]["mean"] > 0.0


def test_global_nearest_projection_uses_only_observed_support():
    projected, metadata = project_numerical_support(
        pd.Series([0.01, 0.02, 0.03]),
        pd.Series([0.011, 0.026, np.nan]),
        mode="global_nearest",
        seed=42,
    )

    assert projected[:2].tolist() == [0.01, 0.03]
    assert np.isnan(projected[2])
    assert metadata["output_exact_training_support_rate"] == 1.0


def test_stochastic_projection_is_deterministic_for_fixed_seed():
    kwargs = {
        "train_values": pd.Series([0.01] * 5 + [0.02] * 3 + [0.03]),
        "generated_values": pd.Series([0.015] * 50),
        "mode": "global_stochastic",
        "seed": 73,
        "stochastic_neighbors": 3,
        "stochastic_temperature": 1.0,
    }
    first, _ = project_numerical_support(**kwargs)
    second, _ = project_numerical_support(**kwargs)

    assert np.array_equal(first, second)
    assert set(first).issubset({0.01, 0.02, 0.03})


def test_entity_projection_uses_entity_bucket_then_global_fallback():
    train_values = pd.Series(
        [0.01, 0.01, 0.02, 0.02, 0.04, 0.04, 0.05]
    )
    train_entities = pd.Series(["a", "a", "a", "b", "b", "b", "c"])
    generated = pd.Series([0.039, 0.049, 0.031])
    query_entities = pd.Series(["a", "b", "unseen"])

    projected, metadata = project_numerical_support(
        train_values,
        generated,
        mode="entity_nearest",
        seed=42,
        train_entities=train_entities,
        query_entities=query_entities,
        min_entity_rows=2,
    )

    assert projected[0] == 0.02
    assert projected[1] == 0.04
    assert projected[2] in set(train_values)
    assert metadata["fallback_hierarchy"] == [
        "entity",
        "entity_frequency_bucket",
        "global",
    ]


def test_learned_bins_remain_on_training_support():
    train = pd.Series(np.repeat(np.linspace(0.0, 1.0, 500), 2))
    generated = pd.Series(np.linspace(-0.1, 1.1, 100))
    projected, metadata = project_numerical_support(
        train,
        generated,
        mode="learned_bins",
        seed=42,
        max_learned_bins=16,
    )

    assert set(projected).issubset(set(train))
    assert metadata["num_learned_bins"] <= 16


def test_genuinely_continuous_high_cardinality_support_is_not_quantized():
    support = np.linspace(0.0, 1.0, 1000)
    counts = np.ones(1000, dtype=int)

    inferred = infer_support_kind(1000, support, counts)

    assert inferred["quantized"] is False
    assert inferred["label"] == "continuous"


def test_precision_diagnostic_does_not_round_values_arbitrarily():
    assert decimal_precision(0.02542370) >= 7
    assert decimal_precision(0.02) == 2


def test_support_projection_preserves_event_spine_columns():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u2"],
            "article_id": ["a1", "a2"],
            "event_time": ["2020-01-01", "2020-01-02"],
            "price": [0.011, 0.029],
        }
    )
    before = frame[["customer_id", "article_id", "event_time"]].copy()
    projected, _ = project_numerical_support(
        pd.Series([0.01, 0.02, 0.03]),
        frame["price"],
        mode="global_nearest",
        seed=42,
    )
    frame["price"] = projected

    pd.testing.assert_frame_equal(
        before,
        frame[["customer_id", "article_id", "event_time"]],
    )
