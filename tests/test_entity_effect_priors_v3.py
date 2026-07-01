from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from reldiff.attributes.entity_effect_priors_v3 import (  # noqa: E402
    estimate_entity_effects_v3,
    fit_entity_priors_v3,
)


def test_rating_effect_vector_is_positive_for_dominant_class_and_centered(tmp_path):
    reviews = pd.DataFrame(
        {
            "customer_id": [f"c{i % 4}" for i in range(40)],
            "product_id": ["p_good"] * 30 + ["p_base"] * 10,
            "review_time": pd.date_range("2020-01-01", periods=40, freq="D"),
            "rating": [5] * 30 + [1, 2, 3, 4, 5] * 2,
            "verified": [True] * 40,
        }
    )
    debug = tmp_path / "debug"
    debug.mkdir()
    pd.DataFrame({"customer_id": reviews["customer_id"].unique(), "customer_block": 0}).to_csv(debug / "customer_blocks.csv", index=False)
    pd.DataFrame({"product_id": reviews["product_id"].unique(), "product_block": 0}).to_csv(debug / "product_blocks.csv", index=False)
    _, products, _ = estimate_entity_effects_v3(reviews, [1, 2, 3, 4, 5], structure_debug_dir=debug)
    good = products.set_index("product_id").loc["p_good"]
    assert good["rating_effect_4"] > 0
    assert abs(np.mean([good[f"rating_effect_{i}"] for i in range(5)])) < 1e-8


def test_v3_prior_sampling_no_nans(tmp_path):
    effects = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(8)],
            "customer_block": [0, 0, 0, 0, 1, 1, 1, 1],
            "degree": [1, 2, 3, 4, 5, 6, 7, 8],
            "rating_effect_0": np.linspace(-0.2, 0.2, 8),
            "rating_effect_1": np.linspace(0.2, -0.2, 8),
            "verified_effect": np.linspace(-0.5, 0.5, 8),
        }
    )
    customer_prior, _ = fit_entity_priors_v3(
        effects,
        effects.rename(columns={"customer_id": "product_id", "customer_block": "product_block"}),
        min_entities_per_cell=2,
    )
    sample = customer_prior.sample(
        pd.DataFrame({"customer_id": ["x", "y"], "customer_block": [0, 99], "degree": [2, 10]}),
        rng=np.random.default_rng(1),
    )
    assert len(sample) == 2
    assert sample.filter(like="sampled_").notna().all().all()
