from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_temporal_nontext_attrs import evaluate_nontext_attrs, load_reviews  # noqa: E402
from reldiff.attributes import TemporalNonTextAttributeDiffusionV3  # noqa: E402
from reldiff.attributes.block_attribute_priors import BlockAttributePrior  # noqa: E402
from reldiff.attributes.temporal_priors import TemporalAttributePrior  # noqa: E402


def make_reviews():
    rows = []
    customers = [f"c{i}" for i in range(5)]
    products = [f"p{i}" for i in range(4)]
    dates = pd.date_range("2020-01-01", periods=64, freq="D")
    for idx in range(64):
        product = products[(idx + idx // 4) % len(products)]
        rating = {"p0": 1, "p1": 2, "p2": 4, "p3": 5}[product]
        if dates[idx].month == 2:
            rating = min(5, rating + 1)
        rows.append((customers[idx % 5], product, dates[idx].strftime("%Y-%m-%d"), rating, product in {"p2", "p3"}))
    return pd.DataFrame(rows, columns=["customer_id", "product_id", "review_time", "rating", "verified"])


def write_debug(path: Path, reviews: pd.DataFrame):
    path.mkdir(parents=True)
    pd.DataFrame({"customer_id": sorted(reviews["customer_id"].unique()), "customer_block": [0, 1, 0, 1, 0]}).to_csv(path / "customer_blocks.csv", index=False)
    pd.DataFrame({"product_id": sorted(reviews["product_id"].unique()), "product_block": [0, 1, 0, 1]}).to_csv(path / "product_blocks.csv", index=False)


def test_base_logits_decomposition():
    gen = TemporalNonTextAttributeDiffusionV3(cat_cols=["rating", "verified"])
    gen.category_values = {"rating": [1, 2], "verified": [False, True]}
    gen.temporal_prior = TemporalAttributePrior([1, 2], temporal_prior_level="global")
    gen.temporal_prior.rating_global_distribution = [0.25, 0.75]
    gen.temporal_prior.verified_global_rate = 0.5
    gen.temporal_prior.per_bucket_rating_distribution = {"global": [0.25, 0.75]}
    gen.temporal_prior.per_bucket_verified_rate = {"global": 0.5}
    gen.block_prior = BlockAttributePrior([1, 2])
    gen.block_prior.block_pair_rating_residuals = {"0:0": [0.1, -0.1]}
    gen.block_prior.block_pair_verified_residuals = {"0:0": 0.2}
    gen.lambda_block = 2.0
    gen.lambda_product_effect = 3.0
    gen.lambda_customer_effect = 4.0
    rows = pd.DataFrame({"customer_id": ["c"], "product_id": ["p"], "review_time": ["2020-01-01"]})
    latent = pd.DataFrame(
        {
            "customer_rating_effect_0": [0.01],
            "customer_rating_effect_1": [0.02],
            "customer_verified_effect": [0.0],
            "product_rating_effect_0": [0.03],
            "product_rating_effect_1": [0.04],
            "product_verified_effect": [0.0],
        }
    )
    rating, verified = gen.compute_base_logits(rows, latent, {"c": 0}, {"p": 0})
    expected = np.log(np.asarray([[0.25, 0.75]])) + 2.0 * np.asarray([[0.1, -0.1]]) + 3.0 * np.asarray([[0.03, 0.04]]) + 4.0 * np.asarray([[0.01, 0.02]])
    assert np.allclose(rating, expected, atol=1e-6)
    assert verified.shape == (1, 2)


def test_v3_train_sample_and_evaluate(tmp_path):
    reviews = make_reviews()
    train_path = tmp_path / "review.csv"
    reviews.to_csv(train_path, index=False)
    debug = tmp_path / "debug"
    write_debug(debug, reviews)
    result = TemporalNonTextAttributeDiffusionV3.train_from_csv(
        train_path,
        output_dir=tmp_path / "v3",
        structure_debug_dir=debug,
        cat_cols=["rating", "verified"],
        epochs=1,
        batch_size=16,
        hidden_dim=64,
        num_layers=2,
        min_entities_per_cell=2,
        seed=3,
    )
    assert result.best_checkpoint.exists()
    assert (tmp_path / "v3" / "temporal_priors.json").exists()
    spine = reviews[["customer_id", "product_id", "review_time"]].iloc[:24].copy()
    spine_path = tmp_path / "spine.csv"
    output_path = tmp_path / "synthetic_review_nontext_v3.csv"
    spine.to_csv(spine_path, index=False)
    synthetic = TemporalNonTextAttributeDiffusionV3.sample_from_checkpoint(
        spine_path,
        result.best_checkpoint,
        output_path,
        structure_debug_dir=debug,
        seed=5,
        num_steps=3,
        cat_sampling_strategy="argmax",
        use_temporal_calibration=True,
    )
    with (tmp_path / "synthetic_review_nontext_v3_metadata.json").open() as handle:
        metadata = json.load(handle)
    assert metadata["method"] == "temporal_nontext_attr_diffusion_v3"
    assert metadata["uses_real_entity_effect_lookup"] is False
    assert metadata["samples_entity_effects_from_prior"] is True
    assert metadata["uses_temporal_calibration"] is True
    assert (tmp_path / "sampled_customer_effects_v3.csv").exists()
    assert (tmp_path / "sampled_product_effects_v3.csv").exists()
    assert set(synthetic["rating"]).issubset(set(reviews["rating"]))
    assert set(synthetic["verified"]).issubset(set(reviews["verified"]))
    metrics = evaluate_nontext_attrs(
        load_reviews(train_path, "review_time"),
        load_reviews(output_path, "review_time"),
        customer_col="customer_id",
        product_col="product_id",
        timestamp_col="review_time",
        cat_cols=["rating", "verified"],
        num_cols=[],
    )
    assert "monthly_rating_distribution_js_mean" in metrics["temporal"]
    assert "monthly_verified_rate_mae" in metrics["temporal"]
    assert "decomposition" in metrics
