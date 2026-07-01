from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_temporal_nontext_attrs import evaluate_nontext_attrs, load_reviews  # noqa: E402
from reldiff.attributes import TemporalNonTextAttributeDiffusionV2  # noqa: E402
from reldiff.attributes.entity_effect_priors import (  # noqa: E402
    fit_customer_product_priors,
    save_customer_product_priors,
)
from reldiff.attributes.entity_latent_effects import (  # noqa: E402
    estimate_entity_latent_effects,
    save_entity_effect_estimate,
)


def make_tiny_reviews():
    customers = [f"c{i}" for i in range(5)]
    products = [f"p{i}" for i in range(4)]
    rows = []
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    product_bias = {"p0": 1, "p1": 2, "p2": 4, "p3": 5}
    for index in range(60):
        customer_id = customers[index % len(customers)]
        product_id = products[(index + index // 4) % len(products)]
        rating = product_bias[product_id]
        verified = product_id in {"p2", "p3"} or customer_id == "c0"
        rows.append((customer_id, product_id, dates[index].strftime("%Y-%m-%d"), rating, verified))
    return pd.DataFrame(
        rows,
        columns=["customer_id", "product_id", "review_time", "rating", "verified"],
    )


def write_block_debug(debug_dir: Path, reviews: pd.DataFrame) -> None:
    debug_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "customer_id": sorted(reviews["customer_id"].unique()),
            "customer_block": [i % 2 for i, _ in enumerate(sorted(reviews["customer_id"].unique()))],
        }
    ).to_csv(debug_dir / "customer_blocks.csv", index=False)
    pd.DataFrame(
        {
            "product_id": sorted(reviews["product_id"].unique()),
            "product_block": [i % 2 for i, _ in enumerate(sorted(reviews["product_id"].unique()))],
        }
    ).to_csv(debug_dir / "product_blocks.csv", index=False)


def train_tiny_priors(reviews: pd.DataFrame, debug_dir: Path, prior_dir: Path) -> None:
    estimate = estimate_entity_latent_effects(reviews, structure_debug_dir=debug_dir)
    save_entity_effect_estimate(estimate, prior_dir)
    customer_prior, product_prior, diagnostics = fit_customer_product_priors(
        estimate.customer_effects,
        estimate.product_effects,
        min_entities_per_cell=2,
    )
    save_customer_product_priors(prior_dir, customer_prior, product_prior, diagnostics)


def test_temporal_nontext_v2_train_sample_no_real_lookup_and_evaluate(tmp_path):
    reviews = make_tiny_reviews()
    train_path = tmp_path / "review.csv"
    reviews.to_csv(train_path, index=False)
    debug_dir = tmp_path / "structure_debug"
    write_block_debug(debug_dir, reviews)
    prior_dir = tmp_path / "entity_effect_priors"
    train_tiny_priors(reviews, debug_dir, prior_dir)

    result = TemporalNonTextAttributeDiffusionV2.train_from_csv(
        train_path,
        output_dir=tmp_path / "nontext_v2",
        structure_debug_dir=debug_dir,
        entity_prior_dir=prior_dir,
        cat_cols=["rating", "verified"],
        epochs=1,
        batch_size=16,
        hidden_dim=64,
        num_layers=2,
        effect_noise_std=0.01,
        effect_dropout=0.0,
        seed=7,
    )
    assert result.best_checkpoint.exists()

    # Poison posterior effect files. Default V2 sampling must not read these by ID.
    customer_effects = pd.read_csv(prior_dir / "entity_effects" / "customer_effects.csv")
    product_effects = pd.read_csv(prior_dir / "entity_effects" / "product_effects.csv")
    customer_effects["rating_effect"] = 999.0
    product_effects["rating_effect"] = 999.0
    customer_effects.to_csv(prior_dir / "entity_effects" / "customer_effects.csv", index=False)
    product_effects.to_csv(prior_dir / "entity_effects" / "product_effects.csv", index=False)

    spine = reviews[["customer_id", "product_id", "review_time"]].iloc[:24].copy()
    spine_path = tmp_path / "spine.csv"
    output_path = tmp_path / "synthetic_review_nontext_v2.csv"
    spine.to_csv(spine_path, index=False)

    synthetic = TemporalNonTextAttributeDiffusionV2.sample_from_checkpoint(
        synthetic_spine_path=spine_path,
        checkpoint_path=result.best_checkpoint,
        output_path=output_path,
        structure_debug_dir=debug_dir,
        entity_prior_dir=prior_dir,
        seed=11,
        num_steps=3,
        cat_sampling_strategy="argmax",
    )
    metadata_path = tmp_path / "synthetic_review_nontext_v2_metadata.json"
    assert output_path.exists()
    assert metadata_path.exists()
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    assert metadata["method"] == "temporal_nontext_attr_diffusion_v2"
    assert metadata["uses_real_entity_effect_lookup"] is False
    assert metadata["samples_entity_effects_from_prior"] is True
    assert (tmp_path / "sampled_customer_effects.csv").exists()
    assert (tmp_path / "sampled_product_effects.csv").exists()
    sampled_products = pd.read_csv(tmp_path / "sampled_product_effects.csv")
    assert sampled_products["sampled_rating_effect"].abs().max() < 10

    assert list(synthetic.columns) == [
        "customer_id",
        "product_id",
        "review_time",
        "rating",
        "verified",
    ]
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
    entity_metrics = metrics["entity_distribution"]
    assert "product_avg_rating_distribution_ks" in entity_metrics
    assert "customer_avg_rating_distribution_ks" in entity_metrics
    assert "product_verified_rate_distribution_ks" in entity_metrics
    assert "customer_verified_rate_distribution_ks" in entity_metrics
    json.dumps(metrics)
