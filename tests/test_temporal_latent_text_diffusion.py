from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_temporal_attr_generation import evaluate_attributes  # noqa: E402
from reldiff.attributes import (  # noqa: E402
    TemporalLatentTextAttributeDiffusion,
    TemporalReviewNeighborSampler,
    TextLatentEncoder,
)
from reldiff.generation import (  # noqa: E402
    ContinuousTime2KSBMPlusGenerator,
    ContinuousTime2KSBMTemporalKDEStubsGenerator,
    ContinuousTime2KSBMTemporalStubsGenerator,
    ContinuousTimeTemporalSBMGenerator,
)


def make_tiny_reviews():
    customers = [f"c{i}" for i in range(5)]
    products = [f"p{i}" for i in range(4)]
    rows = []
    pairs = [
        ("c0", "p0"),
        ("c0", "p1"),
        ("c0", "p2"),
        ("c1", "p0"),
        ("c1", "p1"),
        ("c1", "p2"),
        ("c1", "p3"),
        ("c2", "p0"),
        ("c2", "p1"),
        ("c3", "p1"),
        ("c3", "p2"),
        ("c3", "p3"),
        ("c4", "p0"),
        ("c4", "p1"),
        ("c4", "p2"),
    ]
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    for index in range(30):
        customer_id, product_id = pairs[index % len(pairs)]
        rating = 1 + (index % 5)
        verified = bool(index % 2)
        rows.append(
            {
                "customer_id": customer_id,
                "product_id": product_id,
                "review_time": dates[index].strftime("%Y-%m-%d"),
                "rating": rating,
                "verified": verified,
                "summary": f"short summary {rating}",
                "review_text": f"customer {customer_id} says product {product_id} has rating {rating}",
            }
        )
    return pd.DataFrame(rows), customers, products


def test_text_latent_encoder_runs_and_caches(tmp_path):
    reviews, _, _ = make_tiny_reviews()
    encoder = TextLatentEncoder(backend="hashing", latent_dim=32)
    latents = encoder.fit_transform_columns(
        reviews,
        ["summary", "review_text"],
        cache_dir=tmp_path,
        max_lengths={"summary": 64, "review_text": 128},
    )
    assert latents["summary"].shape == (30, 32)
    assert latents["review_text"].shape == (30, 32)
    assert (tmp_path / "summary_latents.npy").exists()
    assert (tmp_path / "review_text_latents.npy").exists()
    assert (tmp_path / "text_latent_metadata.json").exists()


def test_temporal_neighbor_sampler_has_no_future_reviews():
    reviews, _, _ = make_tiny_reviews()
    sampler = TemporalReviewNeighborSampler(
        reviews,
        temporal_mode="causal_window",
        temporal_window_days=365,
        max_customer_history=8,
        max_product_history=8,
    )
    sampler.assert_no_future(range(len(reviews)))
    for target_index in range(len(reviews)):
        neighborhood = sampler.sample(target_index)
        target_time = pd.to_datetime(reviews.loc[target_index, "review_time"])
        for neighbor_index in neighborhood.review_indices:
            assert pd.to_datetime(reviews.loc[neighbor_index, "review_time"]) <= target_time


def test_temporal_latent_text_diffusion_train_sample_and_evaluate(tmp_path):
    reviews, _, _ = make_tiny_reviews()
    train_path = tmp_path / "review.csv"
    reviews.to_csv(train_path, index=False)

    result = TemporalLatentTextAttributeDiffusion.train_from_csv(
        reviews_path=train_path,
        output_dir=tmp_path / "attr",
        cat_cols=["rating", "verified"],
        text_cols=["summary", "review_text"],
        text_encoder_backend="hashing",
        text_latent_dim=32,
        epochs=1,
        batch_size=8,
        hidden_dim=48,
        seed=7,
    )
    assert result.best_checkpoint.exists()
    assert result.latest_checkpoint.exists()

    spine = reviews[["customer_id", "product_id", "review_time"]].iloc[:12].copy()
    spine_path = tmp_path / "synthetic_spine.csv"
    output_path = tmp_path / "synthetic_full.csv"
    spine.to_csv(spine_path, index=False)

    synthetic = TemporalLatentTextAttributeDiffusion.sample_from_checkpoint(
        synthetic_spine_path=spine_path,
        checkpoint_path=result.best_checkpoint,
        output_path=output_path,
        seed=11,
        num_steps=3,
        batch_size=6,
    )
    assert output_path.exists()
    assert list(synthetic.columns) == [
        "customer_id",
        "product_id",
        "review_time",
        "rating",
        "verified",
        "summary",
        "review_text",
    ]
    assert set(synthetic["rating"]).issubset(set(reviews["rating"]))
    assert set(synthetic["verified"]).issubset(set(reviews["verified"]))
    assert synthetic["summary"].notna().all()
    assert synthetic["review_text"].notna().all()
    assert synthetic["summary"].map(type).eq(str).all()
    assert synthetic["review_text"].map(type).eq(str).all()

    results = evaluate_attributes(
        reviews,
        synthetic,
        customer_col="customer_id",
        product_col="product_id",
        timestamp_col="review_time",
        rating_col="rating",
        verified_col="verified",
        summary_col="summary",
        review_text_col="review_text",
    )
    assert "categorical" in results
    assert "text_lexical" in results
    json.dumps(results)


def test_existing_structural_generators_still_import():
    assert ContinuousTimeTemporalSBMGenerator is not None
    assert ContinuousTime2KSBMPlusGenerator is not None
    assert ContinuousTime2KSBMTemporalStubsGenerator is not None
    assert ContinuousTime2KSBMTemporalKDEStubsGenerator is not None
