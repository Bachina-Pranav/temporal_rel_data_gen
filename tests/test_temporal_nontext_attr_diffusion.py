from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_temporal_nontext_attrs import evaluate_nontext_attrs, load_reviews  # noqa: E402
from reldiff.attributes import TemporalNonTextAttributeDiffusion  # noqa: E402
from reldiff.generation import (  # noqa: E402
    ContinuousTime2KSBMTemporalKDEStubsGenerator,
    ContinuousTimeTemporalSBMGenerator,
)


def make_tiny_reviews():
    customers = [f"c{i}" for i in range(5)]
    products = [f"p{i}" for i in range(4)]
    rows = []
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    for index in range(50):
        customer_id = customers[index % len(customers)]
        product_id = products[(index + index // 5) % len(products)]
        rating = 1 + ((index + products.index(product_id)) % 5)
        verified = bool((index + customers.index(customer_id)) % 2)
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


def test_temporal_nontext_train_sample_and_evaluate(tmp_path):
    reviews = make_tiny_reviews()
    train_path = tmp_path / "review.csv"
    reviews.to_csv(train_path, index=False)
    debug_dir = tmp_path / "structure_debug"
    write_block_debug(debug_dir, reviews)

    result = TemporalNonTextAttributeDiffusion.train_from_csv(
        train_path,
        output_dir=tmp_path / "nontext",
        structure_debug_dir=debug_dir,
        cat_cols=["rating", "verified"],
        num_cols=["helpful_vote"],
        epochs=1,
        batch_size=16,
        hidden_dim=64,
        num_layers=2,
        seed=7,
    )
    assert result.best_checkpoint.exists()
    assert (tmp_path / "nontext" / "category_mappings.json").exists()
    with (tmp_path / "nontext" / "category_mappings.json").open() as handle:
        mappings = json.load(handle)
    assert "rating" in mappings["category_values"]
    assert "verified" in mappings["category_values"]

    spine = reviews[["customer_id", "product_id", "review_time"]].iloc[:20].copy()
    spine_path = tmp_path / "spine.csv"
    output_path = tmp_path / "synthetic_nontext.csv"
    spine.to_csv(spine_path, index=False)

    synthetic = TemporalNonTextAttributeDiffusion.sample_from_checkpoint(
        synthetic_spine_path=spine_path,
        checkpoint_path=result.best_checkpoint,
        output_path=output_path,
        structure_debug_dir=debug_dir,
        seed=11,
        num_steps=3,
        cat_sampling_strategy="argmax",
    )

    assert output_path.exists()
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
    assert "rating_distribution_js" in metrics["categorical"]
    assert "verified_distribution_js" in metrics["categorical"]
    json.dumps(metrics)


def test_existing_structure_generators_still_import():
    assert ContinuousTimeTemporalSBMGenerator is not None
    assert ContinuousTime2KSBMTemporalKDEStubsGenerator is not None
