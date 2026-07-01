from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reldiff.generation.continuous_time_temporal_sbm import (  # noqa: E402
    ContinuousKDESampler,
    ContinuousTimeTemporalSBMGenerator,
)


def make_tiny_amazon():
    customers = pd.DataFrame({"customer_id": [f"c{i}" for i in range(5)]})
    products = pd.DataFrame({"product_id": [f"p{i}" for i in range(4)]})
    review_rows = [
        ("c0", "p0", "2020-01-01 09:00:00"),
        ("c0", "p1", "2020-01-01 09:05:00"),
        ("c1", "p0", "2020-01-01 09:06:00"),
        ("c1", "p1", "2020-01-01 10:30:00"),
        ("c2", "p2", "2020-01-03 12:00:00"),
        ("c2", "p2", "2020-01-03 12:04:00"),
        ("c3", "p2", "2020-01-03 12:08:00"),
        ("c3", "p3", "2020-01-04 20:00:00"),
        ("c4", "p3", "2020-01-05 21:00:00"),
        ("c4", "p0", "2020-01-06 22:00:00"),
        ("c0", "p0", "2020-01-07 08:00:00"),
        ("c1", "p1", "2020-01-07 08:03:00"),
        ("c2", "p2", "2020-01-08 18:00:00"),
        ("c3", "p3", "2020-01-08 18:01:00"),
        ("c4", "p3", "2020-01-08 18:02:00"),
        ("c0", "p1", "2020-01-09 06:00:00"),
        ("c1", "p0", "2020-01-09 06:05:00"),
        ("c2", "p2", "2020-01-10 23:00:00"),
        ("c3", "p2", "2020-01-10 23:02:00"),
        ("c4", "p3", "2020-01-10 23:03:00"),
    ]
    reviews = pd.DataFrame(
        review_rows, columns=["customer_id", "product_id", "review_time"]
    )
    return customers, products, reviews


def test_continuous_time_temporal_sbm_generator_outputs_event_spine(tmp_path):
    customers, products, reviews = make_tiny_amazon()
    output_path = tmp_path / "synthetic_review.csv"
    debug_dir = tmp_path / "debug"

    generator = ContinuousTimeTemporalSBMGenerator(
        customers, products, reviews, seed=7
    )
    generator.fit()
    synthetic = generator.generate(output_path=output_path, debug_dir=debug_dir)

    assert output_path.exists()
    assert list(synthetic.columns) == ["customer_id", "product_id", "review_time"]
    assert len(synthetic) == len(reviews)
    assert set(synthetic["customer_id"]).issubset(set(customers["customer_id"]))
    assert set(synthetic["product_id"]).issubset(set(products["product_id"]))

    synthetic_times = pd.to_datetime(synthetic["review_time"])
    real_times = pd.to_datetime(reviews["review_time"])
    assert synthetic_times.notnull().all()
    assert synthetic_times.min() >= real_times.min()
    assert synthetic_times.max() <= real_times.max()
    assert synthetic_times.nunique() > 1

    summary_path = debug_dir / "temporal_sbm_summary.json"
    assert summary_path.exists()
    with summary_path.open() as handle:
        summary = json.load(handle)
    assert summary["generator"] == "continuous_time_temporal_sbm"
    assert summary["num_real_reviews"] == len(reviews)
    assert summary["num_synthetic_reviews"] == len(reviews)
    assert summary["sbm_block_level_requested"] == "auto"
    assert "sbm_block_level_resolved" in summary
    assert "num_nonzero_block_pairs_real" in summary
    assert "num_nonzero_block_pairs_synthetic" in summary

    for filename in ("customer_blocks.csv", "product_blocks.csv", "block_pair_counts.csv"):
        assert (debug_dir / filename).exists()

    block_pairs = pd.read_csv(debug_dir / "temporal_sbm_block_pairs.csv")
    assert (
        block_pairs["real_event_count"] == block_pairs["synthetic_event_count"]
    ).all()


def test_timestamp_sampler_is_kde_not_coarse_bin_uniform():
    source = inspect.getsource(ContinuousKDESampler.sample)
    assert "normal" in source
    assert "uniform" not in source
