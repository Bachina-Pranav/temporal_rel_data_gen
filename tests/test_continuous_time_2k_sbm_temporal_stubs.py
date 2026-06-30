from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_ct_2k_sbm_temporal_stubs import evaluate_temporal_stubs  # noqa: E402
from reldiff.generation import (  # noqa: E402
    ContinuousTime2KSBMPlusGenerator,
    ContinuousTime2KSBMTemporalStubsGenerator,
    ContinuousTimeTemporalSBMGenerator,
)
from reldiff.generation.continuous_time_temporal_sbm import (  # noqa: E402
    empirical_ks_statistic,
)


def make_midnight_amazon():
    customers = pd.DataFrame({"customer_id": [f"c{i}" for i in range(5)]})
    products = pd.DataFrame({"product_id": [f"p{i}" for i in range(4)]})
    pair_multiplicities = [
        ("c0", "p0", 3),
        ("c0", "p1", 2),
        ("c0", "p2", 1),
        ("c1", "p0", 3),
        ("c1", "p1", 2),
        ("c1", "p2", 1),
        ("c1", "p3", 1),
        ("c2", "p0", 4),
        ("c2", "p1", 2),
        ("c3", "p1", 3),
        ("c3", "p2", 2),
        ("c3", "p3", 1),
        ("c4", "p0", 2),
        ("c4", "p1", 1),
        ("c4", "p2", 2),
    ]
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    rows = []
    date_index = 0
    for customer_id, product_id, multiplicity in pair_multiplicities:
        for _ in range(multiplicity):
            rows.append((customer_id, product_id, dates[date_index].strftime("%Y-%m-%d")))
            date_index += 1
    reviews = pd.DataFrame(rows, columns=["customer_id", "product_id", "review_time"])
    return customers, products, reviews


def degree_counts(df, column):
    return df[column].value_counts().sort_index()


def block_pair_counts(df, customer_blocks, product_blocks):
    annotated = df.copy()
    annotated["customer_block"] = annotated["customer_id"].map(customer_blocks)
    annotated["product_block"] = annotated["product_id"].map(product_blocks)
    return annotated.groupby(["customer_block", "product_block"]).size().sort_index()


def normalized_against_real_range(real_times, synthetic_times):
    min_time = real_times.min()
    max_time = real_times.max()
    span = (max_time - min_time).total_seconds()
    if span <= 0:
        return [0.5] * len(real_times), [0.5] * len(synthetic_times)
    real_x = ((real_times - min_time).dt.total_seconds() / span).to_numpy()
    synthetic_x = ((synthetic_times - min_time).dt.total_seconds() / span).to_numpy()
    return real_x, synthetic_x


def test_temporal_stubs_and_old_methods_import():
    assert ContinuousTimeTemporalSBMGenerator is not None
    assert ContinuousTime2KSBMPlusGenerator is not None
    assert ContinuousTime2KSBMTemporalStubsGenerator is not None


def test_temporal_stubs_preserves_degrees_block_pairs_and_timestamps(tmp_path):
    customers, products, reviews = make_midnight_amazon()
    output_path = tmp_path / "synthetic_review.csv"
    debug_dir = tmp_path / "debug"

    generator = ContinuousTime2KSBMTemporalStubsGenerator(
        customers,
        products,
        reviews,
        seed=13,
        stub_pairing="temporal_window_shuffle",
        timestamp_stub_mode="reuse_block_pair_timestamps",
    )
    generator.fit()
    synthetic = generator.generate(output_path=output_path, debug_dir=debug_dir)

    assert output_path.exists()
    assert list(synthetic.columns) == ["customer_id", "product_id", "review_time"]
    assert len(synthetic) == len(reviews)
    assert set(synthetic["customer_id"]).issubset(set(customers["customer_id"]))
    assert set(synthetic["product_id"]).issubset(set(products["product_id"]))

    real_times = pd.to_datetime(reviews["review_time"])
    synthetic_times = pd.to_datetime(synthetic["review_time"])
    assert synthetic_times.min() >= real_times.min()
    assert synthetic_times.max() <= real_times.max()
    assert (synthetic_times.dt.hour == 0).all()
    assert (synthetic_times.dt.minute == 0).all()
    assert (synthetic_times.dt.second == 0).all()

    assert degree_counts(synthetic, "product_id").equals(
        degree_counts(reviews, "product_id")
    )
    assert degree_counts(synthetic, "customer_id").equals(
        degree_counts(reviews, "customer_id")
    )
    assert block_pair_counts(
        synthetic,
        generator.sbm_result.customer_blocks,
        generator.sbm_result.product_blocks,
    ).equals(
        block_pair_counts(
            reviews,
            generator.sbm_result.customer_blocks,
            generator.sbm_result.product_blocks,
        )
    )

    real_x, synthetic_x = normalized_against_real_range(real_times, synthetic_times)
    assert empirical_ks_statistic(real_x, synthetic_x) == pytest.approx(0.0)

    expected_debug_files = [
        "ct_2k_sbm_temporal_stubs_summary.json",
        "ct_2k_sbm_temporal_stubs_block_pairs.csv",
        "ct_2k_sbm_temporal_stubs_customer_degree_check.csv",
        "ct_2k_sbm_temporal_stubs_product_degree_check.csv",
        "ct_2k_sbm_temporal_stubs_timestamp_diagnostics.json",
        "ct_2k_sbm_temporal_stubs_pair_multiplicity.json",
    ]
    for filename in expected_debug_files:
        assert (debug_dir / filename).exists()

    with (debug_dir / "ct_2k_sbm_temporal_stubs_summary.json").open() as handle:
        summary = json.load(handle)
    assert summary["generator"] == "ct_2k_sbm_temporal_stubs"
    assert summary["timestamp_granularity_mode"] == "date_only"


def test_temporal_stubs_evaluation_metrics_json():
    customers, products, reviews = make_midnight_amazon()
    generator = ContinuousTime2KSBMTemporalStubsGenerator(
        customers, products, reviews, seed=19
    )
    generator.fit()
    synthetic = generator.generate()

    results = evaluate_temporal_stubs(
        pd.DataFrame(reviews).assign(review_time=pd.to_datetime(reviews["review_time"])),
        synthetic,
        customer_col="customer_id",
        product_col="product_id",
        timestamp_col="review_time",
        customer_blocks=generator.sbm_result.customer_blocks,
        product_blocks=generator.sbm_result.product_blocks,
    )

    assert "structural" in results
    assert "temporal" in results
    assert "additional" in results
    assert results["additional"]["product_degree_exact_match_rate"] == 1.0
    assert results["additional"]["customer_degree_exact_match_rate"] == 1.0
    assert results["additional"]["block_pair_count_exact_match_rate"] == 1.0
    assert results["additional"]["block_pair_timestamp_ks_mean"] == pytest.approx(0.0)
    json.dumps(results)
