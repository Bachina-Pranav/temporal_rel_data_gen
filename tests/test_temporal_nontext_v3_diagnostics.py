from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_temporal_nontext_attrs import evaluate_nontext_attrs, load_reviews  # noqa: E402
from reldiff.attributes import TemporalNonTextAttributeDiffusionV3  # noqa: E402


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


def train_tiny_v3(tmp_path: Path):
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
        hidden_dim=32,
        num_layers=1,
        min_entities_per_cell=2,
        seed=7,
    )
    spine_path = tmp_path / "spine.csv"
    reviews[["customer_id", "product_id", "review_time"]].iloc[:12].to_csv(spine_path, index=False)
    return reviews, train_path, spine_path, debug, result.best_checkpoint


def test_v3_sampling_evaluator_diagnostics_and_sweep_smoke(tmp_path):
    _, train_path, spine_path, debug, checkpoint = train_tiny_v3(tmp_path)
    output_path = tmp_path / "synthetic_review_nontext_v3.csv"
    diagnostics_dir = tmp_path / "diagnostics"
    TemporalNonTextAttributeDiffusionV3.sample_from_checkpoint(
        spine_path,
        checkpoint,
        output_path,
        structure_debug_dir=debug,
        seed=11,
        num_steps=2,
        cat_sampling_strategy="argmax",
        use_temporal_calibration=True,
        diagnostics_dir=diagnostics_dir,
        diagnostic_row_sample_size=5,
    )
    with (tmp_path / "synthetic_review_nontext_v3_metadata.json").open() as handle:
        metadata = json.load(handle)
    assert metadata["calibration_applied_before_sampling"] is True
    assert metadata["temporal_calibration_num_groups_calibrated"] > 0

    metrics = evaluate_nontext_attrs(
        load_reviews(train_path, "review_time"),
        load_reviews(output_path, "review_time"),
        customer_col="customer_id",
        product_col="product_id",
        timestamp_col="review_time",
        cat_cols=["rating", "verified"],
        num_cols=[],
        diagnostics_dir=diagnostics_dir,
    )
    decomposition = metrics["decomposition"]
    assert decomposition["average_norm_base_rating_logits"] is not None
    assert decomposition["average_norm_residual_rating_logits"] is not None
    assert decomposition["residual_to_base_norm_ratio"] is not None

    monthly_path = diagnostics_dir / "monthly_real_vs_synthetic.csv"
    assert monthly_path.exists()
    monthly = pd.read_csv(monthly_path)
    required = {
        "month",
        "real_count",
        "synthetic_count",
        "real_avg_rating",
        "synthetic_avg_rating",
        "rating_abs_error",
        "real_verified_rate",
        "synthetic_verified_rate",
        "verified_abs_error",
        "real_rating_dist_1",
        "real_rating_dist_2",
        "real_rating_dist_3",
        "real_rating_dist_4",
        "real_rating_dist_5",
        "synthetic_rating_dist_1",
        "synthetic_rating_dist_2",
        "synthetic_rating_dist_3",
        "synthetic_rating_dist_4",
        "synthetic_rating_dist_5",
        "monthly_rating_distribution_js",
        "synthetic_minus_real_avg_rating",
        "synthetic_minus_real_verified_rate",
    }
    assert required.issubset(set(monthly.columns))
    assert (diagnostics_dir / "decomposition_diagnostics.json").exists()
    assert (diagnostics_dir / "temporal_bucket_consistency.json").exists()
    assert (diagnostics_dir / "v3_row_level_logit_components_sample.csv").exists()
    assert (diagnostics_dir / "temporal_calibration_by_group.csv").exists()

    sweep_root = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "scripts" / "sweep_temporal_nontext_v3.py"),
            "--real-reviews",
            str(train_path),
            "--synthetic-spine",
            str(spine_path),
            "--structure-debug-dir",
            str(debug),
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(sweep_root),
            "--sweep",
            "calibration_strength",
            "--values",
            "0.0",
            "0.5",
            "--num-diffusion-steps",
            "1",
            "--cat-sampling-strategy",
            "argmax",
            "--seed",
            "13",
        ],
        check=True,
    )
    summary = pd.read_csv(sweep_root / "sweep_summary.csv")
    assert len(summary) == 2
    assert set(summary["calibration_strength"]) == {0.0, 0.5}
