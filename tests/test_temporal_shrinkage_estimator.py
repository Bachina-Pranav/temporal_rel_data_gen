from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.temporal_shrinkage_estimator import estimate_temporal_shrinkage_alpha  # noqa: E402
from generators.time_biased_block_stub_matching import TimeBiasedBlockStubMatchingGenerator  # noqa: E402


def tiny_events() -> pd.DataFrame:
    rows = []
    days = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d").tolist()
    customer_by_block = {0: ["c0", "c1"], 1: ["c2", "c3"]}
    product_by_block = {0: ["p0", "p1"], 1: ["p2", "p3", "p4"]}
    pattern = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)]
    for day_index, day in enumerate(days):
        for event_index, (cblock, pblock) in enumerate(pattern):
            rows.append(
                (
                    customer_by_block[cblock][(day_index + event_index) % len(customer_by_block[cblock])],
                    product_by_block[pblock][(day_index + event_index) % len(product_by_block[pblock])],
                    day,
                )
            )
    return pd.DataFrame(rows, columns=["customer_id", "product_id", "review_time"])


def write_blocks(debug: Path) -> None:
    debug.mkdir(parents=True)
    pd.DataFrame({"customer_id": ["c0", "c1", "c2", "c3"], "customer_block": [0, 0, 1, 1]}).to_csv(
        debug / "customer_blocks.csv",
        index=False,
    )
    pd.DataFrame({"product_id": ["p0", "p1", "p2", "p3", "p4"], "product_block": [0, 0, 1, 1, 1]}).to_csv(
        debug / "product_blocks.csv",
        index=False,
    )


def test_empirical_alpha_selected_for_entity_specific_timing():
    frame = pd.DataFrame(
        {
            "entity_id": ["e0", "e0", "e0", "e1", "e1", "e1", "e2", "e2", "e2"],
            "review_time": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
                "2020-01-03",
            ],
        }
    )
    blocks = {"e0": 0, "e1": 0, "e2": 0}

    result = estimate_temporal_shrinkage_alpha(
        frame,
        "entity_id",
        "review_time",
        blocks,
        candidate_alphas=[0.0, 0.1, 1.0, 10.0],
        seed=7,
    )

    assert result["fallback_used"] is False
    assert result["num_holdout_events"] == 3
    assert result["best_alpha"] <= 0.1


def test_sparse_entities_fall_back_to_median_degree_alpha():
    frame = pd.DataFrame(
        {
            "entity_id": ["e0", "e1", "e2"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    blocks = {"e0": 0, "e1": 0, "e2": 0}

    result = estimate_temporal_shrinkage_alpha(frame, "entity_id", "review_time", blocks)

    assert result["fallback_used"] is True
    assert result["num_holdout_events"] == 0
    assert result["best_alpha"] == 1.0


def test_likelihood_only_evaluates_heldout_timestamps():
    frame = pd.DataFrame(
        {
            "entity_id": ["e0", "e0", "e0", "e1", "e1", "e1"],
            "review_time": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
                "2020-01-04",
            ],
        }
    )
    blocks = {"e0": 0, "e1": 1}
    candidates = [0.0, 1.0, 5.0]

    result = estimate_temporal_shrinkage_alpha(
        frame,
        "entity_id",
        "review_time",
        blocks,
        candidate_alphas=candidates,
        seed=11,
    )

    assert result["num_holdout_events"] == 2
    assert result["num_likelihood_evaluations"] == result["num_holdout_events"] * len(candidates)


def test_temporal_shrinkage_estimation_is_deterministic():
    frame = pd.DataFrame(
        {
            "entity_id": ["e0", "e0", "e0", "e1", "e1", "e1", "e2", "e2", "e2"],
            "review_time": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
                "2020-01-03",
                "2020-01-04",
                "2020-01-04",
            ],
        }
    )
    blocks = {"e0": 0, "e1": 0, "e2": 1}

    first = estimate_temporal_shrinkage_alpha(frame, "entity_id", "review_time", blocks, seed=13)
    second = estimate_temporal_shrinkage_alpha(frame, "entity_id", "review_time", blocks, seed=13)

    assert first == second


def test_time_biased_generator_empirical_bayes_metadata(tmp_path):
    real = tiny_events()
    debug_in = tmp_path / "debug_in"
    write_blocks(debug_in)
    generator = TimeBiasedBlockStubMatchingGenerator(
        structure_debug_dir=debug_in,
        rank=2,
        temporal_shrinkage_mode="empirical_bayes",
        seed=9,
    )

    synthetic = generator.fit(real).sample(seed=9)
    metadata = generator.metadata()
    debug_out = tmp_path / "debug_out"
    generator.save_debug(debug_out)

    assert len(synthetic) == len(real)
    assert metadata["temporal_shrinkage_mode"] == "empirical_bayes"
    assert metadata["alpha_selection_uses_synthetic_metrics"] is False
    assert metadata["alpha_customer_time_selected"] is not None
    assert metadata["alpha_product_time_selected"] is not None
    assert metadata["pairing_mode"] == "dynamic_exact_penalized"
    assert metadata["pairing_penalties_fixed_defaults"] is True
    assert (debug_out / "customer_temporal_shrinkage_estimation.json").exists()
    assert (debug_out / "product_temporal_shrinkage_estimation.json").exists()
