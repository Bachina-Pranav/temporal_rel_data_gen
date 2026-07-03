from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.fast_temporal_activity import FastTemporalActivityModel  # noqa: E402
from generators.stub_time_assignment import assign_stubs_to_slots_by_time  # noqa: E402
from generators.temporal_kernel_bandwidth import estimate_temporal_kernel_bandwidths  # noqa: E402
from generators.time_biased_block_stub_matching import TimeBiasedBlockStubMatchingGenerator  # noqa: E402
from generators.time_biased_stub_sampler import sample_desired_times_for_stubs_local_kernel  # noqa: E402


def base_distributions():
    values = {0: np.arange(100, dtype=np.int32)}
    cdf = {0: np.linspace(0.01, 1.0, 100)}
    return values, cdf, np.arange(100, dtype=np.int32), np.linspace(0.01, 1.0, 100)


def test_local_kernel_preserves_identity_timing():
    block_values, block_cdfs, global_values, global_cdf = base_distributions()
    offsets = np.asarray([0, 3, 6], dtype=np.int64)
    observed = np.asarray([10, 11, 12, 90, 91, 92], dtype=np.int32)
    blocks = np.asarray([0, 0], dtype=np.int64)
    stubs = np.asarray([0] * 1000 + [1] * 1000, dtype=np.int64)

    desired = sample_desired_times_for_stubs_local_kernel(
        stubs,
        offsets,
        observed,
        blocks,
        block_values,
        block_cdfs,
        global_values,
        global_cdf,
        100,
        np.random.default_rng(7),
        kernel="discrete_laplace",
        entity_bandwidths=np.asarray([1.0, 1.0]),
        block_bandwidths={0: 1.0},
        global_bandwidth=1.0,
    )

    assert desired[:1000].mean() < desired[1000:].mean()


def test_kernel_none_equals_empirical_exact_values():
    block_values, block_cdfs, global_values, global_cdf = base_distributions()
    desired = sample_desired_times_for_stubs_local_kernel(
        np.asarray([0] * 200, dtype=np.int64),
        np.asarray([0, 4], dtype=np.int64),
        np.asarray([5, 5, 7, 9], dtype=np.int32),
        np.asarray([0], dtype=np.int64),
        block_values,
        block_cdfs,
        global_values,
        global_cdf,
        20,
        np.random.default_rng(9),
        kernel="none",
    )

    assert set(desired.tolist()).issubset({5, 7, 9})


def test_local_kernel_clips_boundary_times():
    block_values, block_cdfs, global_values, global_cdf = base_distributions()
    desired = sample_desired_times_for_stubs_local_kernel(
        np.asarray([0] * 500 + [1] * 500, dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0, 9], dtype=np.int32),
        np.asarray([0, 0], dtype=np.int64),
        block_values,
        block_cdfs,
        global_values,
        global_cdf,
        10,
        np.random.default_rng(10),
        kernel="discrete_gaussian",
        entity_bandwidths=np.asarray([5.0, 5.0]),
        global_bandwidth=5.0,
    )

    assert desired.min() >= 0
    assert desired.max() < 10


def test_assign_stubs_to_slots_by_time_local_kernel_preserves_exact_degrees():
    frame = pd.DataFrame(
        {
            "entity": ["a", "a", "a", "b", "b", "b"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-08", "2020-01-09", "2020-01-10"],
        }
    )
    degrees = {"a": 3, "b": 3}
    blocks = {"a": 0, "b": 0}
    model = FastTemporalActivityModel(alpha=0.0).fit(frame, "entity", "review_time", blocks)
    state = model.get_fast_sampling_state()
    bandwidth = estimate_temporal_kernel_bandwidths(
        state["empirical_offsets"],
        state["empirical_time_values"],
        state["entity_block"],
        num_blocks=1,
    )
    assigned, summary = assign_stubs_to_slots_by_time(
        ["a", "b"],
        degrees,
        blocks,
        np.zeros(6, dtype=int),
        np.asarray([0, 1, 2, 7, 8, 9], dtype=int),
        model,
        np.random.default_rng(12),
        desired_time_sampling_mode="local_kernel",
        local_kernel_state={
            "num_time_codes": 10,
            "entity_bandwidths": bandwidth["entity_bandwidths"],
            "block_bandwidths": bandwidth["block_bandwidths"],
            "global_bandwidth": bandwidth["global_bandwidth"],
            "kernel": "discrete_laplace",
        },
    )

    assert Counter(assigned.tolist()) == Counter(degrees)
    assert summary["exact_degree_preserved"] is True
    assert summary["desired_time_sampling_mode"] == "local_kernel"


def test_time_biased_generator_local_kernel_metadata_and_exact_constraints(tmp_path):
    real = tiny_events()
    debug_in = tmp_path / "debug_in"
    write_blocks(debug_in)
    generator = TimeBiasedBlockStubMatchingGenerator(
        structure_debug_dir=debug_in,
        rank=2,
        desired_time_sampling_mode="local_kernel",
        kernel_type="discrete_laplace",
        seed=14,
    )

    synthetic = generator.fit(real).sample(seed=14)
    metrics = generator.evaluate(real, synthetic)
    metadata = generator.metadata()
    debug_out = tmp_path / "debug_out"
    generator.save_debug(debug_out)

    assert metadata["desired_time_sampling_mode"] == "local_kernel"
    assert metadata["temporal_alpha_used"] is False
    assert metadata["empirical_bayes_used"] is False
    assert metadata["bandwidth_selection_uses_synthetic_metrics"] is False
    assert metrics["customer_degree_exact_match"] is True
    assert metrics["product_degree_exact_match"] is True
    assert metrics["block_pair_time_exact_match"] is True
    assert metrics["daily_count_l1"] == 0.0
    assert (debug_out / "customer_temporal_kernel_bandwidths.json").exists()
    assert (debug_out / "product_temporal_kernel_bandwidths.json").exists()


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
