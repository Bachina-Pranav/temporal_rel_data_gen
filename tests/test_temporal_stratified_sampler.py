from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.temporal_stratified_sampler import TemporalStratifiedSampler  # noqa: E402


def test_temporal_stratified_sampler_hits_all_bins_reasonably():
    timestamps = np.repeat(np.arange(4, dtype=np.int64), 25)
    sampler = TemporalStratifiedSampler(
        timestamps,
        mode="temporal_stratified",
        num_time_bins=4,
        seed=7,
        num_samples=400,
        timestamp_column="review_time",
    )

    rows = list(iter(sampler))
    sampled_bins = timestamps[rows]

    assert len(rows) == 400
    assert set(sampled_bins.tolist()) == {0, 1, 2, 3}
    counts = np.bincount(sampled_bins, minlength=4)
    assert counts.min() > 60
    assert counts.max() < 140
    diagnostics = sampler.diagnostics().to_dict()
    assert diagnostics["timestamp_column"] == "review_time"
    assert diagnostics["num_rows"] == 100
