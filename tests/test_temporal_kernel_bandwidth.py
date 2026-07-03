from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.temporal_kernel_bandwidth import estimate_temporal_kernel_bandwidths  # noqa: E402


def test_block_bandwidth_fallback_for_degree_one_entities():
    offsets = np.asarray([0, 1, 4], dtype=np.int64)
    values = np.asarray([5, 10, 20, 30], dtype=np.int32)
    blocks = np.asarray([0, 0], dtype=np.int64)

    result = estimate_temporal_kernel_bandwidths(
        offsets,
        values,
        blocks,
        num_blocks=1,
        bandwidth_mode="auto_block_iqr",
        bandwidth_scale=0.25,
        min_bandwidth=1.0,
    )

    entity_bandwidths = result["entity_bandwidths"]
    assert len(entity_bandwidths) == 2
    assert entity_bandwidths[0] == result["block_bandwidths"][0]
    assert entity_bandwidths[0] >= 1.0
    assert result["diagnostics"]["bandwidth_selection_uses_synthetic_metrics"] is False


def test_fixed_bandwidth_mode_uses_requested_value():
    offsets = np.asarray([0, 2, 4], dtype=np.int64)
    values = np.asarray([0, 10, 20, 30], dtype=np.int32)
    blocks = np.asarray([0, 1], dtype=np.int64)

    result = estimate_temporal_kernel_bandwidths(
        offsets,
        values,
        blocks,
        num_blocks=2,
        default_bandwidth=3.5,
        bandwidth_mode="fixed",
        min_bandwidth=1.0,
    )

    assert np.allclose(result["entity_bandwidths"], 3.5)
    assert result["block_bandwidths"] == {0: 3.5, 1: 3.5}
    assert result["global_bandwidth"] == 3.5
