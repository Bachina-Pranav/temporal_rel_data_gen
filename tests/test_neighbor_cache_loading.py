from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.neighbor_cache import (  # noqa: E402
    CachedTemporalHistoryIndex,
    validate_cache_temporal_safety,
)
from attribute_generation.conditional_tabdlm.utils import save_json  # noqa: E402


def test_neighbor_cache_loads_cached_indices_and_preserves_past_only(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    save_json(
        {
            "num_rows": 3,
            "max_customer_history": 2,
            "max_product_history": 2,
        },
        root / "metadata.json",
    )
    np.save(root / "customer_hash.npy", np.asarray([11, 11, 22], dtype=np.int64))
    np.save(root / "product_hash.npy", np.asarray([31, 32, 32], dtype=np.int64))
    np.save(root / "timestamp_ns.npy", np.asarray([10, 20, 30], dtype=np.int64))
    np.save(root / "timestamp_seconds.npy", np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    for kind in ["customer", "product"]:
        indices = np.memmap(root / f"{kind}_history_indices.memmap", dtype=np.int64, mode="w+", shape=(3, 2))
        mask = np.memmap(root / f"{kind}_history_mask.memmap", dtype=np.uint8, mode="w+", shape=(3, 2))
        indices[:] = -1
        mask[:] = 0
        indices[1, 0] = 0
        mask[1, 0] = 1
        indices[2, 0] = 1
        mask[2, 0] = 1
        indices.flush()
        mask.flush()

    cache = CachedTemporalHistoryIndex(root)
    batch = cache.build_batch([1, 2], device="cpu")
    diagnostics = validate_cache_temporal_safety(root)

    assert batch["customer_history_row_index"].shape == (2, 2)
    assert batch["target_customer_hash"].tolist() == [11, 22]
    assert diagnostics["future_or_same_time_violations"] == 0
    assert diagnostics["temporal_past_only"] is True
