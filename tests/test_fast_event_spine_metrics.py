from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.fast_event_spine_metrics import evaluate_fast_event_spine  # noqa: E402


def test_fast_event_spine_metrics_include_runtime_and_block_l1(tmp_path):
    real = pd.DataFrame(
        {
            "customer_id": ["c0", "c0", "c1", "c2"],
            "product_id": ["p0", "p1", "p0", "p1"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        }
    )
    synthetic = real.copy()
    debug = tmp_path / "debug"
    debug.mkdir()
    pd.DataFrame({"customer_id": ["c0", "c1", "c2"], "customer_block": [0, 0, 1]}).to_csv(debug / "customer_blocks.csv", index=False)
    pd.DataFrame({"product_id": ["p0", "p1"], "product_block": [0, 1]}).to_csv(debug / "product_blocks.csv", index=False)

    metrics = evaluate_fast_event_spine(
        real,
        synthetic,
        structure_debug_dir=debug,
        metadata={"method": "fast_lowrank_temporal_event", "events_per_second": 100.0},
    )

    assert metrics["method"] == "fast_lowrank_temporal_event"
    assert metrics["events_per_second"] == 100.0
    assert metrics["block_pair_count_l1"] == 0.0
