from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.c2st import featurize_frame  # noqa: E402


def test_c2st_excludes_primary_key_features():
    frame = pd.DataFrame(
        {
            "event_id": ["real-1", "real-2"],
            "value": [1.0, 2.0],
        }
    )
    table = {
        "primary_key": "event_id",
        "columns": {
            "event_id": {"type": "categorical"},
            "value": {"type": "numerical"},
        },
    }

    _, feature_names = featurize_frame(frame, table)

    assert all(not name.startswith("event_id") for name in feature_names)
    assert "value" in feature_names
