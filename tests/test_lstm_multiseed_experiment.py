from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_lstm_multiseed_experiment import (  # noqa: E402
    flatten_numeric_scalars,
    prepare_evaluation_real,
)


def test_smoke_evaluation_real_matches_sample_prefix(tmp_path: Path):
    test_real = tmp_path / "test_real.csv"
    pd.DataFrame(
        {
            "event_id": [f"event-{index}" for index in range(10)],
            "target": list(range(10)),
        }
    ).to_csv(test_real, index=False)

    smoke_path = prepare_evaluation_real(
        test_real,
        tmp_path / "smoke",
        smoke_rows=4,
        dry_run=False,
    )

    smoke = pd.read_csv(smoke_path)
    assert len(smoke) == 4
    assert smoke["event_id"].tolist() == [
        "event-0",
        "event-1",
        "event-2",
        "event-3",
    ]


def test_full_evaluation_reuses_fixed_test_table(tmp_path: Path):
    test_real = tmp_path / "test_real.csv"

    result = prepare_evaluation_real(
        test_real,
        tmp_path / "full",
        smoke_rows=None,
        dry_run=False,
    )

    assert result == test_real


def test_attribute_diagnostics_scalars_are_flattened_for_aggregation():
    flattened = flatten_numeric_scalars(
        {
            "price": {
                "ks_distance": 0.12,
                "quantiles": {"0.5": 0.03},
                "note": "not numeric",
            },
            "valid": True,
        },
        prefix="attribute",
    )

    assert flattened == {
        "attribute.price.ks_distance": 0.12,
        "attribute.price.quantiles.0.5": 0.03,
        "attribute.valid": 1.0,
    }
