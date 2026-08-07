from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.numerical_calibration import (  # noqa: E402
    CalibrationOptions,
    calibrate_numerical_column,
    empirical_rank_map,
)
from scripts.run_lstm_numerical_calibration_diagnostics import (  # noqa: E402
    finalize_existing_results,
    resolve_shared_spine_directory,
)


def test_global_quantile_calibration_preserves_order_and_training_support():
    train = frame(
        destinations=["a"] * 8,
        values=[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    )
    synthetic = frame(
        destinations=["a"] * 4,
        values=[0.1, 0.4, 0.2, 0.3],
    )

    calibrated, metadata = calibrate_numerical_column(
        train,
        synthetic,
        value_column="value",
        destination_column="destination",
        timestamp_column="event_time",
        mode="global",
        options=CalibrationOptions(
            project_to_training_support=True,
        ),
    )

    assert set(calibrated).issubset(set(train["value"]))
    assert np.array_equal(
        np.argsort(synthetic["value"].to_numpy()),
        np.argsort(calibrated),
    )
    assert metadata["mapping_fit_scope"] == "training_split_only"


def test_destination_hierarchy_uses_exact_then_frequency_fallback():
    train = frame(
        destinations=["a"] * 4 + ["b"] * 4 + ["c"] * 2,
        values=[1.0] * 4 + [10.0] * 4 + [20.0] * 2,
    )
    synthetic = frame(
        destinations=["a", "a", "new", "new"],
        values=[0.1, 0.9, 0.2, 0.8],
    )

    calibrated, metadata = calibrate_numerical_column(
        train,
        synthetic,
        value_column="value",
        destination_column="destination",
        timestamp_column="event_time",
        mode="destination_hierarchy",
        options=CalibrationOptions(
            min_destination_rows=3,
            min_bucket_rows=2,
            project_to_training_support=True,
        ),
    )

    assert calibrated[:2].tolist() == [1.0, 1.0]
    assert set(calibrated[2:]).issubset(set(train["value"]))
    assert any(
        key.startswith("destination:a")
        for key in metadata["source_counts"]
    )
    assert not any(
        key.startswith("destination:new")
        for key in metadata["source_counts"]
    )


def test_empirical_rank_map_keeps_nan_and_tie_ranks():
    mapped = empirical_rank_map(
        np.array([3.0, np.nan, 1.0, 1.0, 2.0]),
        np.array([10.0, 20.0, 30.0, 40.0]),
    )

    assert np.isnan(mapped[1])
    assert mapped[2] == mapped[3]
    assert mapped[2] < mapped[4] < mapped[0]


def test_calibration_resolves_multiseed_shared_spine_layout(
    tmp_path: Path,
):
    shared = tmp_path / "shared" / "spines"
    shared.mkdir(parents=True)
    (shared / "train_real.csv").write_text("value\n1\n")
    (shared / "test_real.csv").write_text("value\n1\n")

    assert resolve_shared_spine_directory(tmp_path) == shared


def test_finalize_calibration_report_needs_no_tabulate(
    tmp_path: Path,
):
    pd.DataFrame(
        {
            "calibration": ["Q0"],
            "full_row_c2st_mean": [0.5],
        }
    ).to_csv(
        tmp_path / "calibration_results_aggregate.csv",
        index=False,
    )
    (
        tmp_path / "calibration_interpretation.json"
    ).write_text('{"status": "complete"}')

    finalize_existing_results(tmp_path)

    report = (
        tmp_path / "calibration_report.md"
    ).read_text()
    assert "| calibration | full_row_c2st_mean |" in report
    assert '"status": "complete"' in report


def frame(
    *,
    destinations: list[str],
    values: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "destination": destinations,
            "event_time": pd.date_range(
                "2020-01-01",
                periods=len(values),
                freq="D",
                tz="UTC",
            ),
            "value": values,
        }
    )
