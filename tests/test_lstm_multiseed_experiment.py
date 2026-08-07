from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_lstm_multiseed_experiment import (  # noqa: E402
    attribute_diagnostics_command,
    flatten_numeric_scalars,
    pretokenized_split_counts,
    prepare_evaluation_real,
    resolve_seed_config,
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


def test_attribute_diagnostics_command_receives_evaluation_config():
    command = attribute_diagnostics_command(
        config_path=Path("model.yaml"),
        evaluation_config_path=Path("evaluation.yaml"),
        train_real_path=Path("train.csv"),
        evaluation_real_path=Path("test.csv"),
        synthetic_path=Path("synthetic.csv"),
        graph_history_prefix_path=Path("history.csv"),
        output_path=Path("diagnostics.json"),
        seed=42,
    )

    assert command[
        command.index("--evaluation-config") + 1
    ] == "evaluation.yaml"
    assert "--evaluation-config" in command


def test_seed_config_uses_materialized_training_table_for_numerical_head():
    resolved = resolve_seed_config(
        {"paths": {}, "training": {}, "sampling": {}},
        42,
        Path("run"),
        Path("test_spine.csv"),
        Path("pretokenized"),
        Path("neighbor_cache"),
        numerical_head_training_table=Path("train_real.csv"),
        smoke=False,
    )

    assert (
        resolved["paths"]["numerical_head_training_table_path"]
        == "train_real.csv"
    )


def test_pretokenized_split_counts_read_actual_index_arrays(
    tmp_path: Path,
):
    np.save(tmp_path / "train_indices.npy", np.arange(7))
    np.save(tmp_path / "valid_indices.npy", np.arange(2))
    np.save(tmp_path / "test_indices.npy", np.arange(3))

    assert pretokenized_split_counts(tmp_path) == {
        "train_rows": 7,
        "valid_rows": 2,
        "test_rows": 3,
    }
