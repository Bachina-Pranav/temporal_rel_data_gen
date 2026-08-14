from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_lstm_architecture_finalization_compact import (  # noqa: E402
    architecture_config,
    classify_deltas,
    completed_evaluation,
    completed_run,
    enforce_compact_contract,
    promote_schema_numeric_ordinals,
    select_temporal_candidate,
    temporal_numerical_error,
)


def compact_config():
    return yaml.safe_load(
        (
            ROOT
            / "configs/experiments/"
            "lstm_architecture_finalization_compact.yaml"
        ).read_text(encoding="utf-8")
    )


def test_compact_contract_is_seed_42_and_two_lambdas_only():
    raw = compact_config()

    enforce_compact_contract(raw)
    assert raw["maximum_new_training_runs"] == 3

    bad_seed = copy.deepcopy(raw)
    bad_seed["seeds"] = [17, 42, 73]
    with pytest.raises(RuntimeError, match="exactly one generator seed"):
        enforce_compact_contract(bad_seed)

    bad_grid = copy.deepcopy(raw)
    bad_grid["temporal_lambdas"] = [0.1, 0.25, 0.5]
    with pytest.raises(RuntimeError, match="only lambda_t"):
        enforce_compact_contract(bad_grid)


def test_candidate_config_removes_categorical_prior_and_fixes_support_head():
    matrix = compact_config()
    base = {
        "categorical_heads": {"prior": {"enabled": True}},
        "sampling": {"numerical_temperature": 2.0},
    }

    candidate = architecture_config(
        base,
        matrix,
        temporal_lambda=0.1,
        adaptive=False,
    )

    assert "categorical_heads" not in candidate
    assert candidate["numerical_heads"]["mode"] == "support_prior"
    assert candidate["numerical_heads"]["global_prior"]["residual_weight"] == 0.25
    assert candidate["numerical_heads"]["global_prior"]["temporal_prior"]["lambda_t"] == 0.1
    assert candidate["sampling"]["numerical_temperature"] == 1.0


def test_amazon_finalization_m2_is_compatible_original_category_control():
    matrix = compact_config()
    amazon = yaml.safe_load(
        (ROOT / matrix["transfer"]["amazon_toy"]["base_config"]).read_text(
            encoding="utf-8"
        )
    )
    targets = promote_schema_numeric_ordinals(amazon)["columns"]["target"]

    assert targets["numerical"] == []
    assert "rating" in targets["categorical"]
    assert "categorical_heads" not in amazon


def test_temporal_selection_prefers_weakest_accepted_candidate():
    selected, _ = select_temporal_candidate(
        [
            {"lambda_t": 0.1, "trend_improvement": 0.02, "accepted": True},
            {"lambda_t": 0.25, "trend_improvement": 0.025, "accepted": True},
        ]
    )

    assert selected == 0.1


def test_temporal_selection_uses_stronger_candidate_only_for_clear_gain():
    selected, _ = select_temporal_candidate(
        [
            {"lambda_t": 0.1, "trend_improvement": 0.02, "accepted": True},
            {"lambda_t": 0.25, "trend_improvement": 0.031, "accepted": True},
        ]
    )

    assert selected == 0.25


def test_categorical_prior_requires_cross_dataset_wins_without_regression():
    tolerances = {"full": 0.02, "shape": 0.01}

    wins, regressions = classify_deltas(
        {"full": -0.03, "shape": 0.02},
        tolerances,
    )

    assert wins == ["full"]
    assert regressions == ["shape"]


def test_temporal_numerical_error_uses_only_numeric_timestamp_pairs():
    attribute = {
        "dependency_fidelity": {
            "pairs": [
                {"left": "price", "right": "event_time", "error": 0.2},
                {
                    "left": "sales_channel_id",
                    "right": "event_time",
                    "error": 0.8,
                },
            ]
        }
    }
    config = {
        "columns": {
            "target": {"numerical": ["price"]},
            "condition": {"datetimes": ["event_time"]},
        }
    }

    assert temporal_numerical_error(attribute, config) == 0.2


def test_evaluation_only_reuse_does_not_require_checkpoint(tmp_path):
    run = tmp_path / "runs/seed_42"
    required = [
        run / "samples/synthetic_interactions.csv",
        run / "evaluation/paper_grade/metrics.json",
        run / "evaluation/attribute_diagnostics.json",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    assert completed_evaluation(tmp_path)
    assert not completed_run(tmp_path)
