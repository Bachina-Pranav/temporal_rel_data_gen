from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm import (  # noqa: E402
    posthoc_diagnostics,
)
from attribute_generation.conditional_tabdlm.posthoc_diagnostics import (  # noqa: E402
    c2st_config_for_columns,
    c2st_feature_ablation_suite,
    c2st_sanity_suite,
)
from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMSchema,
)
from scripts.run_lstm_posthoc_diagnostics import (  # noqa: E402
    resolve_requested_phases,
)


def schema() -> ConditionalTABDLMSchema:
    return ConditionalTABDLMSchema(
        foreign_key_columns=("user_id", "item_id"),
        datetime_columns=("event_time",),
        categorical_targets=("channel",),
        numerical_targets=("price",),
    )


def evaluation_config() -> dict:
    return {
        "table": {
            "primary_key": "event_id",
            "columns": {
                "event_id": {"type": "categorical"},
                "user_id": {
                    "type": "foreign_key",
                    "c2st_hash_buckets": 8,
                },
                "item_id": {
                    "type": "foreign_key",
                    "c2st_hash_buckets": 8,
                },
                "event_time": {"type": "datetime"},
                "price": {"type": "numerical"},
                "channel": {"type": "categorical"},
            },
        },
        "evaluation": {
            "random_seed": 42,
            "c2st": {
                "classifiers": ["logistic_regression"],
                "n_splits": 2,
            },
        },
    }


def tiny_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "event_id": range(12),
            "user_id": ["u1"] * 6 + ["u2"] * 6,
            "item_id": ["i1", "i2"] * 6,
            "event_time": pd.date_range("2020-01-01", periods=12),
            "price": [0.01, 0.02, 0.03] * 4,
            "channel": ["1", "2"] * 6,
        }
    )
    test = train.iloc[:8].copy().reset_index(drop=True)
    return train, test


def fake_repeated_c2st(
    real,
    synthetic,
    config,
    *,
    classifier_seeds,
    columns=None,
    max_rows=None,
    generator_seed=None,
    label=None,
):
    chance = label in {
        "S1_identical_real_copy",
        "S2_disjoint_real_splits",
    }
    aggregate = {
        "auc_mean": 0.5 if chance else 0.8,
        "auc_std": 0.0,
        "accuracy_mean": 0.5 if chance else 0.75,
        "c2st_error_mean": 0.0 if chance else 0.6,
        "feature_count": len(columns or config["table"]["columns"]),
        "num_runs": len(classifier_seeds),
    }
    details = [
        {
            "label": label,
            "generator_seed": generator_seed,
            "classifier_seed": seed,
            "auc": aggregate["auc_mean"],
        }
        for seed in classifier_seeds
    ]
    return aggregate, details


def test_c2st_sanity_suite_runs_all_five_controls(monkeypatch):
    train, real = tiny_frames()
    monkeypatch.setattr(
        posthoc_diagnostics,
        "repeated_c2st",
        fake_repeated_c2st,
    )

    summary, details = c2st_sanity_suite(
        train,
        real,
        evaluation_config(),
        schema(),
        classifier_seeds=[1, 2, 3, 4, 5],
        max_rows=None,
        chance_tolerance=0.15,
    )

    assert summary["status"] == "passed"
    assert summary["chance_controls_passed"] is True
    assert set(summary["scenarios"]) == {
        "S1_identical_real_copy",
        "S2_disjoint_real_splits",
        "S3_shuffled_real_attributes",
        "S4_global_empirical_attributes",
        "S5_corrupted_numerical_attribute",
    }
    assert len(details) == 25


def test_feature_ablation_builds_f1_through_f9(monkeypatch):
    train, real = tiny_frames()
    synthetic = real.copy()
    synthetic["price"] = synthetic["price"] + 0.001
    monkeypatch.setattr(
        posthoc_diagnostics,
        "repeated_c2st",
        fake_repeated_c2st,
    )

    rows, details = c2st_feature_ablation_suite(
        train,
        real,
        {42: synthetic},
        evaluation_config(),
        schema(),
        classifier_seeds=[11, 23, 37, 53, 71],
        max_rows=None,
    )

    assert len(rows) == 9
    assert len(details) == 45
    assert set(rows["feature_set"].str.slice(0, 2)) == {
        f"F{index}" for index in range(1, 10)
    }
    f8_columns = rows.loc[
        rows["feature_set"] == "F8_full_without_entity_ids",
        "columns",
    ].iloc[0]
    assert "user_id" not in f8_columns
    assert "item_id" not in f8_columns
    f9_columns = rows.loc[
        rows["feature_set"] == "F9_frequency_buckets_instead_of_ids",
        "columns",
    ].iloc[0]
    assert "__entity_frequency_0" in f9_columns
    assert "__entity_frequency_1" in f9_columns


def test_column_ablation_removes_primary_key_and_limits_c2st_rows():
    resolved = c2st_config_for_columns(
        evaluation_config(),
        columns=["price", "channel"],
        max_rows=123,
    )

    assert resolved["table"]["primary_key"] is None
    assert set(resolved["table"]["columns"]) == {"price", "channel"}
    assert resolved["evaluation"]["c2st"]["max_rows"] == 123


def test_phase_selection_adds_only_required_prerequisites():
    assert resolve_requested_phases(["support"]) == {"audit", "support"}
    assert resolve_requested_phases(["oracle"]) == {
        "audit",
        "sanity",
        "support",
        "projection",
        "oracle",
    }
    assert resolve_requested_phases(["diagnosis"]) == {
        "audit",
        "sanity",
        "feature_ablation",
        "support",
        "projection",
        "oracle",
        "importance",
        "conditional",
        "diagnosis",
    }


def test_tiny_hm_posthoc_command_runs_without_retraining(tmp_path: Path):
    experiment = tmp_path / "experiment"
    shared = experiment / "shared" / "spines"
    run = experiment / "runs" / "seed_17"
    shared.mkdir(parents=True)
    (run / "checkpoints").mkdir(parents=True)
    (run / "samples").mkdir(parents=True)
    (run / "evaluation" / "paper_grade").mkdir(parents=True)

    train, test = tiny_frames()
    validation = train.iloc[8:10].copy()
    train.to_csv(shared / "train_real.csv", index=False)
    validation.to_csv(shared / "validation_real.csv", index=False)
    test.to_csv(shared / "test_real.csv", index=False)
    test[
        ["event_id", "user_id", "item_id", "event_time"]
    ].to_csv(shared / "test_spine.csv", index=False)
    train[
        ["event_id", "user_id", "item_id", "event_time"]
    ].to_csv(shared / "history_prefix_spine.csv", index=False)

    synthetic = test.copy()
    synthetic["price"] = synthetic["price"] + 0.0001
    synthetic.to_csv(
        run / "samples" / "synthetic_interactions.csv",
        index=False,
    )
    model_raw = {
        "paths": {
            "train_data_path": str(shared / "train_real.csv"),
            "synthetic_spine_path": str(shared / "test_spine.csv"),
            "output_dir": str(run),
        },
        "columns": {
            "condition": {
                "foreign_keys": ["user_id", "item_id"],
                "datetimes": ["event_time"],
            },
            "target": {
                "categorical": ["channel"],
                "numerical": ["price"],
                "text": [],
            },
        },
        "text": {"max_length": {}},
    }
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        yaml.safe_dump(model_raw),
        encoding="utf-8",
    )
    (run / "config_resolved.yaml").write_text(
        yaml.safe_dump(model_raw),
        encoding="utf-8",
    )
    eval_config = evaluation_config()
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(
        yaml.safe_dump(eval_config),
        encoding="utf-8",
    )
    (run / "evaluation_config_resolved.yaml").write_text(
        yaml.safe_dump(eval_config),
        encoding="utf-8",
    )
    torch.save(
        {
            "numerical_metadata": {
                "price": {
                    "preprocessing": "standardize",
                    "output_distribution": "gaussian",
                    "mean": float(train["price"].mean()),
                    "std": float(train["price"].std(ddof=0)),
                    "min_train": float(train["price"].min()),
                    "max_train": float(train["price"].max()),
                    "clip_to_train_range": True,
                }
            },
            "categorical_vocabs": {
                "channel": {
                    "column": "channel",
                    "token_to_id": {"<missing>": 0, "1": 1, "2": 2},
                }
            },
            "model_config": {},
            "epoch": 1,
            "valid_metrics": {"total_loss": 1.0},
        },
        run / "checkpoints" / "best.pt",
    )
    (run / "evaluation" / "paper_grade" / "metrics.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (run / "evaluation" / "attribute_diagnostics.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    output = tmp_path / "posthoc"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "scripts" / "run_lstm_posthoc_diagnostics.py"),
            "--experiment-root",
            str(experiment),
            "--model-config",
            str(model_config),
            "--evaluation-config",
            str(evaluation_path),
            "--seeds",
            "17",
            "--output-dir",
            str(output),
            "--phases",
            "support",
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "retraining" not in completed.stderr.lower()
    assert (output / "01_current_experiment_audit.json").is_file()
    assert (output / "04_numerical_support.json").is_file()
    assert not (run / "checkpoints" / "last.pt").exists()
