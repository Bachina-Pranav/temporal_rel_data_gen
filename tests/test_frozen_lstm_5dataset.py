from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    review_text_content_lengths_from_csv,
    resolve_auto_review_text_config,
)
from scripts.run_frozen_lstm_5dataset import (  # noqa: E402
    derive_frozen_dataset_config,
    frozen_config_comparability,
    validate_frozen_contract,
)


def load_yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def frozen_config() -> dict:
    raw = load_yaml("configs/attribute_generation/lstm_hm_10k_customers.yaml")
    raw["numerical_heads"] = {
        "mode": "auto",
        "global_prior": {
            "enabled": True,
            "alpha": 1.0,
            "residual_weight": 0.25,
            "residual_temperature": 1.0,
            "temporal_prior": {"enabled": False, "lambda_t": 0.0},
        },
    }
    return raw


def test_frozen_contract_rejects_architecture_drift():
    frozen = frozen_config()
    lock = {
        "categorical_architecture": "original",
        "temporal_prior_lambda": 0.0,
    }
    validate_frozen_contract(frozen, lock)

    changed = copy.deepcopy(frozen)
    changed["numerical_heads"]["global_prior"]["residual_weight"] = 0.5
    try:
        validate_frozen_contract(changed, lock)
    except RuntimeError as exc:
        assert "gamma" in str(exc)
    else:
        raise AssertionError("Frozen contract accepted changed gamma")

    changed = copy.deepcopy(frozen)
    changed["sampling"]["numerical_temperature"] = 1.5
    try:
        validate_frozen_contract(changed, lock)
    except RuntimeError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("Frozen contract accepted changed temperature")

    changed = copy.deepcopy(frozen)
    changed["categorical_heads"] = {"global_prior": {"enabled": True}}
    try:
        validate_frozen_contract(changed, lock)
    except RuntimeError as exc:
        assert "categorical" in str(exc)
    else:
        raise AssertionError("Frozen contract accepted categorical prior")


def test_yelp_derivation_preserves_schema_and_uses_frozen_text_branch(tmp_path):
    base = load_yaml("configs/attribute_generation/lstm_yelp_100k.yaml")
    text = load_yaml(
        "configs/attribute_generation/"
        "conditional_tabdlm_amazon_toy_exp5_1_lstm_privacy_alignment.yaml"
    )
    frozen_path = tmp_path / "frozen.yaml"
    frozen_path.write_text("frozen: true\n", encoding="utf-8")
    resolved = derive_frozen_dataset_config(
        base,
        frozen_config(),
        text,
        output_root=tmp_path,
        seed=42,
        frozen_config_path=frozen_path,
    )

    assert resolved["columns"] == base["columns"]
    assert resolved["numerical_heads"]["mode"] == "auto"
    assert resolved["numerical_heads"]["global_prior"]["residual_weight"] == 0.25
    assert "categorical_heads" not in resolved
    assert resolved["review_text_decoder"]["condition_on_summary"] is False
    assert resolved["auxiliary_targets"]["categorical"] == [
        "review_text_length_bucket"
    ]
    assert resolved["model"] == text["model"]
    assert resolved["graph_conditioning"]["foreign_key_edges"] == base[
        "graph_conditioning"
    ]["foreign_key_edges"]


def test_retailrocket_derivation_has_only_event_type_target(tmp_path):
    base = load_yaml(
        "configs/attribute_generation/lstm_retailrocket_100k.yaml"
    )
    text = load_yaml(
        "configs/attribute_generation/"
        "conditional_tabdlm_amazon_toy_exp5_1_lstm_privacy_alignment.yaml"
    )
    frozen = frozen_config()
    frozen_path = tmp_path / "frozen.yaml"
    frozen_path.write_text("frozen: true\n", encoding="utf-8")
    resolved = derive_frozen_dataset_config(
        base,
        frozen,
        text,
        output_root=tmp_path,
        seed=42,
        frozen_config_path=frozen_path,
    )

    assert resolved["columns"]["target"] == {
        "categorical": ["event_type"],
        "numerical": [],
        "text": [],
    }
    assert "transaction_id" not in str(resolved)
    assert resolved["model"] == frozen["model"]
    assert resolved["graph_conditioning"]["graph_uses_future_events"] is False


def test_review_text_auto_length_reads_explicit_training_split_only(tmp_path):
    path = tmp_path / "interactions.csv"
    pd.DataFrame(
        {
            "review_text": [
                "short train",
                "three token train",
                "validation " + "long " * 100,
                "test " + "long " * 200,
            ],
            "split": ["train", "train", "validation", "test"],
        }
    ).to_csv(path, index=False)

    values = review_text_content_lengths_from_csv(path, chunk_size=2)

    assert sorted(values.tolist()) == [2, 3]

    raw = load_yaml(
        "configs/attribute_generation/lstm_yelp_100k.yaml"
    )
    raw["paths"]["train_data_path"] = str(path)
    raw["text"]["max_length"]["review_text"] = "auto"
    raw["review_text"] = {
        "max_tokens": "auto",
        "max_feasible_tokens": 128,
    }
    raw["auxiliary_targets"] = {
        "categorical": ["review_text_length_bucket"]
    }
    resolved = resolve_auto_review_text_config(raw)
    metadata = resolved["_auto_text_length_metadata"]["review_text"]
    assert metadata["fit_scope"] == "explicit_train_split"
    assert metadata["training_only"] is True


def test_retailrocket_evaluator_excludes_transaction_identifier():
    config = load_yaml(
        "configs/evaluation/"
        "single_event_table_paper_metrics_retailrocket_100k.yaml"
    )
    columns = set(config["table"]["columns"])

    assert "transactionid" not in columns
    assert "transaction_id" not in columns


def test_frozen_comparability_requires_auto_prior_residual_head():
    comparable, reason = frozen_config_comparability(
        frozen_config(), has_numerical=True
    )
    assert comparable, reason

    old = frozen_config()
    old["numerical_heads"]["mode"] = "support"
    comparable, reason = frozen_config_comparability(old, has_numerical=True)
    assert not comparable
    assert "router" in reason
