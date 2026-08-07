from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_lstm_attribute_diagnostics import numerical_metrics  # noqa: E402
from scripts.materialize_interaction_lstm_splits import resolve_split_labels  # noqa: E402
from scripts.run_lstm_m2_transfer_experiments import (  # noqa: E402
    collect_metrics,
    derive_m2_config,
)
from scripts.run_lstm_multiseed_experiment import (  # noqa: E402
    attribute_diagnostics_command,
    expected_materialized_rows,
    resolve_evaluation_scope,
)
from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMConfig,
    ConditionalTABDLMSchema,
)
from attribute_generation.conditional_tabdlm.lstm_joint import (  # noqa: E402
    JointLSTMRelationalAttributeGenerator,
)
from attribute_generation.conditional_tabdlm.tokenization import (  # noqa: E402
    CategoryVocab,
    SimpleTextTokenizer,
)
M2 = {
    "mode": "support",
    "class_frequency_weighting": "inverse_sqrt",
    "label_smoothing": 0.01,
    "conditioning": {
        "explicit_destination": False,
        "include_support_in_text_context": True,
    },
    "prior": {"enabled": False},
}


def load_yaml(path: str) -> dict:
    with (ROOT / path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_amazon_m2_changes_only_rating_target_family_and_numerical_head():
    base = load_yaml(
        "configs/attribute_generation/"
        "conditional_tabdlm_amazon_toy_exp5_1_lstm_privacy_alignment.yaml"
    )
    derived = derive_m2_config(
        base,
        numerical_attributes=["rating"],
        numerical_head=M2,
        output_root="outputs/new",
        seed=42,
        numerical_sampling_temperature=1.0,
        variant_name="amazon_m2",
    )

    assert derived["columns"]["target"] == {
        "categorical": ["verified"],
        "numerical": ["rating"],
        "text": ["summary", "review_text"],
    }
    assert derived["numerical_heads"] == M2
    assert derived["graph_conditioning"] == base["graph_conditioning"]
    assert derived["text_decoder"] == base["text_decoder"]
    assert derived["review_text_decoder"] == base["review_text_decoder"]
    assert derived["training"] == base["training"]
    assert derived["loss_weights"] == base["loss_weights"]
    assert derived["sampling"]["numerical_temperature"] == 1.0
    assert derived["event_spine"] == {
        "source_fk": "customer_id",
        "destination_fk": "product_id",
        "timestamp": "review_time",
    }


def test_movielens_m2_uses_training_derived_support_without_hardcoded_values():
    base = load_yaml(
        "configs/attribute_generation/lstm_movielens_100k.yaml"
    )
    derived = derive_m2_config(
        base,
        numerical_attributes=["rating"],
        numerical_head=M2,
        output_root="outputs/new",
        seed=42,
        numerical_sampling_temperature=1.0,
        variant_name="movielens_m2",
    )

    assert derived["columns"]["target"]["categorical"] == []
    assert derived["columns"]["target"]["numerical"] == ["rating"]
    assert "support_values" not in derived["numerical_heads"]
    assert derived["graph_conditioning"] == base["graph_conditioning"]
    assert derived["training"] == base["training"]


def test_implicit_split_materialization_matches_legacy_90_5_5():
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range(
                "2020-01-01", periods=20, freq="D"
            )[::-1]
        }
    )

    labels, source = resolve_split_labels(frame, "event_time")

    assert source == "legacy_time_aware_90_5_5"
    assert labels.value_counts().to_dict() == {
        "train": 18,
        "validation": 1,
        "test": 1,
    }
    earliest = frame["event_time"].idxmin()
    latest = frame["event_time"].idxmax()
    assert labels.loc[earliest] == "train"
    assert labels.loc[latest] == "test"


def test_neighbor_cache_expected_rows_use_filtered_split_total():
    assert expected_materialized_rows(
        {
            "train": {"rows": 71_669},
            "validation": {"rows": 3_982},
            "test": {"rows": 3_982},
        }
    ) == 79_633


def test_configured_spine_scope_uses_original_real_and_fixed_spine():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("user_id", "item_id"),
        datetime_columns=("event_time",),
        categorical_targets=(),
        numerical_targets=("value",),
    )
    config = ConditionalTABDLMConfig(
        raw={
            "paths": {
                "train_data_path": "data/full.csv",
                "synthetic_spine_path": "outputs/fixed_spine.csv",
                "output_dir": "outputs/run",
            }
        },
        schema=schema,
    )

    scope = resolve_evaluation_scope(
        config,
        Path("outputs/run/shared"),
        "configured-spine",
    )

    assert scope["real_table"] == Path("data/full.csv")
    assert scope["spine"] == Path("outputs/fixed_spine.csv")
    assert scope["graph_history_prefix"] is None


def test_numerical_metrics_include_support_tv_and_unique_count():
    metrics = numerical_metrics(
        pd.Series([1, 1, 2, 3]),
        pd.Series([1, 1, 2, 3]),
        pd.Series([1, 2, 2, 3]),
    )

    assert metrics["support_total_variation"] == 0.25
    assert metrics["num_unique_real"] == 3
    assert metrics["num_unique_synthetic"] == 3


def test_full_spine_diagnostics_omit_history_prefix():
    command = attribute_diagnostics_command(
        config_path=Path("config.yaml"),
        evaluation_config_path=Path("evaluation.yaml"),
        train_real_path=Path("train.csv"),
        evaluation_real_path=Path("real.csv"),
        synthetic_path=Path("synthetic.csv"),
        graph_history_prefix_path=None,
        output_path=Path("metrics.json"),
        seed=42,
    )

    assert "--graph-history-prefix" not in command


def test_comparison_collects_runtime_and_existing_legacy_metrics(
    tmp_path: Path,
):
    paper = tmp_path / "paper.json"
    legacy = tmp_path / "legacy.json"
    runtime = tmp_path / "runtime.json"
    paper.write_text(
        '{"paper_metrics_summary": {"shape_error": 0.1}, '
        '"shape": {"per_column": {}}}\n',
        encoding="utf-8",
    )
    legacy.write_text('{"rating_ks": 0.2}\n', encoding="utf-8")
    runtime.write_text(
        '{"total_sampling_seconds": 4.0, "rows_per_second": 25.0}\n',
        encoding="utf-8",
    )

    result = collect_metrics(
        "MovieLens-toy",
        "M2 global support",
        str(paper),
        None,
        str(legacy),
        str(runtime),
    )

    assert result["core"]["Sampling Seconds"] == 4.0
    assert result["core"]["Rows Per Second"] == 25.0
    assert result["dataset_specific"][0]["Metric"] == "rating_ks"
    assert result["dataset_specific"][0]["Value"] == 0.2


def test_support_rating_preserves_text_decoder_context_width():
    tokenizer = SimpleTextTokenizer()
    tokenizer.fit(["good product", "bad product"])
    baseline_schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        numerical_targets=(),
        text_targets=("summary",),
        text_max_lengths={"summary": 8},
    )
    m2_schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("verified",),
        numerical_targets=("rating",),
        text_targets=("summary",),
        text_max_lengths={"summary": 8},
    )
    baseline = JointLSTMRelationalAttributeGenerator(
        baseline_schema,
        {
            "rating": CategoryVocab.from_values("rating", [1, 2, 3]),
            "verified": CategoryVocab.from_values(
                "verified", [False, True]
            ),
        },
        tokenizer,
        row_hidden_dim=16,
        latent_noise_dim=4,
        categorical_context_dim=4,
        text_embedding_dim=8,
        text_hidden_dim=12,
        text_num_layers=1,
        use_graph_context=False,
    )
    support_metadata = {
        "event_roles": {
            "source_fk_index": 0,
            "destination_fk_index": 1,
            "timestamp_index": 0,
        },
        "conditioning": {
            "explicit_destination": False,
            "include_support_in_text_context": True,
        },
        "columns": {
            "rating": {
                "resolved_mode": "discrete_support",
                "support_size": 3,
                "support_counts": [2, 1, 1],
                "support_values_standardized": [1.0, 2.0, 3.0],
                "support_values_original": [1, 2, 3],
                "support_output_dtype": "int64",
                "class_weights": [1.0, 1.0, 1.0],
                "label_smoothing": 0.0,
                "ordinal_regularization_weight": 0.0,
                "global_calibration_weight": 0.0,
                "prior": {"enabled": False},
                "global_prior": {"enabled": False},
            }
        },
    }
    m2 = JointLSTMRelationalAttributeGenerator(
        m2_schema,
        {"verified": CategoryVocab.from_values("verified", [False, True])},
        tokenizer,
        row_hidden_dim=16,
        latent_noise_dim=4,
        categorical_context_dim=4,
        text_embedding_dim=8,
        text_hidden_dim=12,
        text_num_layers=1,
        use_graph_context=False,
        numerical_head_metadata=support_metadata,
        numerical_conditioning=support_metadata["conditioning"],
    )

    assert baseline.decoder_context_dim == m2.decoder_context_dim
    assert list(m2.numerical_context_embeddings) == ["rating"]
    row = torch.randn(2, 16)
    context = m2.categorical_context(
        row,
        torch.tensor([[0], [1]]),
        {"rating": torch.tensor([0, 2])},
    )
    assert context.shape == (2, baseline.decoder_context_dim)
