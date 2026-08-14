from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.dataset import (  # noqa: E402
    ConditionalTABDLMDataset,
)
from attribute_generation.conditional_tabdlm.lstm_joint import (  # noqa: E402
    build_lstm_model,
    load_lstm_checkpoint,
    lstm_joint_loss,
    make_lstm_collate_fn,
    prepare_numerical_head_config,
    save_lstm_checkpoint,
)
from attribute_generation.conditional_tabdlm.numerical import (  # noqa: E402
    fit_numerical_transformers,
)
from attribute_generation.conditional_tabdlm.numerical_head import (  # noqa: E402
    GlobalSupportPrior,
    SmoothedSupportPrior,
    fit_global_support_prior,
    fit_numerical_head_metadata,
)
from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMConfig,
    ConditionalTABDLMSchema,
)
from attribute_generation.conditional_tabdlm.tokenization import (  # noqa: E402
    SimpleTextTokenizer,
)


def test_discrete_support_head_decodes_only_training_values_and_backpropagates():
    frame, config, numerical_metadata = fixture("discrete_support")
    model, batch = build_fixture_model(frame, config, numerical_metadata)

    output = model(
        batch["foreign_key_ids"],
        batch["datetime_values"],
        batch["categorical_ids"],
        batch["text_ids"],
        noise=torch.zeros(len(frame), model.latent_noise_dim),
        numerical_values=batch["numerical_values"],
    )
    price_output = output["numerical"]["price"]
    assert price_output["logits"].shape == (len(frame), 3)

    loss, components = lstm_joint_loss(
        output,
        batch,
        config.schema,
        {"price": 1.0},
        SimpleTextTokenizer(),
        config=config,
    )
    loss.backward()

    destination_index = config.schema.foreign_key_columns.index(
        "article_id"
    )
    destination_grad = model.foreign_key_embeddings[
        destination_index
    ].weight.grad
    assert torch.isfinite(loss)
    assert destination_grad is not None
    assert float(destination_grad.abs().sum()) > 0.0
    assert "price_nll" in components

    torch.manual_seed(42)
    sampled = model.sample_support_numerical(
        output["numerical"],
        temperature=1.0,
    )["price"]
    assert set(sampled.tolist()).issubset({0.01, 0.02, 0.03})


def test_explicit_destination_fusion_changes_fixed_latent_logits():
    frame, config, numerical_metadata = fixture("discrete_support")
    model, batch = build_fixture_model(frame, config, numerical_metadata)
    model.eval()
    noise = torch.zeros(len(frame), model.latent_noise_dim)

    first = model(
        batch["foreign_key_ids"],
        batch["datetime_values"],
        batch["categorical_ids"],
        batch["text_ids"],
        noise=noise,
        numerical_values=batch["numerical_values"],
    )["numerical"]["price"]["logits"]
    shuffled_ids = batch["foreign_key_ids"].clone()
    shuffled_ids[:, 1] = shuffled_ids[:, 1].roll(1)
    second = model(
        shuffled_ids,
        batch["datetime_values"],
        batch["categorical_ids"],
        batch["text_ids"],
        noise=noise,
        numerical_values=batch["numerical_values"],
    )["numerical"]["price"]["logits"]

    assert not torch.allclose(first, second)


def test_hierarchical_support_head_has_finite_two_stage_loss():
    frame, config, numerical_metadata = fixture(
        "hierarchical_support",
    )
    model, batch = build_fixture_model(frame, config, numerical_metadata)

    output = model(
        batch["foreign_key_ids"],
        batch["datetime_values"],
        batch["categorical_ids"],
        batch["text_ids"],
        numerical_values=batch["numerical_values"],
    )
    price = output["numerical"]["price"]
    loss, _ = lstm_joint_loss(
        output,
        batch,
        config.schema,
        {"price": 1.0},
        SimpleTextTokenizer(),
        config=config,
    )

    assert price["coarse_logits"].shape[1] >= 2
    assert len(price["fine_groups"]) >= 1
    assert torch.isfinite(loss)


def test_smoothed_prior_normalizes_and_handles_cold_destination():
    prior = SmoothedSupportPrior(
        {
            "enabled": True,
            "global_counts": [2.0, 3.0, 5.0],
            "time_boundaries_seconds": [5.0],
            "time_counts": [[1.0, 2.0, 0.0], [1.0, 1.0, 4.0]],
            "frequency_counts": [[1.0, 1.0, 1.0]],
            "hash_to_frequency": [0, -1],
            "exact_hash_to_row": [-1, -1],
            "exact_support_ids": [],
            "exact_support_counts": [],
            "exact_totals": [],
        }
    )

    probabilities = torch.exp(
        prior(
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0.0, 10.0]),
        )
    )

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.isfinite(probabilities).all()


def test_fitted_hierarchical_prior_is_normalized_for_seen_and_unseen_hashes():
    frame, config, numerical_metadata = fixture("discrete_support")
    config.raw["numerical_heads"]["prior"] = {
        "enabled": True,
        "num_frequency_buckets": 2,
        "num_time_buckets": 2,
        "min_destination_rows": 2,
        "max_exact_support_values": 3,
    }
    metadata = fit_numerical_head_metadata(
        config,
        train_frame=frame,
        numerical_metadata=numerical_metadata,
    )
    prior = SmoothedSupportPrior(
        metadata["columns"]["price"]["prior"]
    )

    probabilities = torch.exp(
        prior(
            torch.tensor([0, 63], dtype=torch.long),
            torch.tensor([1.5e9, 1.6e9]),
        )
    )

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_old_continuous_checkpoint_layout_remains_strictly_loadable():
    frame, config, _ = fixture(
        "continuous_baseline",
        enable_new_feature=False,
    )
    tokenizer = SimpleTextTokenizer()
    first = build_lstm_model(config, {}, tokenizer)
    state = copy.deepcopy(first.state_dict())
    second = build_lstm_model(config, {}, tokenizer)

    result = second.load_state_dict(state, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert list(second.numerical_heads) == ["price"]
    assert list(second.support_numerical_heads) == []


def test_support_head_checkpoint_round_trip_and_fixed_seed_sampling(tmp_path):
    frame, config, numerical_metadata = fixture("discrete_support")
    model, batch = build_fixture_model(frame, config, numerical_metadata)
    checkpoint = tmp_path / "best.pt"
    save_lstm_checkpoint(
        checkpoint,
        model,
        config,
        {},
        SimpleTextTokenizer(),
        epoch=1,
        valid_metrics={"total_loss": 1.0},
    )

    loaded, loaded_config, _, _, _ = load_lstm_checkpoint(
        checkpoint,
        device="cpu",
    )
    loaded.eval()
    first = generate_support_values(loaded, batch, seed=73)
    second = generate_support_values(loaded, batch, seed=73)

    assert loaded.numerical_head_modes["price"] == "discrete_support"
    assert "_numerical_head_metadata" in loaded_config.raw
    assert torch.equal(first, second)


def test_support_metadata_never_adds_unseen_validation_value():
    frame, config, numerical_metadata = fixture("auto")
    metadata = fit_numerical_head_metadata(
        config,
        train_frame=frame,
        numerical_metadata=numerical_metadata,
    )

    support = metadata["columns"]["price"].get(
        "support_values_original",
        [],
    )
    assert 9.99 not in support
    assert set(support).issubset({0.01, 0.02, 0.03})


def test_pretokenized_run_fits_exact_support_from_training_only_table(
    tmp_path: Path,
):
    frame, config, numerical_metadata = fixture("auto")
    training_table = tmp_path / "train_real.csv"
    frame.to_csv(training_table, index=False)
    config.raw["paths"][
        "numerical_head_training_table_path"
    ] = str(training_table)

    metadata = prepare_numerical_head_config(
        config,
        numerical_metadata=numerical_metadata,
        train_frame=None,
        train_dataset=object(),
        metadata_dir=tmp_path,
    )

    assert metadata["fit_source"] == "training_only_table"
    assert set(
        metadata["columns"]["price"]["support_values_original"]
    ) == {0.01, 0.02, 0.03}
    assert config.raw["numerical_columns"]["price"] == {
        "inferred_type": "repeated_or_quantized",
        "selected_head": "support_prior",
        "implementation_mode": "discrete_support",
        "support_size": 3,
        "unique_ratio": 0.5,
        "repeated_mass": 1.0,
        "training_only": True,
    }


def test_integer_support_decodes_as_numeric_integer_tensor():
    frame, config, numerical_metadata = fixture(
        "discrete_support"
    )
    frame["price"] = [1, 1, 2, 2, 3, 3]
    numerical_metadata = fit_numerical_transformers(frame, config)
    config.raw["_numerical_metadata"] = numerical_metadata
    model, batch = build_fixture_model(
        frame,
        config,
        numerical_metadata,
    )

    sampled = generate_support_values(model, batch, seed=42)
    column_metadata = config.raw["_numerical_head_metadata"][
        "columns"
    ]["price"]

    assert column_metadata["support_output_dtype"] == "int64"
    assert sampled.dtype == torch.int64
    assert set(sampled.tolist()).issubset({1, 2, 3})


def test_event_roles_not_foreign_key_positions_drive_destination_fusion():
    frame, config, numerical_metadata = fixture(
        "discrete_support"
    )
    config.schema = ConditionalTABDLMSchema(
        foreign_key_columns=("article_id", "customer_id"),
        datetime_columns=("event_time",),
        categorical_targets=(),
        numerical_targets=("price",),
        text_targets=(),
    )
    config.raw["columns"]["condition"]["foreign_keys"] = [
        "article_id",
        "customer_id",
    ]
    model, batch = build_fixture_model(
        frame,
        config,
        numerical_metadata,
    )
    roles = config.raw["_numerical_head_metadata"]["event_roles"]
    noise = torch.zeros(len(frame), model.latent_noise_dim)
    first = model(
        batch["foreign_key_ids"],
        batch["datetime_values"],
        batch["categorical_ids"],
        batch["text_ids"],
        noise=noise,
        numerical_values=batch["numerical_values"],
    )["numerical"]["price"]["logits"]
    changed = batch["foreign_key_ids"].clone()
    changed[:, roles["destination_fk_index"]] = changed[
        :,
        roles["destination_fk_index"],
    ].roll(1)
    second = model(
        changed,
        batch["datetime_values"],
        batch["categorical_ids"],
        batch["text_ids"],
        noise=noise,
        numerical_values=batch["numerical_values"],
    )["numerical"]["price"]["logits"]

    assert roles["destination_fk_index"] == 0
    assert roles["source_fk_index"] == 1
    assert not torch.allclose(first, second)


def test_support_prior_uses_smoothed_training_counts_and_near_zero_residual():
    frame, config, numerical_metadata = fixture("support_prior")
    config.raw["numerical_heads"]["global_prior"] = {
        "smoothing": 2.0,
        "alpha": 1.0,
        "residual_weight": 0.0,
        "residual_init_scale": 0.001,
    }
    model, batch = build_fixture_model(
        frame,
        config,
        numerical_metadata,
    )
    metadata = config.raw["_numerical_head_metadata"]["columns"]["price"]
    head = model.support_numerical_heads["price"]
    probability = torch.as_tensor(
        metadata["global_prior"]["probabilities"]
    )

    assert metadata["global_prior"]["training_only"] is True
    assert torch.allclose(probability.sum(), torch.tensor(1.0))
    assert metadata["selected_head"] == "support_prior"
    assert float(head.linear.weight.abs().max()) < 0.01

    condition = model.encode_condition(
        batch["foreign_key_ids"],
        batch["datetime_values"],
    )
    row = model.row_latent(
        condition,
        noise=torch.zeros(len(frame), model.latent_noise_dim),
    )
    output = model.numerical_params(
        row,
        batch["foreign_key_ids"],
        batch["datetime_values"],
    )["price"]
    predicted = torch.softmax(output["logits"], dim=-1)
    assert torch.allclose(
        predicted,
        probability.unsqueeze(0).expand_as(predicted),
        atol=1e-6,
    )


def test_global_prior_smoothing_handles_zero_counts():
    metadata = fit_global_support_prior(
        torch.tensor([4.0, 0.0, 2.0]).numpy(),
        {"enabled": True, "smoothing": 1.0},
    )
    probability = torch.tensor(metadata["probabilities"])

    assert torch.all(probability > 0)
    assert torch.allclose(probability.sum(), torch.tensor(1.0))


def test_global_prior_runtime_bias_is_deterministic_and_normalized():
    prior = GlobalSupportPrior(
        {
            "enabled": True,
            "probabilities": [0.2, 0.3, 0.5],
            "runtime_logit_bias": [0.1, -0.2, 0.3],
            "residual_weight": 0.0,
        }
    )
    logits = prior.combine(torch.randn(4, 3))
    first = torch.softmax(logits, dim=-1)
    second = torch.softmax(prior.combine(torch.randn(4, 3)), dim=-1)

    assert torch.allclose(first, second)
    assert torch.allclose(first.sum(dim=1), torch.ones(4))


def test_temporal_support_prior_uses_training_quantile_buckets():
    values = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).numpy()
    timestamps = torch.tensor([1.0, 2.0, 3.0, 100.0, 101.0, 102.0]).numpy()
    metadata = fit_global_support_prior(
        torch.tensor([3.0, 3.0]).numpy(),
        {
            "enabled": True,
            "smoothing": 0.0,
            "residual_weight": 0.0,
            "temporal_prior": {
                "enabled": True,
                "lambda_t": 1.0,
                "num_time_buckets": 2,
                "backoff_strength": 0.0,
                "min_bucket_rows": 1,
            },
        },
        training_values=values,
        support=torch.tensor([0.0, 1.0]).numpy(),
        timestamps=timestamps,
    )
    prior = GlobalSupportPrior(metadata)
    logits = prior.combine(
        torch.randn(2, 2),
        torch.tensor([2.0, 101.0]),
    )
    probability = torch.softmax(logits, dim=-1)

    assert metadata["temporal_prior"]["training_only"] is True
    assert probability[0, 0] > 0.99
    assert probability[1, 1] > 0.99


def test_temporal_support_prior_sparse_bucket_backs_off_to_global():
    metadata = fit_global_support_prior(
        torch.tensor([3.0, 1.0]).numpy(),
        {
            "enabled": True,
            "smoothing": 0.0,
            "residual_weight": 0.0,
            "temporal_prior": {
                "enabled": True,
                "lambda_t": 0.25,
                "num_time_buckets": 2,
                "backoff_strength": 0.0,
                "min_bucket_rows": 10,
            },
        },
        training_values=torch.tensor([0.0, 0.0, 0.0, 1.0]).numpy(),
        support=torch.tensor([0.0, 1.0]).numpy(),
        timestamps=torch.tensor([1.0, 2.0, 100.0, 101.0]).numpy(),
    )
    temporal = torch.tensor(
        metadata["temporal_prior"]["bucket_probabilities"]
    )
    global_probability = torch.tensor(metadata["probabilities"])

    assert torch.allclose(
        temporal,
        global_probability.unsqueeze(0).expand_as(temporal),
    )


def test_hierarchical_support_applies_runtime_bias_without_training_prior():
    frame, config, numerical_metadata = fixture(
        "hierarchical_support"
    )
    model, batch = build_fixture_model(
        frame,
        config,
        numerical_metadata,
    )
    head = model.support_numerical_heads["price"]
    condition = model.encode_condition(
        batch["foreign_key_ids"],
        batch["datetime_values"],
    )
    row = model.row_latent(
        condition,
        noise=torch.zeros(len(frame), model.latent_noise_dim),
    )
    before = head(
        row,
        batch["foreign_key_ids"][:, 1],
        batch["datetime_values"][:, 0],
    )["coarse_logits"]
    head.global_prior.set_runtime_logit_bias([3.0, 0.0, -3.0])
    after = head(
        row,
        batch["foreign_key_ids"][:, 1],
        batch["datetime_values"][:, 0],
    )["coarse_logits"]

    assert head.global_prior.enabled is False
    assert head.global_prior.has_runtime_logit_bias is True
    assert not torch.allclose(before, after)


def test_auto_router_selects_support_prior_and_persists_profile():
    frame, config, numerical_metadata = fixture("auto")
    metadata = fit_numerical_head_metadata(
        config,
        train_frame=frame,
        numerical_metadata=numerical_metadata,
    )
    report = metadata["columns"]["price"]

    assert report["selected_head"] == "support_prior"
    assert report["resolved_mode"] == "discrete_support"
    assert report["global_prior"]["enabled"] is True


def test_singular_legacy_continuous_override_preserves_linear_layout():
    frame, config, numerical_metadata = fixture(
        "continuous_baseline",
        enable_new_feature=False,
    )
    config.raw["numerical_head"] = {"mode": "continuous"}
    metadata = fit_numerical_head_metadata(
        config,
        train_frame=frame,
        numerical_metadata=numerical_metadata,
    )
    config.raw["_numerical_head_metadata"] = metadata
    model = build_lstm_model(
        config,
        {},
        SimpleTextTokenizer(),
    )

    assert metadata["columns"]["price"]["selected_head"] == "continuous"
    assert list(model.numerical_heads) == ["price"]
    assert list(model.support_numerical_heads) == []


def build_fixture_model(
    frame: pd.DataFrame,
    config: ConditionalTABDLMConfig,
    numerical_metadata: dict,
):
    metadata = fit_numerical_head_metadata(
        config,
        train_frame=frame,
        numerical_metadata=numerical_metadata,
    )
    config.raw["_numerical_head_metadata"] = metadata
    dataset = ConditionalTABDLMDataset(
        frame,
        config.schema,
        {},
        SimpleTextTokenizer(),
        num_hash_buckets=64,
        numerical_metadata=numerical_metadata,
    )
    batch = make_lstm_collate_fn(
        [dataset[index] for index in range(len(dataset))]
    )
    model = build_lstm_model(config, {}, SimpleTextTokenizer())
    return model, batch


def generate_support_values(model, batch, *, seed: int):
    torch.manual_seed(seed)
    with torch.no_grad():
        condition = model.encode_condition(
            batch["foreign_key_ids"],
            batch["datetime_values"],
        )
        row = model.row_latent(condition)
        output = model.numerical_params(
            row,
            batch["foreign_key_ids"],
            batch["datetime_values"],
        )
        return model.sample_support_numerical(
            output,
            temperature=1.0,
        )["price"]


def fixture(
    mode: str,
    *,
    enable_new_feature: bool = True,
):
    frame = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2", "c2", "c3", "c3"],
            "article_id": ["a1", "a1", "a2", "a2", "a3", "a3"],
            "event_time": pd.date_range(
                "2020-01-01",
                periods=6,
                freq="D",
            ),
            "price": [0.01, 0.01, 0.02, 0.02, 0.03, 0.03],
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "article_id"),
        datetime_columns=("event_time",),
        categorical_targets=(),
        numerical_targets=("price",),
        text_targets=(),
    )
    raw = {
        "paths": {
            "train_data_path": "unused.csv",
            "synthetic_spine_path": "unused.csv",
            "output_dir": "unused",
        },
        "event_spine": {
            "source_fk": "customer_id",
            "destination_fk": "article_id",
            "timestamp": "event_time",
        },
        "columns": {
            "condition": {
                "foreign_keys": ["customer_id", "article_id"],
                "datetimes": ["event_time"],
            },
            "target": {
                "categorical": [],
                "numerical": ["price"],
                "text": [],
            },
        },
        "schema": {
            "fields": {
                "customer_id": {"role": "source_foreign_key"},
                "article_id": {"role": "destination_foreign_key"},
                "event_time": {"role": "timestamp"},
                "price": {
                    "role": "generated_attribute",
                    "semantic_type": "continuous_numerical",
                },
            }
        },
        "id_encoding": {"num_buckets": 64, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
        "model": {
            "row_hidden_dim": 16,
            "latent_noise_dim": 4,
            "categorical_context_dim": 4,
            "dropout": 0.0,
            "use_graph_context": False,
        },
        "text_decoder": {
            "enabled": False,
            "type": "none",
            "embedding_dim": 0,
            "hidden_dim": 0,
            "num_layers": 0,
        },
        "training": {"seed": 42},
    }
    if enable_new_feature:
        raw["numerical_heads"] = {
            "mode": mode,
            "hierarchical_num_bins": 2,
            "label_smoothing": 0.01,
            "conditioning": {"explicit_destination": True},
            "type_inference": {
                "low_cardinality_max_values": 2,
            },
        }
    config = ConditionalTABDLMConfig(raw=raw, schema=schema)
    numerical_metadata = fit_numerical_transformers(frame, config)
    config.raw["_numerical_metadata"] = numerical_metadata
    return frame, config, numerical_metadata
