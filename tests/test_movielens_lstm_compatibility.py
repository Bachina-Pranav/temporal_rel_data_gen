from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.constrained import decode_category_id, validate_output_categoricals  # noqa: E402
from attribute_generation.conditional_tabdlm.dataset import (  # noqa: E402
    ConditionalTABDLMDataset,
    split_metadata,
    split_prepared_frame,
)
from attribute_generation.conditional_tabdlm.lstm_joint import (  # noqa: E402
    build_lstm_model,
    lstm_joint_loss,
    make_lstm_collate_fn,
    rating_ordinal_auxiliary_loss,
)
from attribute_generation.conditional_tabdlm.evaluate import normalize_rating_series  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402
from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402
from evaluation.paper_metrics.shape_trend import column_shape_metric  # noqa: E402

_baseline_spec = importlib.util.spec_from_file_location(
    "run_interaction_rating_baselines",
    ROOT / "src" / "scripts" / "run_interaction_rating_baselines.py",
)
_baseline_module = importlib.util.module_from_spec(_baseline_spec)
assert _baseline_spec.loader is not None
_baseline_spec.loader.exec_module(_baseline_module)
sample_grouped = _baseline_module.sample_grouped
sample_user_movie_mixture = _baseline_module.sample_user_movie_mixture


def movielens_schema() -> ConditionalTABDLMSchema:
    return ConditionalTABDLMSchema(
        foreign_key_columns=("user_id", "movie_id"),
        datetime_columns=("event_time",),
        categorical_targets=("rating",),
        numerical_targets=(),
        text_targets=(),
    )


def movielens_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "user_id": ["u1", "u1", "u2", "u3", "u4", "u4"],
            "movie_id": ["m1", "m2", "m2", "m3", "m1", "m4"],
            "event_time": pd.date_range("2020-01-01", periods=6, freq="D"),
            "rating": ["0.5", "3.5", "5.0", "4.0", "2.5", "1.0"],
            "split": ["train", "train", "train", "validation", "test", "test"],
        }
    )


def movielens_config(schema: ConditionalTABDLMSchema) -> ConditionalTABDLMConfig:
    raw = {
        "paths": {"train_data_path": "unused.csv", "synthetic_spine_path": "unused.csv", "output_dir": "unused"},
        "id_encoding": {"num_buckets": 128, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
        "model": {
            "row_hidden_dim": 32,
            "latent_noise_dim": 8,
            "categorical_context_dim": 4,
            "dropout": 0.0,
            "use_graph_context": False,
        },
        "text_decoder": {"enabled": False, "type": "none", "embedding_dim": 0, "hidden_dim": 0, "num_layers": 0},
    }
    return ConditionalTABDLMConfig(raw=raw, schema=schema)


def test_explicit_movielens_split_is_preserved_and_reports_cold_start():
    schema = movielens_schema()
    train, valid, test = split_prepared_frame(movielens_frame(), schema)

    assert len(train) == 3
    assert len(valid) == 1
    assert len(test) == 2
    assert train["split"].unique().tolist() == ["train"]
    assert valid["split"].unique().tolist() == ["validation"]

    metadata = split_metadata(train, valid, test, schema)
    assert metadata["split_source"] == "explicit_split_column"
    assert metadata["row_counts"] == {"train": 3, "validation": 1, "test": 2}
    assert metadata["cold_start_foreign_keys"]["user_id"]["validation"]["num_first_seen_after_train"] == 1
    assert metadata["cold_start_foreign_keys"]["movie_id"]["test"]["num_first_seen_after_train"] == 1


def test_movielens_rating_vocab_allows_half_star_values():
    vocab = CategoryVocab.from_values("rating", ["0.5", "1.0", "3.5", "5.0"])

    assert decode_category_id("rating", vocab, vocab.token_to_id["0.5"]) == 0.5
    assert decode_category_id("rating", vocab, vocab.token_to_id["5.0"]) == 5

    frame = pd.DataFrame({"rating": ["0.5", "3.5", 5.0]})
    validated = validate_output_categoricals(frame, {"rating": vocab})
    assert validated["rating"].tolist() == [0.5, 3.5, 5]


def test_movielens_evaluator_preserves_half_star_domain():
    normalized, invalid = normalize_rating_series(
        pd.Series(["0.5", "1.0", "1.5", "5.0", "bad"]),
        valid_rating_values=[0.5, 1.0, 1.5, 5.0],
    )

    assert normalized.tolist()[:4] == [0.5, 1, 1.5, 5]
    assert invalid.tolist() == [False, False, False, False, True]


def test_movielens_lstm_instantiates_no_text_modules_and_trains_one_step():
    schema = movielens_schema()
    frame = movielens_frame().iloc[:4].copy()
    vocabs = {"rating": CategoryVocab.from_values("rating", frame["rating"])}
    tokenizer = SimpleTextTokenizer()
    config = movielens_config(schema)
    dataset = ConditionalTABDLMDataset(frame, schema, vocabs, tokenizer, num_hash_buckets=128)
    batch = make_lstm_collate_fn([dataset[0], dataset[1], dataset[2], dataset[3]])
    model = build_lstm_model(config, vocabs, tokenizer)

    assert model.text_embedding is None
    assert len(model.text_decoders) == 0
    assert len(model.text_heads) == 0
    assert model.summary_condition_projector is None
    assert list(model.numerical_heads) == []

    logits = model(batch["foreign_key_ids"], batch["datetime_values"], batch["categorical_ids"], batch["text_ids"])
    loss, component = lstm_joint_loss(logits, batch, schema, {"rating": 1.0}, tokenizer)
    assert torch.isfinite(loss)
    assert component["rating"]["count"] == 4

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        generated = model.generate(batch["foreign_key_ids"], batch["datetime_values"], vocabs, tokenizer)
    assert set(generated["categorical"]["rating"]).issubset({0.5, 3.5, 4, 5})


def test_movielens_paper_shape_metric_reports_ordinal_wasserstein_for_rating():
    metric = column_shape_metric(
        pd.Series([0.5, 1.0, 5.0]),
        pd.Series([0.5, 3.0, 5.0]),
        "categorical",
        {
            "semantic_type": "ordinal_categorical",
            "ordered": True,
            "dtype": "float",
            "valid_values": [0.5, 1.0, 3.0, 5.0],
        },
    )

    assert metric["primary_statistic"] == "total_variation"
    assert metric["secondary_statistics"]["ordinal_wasserstein_distance"] > 0


def test_recent_temporal_history_uses_capped_strict_past_suffix():
    frame = pd.DataFrame(
        {
            "user_id": ["u"] * 20,
            "movie_id": ["m"] * 20,
            "event_time": pd.date_range("2020-01-01", periods=20, freq="D"),
        }
    )
    history = TemporalHistoryIndex(
        frame,
        customer_col="user_id",
        product_col="movie_id",
        timestamp_col="event_time",
        num_hash_buckets=128,
        max_customer_history=3,
        max_product_history=4,
        history_sampling_strategy="recent",
    )

    assert history.history_for_row(10, kind="customer", deterministic=False) == [7, 8, 9]
    assert history.history_for_row(10, kind="product", deterministic=True) == [6, 7, 8, 9]


def test_movielens_ordinal_auxiliary_loss_is_finite_for_half_star_spacing():
    vocab = CategoryVocab.from_values("rating", ["0.5", "1.0", "2.5", "5.0"])
    raw = {
        "paths": {"train_data_path": "unused.csv", "synthetic_spine_path": "unused.csv", "output_dir": "unused"},
        "columns": {"condition": {"foreign_keys": ["user_id"], "datetimes": ["event_time"]}, "target": {"categorical": ["rating"], "numerical": [], "text": []}},
        "rating_head": {"ordinal_auxiliary": {"enabled": True, "method": "cumulative_emd", "weight": 0.1}},
    }
    config = ConditionalTABDLMConfig(raw=raw, schema=ConditionalTABDLMSchema.from_config_dict(raw))
    logits = torch.randn(4, vocab.size)
    labels = torch.tensor([vocab.encode("0.5"), vocab.encode("1.0"), vocab.encode("2.5"), vocab.encode("5.0")])

    result = rating_ordinal_auxiliary_loss(logits, labels, "rating", config, vocab)

    assert result is not None
    loss_sum, count, weight = result
    assert torch.isfinite(loss_sum)
    assert count == 4
    assert weight == 0.1


def test_empirical_user_baseline_falls_back_for_unseen_user():
    rng = __import__("numpy").random.default_rng(7)
    samples = sample_grouped(
        pd.Series(["unseen"] * 20),
        grouped={"seen": {"5.0": 1.0}},
        fallback={"0.5": 1.0},
        smoothing=1.0,
        rng=rng,
        group_counts={"seen": 10},
        min_group_count=2,
    )

    assert set(samples) == {"0.5"}


def test_user_movie_mixture_samples_from_normalized_weights():
    rng = __import__("numpy").random.default_rng(11)
    frame = pd.DataFrame({"user_id": ["u"], "movie_id": ["m"]})

    samples = sample_user_movie_mixture(
        frame,
        user_col="user_id",
        movie_col="movie_id",
        user_dist={"u": {"5.0": 1.0}},
        movie_dist={"m": {"0.5": 1.0}},
        global_dist={"3.0": 1.0},
        user_counts={"u": 10},
        movie_counts={"m": 10},
        smoothing=1.0,
        min_group_count=1,
        rng=rng,
    )

    assert samples[0] in {"0.5", "3.0", "5.0"}
