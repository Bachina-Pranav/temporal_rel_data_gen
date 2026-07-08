from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_joint import build_lstm_model  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def make_lstm_fast_fixture():
    frame = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "product_id": ["p1", "p2", "p3", "p4"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "rating": ["5", "4", "2", "1"],
            "verified": ["True", "False", "True", "False"],
            "summary": ["great item", "fine item", "poor fit", "bad fit"],
            "review_text": ["great item works well", "fine item overall", "poor fit for me", "bad fit"],
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket", "review_text_length_bucket"),
        text_targets=("summary", "review_text"),
        text_max_lengths={"summary": 8, "review_text": 16},
        summary_length_enabled=True,
        summary_length_buckets={"short": (0, 2), "long": (3, 6)},
        review_text_length_buckets={"short": (0, 3), "long": (8, 14)},
    )
    raw = {
        "paths": {"train_data_path": "unused.csv", "synthetic_spine_path": "unused.csv", "output_dir": "unused"},
        "columns": {
            "condition": {"foreign_keys": ["customer_id", "product_id"], "datetimes": ["review_time"]},
            "target": {"categorical": ["rating", "verified"], "numerical": [], "text": ["summary", "review_text"]},
        },
        "auxiliary_targets": {"categorical": ["summary_length_bucket", "review_text_length_bucket"]},
        "text": {"max_length": {"summary": 8, "review_text": 16}},
        "summary_length": {
            "enabled": True,
            "use_length_bucket_in_sampling": True,
            "buckets": {"short": [0, 2], "long": [3, 6]},
        },
        "review_text_length": {
            "enabled": True,
            "use_length_bucket_in_sampling": True,
            "buckets": {"short": [0, 3], "long": [8, 14]},
        },
        "model": {"row_hidden_dim": 32, "latent_noise_dim": 8, "categorical_context_dim": 4, "dropout": 0.0, "use_graph_context": False},
        "text_decoder": {"embedding_dim": 16, "hidden_dim": 24, "num_layers": 1, "dropout": 0.0, "type": "lstm"},
        "id_encoding": {"num_buckets": 64, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
        "graph_conditioning": {
            "enabled": True,
            "mode": "structure_only_temporal",
            "temporal_filter": {"enabled": True, "mode": "past_only"},
            "leakage_policy": {
                "graph_uses_future_events": False,
                "graph_uses_target_attributes": False,
                "real_graph_used_at_sampling": False,
            },
        },
    }
    config = ConditionalTABDLMConfig(raw=raw, schema=schema, config_path=None)
    vocabs = {
        "rating": CategoryVocab.from_values("rating", frame["rating"]),
        "verified": CategoryVocab.from_values("verified", frame["verified"]),
        "summary_length_bucket": CategoryVocab.from_values("summary_length_bucket", ["short", "long"]),
        "review_text_length_bucket": CategoryVocab.from_values("review_text_length_bucket", ["short", "long"]),
    }
    tokenizer = SimpleTextTokenizer().fit(frame["summary"].tolist() + frame["review_text"].tolist())
    model = build_lstm_model(config, vocabs, tokenizer)
    return frame, config, vocabs, tokenizer, model
