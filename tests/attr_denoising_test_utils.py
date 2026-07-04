from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def make_v3_config_and_components():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2"],
            "product_id": ["i1", "i2", "i1"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "rating": ["5", "4", "1"],
            "verified": ["True", "True", "False"],
            "summary": ["great product", "nice fit", "bad"],
        }
    )
    raw = {
        "paths": {"train_data_path": "unused.csv", "synthetic_spine_path": "unused.csv", "output_dir": "unused"},
        "columns": {
            "condition": {"foreign_keys": ["customer_id", "product_id"], "datetimes": ["review_time"]},
            "target": {"categorical": ["rating", "verified"], "numerical": [], "text": ["summary"]},
        },
        "auxiliary_targets": {"categorical": ["summary_length_bucket"]},
        "text": {"max_length": {"summary": 6}},
        "summary_length": {"enabled": True, "buckets": {"len_1_2": [1, 2], "len_3_5": [3, 5]}},
        "id_encoding": {"num_buckets": 64, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
        "model": {"hidden_dim": 32, "num_layers": 1, "num_heads": 4, "condition_dim": 16, "use_graph_context": True, "graph_context_dim": 12},
        "graph_conditioning": {
            "enabled": True,
            "mode": "temporal_attribute_denoising",
            "temporal_filter": {"enabled": True, "mode": "past_only", "timestamp_column": "review_time", "max_history_events_per_customer": 2, "max_history_events_per_product": 2},
            "graph_encoder": {"hidden_dim": 16, "output_dim": 12, "num_layers": 1, "dropout": 0.0},
            "leakage_policy": {"graph_uses_future_events": False, "graph_uses_clean_target_attributes": False, "graph_uses_clean_future_attributes": False},
        },
        "attribute_denoising": {
            "enabled": True,
            "history_attribute_corruption": {"enabled": True, "mask_prob": 0.15},
            "auxiliary_neighbor_denoising_loss": {"enabled": True, "weight": 0.25, "max_neighbor_nodes_for_loss": 8},
            "attribute_embedding": {"rating_dim": 4, "verified_dim": 4, "summary_length_dim": 4, "summary_dim": 8, "dropout": 0.0},
        },
    }
    schema = ConditionalTABDLMSchema.from_config_dict(raw)
    config = ConditionalTABDLMConfig(raw=raw, schema=schema)
    vocabs = {
        "rating": CategoryVocab.from_values("rating", frame["rating"]),
        "verified": CategoryVocab.from_values("verified", frame["verified"]),
        "summary_length_bucket": CategoryVocab.from_values("summary_length_bucket", ["len_1_2", "len_3_5"]),
    }
    tokenizer = SimpleTextTokenizer().fit(frame["summary"])
    return config, frame, vocabs, tokenizer
