from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.model import ConditionalTABDLM  # noqa: E402
from attribute_generation.conditional_tabdlm.sample import sample_attributes  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def test_sampling_produces_valid_categorical_values():
    frame = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "product_id": ["p1", "p2"],
            "review_time": ["2020-01-01", "2020-01-02"],
            "rating": ["5", "1"],
            "verified": ["True", "False"],
            "summary": ["great product", "bad fit"],
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        text_targets=("summary",),
        text_max_lengths={"summary": 4},
    )
    raw = {
        "paths": {
            "train_data_path": "unused.csv",
            "synthetic_spine_path": "unused.csv",
            "output_dir": "unused",
        },
        "columns": {
            "condition": {"foreign_keys": ["customer_id", "product_id"], "datetimes": ["review_time"]},
            "target": {"categorical": ["rating", "verified"], "numerical": [], "text": ["summary"]},
        },
        "text": {"max_length": {"summary": 4}},
        "id_encoding": {"num_buckets": 64, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
        "model": {"hidden_dim": 32, "num_layers": 1, "num_heads": 4, "condition_dim": 16},
        "diffusion": {"timesteps": 2},
    }
    vocabs = {
        "rating": CategoryVocab.from_values("rating", frame["rating"]),
        "verified": CategoryVocab.from_values("verified", frame["verified"]),
    }
    tokenizer = SimpleTextTokenizer().fit(frame["summary"])
    model = ConditionalTABDLM(
        schema,
        vocabs,
        tokenizer,
        num_hash_buckets=64,
        id_embedding_dim=8,
        datetime_embedding_dim=8,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        condition_dim=16,
    )
    attrs = sample_attributes(
        model,
        schema,
        vocabs,
        tokenizer,
        frame[["customer_id", "product_id", "review_time"]],
        config=ConditionalTABDLMConfig(raw=raw, schema=schema),
        batch_size=2,
        temperature=1.0,
        top_p=0.9,
        device="cpu",
    )
    assert set(attrs["rating"]).issubset(set(vocabs["rating"].token_to_id))
    assert set(attrs["verified"]).issubset(set(vocabs["verified"].token_to_id))
    assert len(attrs["summary"]) == 2

