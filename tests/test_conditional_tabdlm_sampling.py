from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.model import ConditionalTABDLM  # noqa: E402
from attribute_generation.conditional_tabdlm.sample import enforce_summary_length_constraints, sample_attributes  # noqa: E402
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
    assert set(attrs["rating"]).issubset({1, 5})
    assert set(attrs["verified"]).issubset(set(vocabs["verified"].token_to_id))
    assert len(attrs["summary"]) == 2


def test_summary_length_constraint_forces_eos_and_pad():
    frame = pd.DataFrame(
        {
            "customer_id": ["c1"],
            "product_id": ["p1"],
            "review_time": ["2020-01-01"],
            "rating": ["5"],
            "verified": ["True"],
            "summary": ["great product"],
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket",),
        text_targets=("summary",),
        text_max_lengths={"summary": 8},
        summary_length_enabled=True,
        use_length_bucket_in_sampling=True,
        force_eos_after_sampled_length=True,
        force_pad_after_eos=True,
        summary_length_buckets={"len_1_2": (1, 2)},
    )
    vocabs = {
        "rating": CategoryVocab.from_values("rating", frame["rating"]),
        "verified": CategoryVocab.from_values("verified", frame["verified"]),
        "summary_length_bucket": CategoryVocab.from_values("summary_length_bucket", ["len_1_2"]),
    }
    tokenizer = SimpleTextTokenizer().fit(frame["summary"])
    cat_input = pd.Series([vocabs["rating"].encode("5"), vocabs["verified"].encode("True"), vocabs["summary_length_bucket"].encode("len_1_2")])
    import torch
    import random

    cat_tensor = torch.tensor([cat_input.tolist()], dtype=torch.long)
    text_input = {"summary": torch.full((1, 8), tokenizer.mask_id, dtype=torch.long)}
    text_input["summary"][:, 0] = tokenizer.bos_id
    text_logits = {"summary": torch.zeros((1, 8, tokenizer.vocab_size), dtype=torch.float32)}
    enforce_summary_length_constraints(
        schema=schema,
        categorical_vocabs=vocabs,
        text_tokenizer=tokenizer,
        cat_input=cat_tensor,
        text_input=text_input,
        text_logits=text_logits,
        rng=random.Random(1),
        temperature=1.0,
        top_p=1.0,
    )
    ids = text_input["summary"][0].tolist()
    assert ids[0] == tokenizer.bos_id
    eos_pos = ids.index(tokenizer.eos_id)
    assert eos_pos in {2, 3}
    assert all(token_id == tokenizer.pad_id for token_id in ids[eos_pos + 1 :])
