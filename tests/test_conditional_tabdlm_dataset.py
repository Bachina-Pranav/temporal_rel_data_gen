from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.dataset import (  # noqa: E402
    ConditionalTABDLMDataset,
    collate_and_mask,
)
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def make_fixture():
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
        text_max_lengths={"summary": 6},
    )
    vocabs = {
        "rating": CategoryVocab.from_values("rating", frame["rating"]),
        "verified": CategoryVocab.from_values("verified", frame["verified"]),
    }
    tokenizer = SimpleTextTokenizer().fit(frame["summary"])
    return frame, schema, vocabs, tokenizer


def test_dataset_encodes_conditions_and_targets():
    frame, schema, vocabs, tokenizer = make_fixture()
    dataset = ConditionalTABDLMDataset(frame, schema, vocabs, tokenizer, num_hash_buckets=128)
    item = dataset[0]
    assert item["foreign_key_ids"].shape[0] == 2
    assert item["datetime_values"].shape[0] == 1
    assert item["categorical_ids"].shape[0] == 2
    assert item["text_ids"]["summary"].shape[0] == 6


def test_collate_never_masks_condition_columns():
    frame, schema, vocabs, tokenizer = make_fixture()
    dataset = ConditionalTABDLMDataset(frame, schema, vocabs, tokenizer, num_hash_buckets=128)
    samples = [dataset[0], dataset[1]]
    batch = collate_and_mask(
        samples,
        schema=schema,
        categorical_vocabs=vocabs,
        text_tokenizer=tokenizer,
        min_mask_prob=1.0,
        max_mask_prob=1.0,
        mask_schedule="linear",
    )
    assert "foreign_key_labels" not in batch
    assert "datetime_labels" not in batch
    assert batch["foreign_key_ids"].equal(samples[0]["foreign_key_ids"].new_tensor([s["foreign_key_ids"].tolist() for s in samples]))
    assert (batch["categorical_labels"] != -100).all()
    assert (batch["text_labels"]["summary"][batch["text_attention"]["summary"].bool()] != -100).all()

