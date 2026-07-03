from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.dataset import ConditionalTABDLMDataset, collate_and_mask  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def test_masking_forces_at_least_one_target_mask_per_row():
    frame = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3"],
            "product_id": ["p1", "p2", "p3"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "rating": ["5", "4", "3"],
            "verified": ["True", "True", "False"],
            "summary": ["great", "okay", "bad"],
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        text_targets=("summary",),
        text_max_lengths={"summary": 4},
    )
    vocabs = {
        "rating": CategoryVocab.from_values("rating", frame["rating"]),
        "verified": CategoryVocab.from_values("verified", frame["verified"]),
    }
    tokenizer = SimpleTextTokenizer().fit(frame["summary"])
    dataset = ConditionalTABDLMDataset(frame, schema, vocabs, tokenizer, num_hash_buckets=64)
    batch = collate_and_mask(
        [dataset[i] for i in range(len(dataset))],
        schema=schema,
        categorical_vocabs=vocabs,
        text_tokenizer=tokenizer,
        min_mask_prob=0.0,
        max_mask_prob=0.0,
    )
    masked = (batch["categorical_labels"] != -100).any(dim=1)
    masked |= (batch["text_labels"]["summary"] != -100).any(dim=1)
    assert masked.all()

