from __future__ import annotations

import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.sample import enforce_length_constraints  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def test_full_review_text_length_enforcement_handles_both_text_fields():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket", "review_text_length_bucket"),
        text_targets=("summary", "review_text"),
        text_max_lengths={"summary": 8, "review_text": 16},
        summary_length_enabled=True,
        use_length_bucket_in_sampling=True,
        force_eos_after_sampled_length=True,
        force_pad_after_eos=True,
        summary_length_buckets={"len_1_2": (1, 2)},
        review_text_length_buckets={"q0_q20": (3, 5)},
    )
    tokenizer = SimpleTextTokenizer().fit(["alpha beta gamma delta epsilon zeta"])
    vocabs = {
        "rating": CategoryVocab.from_values("rating", ["5"]),
        "verified": CategoryVocab.from_values("verified", ["True"]),
        "summary_length_bucket": CategoryVocab.from_values("summary_length_bucket", ["len_1_2"]),
        "review_text_length_bucket": CategoryVocab.from_values("review_text_length_bucket", ["q0_q20"]),
    }
    cat_input = torch.tensor(
        [[
            vocabs["rating"].encode("5"),
            vocabs["verified"].encode("True"),
            vocabs["summary_length_bucket"].encode("len_1_2"),
            vocabs["review_text_length_bucket"].encode("q0_q20"),
        ]],
        dtype=torch.long,
    )
    text_input = {
        "summary": torch.full((1, 8), tokenizer.mask_id, dtype=torch.long),
        "review_text": torch.full((1, 16), tokenizer.mask_id, dtype=torch.long),
    }
    text_input["summary"][0, 0] = tokenizer.bos_id
    text_input["review_text"][0, 0] = tokenizer.bos_id
    text_logits = {
        "summary": torch.zeros((1, 8, tokenizer.vocab_size), dtype=torch.float32),
        "review_text": torch.zeros((1, 16, tokenizer.vocab_size), dtype=torch.float32),
    }

    debug = enforce_length_constraints(
        schema,
        vocabs,
        tokenizer,
        cat_input,
        text_input,
        text_logits,
        rng=random.Random(7),
        temperature=1.0,
        top_p=1.0,
        repetition_penalties={"summary": 1.0, "review_text": 1.0},
        min_content_tokens={"summary": 1, "review_text": 1},
    )

    assert set(debug[0]) == {"summary", "review_text"}
    assert debug[0]["summary"]["target_bucket_respected"] == 1
    assert debug[0]["review_text"]["target_bucket_respected"] == 1
    assert tokenizer.decode(text_input["summary"][0].tolist())
    assert tokenizer.decode(text_input["review_text"][0].tolist())

