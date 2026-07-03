from __future__ import annotations

import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.sample import enforce_summary_length_constraints  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def test_soft_length_sampling_respects_bucket_and_pads_after_eos():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket",),
        text_targets=("summary",),
        text_max_lengths={"summary": 8},
        summary_length_enabled=True,
        use_length_bucket_in_sampling=True,
        force_eos_after_sampled_length="soft",
        force_pad_after_eos=True,
        summary_length_buckets={"len_3_5": (3, 5)},
    )
    tokenizer = SimpleTextTokenizer().fit(["alpha beta gamma delta epsilon"])
    vocabs = {
        "rating": CategoryVocab.from_values("rating", ["5"]),
        "verified": CategoryVocab.from_values("verified", ["True"]),
        "summary_length_bucket": CategoryVocab.from_values("summary_length_bucket", ["len_3_5"]),
    }
    cat_input = torch.tensor(
        [[
            vocabs["rating"].encode("5"),
            vocabs["verified"].encode("True"),
            vocabs["summary_length_bucket"].encode("len_3_5"),
        ]],
        dtype=torch.long,
    )
    text_input = {"summary": torch.full((1, 8), tokenizer.mask_id, dtype=torch.long)}
    text_input["summary"][0, 0] = tokenizer.bos_id
    text_logits = {"summary": torch.zeros((1, 8, tokenizer.vocab_size), dtype=torch.float32)}

    debug_rows = enforce_summary_length_constraints(
        schema=schema,
        categorical_vocabs=vocabs,
        text_tokenizer=tokenizer,
        cat_input=cat_input,
        text_input=text_input,
        text_logits=text_logits,
        rng=random.Random(3),
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.15,
        min_content_tokens=1,
    )

    ids = text_input["summary"][0].tolist()
    content_length = tokenizer.content_length(ids)
    eos_pos = ids.index(tokenizer.eos_id)
    assert 3 <= content_length <= 5
    assert all(token_id == tokenizer.pad_id for token_id in ids[eos_pos + 1 :])
    assert debug_rows[0]["target_bucket_respected"] == 1

