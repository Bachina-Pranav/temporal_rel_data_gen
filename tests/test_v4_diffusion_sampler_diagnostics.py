from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.model import ConditionalTABDLM  # noqa: E402
from attribute_generation.conditional_tabdlm.sample import (  # noqa: E402
    enforce_length_constraints,
    masked_denoising_schedule,
    resolve_sampling_steps,
    sample_attributes,
)
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema, load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def tiny_fixture() -> tuple[pd.DataFrame, ConditionalTABDLMSchema, dict[str, CategoryVocab], SimpleTextTokenizer, ConditionalTABDLMConfig]:
    frame = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4", "c5"],
            "product_id": ["p1", "p2", "p3", "p4", "p5"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            "rating": ["5", "1", "4", "5", "2"],
            "verified": ["True", "False", "True", "True", "False"],
            "title": ["great fit", "bad fit", "works well", "nice material", "too small"],
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        text_targets=("title",),
        text_max_lengths={"title": 8},
        force_pad_after_eos=True,
    )
    raw = {
        "paths": {"train_data_path": "unused.csv", "synthetic_spine_path": "unused.csv", "output_dir": "unused"},
        "columns": {
            "condition": {"foreign_keys": ["customer_id", "product_id"], "datetimes": ["review_time"]},
            "target": {"categorical": ["rating", "verified"], "numerical": [], "text": ["title"]},
        },
        "text": {"max_length": {"title": 8}},
        "id_encoding": {"num_buckets": 64, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
        "model": {"hidden_dim": 32, "num_layers": 1, "num_heads": 4, "condition_dim": 16},
        "diffusion": {"timesteps": 4, "sampling_steps": 4},
    }
    vocabs = {
        "rating": CategoryVocab.from_values("rating", frame["rating"]),
        "verified": CategoryVocab.from_values("verified", frame["verified"]),
    }
    tokenizer = SimpleTextTokenizer().fit(frame["title"])
    return frame, schema, vocabs, tokenizer, ConditionalTABDLMConfig(raw=raw, schema=schema)


def build_tiny_model(schema: ConditionalTABDLMSchema, vocabs: dict[str, CategoryVocab], tokenizer: SimpleTextTokenizer) -> ConditionalTABDLM:
    return ConditionalTABDLM(
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


def test_resolve_sampling_steps_supports_full_and_caps_to_training_horizon():
    assert resolve_sampling_steps("full", 50) == 50
    assert resolve_sampling_steps("25", 50) == 25
    assert resolve_sampling_steps(100, 50) == 50
    assert masked_denoising_schedule(total_timesteps=4, sampling_steps=resolve_sampling_steps("full", 4)) == [4, 3, 2, 1]


def test_sampling_with_25_50_and_full_requests_produces_requested_row_count_and_call_counts():
    frame, schema, vocabs, tokenizer, config = tiny_fixture()
    for request in [25, 50, "full"]:
        model = build_tiny_model(schema, vocabs, tokenizer)
        calls = {"count": 0}
        original_forward = model.forward

        def wrapped_forward(*args, **kwargs):
            calls["count"] += 1
            return original_forward(*args, **kwargs)

        model.forward = wrapped_forward  # type: ignore[method-assign]
        resolved = resolve_sampling_steps(request, 4)
        attrs = sample_attributes(
            model,
            schema,
            vocabs,
            tokenizer,
            frame[["customer_id", "product_id", "review_time"]],
            config=config,
            batch_size=2,
            temperature=1.0,
            top_p=1.0,
            device="cpu",
            seed=123,
            sampling_steps=resolved,
        )
        diagnostics = attrs["_sampling_diagnostics"]
        assert len(attrs["rating"]) == len(frame)
        assert len(attrs["title"]) == len(frame)
        assert diagnostics["num_batches"] == 3
        assert diagnostics["num_denoising_steps"] == 4
        assert diagnostics["model_forward_passes_total"] == 15
        assert calls["count"] == 15


def test_fixed_seed_sampling_is_reproducible_for_tiny_fixture():
    frame, schema, vocabs, tokenizer, config = tiny_fixture()
    torch.manual_seed(0)
    model_a = build_tiny_model(schema, vocabs, tokenizer)
    state = model_a.state_dict()
    model_b = build_tiny_model(schema, vocabs, tokenizer)
    model_b.load_state_dict(state)
    kwargs = dict(
        schema=schema,
        categorical_vocabs=vocabs,
        text_tokenizer=tokenizer,
        spine=frame[["customer_id", "product_id", "review_time"]],
        config=config,
        batch_size=2,
        temperature=1.0,
        top_p=1.0,
        device="cpu",
        seed=99,
        sampling_steps=4,
    )
    attrs_a = sample_attributes(model_a, **kwargs)
    attrs_b = sample_attributes(model_b, **kwargs)
    assert attrs_a["rating"] == attrs_b["rating"]
    assert attrs_a["title"] == attrs_b["title"]


def test_explicit_text_length_targets_force_eos_pad_and_preserve_structured_fields():
    frame, schema, vocabs, tokenizer, _ = tiny_fixture()
    cat_input = torch.tensor([[vocabs["rating"].encode("5"), vocabs["verified"].encode("True")]], dtype=torch.long)
    text_input = {"title": torch.full((1, 8), tokenizer.mask_id, dtype=torch.long)}
    text_input["title"][0, 0] = tokenizer.bos_id
    text_logits = {"title": torch.zeros((1, 8, tokenizer.vocab_size), dtype=torch.float32)}
    debug = enforce_length_constraints(
        schema=schema,
        categorical_vocabs=vocabs,
        text_tokenizer=tokenizer,
        cat_input=cat_input,
        text_input=text_input,
        text_logits=text_logits,
        rng=random.Random(1),
        temperature=1.0,
        top_p=1.0,
        target_text_lengths={"title": torch.tensor([3], dtype=torch.long)},
    )
    ids = text_input["title"][0].tolist()
    assert ids[0] == tokenizer.bos_id
    assert ids[1] not in tokenizer.special_ids
    assert ids[2] not in tokenizer.special_ids
    assert ids[3] not in tokenizer.special_ids
    assert ids[4] == tokenizer.eos_id
    assert all(token_id == tokenizer.pad_id for token_id in ids[5:])
    assert tokenizer.content_length(ids) == 3
    assert debug[0]["title"]["target_length_source"] == "explicit"
    assert int(cat_input[0, 0]) == vocabs["rating"].encode("5")
    assert int(cat_input[0, 1]) == vocabs["verified"].encode("True")


def test_oracle_zero_length_is_not_overridden_by_min_content_tokens():
    _, schema, vocabs, tokenizer, _ = tiny_fixture()
    cat_input = torch.tensor([[vocabs["rating"].encode("5"), vocabs["verified"].encode("True")]], dtype=torch.long)
    text_input = {"title": torch.full((1, 8), tokenizer.mask_id, dtype=torch.long)}
    text_input["title"][0, 0] = tokenizer.bos_id
    text_logits = {"title": torch.zeros((1, 8, tokenizer.vocab_size), dtype=torch.float32)}
    enforce_length_constraints(
        schema=schema,
        categorical_vocabs=vocabs,
        text_tokenizer=tokenizer,
        cat_input=cat_input,
        text_input=text_input,
        text_logits=text_logits,
        rng=random.Random(1),
        temperature=1.0,
        top_p=1.0,
        min_content_tokens={"title": 3},
        target_text_lengths={"title": torch.tensor([0], dtype=torch.long)},
    )
    ids = text_input["title"][0].tolist()
    assert ids[0] == tokenizer.bos_id
    assert ids[1] == tokenizer.eos_id
    assert all(token_id == tokenizer.pad_id for token_id in ids[2:])
    assert tokenizer.content_length(ids) == 0


def test_existing_v4_config_still_loads():
    if not Path("data/original/rel-amazon-toy/review.csv").exists():
        return
    config = load_config("configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml")
    assert "review_text" in config.schema.text_targets
    assert int(config.raw["diffusion"]["timesteps"]) == 50
