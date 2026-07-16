from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.hierarchical_sample import (  # noqa: E402
    build_graph_context,
    hierarchical_sample_attributes,
    initial_length_masked_text_inputs,
)
from attribute_generation.conditional_tabdlm.hierarchical_schema import generation_plan_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.hierarchical_train import choose_text_conditioning_mode  # noqa: E402
from attribute_generation.conditional_tabdlm.model import ConditionalTABDLM  # noqa: E402
from attribute_generation.conditional_tabdlm.sample import encode_conditions  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def fixture():
    frame = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3"],
            "product_id": ["p1", "p2", "p3"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "rating": ["5", "1", "4"],
            "verified": ["True", "False", "True"],
            "summary": ["great product", "bad fit", "works fine"],
            "review_text": ["great product for daily use", "bad fit and rough", "works fine for me"],
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket", "review_text_length_bucket"),
        text_targets=("summary", "review_text"),
        text_max_lengths={"summary": 8, "review_text": 12},
        summary_length_enabled=True,
        use_length_bucket_in_sampling=True,
        force_pad_after_eos=True,
        summary_length_buckets={"short": (0, 3), "long": (4, 6)},
        review_text_length_buckets={"short": (0, 4), "long": (5, 10)},
    )
    raw = {
        "paths": {"train_data_path": "unused.csv", "synthetic_spine_path": "unused.csv", "output_dir": "unused"},
        "columns": {
            "condition": {"foreign_keys": ["customer_id", "product_id"], "datetimes": ["review_time"]},
            "target": {"categorical": ["rating", "verified"], "numerical": [], "text": ["summary", "review_text"]},
        },
        "auxiliary_targets": {"categorical": ["summary_length_bucket", "review_text_length_bucket"]},
        "schema": {
            "fields": {
                "summary": {"type": "text", "generation_role": "text", "length_field": "summary_length_bucket"},
                "review_text": {"type": "text", "generation_role": "text", "length_field": "review_text_length_bucket"},
            }
        },
        "generation": {
            "stages": [
                {"name": "structured", "fields": ["rating", "verified", "summary_length_bucket", "review_text_length_bucket"], "condition_on": ["event_context"]},
                {"name": "text", "fields": ["summary", "review_text"], "condition_on": ["structured", "event_context"]},
            ]
        },
        "text": {"max_length": {"summary": 8, "review_text": 12}},
        "id_encoding": {"num_buckets": 64, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
        "model": {"hidden_dim": 32, "num_layers": 1, "num_heads": 4, "condition_dim": 16},
        "diffusion": {"timesteps": 3},
    }
    vocabs = {
        "rating": CategoryVocab.from_values("rating", frame["rating"]),
        "verified": CategoryVocab.from_values("verified", frame["verified"]),
        "summary_length_bucket": CategoryVocab.from_values("summary_length_bucket", ["short", "long"]),
        "review_text_length_bucket": CategoryVocab.from_values("review_text_length_bucket", ["short", "long"]),
    }
    tokenizer = SimpleTextTokenizer().fit(frame["summary"].tolist() + frame["review_text"].tolist())
    config = ConditionalTABDLMConfig(raw=raw, schema=schema)
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
    return frame, schema, raw, config, vocabs, tokenizer, model


def test_generation_plan_validates_stages_and_dependencies():
    _, schema, raw, *_ = fixture()
    plan = generation_plan_from_config(raw, schema)
    assert plan.stage_names() == ("structured", "text")
    bad = dict(raw)
    bad["generation"] = {"stages": [{"name": "text", "fields": ["summary"], "condition_on": ["structured"]}]}
    with pytest.raises(ValueError):
        generation_plan_from_config(bad, schema)


def test_length_masked_text_inputs_place_eos_and_pad_without_spillover():
    _, schema, _, _, _, tokenizer, _ = fixture()
    lengths = {"summary": torch.tensor([2]), "review_text": torch.tensor([4])}
    text_input, attention, remaining = initial_length_masked_text_inputs(schema, tokenizer, lengths, "cpu", 1)
    summary = text_input["summary"][0].tolist()
    review = text_input["review_text"][0].tolist()
    assert summary[0] == tokenizer.bos_id
    assert summary[1:3] == [tokenizer.mask_id, tokenizer.mask_id]
    assert summary[3] == tokenizer.eos_id
    assert all(token == tokenizer.pad_id for token in summary[4:])
    assert review[5] == tokenizer.eos_id
    assert int(remaining["summary"].sum()) == 2
    assert int(remaining["review_text"].sum()) == 4
    assert attention["summary"].shape[1] == schema.text_max_lengths["summary"]
    assert attention["review_text"].shape[1] == schema.text_max_lengths["review_text"]


def test_hierarchical_sampling_generates_requested_rows_and_valid_domains():
    frame, schema, raw, config, vocabs, tokenizer, model = fixture()
    plan = generation_plan_from_config(raw, schema)
    attrs = hierarchical_sample_attributes(
        model=model,
        config=config,
        plan=plan,
        categorical_vocabs=vocabs,
        tokenizer=tokenizer,
        spine=frame[["customer_id", "product_id", "review_time"]],
        batch_size=2,
        temperature=1.0,
        top_p=1.0,
        text_top_k=4,
        device="cpu",
        seed=7,
        structured_steps=2,
        text_steps=2,
        timestep_spacing="uniform",
        inference_dtype="float32",
        graph_encoder=None,
        graph_history_index=None,
        graph_mode_override="no_graph",
    )
    assert len(attrs["rating"]) == len(frame)
    assert len(attrs["summary"]) == len(frame)
    assert set(attrs["rating"]).issubset({1, 4, 5})
    assert attrs["_sampling_diagnostics"]["uses_generated_structured_attributes_for_text"] is True


def test_text_logits_change_when_structured_conditioning_changes():
    frame, schema, _, _, vocabs, tokenizer, model = fixture()
    foreign_key_ids, datetime_values = encode_conditions(frame.head(1), schema, 64, "cpu")
    text_input, text_attention, _ = initial_length_masked_text_inputs(
        schema,
        tokenizer,
        {"summary": torch.tensor([2]), "review_text": torch.tensor([3])},
        "cpu",
        1,
    )
    cat_a = torch.tensor([[vocabs["rating"].encode("5"), vocabs["verified"].encode("True"), vocabs["summary_length_bucket"].encode("short"), vocabs["review_text_length_bucket"].encode("short")]])
    cat_b = torch.tensor([[vocabs["rating"].encode("1"), vocabs["verified"].encode("False"), vocabs["summary_length_bucket"].encode("short"), vocabs["review_text_length_bucket"].encode("short")]])
    logits_a = model(foreign_key_ids, datetime_values, cat_a, text_input, text_attention, torch.ones(1))
    logits_b = model(foreign_key_ids, datetime_values, cat_b, text_input, text_attention, torch.ones(1))
    diff = (logits_a["text"]["summary"] - logits_b["text"]["summary"]).abs().mean()
    assert float(diff) > 1e-6


def test_text_logits_change_when_graph_context_changes_with_gated_fusion():
    frame, schema, _, _, vocabs, tokenizer, _ = fixture()
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
        use_graph_context=True,
        graph_context_dim=6,
        graph_fusion_method="gated_residual",
    )
    foreign_key_ids, datetime_values = encode_conditions(frame.head(1), schema, 64, "cpu")
    text_input, text_attention, _ = initial_length_masked_text_inputs(
        schema,
        tokenizer,
        {"summary": torch.tensor([2]), "review_text": torch.tensor([3])},
        "cpu",
        1,
    )
    cat = torch.tensor([[0, 0, 0, 0]])
    zero = torch.zeros((1, 6))
    one = torch.ones((1, 6))
    logits_zero = model(foreign_key_ids, datetime_values, cat, text_input, text_attention, torch.ones(1), zero)
    logits_one = model(foreign_key_ids, datetime_values, cat, text_input, text_attention, torch.ones(1), one)
    diff = (logits_zero["text"]["review_text"] - logits_one["text"]["review_text"]).abs().mean()
    assert float(diff) > 1e-6


def test_validation_defaults_to_generated_structured_conditioning():
    assert choose_text_conditioning_mode({"mode": "mixed"}, training=False) == "generated"
    assert choose_text_conditioning_mode({"mode": "mixed", "validation_mode": "clean"}, training=False) == "clean"


class DummyGraphEncoder(torch.nn.Module):
    def forward(self, graph_batch):
        return graph_batch["values"].float()


class DummyGraphHistory:
    def build_batch(self, row_indices, *, device, deterministic=True):
        values = torch.tensor([[idx, idx + 1] for idx in row_indices], dtype=torch.float32, device=device)
        return {"values": values}


def test_graph_context_ablation_modes_work():
    encoder = DummyGraphEncoder()
    history = DummyGraphHistory()
    correct = build_graph_context(encoder, history, row_indices=[0, 1, 2], device="cpu", mode="correct")
    zero = build_graph_context(encoder, history, row_indices=[0, 1, 2], device="cpu", mode="zero")
    shuffled = build_graph_context(encoder, history, row_indices=[0, 1, 2], device="cpu", mode="shuffled")
    no_graph = build_graph_context(encoder, history, row_indices=[0, 1, 2], device="cpu", mode="no_graph")

    assert torch.equal(correct, torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]))
    assert torch.equal(zero, torch.zeros_like(correct))
    assert sorted(map(tuple, shuffled.tolist())) == sorted(map(tuple, correct.tolist()))
    assert no_graph is None
