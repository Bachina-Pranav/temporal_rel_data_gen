from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.dataset import ConditionalTABDLMDataset, collate_and_mask  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_encoder import TemporalStructureOnlyGraphEncoder  # noqa: E402
from attribute_generation.conditional_tabdlm.model import ConditionalTABDLM, GraphConditionedConditionalTABDLM  # noqa: E402
from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def make_tiny_components():
    frame = pd.DataFrame(
        {
            "customer_id": ["c1", "c1"],
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
        auxiliary_categorical_targets=("summary_length_bucket",),
        text_targets=("summary",),
        text_max_lengths={"summary": 5},
        summary_length_enabled=True,
        summary_length_buckets={"len_1_2": (1, 2), "len_3_5": (3, 5)},
    )
    vocabs = {
        "rating": CategoryVocab.from_values("rating", frame["rating"]),
        "verified": CategoryVocab.from_values("verified", frame["verified"]),
        "summary_length_bucket": CategoryVocab.from_values("summary_length_bucket", ["len_1_2", "len_3_5"]),
    }
    tokenizer = SimpleTextTokenizer().fit(frame["summary"])
    dataset = ConditionalTABDLMDataset(frame, schema, vocabs, tokenizer, num_hash_buckets=64)
    batch = collate_and_mask(
        [dataset[0], dataset[1]],
        schema=schema,
        categorical_vocabs=vocabs,
        text_tokenizer=tokenizer,
        min_mask_prob=1.0,
        max_mask_prob=1.0,
    )
    return frame, schema, vocabs, tokenizer, batch


def test_model_forward_with_graph_context():
    frame, schema, vocabs, tokenizer, batch = make_tiny_components()
    history = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64, max_customer_history=2, max_product_history=2)
    graph_encoder = TemporalStructureOnlyGraphEncoder(
        num_hash_buckets=64,
        entity_embedding_dim=8,
        datetime_embedding_dim=8,
        hidden_dim=16,
        output_dim=12,
        num_layers=1,
    )
    graph_context = graph_encoder(history.build_batch(batch["row_id"], device="cpu"))
    model = GraphConditionedConditionalTABDLM(
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
        graph_context_dim=12,
    )
    output = model(
        foreign_key_ids=batch["foreign_key_ids"],
        datetime_values=batch["datetime_values"],
        categorical_input_ids=batch["categorical_input_ids"],
        text_input_ids=batch["text_input_ids"],
        text_attention=batch["text_attention"],
        diffusion_t=batch["diffusion_t"],
        graph_context=graph_context,
    )
    assert output["categorical"]["rating"].shape == (2, vocabs["rating"].size)
    assert output["categorical"]["verified"].shape == (2, vocabs["verified"].size)
    assert output["categorical"]["summary_length_bucket"].shape == (2, vocabs["summary_length_bucket"].size)
    assert output["text"]["summary"].shape[:2] == (2, 5)


def test_model_forward_without_graph_context_still_works():
    _, schema, vocabs, tokenizer, batch = make_tiny_components()
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
        use_graph_context=False,
    )
    output = model(
        foreign_key_ids=batch["foreign_key_ids"],
        datetime_values=batch["datetime_values"],
        categorical_input_ids=batch["categorical_input_ids"],
        text_input_ids=batch["text_input_ids"],
        text_attention=batch["text_attention"],
        diffusion_t=batch["diffusion_t"],
    )
    assert output["categorical"]["rating"].shape == (2, vocabs["rating"].size)
