from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.attribute_corruption import GraphAttributeStore, build_attribute_graph_batch  # noqa: E402
from attribute_generation.conditional_tabdlm.dataset import ConditionalTABDLMDataset, collate_and_mask  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_encoder import TemporalAttributeDenoisingGraphEncoder  # noqa: E402
from attribute_generation.conditional_tabdlm.model import GraphConditionedConditionalTABDLM  # noqa: E402
from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402
sys.path.insert(0, str(ROOT / "tests"))

from attr_denoising_test_utils import make_v3_config_and_components  # noqa: E402


def test_attr_denoising_graph_forward_and_model_forward():
    config, frame, vocabs, tokenizer = make_v3_config_and_components()
    dataset = ConditionalTABDLMDataset(frame, config.schema, vocabs, tokenizer, num_hash_buckets=64)
    batch = collate_and_mask(
        [dataset[1], dataset[2]],
        schema=config.schema,
        categorical_vocabs=vocabs,
        text_tokenizer=tokenizer,
        min_mask_prob=1.0,
        max_mask_prob=1.0,
    )
    history = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64, max_customer_history=2, max_product_history=2)
    graph_batch = history.build_batch(batch["row_id"], device="cpu")
    store = GraphAttributeStore.from_frame(frame, config, vocabs, tokenizer)
    attr_batch, _ = build_attribute_graph_batch(graph_batch, batch, store, config, device="cpu", training=True)
    graph_batch.update(attr_batch)
    encoder = TemporalAttributeDenoisingGraphEncoder.from_config(config.raw, config.schema, vocabs, tokenizer)
    context = encoder(graph_batch)
    assert context.shape == (2, 12)
    assert not torch.isnan(context).any()

    model = GraphConditionedConditionalTABDLM(
        config.schema,
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
        graph_context=context,
    )
    assert output["categorical"]["rating"].shape == (2, vocabs["rating"].size)
    assert output["categorical"]["verified"].shape == (2, vocabs["verified"].size)
    assert output["categorical"]["summary_length_bucket"].shape == (2, vocabs["summary_length_bucket"].size)
    assert output["text"]["summary"].shape[:2] == (2, 6)
