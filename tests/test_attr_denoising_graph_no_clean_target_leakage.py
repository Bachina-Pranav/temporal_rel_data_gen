from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.attribute_corruption import GraphAttributeStore, build_attribute_graph_batch  # noqa: E402
from attribute_generation.conditional_tabdlm.dataset import ConditionalTABDLMDataset, collate_and_mask  # noqa: E402
from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402
from attr_denoising_test_utils import make_v3_config_and_components  # noqa: E402


def test_target_event_gnn_input_uses_noised_target_state_not_clean_targets():
    config, frame, vocabs, tokenizer = make_v3_config_and_components()
    dataset = ConditionalTABDLMDataset(frame, config.schema, vocabs, tokenizer, num_hash_buckets=64)
    batch = collate_and_mask(
        [dataset[1]],
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
    assert torch.equal(attr_batch["target_categorical_ids"], batch["categorical_input_ids"])
    assert not torch.equal(attr_batch["target_categorical_ids"], batch["categorical_clean_ids"])
    assert "customer_history_clean_categorical_ids" in attr_batch
    assert "target_clean_categorical_ids" not in attr_batch
