from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.attribute_corruption import GraphAttributeStore, build_attribute_graph_batch  # noqa: E402
from attribute_generation.conditional_tabdlm.dataset import ConditionalTABDLMDataset, collate_and_mask  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_encoder import TemporalAttributeDenoisingGraphEncoder  # noqa: E402
from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig  # noqa: E402
from attr_denoising_test_utils import make_v3_config_and_components  # noqa: E402


def test_auxiliary_neighbor_loss_components_follow_graph_inputs():
    config, frame, vocabs, tokenizer = make_v3_config_and_components()

    assert aux_components(with_inputs(config, ["rating", "verified"], include_length=False, include_summary=False), frame, vocabs, tokenizer) == {
        "rating",
        "verified",
    }
    assert aux_components(
        with_inputs(config, ["rating", "verified", "summary_length_bucket"], include_length=True, include_summary=False),
        frame,
        vocabs,
        tokenizer,
    ) == {"rating", "verified", "summary_length"}
    assert aux_components(
        with_inputs(config, ["rating", "verified", "summary_length_bucket", "summary"], include_length=True, include_summary=True),
        frame,
        vocabs,
        tokenizer,
    ) == {"rating", "verified", "summary_length", "summary"}


def aux_components(config: ConditionalTABDLMConfig, frame, vocabs, tokenizer) -> set[str]:
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
    loss, component = encoder.auxiliary_neighbor_loss(graph_batch, max_nodes=8)
    assert torch.isfinite(loss)
    return set(component["components"])


def with_inputs(config: ConditionalTABDLMConfig, inputs: list[str], *, include_length: bool, include_summary: bool) -> ConditionalTABDLMConfig:
    raw = copy.deepcopy(config.raw)
    raw["attribute_denoising"]["review_event_attribute_inputs"] = list(inputs)
    raw["attribute_denoising"]["include_summary_length_in_graph"] = bool(include_length)
    raw["attribute_denoising"]["include_summary_tokens_in_graph"] = bool(include_summary)
    return ConditionalTABDLMConfig(raw=raw, schema=config.schema)
