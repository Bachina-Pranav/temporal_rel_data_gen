from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.attribute_corruption import GraphAttributeStore, build_attribute_graph_batch  # noqa: E402
from attribute_generation.conditional_tabdlm.dataset import ConditionalTABDLMDataset, collate_and_mask  # noqa: E402
from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig  # noqa: E402
from attr_denoising_test_utils import make_v3_config_and_components  # noqa: E402


def test_variant_b_graph_batch_excludes_summary_token_inputs():
    config, frame, vocabs, tokenizer = make_v3_config_and_components()
    config = with_graph_inputs(config, ["rating", "verified", "summary_length_bucket"], include_length=True, include_summary=False)
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

    assert "target_text_ids_summary" not in attr_batch
    assert "customer_history_text_ids_summary" not in attr_batch
    assert "product_history_text_ids_summary" not in attr_batch
    assert attr_batch["target_categorical_ids"].shape[1] == 3


def with_graph_inputs(config: ConditionalTABDLMConfig, inputs: list[str], *, include_length: bool, include_summary: bool) -> ConditionalTABDLMConfig:
    raw = copy.deepcopy(config.raw)
    raw["attribute_denoising"]["review_event_attribute_inputs"] = list(inputs)
    raw["attribute_denoising"]["include_summary_length_in_graph"] = bool(include_length)
    raw["attribute_denoising"]["include_summary_tokens_in_graph"] = bool(include_summary)
    return ConditionalTABDLMConfig(raw=raw, schema=config.schema)
