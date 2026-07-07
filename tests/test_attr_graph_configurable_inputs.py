from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.graph_encoder import TemporalAttributeDenoisingGraphEncoder  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_schema import graph_attribute_inputs  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, load_config  # noqa: E402
from attr_denoising_test_utils import make_v3_config_and_components  # noqa: E402


CONFIGS = {
    "a": "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp3_1a_attr_graph_rating_verified.yaml",
    "b": "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp3_1b_attr_graph_rating_verified_length.yaml",
    "c": "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp3_1c_attr_graph_rating_verified_length_summary_gated.yaml",
}


def test_v31_config_files_resolve_expected_graph_inputs():
    config_a = load_config(ROOT / CONFIGS["a"])
    config_b = load_config(ROOT / CONFIGS["b"])
    config_c = load_config(ROOT / CONFIGS["c"])

    assert graph_attribute_inputs(config_a.raw, config_a.schema)["graph_attr_inputs"] == ["rating", "verified"]
    assert graph_attribute_inputs(config_b.raw, config_b.schema)["graph_attr_inputs"] == ["rating", "verified", "summary_length_bucket"]
    assert graph_attribute_inputs(config_c.raw, config_c.schema)["graph_attr_inputs"] == ["rating", "verified", "summary_length_bucket", "summary"]


def test_temporal_attr_encoder_uses_configurable_input_sets():
    base_config, _, vocabs, tokenizer = make_v3_config_and_components()
    config_a = with_graph_inputs(base_config, ["rating", "verified"], include_length=False, include_summary=False)
    config_b = with_graph_inputs(base_config, ["rating", "verified", "summary_length_bucket"], include_length=True, include_summary=False)
    config_c = with_graph_inputs(
        base_config,
        ["rating", "verified", "summary_length_bucket", "summary"],
        include_length=True,
        include_summary=True,
        gate_init=0.05,
    )

    encoder_a = TemporalAttributeDenoisingGraphEncoder.from_config(config_a.raw, config_a.schema, vocabs, tokenizer)
    encoder_b = TemporalAttributeDenoisingGraphEncoder.from_config(config_b.raw, config_b.schema, vocabs, tokenizer)
    encoder_c = TemporalAttributeDenoisingGraphEncoder.from_config(config_c.raw, config_c.schema, vocabs, tokenizer)

    assert encoder_a.graph_attr_inputs == ["rating", "verified"]
    assert encoder_a.attr_state_encoder.text_columns == []
    assert "summary_length_bucket" not in encoder_a.aux_categorical_heads

    assert encoder_b.graph_attr_inputs == ["rating", "verified", "summary_length_bucket"]
    assert encoder_b.attr_state_encoder.text_columns == []
    assert "summary_length_bucket" in encoder_b.aux_categorical_heads

    assert encoder_c.graph_attr_inputs == ["rating", "verified", "summary_length_bucket", "summary"]
    assert encoder_c.attr_state_encoder.text_columns == ["summary"]
    assert "summary" in encoder_c.aux_text_heads


def with_graph_inputs(
    config: ConditionalTABDLMConfig,
    inputs: list[str],
    *,
    include_length: bool,
    include_summary: bool,
    gate_init: float = 1.0,
) -> ConditionalTABDLMConfig:
    raw = copy.deepcopy(config.raw)
    raw["attribute_denoising"]["review_event_attribute_inputs"] = list(inputs)
    raw["attribute_denoising"]["include_summary_length_in_graph"] = bool(include_length)
    raw["attribute_denoising"]["include_summary_tokens_in_graph"] = bool(include_summary)
    raw["attribute_denoising"]["summary_attr_gate_init"] = float(gate_init)
    raw["attribute_denoising"]["learnable_summary_attr_gate"] = bool(include_summary)
    return ConditionalTABDLMConfig(raw=raw, schema=config.schema)
