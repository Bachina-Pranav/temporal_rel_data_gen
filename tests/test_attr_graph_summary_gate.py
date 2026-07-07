from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.graph_encoder import TemporalAttributeDenoisingGraphEncoder  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig  # noqa: E402
from attr_denoising_test_utils import make_v3_config_and_components  # noqa: E402


def test_summary_attr_gate_initializes_to_configured_value():
    config, _, vocabs, tokenizer = make_v3_config_and_components()
    config = with_summary_gate(config, gate_init=0.05)

    encoder = TemporalAttributeDenoisingGraphEncoder.from_config(config.raw, config.schema, vocabs, tokenizer)

    assert encoder.attr_state_encoder.text_columns == ["summary"]
    assert encoder.summary_attr_gate_value() == pytest.approx(0.05, abs=1e-6)
    assert encoder.summary_attr_gate_regularization_loss() is not None


def with_summary_gate(config: ConditionalTABDLMConfig, *, gate_init: float) -> ConditionalTABDLMConfig:
    raw = copy.deepcopy(config.raw)
    raw["attribute_denoising"]["review_event_attribute_inputs"] = ["rating", "verified", "summary_length_bucket", "summary"]
    raw["attribute_denoising"]["include_summary_length_in_graph"] = True
    raw["attribute_denoising"]["include_summary_tokens_in_graph"] = True
    raw["attribute_denoising"]["summary_token_graph_dropout"] = 0.5
    raw["attribute_denoising"]["learnable_summary_attr_gate"] = True
    raw["attribute_denoising"]["summary_attr_gate_init"] = float(gate_init)
    raw["attribute_denoising"]["summary_attr_gate_regularization"] = 0.001
    return ConditionalTABDLMConfig(raw=raw, schema=config.schema)
