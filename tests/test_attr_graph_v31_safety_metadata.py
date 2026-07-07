from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.graph_schema import graph_metadata  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


def test_v31_configs_preserve_temporal_safety_metadata():
    paths = [
        ROOT / "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp3_1a_attr_graph_rating_verified.yaml",
        ROOT / "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp3_1b_attr_graph_rating_verified_length.yaml",
        ROOT / "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp3_1c_attr_graph_rating_verified_length_summary_gated.yaml",
    ]
    for path in paths:
        config = load_config(path)
        metadata = graph_metadata(config.raw, real_graph_used_at_sampling=False)

        assert metadata["graph_conditioning_mode"] == "temporal_attribute_denoising"
        assert metadata["graph_attribute_input_mode"] == "noised_or_generated_past"
        assert metadata["temporal_filter_enabled"] is True
        assert metadata["temporal_filter_mode"] == "past_only"
        assert metadata["graph_uses_future_events"] is False
        assert metadata["graph_uses_clean_target_attributes"] is False
        assert metadata["graph_uses_clean_future_attributes"] is False
        assert metadata["real_graph_used_at_sampling"] is False
        assert metadata["history_source_sampling"] == "generated_past_synthetic_attributes"
        assert metadata["sampling_chronological"] is True
