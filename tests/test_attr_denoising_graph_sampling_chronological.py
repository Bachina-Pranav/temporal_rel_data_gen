from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.graph_schema import graph_metadata  # noqa: E402
from attr_denoising_test_utils import make_v3_config_and_components  # noqa: E402


def test_v3_sampling_metadata_is_chronological_and_synthetic_only():
    config, _, _, _ = make_v3_config_and_components()
    metadata = graph_metadata(config.raw, real_graph_used_at_sampling=False)
    assert metadata["graph_conditioning_mode"] == "temporal_attribute_denoising"
    assert metadata["history_source_sampling"] == "generated_past_synthetic_attributes"
    assert metadata["sampling_chronological"] is True
    assert metadata["real_graph_used_at_sampling"] is False
    assert metadata["graph_uses_clean_target_attributes"] is False
    assert metadata["graph_uses_clean_future_attributes"] is False
