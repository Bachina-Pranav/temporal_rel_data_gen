from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_rel_amazon_scaling_patch_does_not_change_model_architecture():
    config = yaml.safe_load(
        (ROOT / "configs/attribute_generation/conditional_tabdlm_rel_amazon_exp5_3_lstm_length_preserving.yaml").read_text()
    )

    assert config["model_family"] == "conditional_tabdlm_lstm_joint_full_text"
    assert config["graph_conditioning_mode"] == "structure_only_temporal"
    assert config["joint_generation"] is True
    assert config["review_text_generated_jointly"] is True
    assert config["review_text_separate_stage"] is False
    assert config["architecture_changed_from_amazon_toy"] is False
    assert config["training"]["epoch_mode"] is False
    assert config["training"]["sampling_mode"] == "temporal_stratified"
