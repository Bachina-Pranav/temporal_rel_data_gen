from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_joint import write_lstm_model_metadata  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import load_json  # noqa: E402


def test_lstm_joint_metadata_says_review_text_is_joint(tmp_path):
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket", "review_text_length_bucket"),
        text_targets=("summary", "review_text"),
        text_max_lengths={"summary": 8, "review_text": 64},
        summary_length_enabled=True,
        summary_length_buckets={"len_1_2": (1, 2)},
        review_text_length_buckets={"q0_q20": (1, 10)},
    )
    config = ConditionalTABDLMConfig(
        raw={
            "experiment_name": "conditional_tabdlm_exp5_lstm_joint_full_review_text",
            "paths": {"output_dir": "unused"},
            "loss_weights": {"rating": 3.0, "verified": 3.0, "summary_length": 2.0, "review_text_length": 2.0, "summary_text": 1.0, "review_text": 1.0},
            "graph_conditioning": {"enabled": True, "mode": "structure_only_temporal"},
            "text_decoder": {"type": "lstm"},
        },
        schema=schema,
    )
    write_lstm_model_metadata(config, tmp_path)
    metadata = load_json(tmp_path / "model_metadata.json")
    assert metadata["joint_generation"] is True
    assert metadata["review_text_generated_jointly"] is True
    assert metadata["review_text_separate_stage"] is False
    assert metadata["uses_diffusion"] is False
    assert metadata["loss_weighting"]["mode"] == "fixed_manual"
    assert metadata["loss_weighting"]["mgda_enabled"] is False
