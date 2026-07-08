from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_joint import write_lstm_model_metadata  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import load_json  # noqa: E402


def test_lstm_graph_safety_metadata(tmp_path):
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
            "paths": {"output_dir": str(tmp_path)},
            "graph_conditioning": {"enabled": True, "mode": "structure_only_temporal"},
            "loss_weights": {},
            "text_decoder": {"type": "lstm"},
        },
        schema=schema,
    )
    write_lstm_model_metadata(config, tmp_path)
    metadata = load_json(tmp_path / "model_metadata.json")
    assert metadata["graph_conditioning_mode"] == "structure_only_temporal"
    assert metadata["temporal_filter_mode"] == "past_only"
    assert metadata["graph_uses_future_events"] is False
    assert metadata["graph_uses_target_attributes"] is False
    assert metadata["real_graph_used_at_sampling"] is False

