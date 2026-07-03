from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.graph_dataset import temporal_graph_metadata  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402


def make_config() -> ConditionalTABDLMConfig:
    raw = {
        "paths": {
            "train_data_path": "unused.csv",
            "synthetic_spine_path": "unused.csv",
            "output_dir": "unused",
        },
        "columns": {
            "condition": {"foreign_keys": ["customer_id", "product_id"], "datetimes": ["review_time"]},
            "target": {"categorical": ["rating", "verified"], "numerical": [], "text": ["summary"]},
        },
        "text": {"max_length": {"summary": 8}},
        "id_encoding": {"num_buckets": 64, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
        "model": {"hidden_dim": 32, "num_layers": 1, "num_heads": 4, "condition_dim": 16, "use_graph_context": True},
        "graph_conditioning": {
            "enabled": True,
            "mode": "structure_only_temporal",
            "temporal_filter": {"enabled": True, "mode": "past_only", "timestamp_column": "review_time"},
            "forbidden_node_features": ["rating", "verified", "summary", "review_text"],
            "graph_uses_future_events": False,
            "graph_uses_target_attributes": False,
        },
    }
    return ConditionalTABDLMConfig(raw=raw, schema=ConditionalTABDLMSchema.from_config_dict(raw))


def test_sampling_graph_metadata_uses_synthetic_spine_only():
    synthetic_spine = pd.DataFrame(
        {
            "customer_id": ["synth_u1", "synth_u1"],
            "product_id": ["synth_i1", "synth_i2"],
            "review_time": ["2021-01-01", "2021-01-02"],
        }
    )
    metadata = temporal_graph_metadata(
        synthetic_spine,
        make_config(),
        source="synthetic_spine",
        real_graph_used_at_sampling=False,
    )
    assert metadata["graph_history_source"] == "synthetic_spine"
    assert metadata["real_graph_used_at_sampling"] is False
    assert metadata["graph_uses_target_attributes"] is False
    assert metadata["graph_uses_future_events"] is False
    assert metadata["num_review_event_nodes"] == 2
