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


def test_temporal_graph_metadata_counts_nodes_and_edges():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2"],
            "product_id": ["i1", "i2", "i1"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "rating": ["5", "4", "1"],
            "verified": ["True", "True", "False"],
            "summary": ["good", "nice", "bad"],
        }
    )
    metadata = temporal_graph_metadata(frame, make_config(), source="test")
    assert metadata["num_customer_nodes"] == 2
    assert metadata["num_product_nodes"] == 2
    assert metadata["num_review_event_nodes"] == 3
    assert metadata["num_edges_by_type"]["customer_to_review"] == 3
    assert metadata["num_edges_by_type"]["product_to_review"] == 3
    assert metadata["uses_target_attributes_as_graph_features"] is False
    assert metadata["graph_uses_target_attributes"] is False
