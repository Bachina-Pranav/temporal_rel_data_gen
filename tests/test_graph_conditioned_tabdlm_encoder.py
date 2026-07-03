from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.graph_encoder import TemporalStructureOnlyGraphEncoder  # noqa: E402
from attribute_generation.conditional_tabdlm.neighbor_sampling import TemporalHistoryIndex  # noqa: E402


def test_temporal_graph_encoder_outputs_context_without_nans():
    frame = pd.DataFrame(
        {
            "customer_id": ["u1", "u1", "u2"],
            "product_id": ["i1", "i2", "i1"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    history = TemporalHistoryIndex(frame, "customer_id", "product_id", "review_time", 64, max_customer_history=2, max_product_history=2)
    batch = history.build_batch([1, 2], device="cpu")
    encoder = TemporalStructureOnlyGraphEncoder(
        num_hash_buckets=64,
        entity_embedding_dim=8,
        datetime_embedding_dim=8,
        hidden_dim=16,
        output_dim=12,
        num_layers=1,
    )
    output = encoder(batch)
    assert output.shape == (2, 12)
    assert not torch.isnan(output).any()
