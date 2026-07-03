"""Graph metadata and history-index helpers for graph-conditioned TABDLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .graph_schema import assert_valid_graph_conditioning, graph_metadata
from .neighbor_sampling import TemporalHistoryIndex
from .schema import ConditionalTABDLMConfig
from .utils import ensure_dir, save_json


def build_temporal_history_index(
    frame: pd.DataFrame,
    config: ConditionalTABDLMConfig,
    *,
    seed: int = 42,
) -> TemporalHistoryIndex:
    assert_valid_graph_conditioning(config.raw)
    id_cfg = config.raw.get("id_encoding", {})
    schema = config.schema
    customer_col = schema.foreign_key_columns[0]
    product_col = schema.foreign_key_columns[1] if len(schema.foreign_key_columns) > 1 else schema.foreign_key_columns[0]
    return TemporalHistoryIndex.from_config(
        frame,
        config.raw,
        customer_col=customer_col,
        product_col=product_col,
        num_hash_buckets=int(id_cfg.get("num_buckets", 262144)),
        seed=seed,
    )


def temporal_graph_metadata(
    frame: pd.DataFrame,
    config: ConditionalTABDLMConfig,
    *,
    source: str,
    seed: int = 42,
    real_graph_used_at_sampling: bool | None = None,
) -> dict[str, Any]:
    index = build_temporal_history_index(frame, config, seed=seed)
    stats = index.diagnostics(sample_size=20000).to_dict()
    metadata = graph_metadata(config.raw, real_graph_used_at_sampling=real_graph_used_at_sampling)
    metadata.update(
        {
            "graph_history_source": source,
            "uses_target_attributes_as_graph_features": False,
            "forbidden_features_checked": True,
            "num_customer_nodes": int(pd.Series(index.customers).nunique()),
            "num_product_nodes": int(pd.Series(index.products).nunique()),
            "num_review_event_nodes": int(index.num_rows),
            "num_edges_by_type": {
                "customer_to_review": int(index.num_rows),
                "review_to_customer": int(index.num_rows),
                "product_to_review": int(index.num_rows),
                "review_to_product": int(index.num_rows),
            },
            **stats,
        }
    )
    return metadata


def write_temporal_graph_metadata(
    frame: pd.DataFrame,
    config: ConditionalTABDLMConfig,
    output_dir: str | Path | None = None,
    *,
    source: str = "real_training_rows",
    seed: int = 42,
    real_graph_used_at_sampling: bool | None = None,
) -> Path:
    graph_dir = ensure_dir(Path(output_dir) if output_dir is not None else config.output_dir / "graph")
    metadata = temporal_graph_metadata(
        frame,
        config,
        source=source,
        seed=seed,
        real_graph_used_at_sampling=real_graph_used_at_sampling,
    )
    path = graph_dir / "graph_metadata.json"
    save_json(metadata, path)
    return path
