"""Graph-conditioning config helpers for Conditional TABDLM."""

from __future__ import annotations

from typing import Any


FORBIDDEN_GRAPH_FEATURES = {
    "rating",
    "verified",
    "summary",
    "review_text",
}

GRAPH_METADATA_FLAGS = {
    "graph_conditioning_mode": "structure_only_temporal",
    "temporal_filter_enabled": True,
    "temporal_filter_mode": "past_only",
    "graph_uses_future_events": False,
    "graph_uses_target_attributes": False,
}


def graph_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    return dict(raw_config.get("graph_conditioning", {}) or {})


def graph_conditioning_enabled(raw_config: dict[str, Any]) -> bool:
    cfg = graph_config(raw_config)
    return bool(cfg.get("enabled", False))


def temporal_filter_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    cfg = graph_config(raw_config)
    temporal = dict(cfg.get("temporal_filter", {}) or {})
    temporal.setdefault("enabled", True)
    temporal.setdefault("mode", "past_only")
    temporal.setdefault("timestamp_column", "review_time")
    temporal.setdefault("allow_same_timestamp_events", False)
    temporal.setdefault("same_timestamp_tiebreak", "row_index")
    temporal.setdefault("exclude_target_event_from_neighbors", True)
    temporal.setdefault("max_history_events_per_customer", 50)
    temporal.setdefault("max_history_events_per_product", 100)
    temporal.setdefault("history_sampling_strategy", "recent")
    temporal.setdefault("recent_fraction_if_mixed", 0.7)
    temporal.setdefault("deterministic_eval", True)
    return temporal


def graph_encoder_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    cfg = graph_config(raw_config)
    encoder = dict(cfg.get("graph_encoder", {}) or {})
    encoder.setdefault("type", "temporal_structure_ego_encoder")
    encoder.setdefault("hidden_dim", 256)
    encoder.setdefault("output_dim", raw_config.get("model", {}).get("graph_context_dim", 256))
    encoder.setdefault("num_layers", 2)
    encoder.setdefault("dropout", 0.1)
    encoder.setdefault("normalize_output", True)
    return encoder


def assert_valid_graph_conditioning(raw_config: dict[str, Any]) -> None:
    if not graph_conditioning_enabled(raw_config):
        return
    cfg = graph_config(raw_config)
    mode = str(cfg.get("mode", "structure_only_temporal"))
    if mode != "structure_only_temporal":
        raise ValueError(f"Graph conditioning mode must be structure_only_temporal, got {mode!r}")
    temporal = temporal_filter_config(raw_config)
    if not bool(temporal.get("enabled", True)):
        raise ValueError("Graph conditioning requires temporal_filter.enabled=true")
    if str(temporal.get("mode")) != "past_only":
        raise ValueError("Graph conditioning requires temporal_filter.mode=past_only")
    forbidden = set(str(value) for value in cfg.get("forbidden_node_features", []))
    forbidden |= set(str(value) for value in cfg.get("forbidden_edge_features", []))
    if not FORBIDDEN_GRAPH_FEATURES.issubset(forbidden):
        missing = sorted(FORBIDDEN_GRAPH_FEATURES.difference(forbidden))
        raise ValueError(f"Graph config must explicitly forbid target/text features: {missing}")
    for section in ("allowed_node_features", "allowed_edge_features"):
        values = cfg.get(section, {})
        if isinstance(values, dict):
            flat = {str(item) for feature_list in values.values() for item in feature_list}
        else:
            flat = {str(item) for item in values}
        leaked = sorted(FORBIDDEN_GRAPH_FEATURES.intersection(flat))
        if leaked:
            raise ValueError(f"Forbidden target/text graph features in {section}: {leaked}")
    unsafe_flags = {
        "graph_uses_future_events": bool(cfg.get("graph_uses_future_events", False)),
        "graph_uses_target_attributes": bool(cfg.get("graph_uses_target_attributes", False)),
    }
    unsafe = [key for key, value in unsafe_flags.items() if value]
    if unsafe:
        raise ValueError(f"Unsafe graph conditioning flags must be false: {unsafe}")


def graph_metadata(raw_config: dict[str, Any], *, real_graph_used_at_sampling: bool | None = None) -> dict[str, Any]:
    metadata = dict(GRAPH_METADATA_FLAGS)
    metadata["uses_graph_context"] = graph_conditioning_enabled(raw_config)
    metadata["graph_node_types"] = list(graph_config(raw_config).get("node_types", ["customer", "product", "review_event"]))
    metadata["graph_edge_types"] = [
        "customer_to_review",
        "review_to_customer",
        "product_to_review",
        "review_to_product",
        "customer_history_to_target",
        "product_history_to_target",
    ]
    metadata["graph_encoder_type"] = graph_encoder_config(raw_config).get("type")
    metadata["graph_num_layers"] = int(graph_encoder_config(raw_config).get("num_layers", 2))
    metadata["graph_hidden_dim"] = int(graph_encoder_config(raw_config).get("hidden_dim", 256))
    metadata["graph_output_dim"] = int(graph_encoder_config(raw_config).get("output_dim", 256))
    if real_graph_used_at_sampling is not None:
        metadata["real_graph_used_at_sampling"] = bool(real_graph_used_at_sampling)
    return metadata
