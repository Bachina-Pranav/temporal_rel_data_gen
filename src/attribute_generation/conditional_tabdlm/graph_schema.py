"""Graph-conditioning config helpers for Conditional TABDLM."""

from __future__ import annotations

from typing import Any


FORBIDDEN_GRAPH_FEATURES = {
    "rating",
    "verified",
    "summary",
    "review_text",
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
    default_type = "temporal_attr_denoising_ego_encoder" if graph_mode(raw_config) == "temporal_attribute_denoising" else "temporal_structure_ego_encoder"
    encoder.setdefault("type", default_type)
    encoder.setdefault("hidden_dim", 256)
    encoder.setdefault("output_dim", raw_config.get("model", {}).get("graph_context_dim", 256))
    encoder.setdefault("num_layers", 2)
    encoder.setdefault("dropout", 0.1)
    encoder.setdefault("normalize_output", True)
    return encoder


def graph_mode(raw_config: dict[str, Any]) -> str:
    return str(graph_config(raw_config).get("mode", "structure_only_temporal"))


def attribute_denoising_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(raw_config.get("attribute_denoising", {}) or {})
    cfg.setdefault("enabled", graph_mode(raw_config) == "temporal_attribute_denoising")
    cfg.setdefault("input_mode_training", "noised_attributes")
    cfg.setdefault("input_mode_sampling", "generated_past_attributes_plus_current_noised_state")
    cfg.setdefault("corrupt_target_event_attributes", True)
    cfg.setdefault("corrupt_history_event_attributes", True)
    history = dict(cfg.get("history_attribute_corruption", {}) or {})
    history.setdefault("enabled", True)
    history.setdefault("mode", "mask_dropout")
    history.setdefault("mask_prob", 0.15)
    cfg["history_attribute_corruption"] = history
    sampling = dict(cfg.get("sampling_history_attribute_corruption", {}) or {})
    sampling.setdefault("enabled", True)
    sampling.setdefault("mode", "mask_dropout")
    sampling.setdefault("mask_prob", 0.0)
    cfg["sampling_history_attribute_corruption"] = sampling
    aux = dict(cfg.get("auxiliary_neighbor_denoising_loss", {}) or {})
    aux.setdefault("enabled", True)
    aux.setdefault("weight", 0.25)
    aux.setdefault("max_neighbor_nodes_for_loss", 256)
    cfg["auxiliary_neighbor_denoising_loss"] = aux
    embedding = dict(cfg.get("attribute_embedding", {}) or {})
    embedding.setdefault("rating_dim", 64)
    embedding.setdefault("verified_dim", 32)
    embedding.setdefault("summary_length_dim", 32)
    embedding.setdefault("summary_dim", 128)
    embedding.setdefault("combined_attr_dim", graph_encoder_config(raw_config).get("hidden_dim", 256))
    embedding.setdefault("dropout", 0.1)
    cfg["attribute_embedding"] = embedding
    return cfg


def assert_valid_graph_conditioning(raw_config: dict[str, Any]) -> None:
    if not graph_conditioning_enabled(raw_config):
        return
    cfg = graph_config(raw_config)
    mode = graph_mode(raw_config)
    if mode not in {"structure_only_temporal", "temporal_attribute_denoising"}:
        raise ValueError(f"Unsupported graph conditioning mode: {mode!r}")
    temporal = temporal_filter_config(raw_config)
    if not bool(temporal.get("enabled", True)):
        raise ValueError("Graph conditioning requires temporal_filter.enabled=true")
    if str(temporal.get("mode")) != "past_only":
        raise ValueError("Graph conditioning requires temporal_filter.mode=past_only")
    forbidden = set(str(value) for value in cfg.get("forbidden_node_features", []))
    forbidden |= set(str(value) for value in cfg.get("forbidden_edge_features", []))
    if mode == "structure_only_temporal" and not FORBIDDEN_GRAPH_FEATURES.issubset(forbidden):
        missing = sorted(FORBIDDEN_GRAPH_FEATURES.difference(forbidden))
        raise ValueError(f"Graph config must explicitly forbid target/text features: {missing}")
    for section in ("allowed_node_features", "allowed_edge_features"):
        values = cfg.get(section, {})
        if isinstance(values, dict):
            flat = {str(item) for feature_list in values.values() for item in feature_list}
        else:
            flat = {str(item) for item in values}
        leaked = sorted(FORBIDDEN_GRAPH_FEATURES.intersection(flat))
        if leaked and mode == "structure_only_temporal":
            raise ValueError(f"Forbidden target/text graph features in {section}: {leaked}")
    leakage = dict(cfg.get("leakage_policy", {}) or {})
    unsafe_flags = {"graph_uses_future_events": bool(cfg.get("graph_uses_future_events", leakage.get("graph_uses_future_events", False)))}
    if mode == "structure_only_temporal":
        unsafe_flags["graph_uses_target_attributes"] = bool(cfg.get("graph_uses_target_attributes", False))
    else:
        unsafe_flags["graph_uses_clean_target_attributes"] = bool(leakage.get("graph_uses_clean_target_attributes", False))
        unsafe_flags["graph_uses_clean_future_attributes"] = bool(leakage.get("graph_uses_clean_future_attributes", False))
    unsafe = [key for key, value in unsafe_flags.items() if value]
    if unsafe:
        raise ValueError(f"Unsafe graph conditioning flags must be false: {unsafe}")


def graph_metadata(raw_config: dict[str, Any], *, real_graph_used_at_sampling: bool | None = None) -> dict[str, Any]:
    mode = graph_mode(raw_config)
    leakage = dict(graph_config(raw_config).get("leakage_policy", {}) or {})
    metadata = {
        "graph_conditioning_mode": mode,
        "temporal_filter_enabled": True,
        "temporal_filter_mode": "past_only",
        "graph_uses_future_events": False,
    }
    if mode == "temporal_attribute_denoising":
        metadata.update(
            {
                "graph_uses_target_attributes": True,
                "graph_attribute_input_mode": "noised_or_generated_past",
                "graph_uses_clean_target_attributes": False,
                "graph_uses_clean_future_attributes": False,
                "synthetic_graph_history_source": leakage.get("synthetic_graph_history_source", "synthetic_spine"),
                "history_source_sampling": "generated_past_synthetic_attributes",
                "sampling_chronological": True,
                "target_attributes_visible_to_gnn_training": "noised_only",
                "history_attributes_visible_to_gnn_training": "corrupted_past_only",
                "target_attributes_visible_to_gnn_sampling": "noised_current_state_only",
                "history_attributes_visible_to_gnn_sampling": "generated_past_only",
            }
        )
    else:
        metadata["graph_uses_target_attributes"] = False
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
