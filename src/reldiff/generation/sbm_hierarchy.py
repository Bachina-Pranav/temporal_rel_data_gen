"""Diagnostics for graph-tool nested SBM hierarchy extraction."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .block_diagnostics import block_pair_counts, block_size_summary


CURRENT_GENERATOR_BLOCK_LEVEL = 0


def preprocess_reviews_for_sbm(
    reviews: pd.DataFrame,
    customer_col: str,
    product_col: str,
    timestamp_col: Optional[str] = None,
    customers: Optional[pd.DataFrame] = None,
    products: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    required = [customer_col, product_col]
    if timestamp_col is not None and timestamp_col in reviews.columns:
        required.append(timestamp_col)
    reviews = reviews.copy()
    if timestamp_col is not None and timestamp_col in reviews.columns:
        reviews[timestamp_col] = pd.to_datetime(reviews[timestamp_col], errors="coerce")
    reviews = reviews.dropna(subset=required).copy()
    if customers is not None and customer_col in customers.columns:
        reviews = reviews[reviews[customer_col].isin(customers[customer_col])]
    if products is not None and product_col in products.columns:
        reviews = reviews[reviews[product_col].isin(products[product_col])]
    if timestamp_col is not None and timestamp_col in reviews.columns:
        reviews = reviews.sort_values(timestamp_col, kind="mergesort")
    return reviews.reset_index(drop=True)


def graph_construction_diagnostics(
    reviews: pd.DataFrame,
    customer_col: str,
    product_col: str,
    graph_is_directed: bool = True,
    graph_is_simple: bool = True,
    edge_multiplicity_used_for_sbm: bool = False,
) -> Dict[str, Any]:
    unique_pairs = reviews[[customer_col, product_col]].drop_duplicates()
    customer_unique_degree = unique_pairs.groupby(customer_col).size()
    product_unique_degree = unique_pairs.groupby(product_col).size()
    customer_review_degree = reviews.groupby(customer_col).size()
    product_review_degree = reviews.groupby(product_col).size()
    parallel_edges_count = int(len(reviews) - len(unique_pairs))
    return {
        "num_review_rows": int(len(reviews)),
        "num_unique_customer_ids": int(reviews[customer_col].nunique()),
        "num_unique_product_ids": int(reviews[product_col].nunique()),
        "num_unique_customer_product_pairs": int(len(unique_pairs)),
        "duplicate_pair_rate": duplicate_pair_rate(reviews, customer_col, product_col),
        "graph_is_simple": bool(graph_is_simple),
        "graph_is_directed": bool(graph_is_directed),
        "graph_is_bipartite": True,
        "num_customer_nodes": int(reviews[customer_col].nunique()),
        "num_product_nodes": int(reviews[product_col].nunique()),
        "num_total_nodes": int(
            reviews[customer_col].nunique() + reviews[product_col].nunique()
        ),
        "num_edges_passed_to_sbm": int(
            len(reviews) if edge_multiplicity_used_for_sbm else len(unique_pairs)
        ),
        "edge_multiplicity_used_for_sbm": bool(edge_multiplicity_used_for_sbm),
        "self_loops_count": 0,
        "parallel_edges_count": parallel_edges_count,
        **degree_summary_fields("customer", customer_unique_degree),
        **degree_summary_fields("product", product_unique_degree),
        "customer_review_count_degree_mean": float(customer_review_degree.mean())
        if len(customer_review_degree)
        else 0.0,
        "product_review_count_degree_mean": float(product_review_degree.mean())
        if len(product_review_degree)
        else 0.0,
    }


def graph_degree_summary_frame(
    reviews: pd.DataFrame, customer_col: str, product_col: str
) -> pd.DataFrame:
    unique_pairs = reviews[[customer_col, product_col]].drop_duplicates()
    customer_unique_degree = unique_pairs.groupby(customer_col).size()
    product_unique_degree = unique_pairs.groupby(product_col).size()
    customer_review_degree = reviews.groupby(customer_col).size()
    product_review_degree = reviews.groupby(product_col).size()
    rows = []
    for customer_id in sorted(reviews[customer_col].dropna().unique(), key=str):
        rows.append(
            {
                "node_id": customer_id,
                "node_type": "customer",
                "degree_unique_pairs": int(customer_unique_degree.get(customer_id, 0)),
                "degree_review_count": int(customer_review_degree.get(customer_id, 0)),
            }
        )
    for product_id in sorted(reviews[product_col].dropna().unique(), key=str):
        rows.append(
            {
                "node_id": product_id,
                "node_type": "product",
                "degree_unique_pairs": int(product_unique_degree.get(product_id, 0)),
                "degree_review_count": int(product_review_degree.get(product_id, 0)),
            }
        )
    return pd.DataFrame(rows)


def degree_summary_fields(prefix: str, degree: pd.Series) -> Dict[str, Any]:
    values = degree.to_numpy(dtype=float)
    if len(values) == 0:
        return {
            f"{prefix}_degree_min": 0,
            f"{prefix}_degree_median": 0.0,
            f"{prefix}_degree_mean": 0.0,
            f"{prefix}_degree_max": 0,
        }
    return {
        f"{prefix}_degree_min": int(values.min()),
        f"{prefix}_degree_median": float(np.median(values)),
        f"{prefix}_degree_mean": float(values.mean()),
        f"{prefix}_degree_max": int(values.max()),
    }


def duplicate_pair_rate(df: pd.DataFrame, customer_col: str, product_col: str) -> float:
    if len(df) == 0:
        return 0.0
    unique_pairs = df[[customer_col, product_col]].drop_duplicates()
    return float(1.0 - len(unique_pairs) / len(df))


def fit_graph_tool_nested_sbm(
    reviews: pd.DataFrame,
    customer_col: str,
    product_col: str,
    seed: int,
    use_multigraph: bool = False,
    use_edge_counts: bool = False,
) -> Dict[str, Any]:
    try:
        import graph_tool.all as gt
    except ImportError as exc:
        return {
            "graph_tool_available": False,
            "warning": f"graph-tool unavailable: {exc}",
        }

    customer_ids = list(pd.unique(reviews[customer_col]))
    product_ids = list(pd.unique(reviews[product_col]))
    edge_df = (
        reviews[[customer_col, product_col]]
        if use_multigraph
        else reviews[[customer_col, product_col]].drop_duplicates()
    )

    gt.seed_rng(seed)
    graph = gt.Graph(directed=True)
    vertex_map: Dict[Tuple[str, Any], Any] = {}
    type_label = graph.new_vertex_property("int")
    node_types: Dict[int, str] = {}
    node_ids: Dict[int, Any] = {}

    for customer_id in tqdm(customer_ids, desc="Adding customer vertices", unit="customer"):
        vertex = graph.add_vertex()
        vertex_map[("customer", customer_id)] = vertex
        type_label[vertex] = 0
        node_types[int(vertex)] = "customer"
        node_ids[int(vertex)] = customer_id
    for product_id in tqdm(product_ids, desc="Adding product vertices", unit="product"):
        vertex = graph.add_vertex()
        vertex_map[("product", product_id)] = vertex
        type_label[vertex] = 1
        node_types[int(vertex)] = "product"
        node_ids[int(vertex)] = product_id
    graph.vertex_properties["block"] = type_label

    edge_weight = graph.new_edge_property("int") if use_edge_counts else None
    if use_edge_counts and not use_multigraph:
        pair_counts = reviews.groupby([customer_col, product_col]).size()
        iterator = pair_counts.items()
        total = len(pair_counts)
        for (customer_id, product_id), count in tqdm(
            iterator, total=total, desc="Adding weighted aggregate edges", unit="edge"
        ):
            edge = graph.add_edge(
                vertex_map[("customer", customer_id)],
                vertex_map[("product", product_id)],
            )
            edge_weight[edge] = int(count)
        graph.edge_properties["count"] = edge_weight
    else:
        for row in tqdm(
            edge_df[[customer_col, product_col]].itertuples(index=False),
            total=len(edge_df),
            desc="Adding aggregate review edges",
            unit="edge",
        ):
            customer_id, product_id = row
            graph.add_edge(
                vertex_map[("customer", customer_id)],
                vertex_map[("product", product_id)],
            )

    if graph.num_edges() == 0:
        return {
            "graph_tool_available": True,
            "warning": "graph has no edges; SBM fit skipped",
        }

    state_args = {"deg_corr": True, "clabel": graph.vp["block"]}
    if use_edge_counts:
        # The existing generator does not use edge covariates. This property is
        # recorded for diagnostics and future experiments, not passed to the
        # current default fit.
        pass
    state = gt.minimize_nested_blockmodel_dl(graph, state_args=state_args)
    return {
        "graph_tool_available": True,
        "graph": graph,
        "state": state,
        "vertex_map": vertex_map,
        "node_types": node_types,
        "node_ids": node_ids,
        "customer_ids": customer_ids,
        "product_ids": product_ids,
        "type_label_values": [0, 1],
        "clabel_used": True,
        "block_property_seeded_with_type_labels": True,
    }


def raw_assignments_for_level(
    state: Any,
    vertex_map: Dict[Tuple[str, Any], Any],
    customer_ids: Iterable[Any],
    product_ids: Iterable[Any],
    level: int,
) -> Dict[Tuple[str, Any], int]:
    level = int(level)
    if level < 0:
        raise ValueError("level must be nonnegative")
    assignments: Dict[Tuple[str, Any], int] = {}
    bottom = state.levels[0].b.a
    for customer_id in customer_ids:
        vertex = vertex_map[("customer", customer_id)]
        assignments[("customer", customer_id)] = int(bottom[int(vertex)])
    for product_id in product_ids:
        vertex = vertex_map[("product", product_id)]
        assignments[("product", product_id)] = int(bottom[int(vertex)])

    for current_level in range(1, min(level, len(state.levels) - 1) + 1):
        parent = state.levels[current_level].b.a
        assignments = {
            key: int(parent[value]) if int(value) < len(parent) else int(value)
            for key, value in assignments.items()
        }
    return assignments


def split_raw_assignments(
    raw_assignments: Dict[Tuple[str, Any], int]
) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    customer_raw = [
        (entity_id, block)
        for (node_type, entity_id), block in raw_assignments.items()
        if node_type == "customer"
    ]
    product_raw = [
        (entity_id, block)
        for (node_type, entity_id), block in raw_assignments.items()
        if node_type == "product"
    ]
    return compact_labels(customer_raw), compact_labels(product_raw)


def compact_labels(values: Iterable[Tuple[Any, int]]) -> Dict[Any, int]:
    mapping: Dict[int, int] = {}
    compact: Dict[Any, int] = {}
    for key, value in values:
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        compact[key] = mapping[value]
    return compact


def summarize_raw_assignments(
    raw_assignments: Dict[Tuple[str, Any], int],
    level: int,
    description_length: Optional[float] = None,
) -> Dict[str, Any]:
    block_types: Dict[int, set[str]] = {}
    block_customer_sizes: Counter = Counter()
    block_product_sizes: Counter = Counter()
    for (node_type, _), block in raw_assignments.items():
        block = int(block)
        block_types.setdefault(block, set()).add(node_type)
        if node_type == "customer":
            block_customer_sizes[block] += 1
        else:
            block_product_sizes[block] += 1
    mixed_blocks = [
        block for block, types in block_types.items() if {"customer", "product"} <= types
    ]
    customer_blocks = [block for block, count in block_customer_sizes.items() if count > 0]
    product_blocks = [block for block, count in block_product_sizes.items() if count > 0]
    return {
        "level": int(level),
        "num_total_blocks": int(len(block_types)),
        "num_customer_blocks": int(len(customer_blocks)),
        "num_product_blocks": int(len(product_blocks)),
        "num_mixed_blocks": int(len(mixed_blocks)),
        "num_nonempty_blocks": int(len(block_types)),
        **block_size_summary("customer", block_customer_sizes.values()),
        **block_size_summary("product", block_product_sizes.values()),
        "description_length_if_available": description_length,
        "warnings": hierarchy_level_warnings(len(mixed_blocks)),
    }


def hierarchy_level_warnings(num_mixed_blocks: int) -> list[str]:
    if num_mixed_blocks > 0:
        return [
            f"Mixed customer/product blocks detected ({num_mixed_blocks}); "
            "type constraints are not preventing mixed blocks."
        ]
    return []


def inspect_state_levels(
    state: Any,
    vertex_map: Dict[Tuple[str, Any], Any],
    customer_ids: Iterable[Any],
    product_ids: Iterable[Any],
    max_levels: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[int, Dict[Tuple[str, Any], int]]]:
    total_levels = len(state.levels)
    if max_levels is not None:
        total_levels = min(total_levels, int(max_levels))
    rows = []
    assignments_by_level: Dict[int, Dict[Tuple[str, Any], int]] = {}
    for level in range(total_levels):
        raw = raw_assignments_for_level(
            state, vertex_map, customer_ids, product_ids, level
        )
        assignments_by_level[level] = raw
        description_length = None
        try:
            description_length = float(state.levels[level].entropy())
        except Exception:
            try:
                description_length = float(state.entropy())
            except Exception:
                description_length = None
        rows.append(
            summarize_raw_assignments(
                raw, level=level, description_length=description_length
            )
        )
    return pd.DataFrame(rows), assignments_by_level


def select_block_level(
    level_summaries: Union[pd.DataFrame, List[Dict[str, Any]]],
    mode: Union[str, int] = "current",
) -> Tuple[int, list[str]]:
    if not isinstance(level_summaries, pd.DataFrame):
        level_summaries = pd.DataFrame(level_summaries)
    if level_summaries.empty:
        return CURRENT_GENERATOR_BLOCK_LEVEL, ["No SBM hierarchy levels available."]

    warnings_list: list[str] = []
    max_level = int(level_summaries["level"].max())
    if isinstance(mode, int):
        return max(0, min(int(mode), max_level)), warnings_list
    mode_str = str(mode)
    if mode_str.isdigit():
        return max(0, min(int(mode_str), max_level)), warnings_list
    if mode_str in {"current", "bottom"}:
        return CURRENT_GENERATOR_BLOCK_LEVEL, warnings_list
    if mode_str == "top":
        return max_level, warnings_list
    if mode_str != "auto":
        raise ValueError("block_level must be current, top, bottom, auto, or an integer.")

    usable = level_summaries[level_summaries["num_mixed_blocks"] == 0].sort_values("level")
    strong = usable[
        (usable["num_customer_blocks"] >= 2)
        & (usable["num_product_blocks"] >= 2)
    ]
    if not strong.empty:
        return int(strong.iloc[0]["level"]), warnings_list
    weak = usable[
        (usable["num_customer_blocks"] > 1)
        | (usable["num_product_blocks"] > 1)
    ]
    if not weak.empty:
        return int(weak.iloc[0]["level"]), warnings_list
    warnings_list.append("SBM genuinely collapsed under current graph construction.")
    return CURRENT_GENERATOR_BLOCK_LEVEL, warnings_list


def block_pair_diagnostics_by_level(
    reviews: pd.DataFrame,
    assignments_by_level: Dict[int, Dict[Tuple[str, Any], int]],
    customer_col: str,
    product_col: str,
) -> pd.DataFrame:
    rows = []
    for level, raw in sorted(assignments_by_level.items()):
        customer_blocks, product_blocks = split_raw_assignments(raw)
        counts = block_pair_counts(
            reviews, customer_blocks, product_blocks, customer_col, product_col
        )
        count_values = counts.to_numpy(dtype=float)
        top_shares = []
        if len(count_values) > 0 and count_values.sum() > 0:
            top_shares = sorted(count_values / count_values.sum(), reverse=True)[:10]
        rows.append(
            {
                "level": int(level),
                "num_customer_blocks": int(len(set(customer_blocks.values()))),
                "num_product_blocks": int(len(set(product_blocks.values()))),
                "num_nonzero_block_pairs_real": int((counts > 0).sum()),
                "num_possible_block_pairs": int(
                    len(set(customer_blocks.values())) * len(set(product_blocks.values()))
                ),
                "fraction_events_in_largest_block_pair": float(top_shares[0])
                if top_shares
                else None,
                "top_10_block_pair_event_shares": json.dumps(
                    [float(value) for value in top_shares]
                ),
                "block_pair_count_min": int(count_values.min())
                if len(count_values)
                else 0,
                "block_pair_count_median": float(np.median(count_values))
                if len(count_values)
                else 0.0,
                "block_pair_count_mean": float(count_values.mean())
                if len(count_values)
                else 0.0,
                "block_pair_count_max": int(count_values.max())
                if len(count_values)
                else 0,
            }
        )
    return pd.DataFrame(rows)


def write_level_assignment_files(
    output_dir: str | Path,
    assignments_by_level: Dict[int, Dict[Tuple[str, Any], int]],
    customer_col: str,
    product_col: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for level, raw in sorted(assignments_by_level.items()):
        customer_blocks, product_blocks = split_raw_assignments(raw)
        pd.DataFrame(
            [
                {customer_col: customer_id, "customer_block": int(block)}
                for customer_id, block in sorted(customer_blocks.items(), key=lambda item: str(item[0]))
            ]
        ).to_csv(output_dir / f"block_assignments_level_{level}_customers.csv", index=False)
        pd.DataFrame(
            [
                {product_col: product_id, "product_block": int(block)}
                for product_id, block in sorted(product_blocks.items(), key=lambda item: str(item[0]))
            ]
        ).to_csv(output_dir / f"block_assignments_level_{level}_products.csv", index=False)


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
