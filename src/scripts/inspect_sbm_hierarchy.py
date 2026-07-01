#!/usr/bin/env python3
"""Inspect graph-tool nested SBM hierarchy extraction for temporal review graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.generation.sbm_hierarchy import (  # noqa: E402
    CURRENT_GENERATOR_BLOCK_LEVEL,
    block_pair_diagnostics_by_level,
    fit_graph_tool_nested_sbm,
    graph_construction_diagnostics,
    graph_degree_summary_frame,
    inspect_state_levels,
    preprocess_reviews_for_sbm,
    select_block_level,
    write_json,
    write_level_assignment_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect nested SBM hierarchy levels for Amazon-style review graphs."
    )
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--customers", default=None)
    parser.add_argument("--products", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-multigraph", action="store_true")
    parser.add_argument("--use-edge-counts", action="store_true")
    parser.add_argument("--simple-graph", action="store_true")
    parser.add_argument("--max-levels", type=int, default=None)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--block-level", default="current")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reviews = pd.read_csv(args.reviews)
    customers = pd.read_csv(args.customers) if args.customers else None
    products = pd.read_csv(args.products) if args.products else None
    reviews = preprocess_reviews_for_sbm(
        reviews,
        args.customer_id_col,
        args.product_id_col,
        timestamp_col=args.timestamp_col,
        customers=customers,
        products=products,
    )

    graph_simple = not args.use_multigraph
    graph_diagnostics = graph_construction_diagnostics(
        reviews,
        args.customer_id_col,
        args.product_id_col,
        graph_is_simple=graph_simple,
        graph_is_directed=True,
        edge_multiplicity_used_for_sbm=args.use_multigraph
        or args.use_edge_counts,
    )
    write_json(output_dir / "graph_construction_diagnostics.json", graph_diagnostics)
    graph_degree_summary_frame(
        reviews, args.customer_id_col, args.product_id_col
    ).to_csv(output_dir / "graph_degree_summary.csv", index=False)

    fit = fit_graph_tool_nested_sbm(
        reviews,
        args.customer_id_col,
        args.product_id_col,
        seed=args.seed,
        use_multigraph=args.use_multigraph,
        use_edge_counts=args.use_edge_counts,
    )
    if not fit.get("graph_tool_available"):
        summary = {
            "graph_tool_available": False,
            "hierarchy_depth": 0,
            "extracted_level_currently_used_by_generators": CURRENT_GENERATOR_BLOCK_LEVEL,
            "recommended_bottom_level": None,
            "recommended_nontrivial_level": None,
            "warnings": [
                fit.get("warning", "graph-tool unavailable"),
                "Temporal generators run with type-only fallback when graph-tool is unavailable.",
            ],
        }
        write_json(output_dir / "sbm_hierarchy_summary.json", summary)
        print(json.dumps(summary, indent=2))
        return

    levels, assignments_by_level = inspect_state_levels(
        fit["state"],
        fit["vertex_map"],
        fit["customer_ids"],
        fit["product_ids"],
        max_levels=args.max_levels,
    )
    levels.to_csv(output_dir / "sbm_hierarchy_levels.csv", index=False)
    level_block_pairs = block_pair_diagnostics_by_level(
        reviews,
        assignments_by_level,
        args.customer_id_col,
        args.product_id_col,
    )
    level_block_pairs.to_csv(
        output_dir / "sbm_level_block_pair_diagnostics.csv", index=False
    )
    write_level_assignment_files(
        output_dir,
        assignments_by_level,
        args.customer_id_col,
        args.product_id_col,
    )

    recommended_level, recommendation_warnings = select_block_level(levels, "auto")
    requested_level, requested_warnings = select_block_level(levels, args.block_level)
    bottom = levels[levels["level"] == CURRENT_GENERATOR_BLOCK_LEVEL].iloc[0]
    top = levels[levels["level"] == levels["level"].max()].iloc[0]
    current = levels[levels["level"] == requested_level].iloc[0]
    warnings_list = []
    warnings_list.extend(recommendation_warnings)
    warnings_list.extend(requested_warnings)
    if fit.get("block_property_seeded_with_type_labels"):
        warnings_list.append(
            "Graph vertex property 'block' stores node type labels and is passed as clabel. "
            "This should constrain types while still allowing subdivisions; if all levels "
            "remain 1+1, collapse is likely genuine under current graph construction."
        )
    if int(bottom["num_customer_blocks"]) == 1 and int(bottom["num_product_blocks"]) == 1:
        warnings_list.append(
            "Bottom/current SBM level has one customer block and one product block."
        )
    if int(levels["num_mixed_blocks"].max()) > 0:
        warnings_list.append("Mixed customer/product blocks detected in the hierarchy.")

    summary = {
        "graph_tool_available": True,
        "extracted_level_currently_used_by_generators": CURRENT_GENERATOR_BLOCK_LEVEL,
        "requested_block_level": args.block_level,
        "requested_block_level_resolved": int(requested_level),
        "recommended_bottom_level": CURRENT_GENERATOR_BLOCK_LEVEL,
        "recommended_nontrivial_level": int(recommended_level),
        "current_num_customer_blocks": int(current["num_customer_blocks"]),
        "current_num_product_blocks": int(current["num_product_blocks"]),
        "bottom_num_customer_blocks": int(bottom["num_customer_blocks"]),
        "bottom_num_product_blocks": int(bottom["num_product_blocks"]),
        "top_num_customer_blocks": int(top["num_customer_blocks"]),
        "top_num_product_blocks": int(top["num_product_blocks"]),
        "hierarchy_depth": int(len(levels)),
        "clabel_used": bool(fit.get("clabel_used")),
        "type_label_values": fit.get("type_label_values"),
        "blocks_allowed_to_split_within_node_type": True,
        "warnings": warnings_list,
    }
    write_json(output_dir / "sbm_hierarchy_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
