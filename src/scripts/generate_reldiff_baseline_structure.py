#!/usr/bin/env python3
"""Thin generic CLI over RelDiff's bundled D2K+SBM structure functions."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_DIR = ROOT / "src/structure"
if str(STRUCTURE_DIR) not in sys.path:
    sys.path.insert(0, str(STRUCTURE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--split-by-subgraphs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import graph_tool.all as gt
    from syntherela.metadata import Metadata

    from generate_d2k_plus_sbm import (
        generate_new_graph,
        postprocess,
        preprocess,
        sort_nodes_by_table,
    )
    from nx2gt import gt2nx, nx2gt

    np.random.seed(args.seed)
    gt.seed_rng(args.seed)
    data_dir = Path(args.data_dir)
    with (data_dir / "structure" / f"{args.dataset_name}_graph.pkl").open("rb") as handle:
        original = pickle.load(handle)
    metadata = Metadata().load_from_json(
        str(data_dir / "original" / args.dataset_name / "metadata.json")
    )

    print(
        f"[structure] loaded nodes={original.number_of_nodes():,} "
        f"edges={original.number_of_edges():,}",
        flush=True,
    )
    start = time.perf_counter()
    graph = nx2gt(original)
    transformed, blocks, reverse_fk = preprocess(
        graph,
        fk_only_tables=None,
        split_by_subgraphs=args.split_by_subgraphs,
        stub_tables=None,
    )
    print(
        f"[structure] transformed vertices={transformed.num_vertices():,} "
        f"edges={transformed.num_edges():,}; fitting nested SBM",
        flush=True,
    )
    fit_start = time.perf_counter()
    state = gt.minimize_nested_blockmodel_dl(
        transformed,
        state_args={"deg_corr": True, "clabel": transformed.vp["block"]},
    )
    fit_seconds = time.perf_counter() - fit_start
    print(f"[structure] nested SBM fit completed in {fit_seconds:.2f}s", flush=True)
    sample_start = time.perf_counter()
    generated = generate_new_graph(
        transformed,
        micro_degs=True,
        micro_ers=True,
        hstate=state,
        max_retries=args.max_retries,
    )
    sample_seconds = time.perf_counter() - sample_start
    generated = postprocess(
        generated, fk_only_tables=None, metadata=metadata, reverse_fk=reverse_fk
    )
    generated_nx = sort_nodes_by_table(gt2nx(generated, multiedges=False))

    original_mixing = nx.assortativity.degree_mixing_dict(
        original, x="out", y="in", normalized=False
    )
    generated_mixing = nx.assortativity.degree_mixing_dict(
        generated_nx, x="out", y="in", normalized=False
    )
    if original_mixing != generated_mixing:
        raise AssertionError("D2K+SBM output did not preserve the joint degree distribution")
    if original.number_of_nodes() != generated_nx.number_of_nodes():
        raise AssertionError("D2K+SBM output changed node cardinality")
    if original.number_of_edges() != generated_nx.number_of_edges():
        raise AssertionError("D2K+SBM output changed edge cardinality")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(generated_nx, handle)
    runtime = {
        "seed": args.seed,
        "graph_structure_fit_seconds": fit_seconds,
        "graph_structure_sample_seconds": sample_seconds,
        "postprocessing_seconds": time.perf_counter() - sample_start - sample_seconds,
        "total_seconds": time.perf_counter() - start,
        "num_nodes": generated_nx.number_of_nodes(),
        "num_edges": generated_nx.number_of_edges(),
        "fk_only_tables": [],
        "interaction_representation": "explicit attributed row nodes",
        "split_by_subgraphs": args.split_by_subgraphs,
        "output": str(output),
    }
    runtime_path = Path(args.runtime_output)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    print(json.dumps(runtime, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
