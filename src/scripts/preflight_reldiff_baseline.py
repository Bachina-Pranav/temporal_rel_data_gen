#!/usr/bin/env python3
"""Scientific compatibility gate for the bundled RelDiff baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # Python 3.7 developer environments.
    import importlib_metadata


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.reldiff.adapter import (  # noqa: E402
    prepare_training_database,
    read_interaction_splits,
    reconstruct_foreign_keys,
    repeated_pair_summary,
    write_json,
)
from baselines.reldiff.schema import RelDiffDatasetConfig, load_dataset_config  # noqa: E402


UPSTREAM_REPOSITORY = "https://github.com/ValterH/RelDiff"
UPSTREAM_LINEAGE_COMMIT = "792f4982c9c82c1944dee7099e9fe5dc6eda1094"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/reldiff_baseline.yaml",
    )
    parser.add_argument(
        "--output-dir", default="outputs/baselines/reldiff/preflight"
    )
    parser.add_argument(
        "--skip-real-statistics",
        action="store_true",
        help="Developer-only option; a skipped audit can never pass the training gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = yaml.safe_load(Path(args.experiment_config).read_text())
    configs = [load_dataset_config(path) for path in experiment["datasets"]]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dependency_error = None
    try:
        unit = run_native_repeated_pair_unit_test(output)
    except Exception as exc:  # Write a useful blocker rather than starting training.
        dependency_error = f"{type(exc).__name__}: {exc}"
        unit = failed_unit_result(dependency_error)
    write_json(unit, output / "repeated_pair_unit_test.json")

    statistic_rows: list[dict[str, Any]] = []
    example_frames: list[pd.DataFrame] = []
    repeated_fractions: dict[str, float | None] = {}
    if not args.skip_real_statistics:
        for config in configs:
            splits, _ = read_interaction_splits(config)
            complete = pd.concat(list(splits.values()), ignore_index=True)
            for split_name, frame in (("train", splits["train"]), ("complete", complete)):
                summary, top, examples = repeated_pair_summary(
                    frame, config.source_fk, config.destination_fk
                )
                statistic_rows.append(
                    {"dataset": config.key, "split": split_name, **summary}
                )
                top = annotate_audit_frame(
                    top,
                    dataset=config.key,
                    split=split_name,
                    record_type="top_20_pair_multiplicity",
                )
                example_frames.append(top)
                if len(examples):
                    examples = annotate_audit_frame(
                        examples,
                        dataset=config.key,
                        split=split_name,
                        record_type="repeated_pair_event_example",
                    )
                    example_frames.append(examples)
                if split_name == "train":
                    repeated_fractions[config.key] = summary[
                        "fraction_rows_in_repeated_pairs"
                    ]
    else:
        repeated_fractions = {config.key: None for config in configs}

    pd.DataFrame(statistic_rows).to_csv(
        output / "repeated_pair_statistics.csv", index=False
    )
    if example_frames:
        pd.concat(example_frames, ignore_index=True, sort=False).to_csv(
            output / "repeated_pair_examples.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output / "repeated_pair_examples.csv", index=False)

    graph_tool_parallel = graph_tool_parallel_edge_test()
    audit = code_audit(unit, graph_tool_parallel, dependency_error)
    (output / "reldiff_code_audit.md").write_text(
        render_code_audit(audit), encoding="utf-8"
    )

    checks = {
        "parallel_edges_supported": bool(graph_tool_parallel["supported"]),
        "bridge_row_count_preserved": unit["checks"]["row_count_preserved"],
        "multiplicity_preserved": unit["checks"]["multiplicity_preserved"],
        "per_event_attributes_preserved": unit["checks"]["per_event_attributes_preserved"],
        "inverse_transform_preserves_events": unit["checks"]["inverse_transform_preserves_events"],
        "real_statistics_complete": not args.skip_real_statistics
        and len(statistic_rows) == len(configs) * 2,
    }
    verdict = "PASS" if all(checks.values()) else "BLOCK"
    gate = {
        "verdict": verdict,
        "checks": checks,
        "representation": "explicit_attributed_interaction_row_nodes",
        "bridge_conversion_used_for_interactions": False,
        "repeated_pair_fraction_train": repeated_fractions,
        "dependency_error": dependency_error,
    }
    write_json(gate, output / "repeated_pair_gate.json")
    if verdict == "BLOCK":
        (output / "repeated_pair_blocker.md").write_text(
            render_blocker(gate, unit), encoding="utf-8"
        )
    else:
        (output / "repeated_pair_blocker.md").unlink(missing_ok=True)

    print_console_gate(gate)
    if verdict != "PASS":
        raise SystemExit(2)


def run_native_repeated_pair_unit_test(output: Path) -> dict[str, Any]:
    from syntherela.data import load_tables, remove_sdv_columns
    from syntherela.metadata import Metadata
    from torch_geometric.utils import to_networkx

    from reldiff.data.dataset import create_dataset, dataset_from_graph
    from reldiff.data.preprocessing import process_data

    tiny_root = output / "tiny_native"
    tiny_root.mkdir(parents=True, exist_ok=True)
    users = pd.DataFrame({"user_id": ["u1", "u2"]})
    items = pd.DataFrame({"item_id": ["i1", "i2"]})
    events = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5],
            "user_id": ["u1", "u1", "u1", "u1", "u2"],
            "item_id": ["i1", "i1", "i1", "i2", "i1"],
            "timestamp": [10, 20, 30, 40, 50],
            "value": ["A", "B", "C", "D", "E"],
            "split": ["train"] * 5,
        }
    )
    users.to_csv(tiny_root / "users_input.csv", index=False)
    items.to_csv(tiny_root / "items_input.csv", index=False)
    events.to_csv(tiny_root / "events_input.csv", index=False)
    config = RelDiffDatasetConfig(
        key="tiny_repeated_pair",
        display_name="Tiny repeated-pair test",
        interaction_path=tiny_root / "events_input.csv",
        source_entity_path=tiny_root / "users_input.csv",
        destination_entity_path=tiny_root / "items_input.csv",
        source_table="users",
        destination_table="items",
        interaction_table="events",
        source_pk="user_id",
        destination_pk="item_id",
        source_fk="user_id",
        destination_fk="item_id",
        event_id="event_id",
        timestamp="timestamp",
        numerical_attributes=(),
        categorical_attributes=("value",),
        ignored_attributes=(),
        evaluation_config=Path("unused.yaml"),
    )
    config.validate()

    data_root = tiny_root / "data"
    provenance = tiny_root / "provenance"
    prepare_training_database(
        config,
        data_root=data_root,
        staged_name="tiny_repeated_pair",
        provenance_dir=provenance,
    )
    metadata_path = data_root / "original/tiny_repeated_pair/metadata.json"
    metadata = Metadata().load_from_json(str(metadata_path))
    tables = load_tables(str(metadata_path.parent), metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)
    for table_name, table in tables.items():
        process_data(
            table,
            name=table_name,
            metadata=metadata.get_table_meta(table_name, to_dict=False),
            data_path=str(data_root),
            dataset_name="tiny_repeated_pair",
            normalization="quantile",
            standardize=False,
            sigma_data=1.0,
        )

    processed = data_root / "processed/tiny_repeated_pair"
    dataset = create_dataset(
        metadata,
        str(processed),
        transform_fk_tables=False,
        add_reverse_edges=True,
    )
    row_count = int(dataset["events"].num_nodes)
    reconstructed = reconstruct_foreign_keys(dataset, metadata, "events", row_count)
    multiplicity = int(
        ((reconstructed["user_id"] == 0) & (reconstructed["item_id"] == 0)).sum()
    )

    forward = create_dataset(
        metadata,
        str(processed),
        transform_fk_tables=False,
        add_reverse_edges=False,
    )
    graph = to_networkx(forward, to_multi=False)
    roundtrip = dataset_from_graph(
        graph,
        forward,
        metadata,
        add_reverse_edges=True,
        transform_fk_tables=False,
        dimension_tables=["users", "items"],
    )
    inverse = reconstruct_foreign_keys(roundtrip, metadata, "events", row_count)
    inverse_multiplicity = int(
        ((inverse["user_id"] == 0) & (inverse["item_id"] == 0)).sum()
    )
    encoded_values = np.load(processed / "events/X_cat.npy")
    distinct_value_slots = int(np.unique(encoded_values[:, 0]).size)

    checks = {
        "row_count_preserved": row_count == 5,
        "relationship_representation_count_preserved": (
            dataset[("events", "user_id", "users")].edge_index.shape[1] == 5
            and dataset[("events", "item_id", "items")].edge_index.shape[1] == 5
        ),
        "multiplicity_preserved": multiplicity == 3,
        "per_event_attributes_preserved": distinct_value_slots == 5,
        "inverse_transform_preserves_events": len(inverse) == 5
        and inverse_multiplicity == 3,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "executed_code": {
            "preprocessing": "reldiff.data.preprocessing.process_data",
            "graph_construction": "reldiff.data.dataset.create_dataset",
            "graph_inverse": "reldiff.data.dataset.dataset_from_graph",
            "foreign_key_inverse": "MultiTableSampler child-sorted edge-index semantics",
        },
        "representation": "one explicit events node per attributed interaction row",
        "transform_fk_tables": False,
        "original_rows": 5,
        "event_nodes": row_count,
        "u1_i1_multiplicity_before": 3,
        "u1_i1_multiplicity_after_create_dataset": multiplicity,
        "u1_i1_multiplicity_after_graph_roundtrip": inverse_multiplicity,
        "distinct_value_attribute_slots": distinct_value_slots,
        "checks": checks,
    }


def graph_tool_parallel_edge_test() -> dict[str, Any]:
    try:
        import graph_tool.all as gt

        graph = gt.Graph(directed=True)
        graph.add_vertex(2)
        graph.add_edge(0, 1)
        graph.add_edge(0, 1)
        parallel = int(gt.label_parallel_edges(graph, mark_only=True).a.sum())
        return {
            "supported": graph.num_edges() == 2 and parallel > 0,
            "graph_type": type(graph).__name__,
            "num_edges_after_adding_duplicate": int(graph.num_edges()),
            "parallel_edge_marks": parallel,
            "version": _package_version("graph-tool"),
        }
    except Exception as exc:
        return {
            "supported": False,
            "graph_type": "graph_tool.Graph",
            "error": f"{type(exc).__name__}: {exc}",
            "version": None,
        }


def annotate_audit_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    record_type: str,
) -> pd.DataFrame:
    """Add audit provenance without colliding with source-table columns."""

    annotated = frame.drop(
        columns=["dataset", "split", "record_type"], errors="ignore"
    ).copy()
    annotated.insert(0, "dataset", dataset)
    annotated.insert(1, "split", split)
    annotated.insert(2, "record_type", record_type)
    return annotated


def code_audit(
    unit: dict[str, Any], graph_tool: dict[str, Any], dependency_error: str | None
) -> dict[str, Any]:
    return {
        "RELDIFF_REPOSITORY": UPSTREAM_REPOSITORY,
        "RELDIFF_COMMIT": git_value("rev-parse", "HEAD"),
        "RELDIFF_BUNDLED_UPSTREAM_LINEAGE_COMMIT": UPSTREAM_LINEAGE_COMMIT,
        "RELDIFF_WORKTREE_STATUS": git_value("status", "--short"),
        "RELDIFF_PAPER": "RelDiff: Relational Data Generative Modeling with Graph-Based Diffusion Models, arXiv:2506.00710",
        "RELDIFF_ENVIRONMENT": {
            "environment_file": "reldiff.yaml",
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "torch_geometric": _package_version("torch-geometric"),
            "syntherela": _package_version("syntherela"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "RELDIFF_GRAPH_LIBRARY": "PyTorch Geometric plus graph-tool",
        "RELDIFF_GRAPH_OBJECT_TYPE": "torch_geometric.data.HeteroData; graph_tool.Graph during 2K+SBM",
        "RELDIFF_SUPPORTS_PARALLEL_EDGES": graph_tool,
        "RELDIFF_SUPPORTS_EDGE_ATTRIBUTES": (
            "Structure edges carry relationship type. Attributed interactions are "
            "represented as explicit row nodes with independent x_num/x_cat slots."
        ),
        "RELDIFF_BRIDGE_TABLE_TRANSFORM": (
            "transform_foreign_key_tables converts only zero-attribute, no-child tables "
            "to direct parent-parent edges. It is deliberately disabled for this adapter."
        ),
        "RELDIFF_BRIDGE_TABLE_INVERSE": (
            "dataset_from_graph reconstructs typed edges; MultiTableSampler reconstructs "
            "child FKs from child-sorted edge indices."
        ),
        "RELDIFF_TIMESTAMP_COMPATIBILITY": (
            "Adapter encodes seconds from the training-only minimum as an ordinary numerical "
            "attribute, then uses the official numerical transform and inverse."
        ),
        "EXECUTED_REPRESENTATION": unit.get("representation"),
        "UPSTREAM_CORE_CHANGES_ALREADY_PRESENT": [
            "src/reldiff/models/model.py: float32 positional-embedding compatibility cast",
            "src/reldiff/trainer.py: cross-version AMP compatibility",
            "src/reldiff/models/joint.py: bias-only equivalent for zero-feature dimension projections",
        ],
        "BASELINE_COMPATIBILITY_CHANGES": [
            "train/sample CLI seed option",
            "train/sample CLI option to preserve explicit table nodes",
        ],
        "CORE_SOURCE_HASHES": core_source_hashes(),
        "dependency_error": dependency_error,
    }


def render_code_audit(audit: dict[str, Any]) -> str:
    lines = [
        "# RelDiff Code Audit",
        "",
        "This audit traces the bundled official implementation rather than reimplementing RelDiff.",
        "",
    ]
    for key, value in audit.items():
        rendered = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        lines.append(f"- **{key}**: {rendered}")
    lines.extend(
        [
            "",
            "## Executed Path",
            "",
            "1. The adapter materializes ID-only source/destination tables and a train-only attributed interaction table.",
            "2. `process_data` applies the official categorical and QuantileTransformer-based numerical preprocessing.",
            "3. `create_dataset(..., transform_fk_tables=False)` retains one interaction node per row and two typed FK edges per event.",
            "4. The official graph-tool D2K+SBM generator samples the typed relational graph while retaining node/edge cardinalities and typed degree sequences.",
            "5. `dataset_from_graph` and `MultiTableSampler.sample_database` reconstruct tables and foreign keys.",
            "",
            "The explicit-node flag is necessary because ID-only parent tables have zero generated attributes; the default bridge-table heuristic otherwise attempts to classify every zero-attribute table as a bridge table. This is an adapter compatibility choice, not a change to RelDiff's graph or diffusion architecture.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_blocker(gate: dict[str, Any], unit: dict[str, Any]) -> str:
    return (
        "# RelDiff Repeated-Pair Blocker\n\n"
        "Expensive training was blocked because the scientific preflight did not pass.\n\n"
        f"```json\n{json.dumps({'gate': gate, 'unit_test': unit}, indent=2)}\n```\n"
    )


def failed_unit_result(error: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": error,
        "checks": {
            "row_count_preserved": False,
            "relationship_representation_count_preserved": False,
            "multiplicity_preserved": False,
            "per_event_attributes_preserved": False,
            "inverse_transform_preserves_events": False,
        },
    }


def print_console_gate(gate: dict[str, Any]) -> None:
    fractions = gate["repeated_pair_fraction_train"]
    yes_no = lambda value: "YES" if value else "NO"
    print("=" * 60)
    print("RELDIFF REPEATED-PAIR PREFLIGHT")
    print("=" * 60)
    print(f"PARALLEL EDGES SUPPORTED: {yes_no(gate['checks']['parallel_edges_supported'])}")
    print(f"BRIDGE ROW COUNT PRESERVED: {yes_no(gate['checks']['bridge_row_count_preserved'])}")
    print(f"MULTIPLICITY PRESERVED: {yes_no(gate['checks']['multiplicity_preserved'])}")
    print(f"PER-EVENT ATTRIBUTES PRESERVED: {yes_no(gate['checks']['per_event_attributes_preserved'])}")
    print(f"INVERSE TRANSFORM PRESERVES EVENTS: {yes_no(gate['checks']['inverse_transform_preserves_events'])}")
    for key, label in (
        ("amazon_toy", "AMAZON"),
        ("movielens_100k", "MOVIELENS"),
        ("rel_hm", "REL-HM"),
    ):
        print(f"{label} REPEATED-PAIR FRACTION: {fractions.get(key)}")
    print(f"VERDICT: {gate['verdict']}")
    print("=" * 60)


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def core_source_hashes() -> dict[str, str]:
    from baselines.reldiff.adapter import file_sha256

    paths = [
        "src/reldiff/data/dataset.py",
        "src/reldiff/data/preprocessing.py",
        "src/reldiff/diffusion/unified_ctime_diffusion.py",
        "src/reldiff/models/joint.py",
        "src/reldiff/sampler.py",
        "src/reldiff/trainer.py",
        "src/structure/generate_d2k_plus_sbm.py",
    ]
    return {path: file_sha256(ROOT / path) for path in paths}


if __name__ == "__main__":
    main()
