#!/usr/bin/env python3
"""Prepare Amazon-toy data for v3 temporal attribute-denoising graph TABDLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.dataset import load_prepared_tables, prepare_rel_amazon_data  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_dataset import write_temporal_graph_metadata  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare v3 temporal attribute-denoising graph TABDLM inputs.")
    parser.add_argument(
        "--config",
        default="configs/attribute_generation/conditional_tabdlm_amazon_toy_exp3_temporal_attr_denoising_graph.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_rel_amazon_data(config)
    train_frame, _, _ = load_prepared_tables(config)
    graph_path = write_temporal_graph_metadata(train_frame, config, config.output_dir / "graph", source="real_training_rows")
    print(prepared)
    print(graph_path)


if __name__ == "__main__":
    main()
