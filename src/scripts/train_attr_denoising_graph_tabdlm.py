#!/usr/bin/env python3
"""Train v3 temporal attribute-denoising graph Conditional TABDLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.train import train_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v3 temporal attribute-denoising graph Conditional TABDLM.")
    parser.add_argument(
        "--config",
        default="configs/attribute_generation/conditional_tabdlm_amazon_toy_exp3_temporal_attr_denoising_graph.yaml",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(train_from_config(load_config(args.config), device=args.device))


if __name__ == "__main__":
    main()
