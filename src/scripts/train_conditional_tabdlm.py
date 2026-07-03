#!/usr/bin/env python3
"""Train Conditional TABDLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.train import train_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Conditional TABDLM.")
    parser.add_argument("--config", default="configs/attribute_generation/conditional_tabdlm_rel_amazon_exp1.yaml")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best = train_from_config(load_config(args.config), device=args.device)
    print(best)


if __name__ == "__main__":
    main()

