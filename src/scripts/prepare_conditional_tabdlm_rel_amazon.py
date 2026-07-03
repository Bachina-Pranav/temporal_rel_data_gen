#!/usr/bin/env python3
"""Prepare Rel-Amazon review rows for Conditional TABDLM Exp1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.dataset import prepare_rel_amazon_data  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Conditional TABDLM Rel-Amazon Exp1 data.")
    parser.add_argument("--config", default="configs/attribute_generation/conditional_tabdlm_rel_amazon_exp1.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepared = prepare_rel_amazon_data(config)
    print(f"Wrote train data: {prepared.train_path}")
    print(f"Wrote valid data: {prepared.valid_path}")
    print(f"Wrote test data: {prepared.test_path}")
    print(f"Wrote schema: {prepared.schema_path}")
    print(f"Wrote tokenizer metadata: {prepared.tokenizer_path}")


if __name__ == "__main__":
    main()

