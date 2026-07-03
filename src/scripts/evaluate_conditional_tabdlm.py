#!/usr/bin/env python3
"""Evaluate Conditional TABDLM attribute samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.evaluate import evaluate_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Conditional TABDLM attributes.")
    parser.add_argument("--config", default="configs/attribute_generation/conditional_tabdlm_rel_amazon_exp1_2.yaml")
    parser.add_argument("--real-reviews", default=None)
    parser.add_argument("--synthetic-reviews", default=None)
    parser.add_argument("--condition-cols", nargs="*", default=None)
    parser.add_argument("--categorical-cols", nargs="*", default=None)
    parser.add_argument("--text-cols", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_from_config(
        load_config(args.config),
        synthetic_reviews_path=args.synthetic_reviews,
        real_reviews_path=args.real_reviews,
        output_path=args.output,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
