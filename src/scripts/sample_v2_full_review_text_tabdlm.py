#!/usr/bin/env python3
"""Sample attributes from v2 full-review-text Conditional TABDLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.sample import sample_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp4_v2_full_review_text.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample v2 full-review-text Conditional TABDLM attributes.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--synthetic-spine", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-rows", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug-write-aux-targets", action="store_true")
    parser.add_argument("--disable-length-calibration", action="store_true")
    parser.add_argument("--repair-invalid-categoricals", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_rows = args.num_rows
    if isinstance(num_rows, str) and num_rows.isdigit():
        num_rows = int(num_rows)
    sample_from_config(
        load_config(args.config),
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_rows=num_rows,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        seed=args.seed,
        synthetic_spine_path=args.synthetic_spine,
        debug_write_aux_targets=args.debug_write_aux_targets,
        disable_length_calibration=args.disable_length_calibration,
        repair_invalid_categoricals=args.repair_invalid_categoricals,
    )


if __name__ == "__main__":
    main()
