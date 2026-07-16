#!/usr/bin/env python3
"""Sample the hierarchical v4.1 Conditional TABDLM diffusion model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.hierarchical_sample import hierarchical_sample_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_hierarchical_v41.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample hierarchical v4.1 Conditional TABDLM attributes.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--synthetic-spine", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-rows", default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--structured-steps", default=None)
    parser.add_argument("--text-steps", default=None)
    parser.add_argument("--timestep-spacing", choices=["uniform", "quadratic", "leading", "trailing"], default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--text-top-k", type=int, default=None)
    parser.add_argument("--inference-dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--graph-mode", choices=["correct", "zero", "shuffled", "no_graph"], default="correct")
    parser.add_argument("--oracle-structured-table", default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug-write-aux-targets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_rows = args.num_rows
    if isinstance(num_rows, str) and num_rows.isdigit():
        num_rows = int(num_rows)
    hierarchical_sample_from_config(
        load_config(args.config),
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_rows=num_rows,
        sample_batch_size=args.sample_batch_size,
        structured_steps=args.structured_steps,
        text_steps=args.text_steps,
        timestep_spacing=args.timestep_spacing,
        inference_dtype=args.inference_dtype,
        text_top_k=args.text_top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        graph_mode_override=args.graph_mode,
        device=args.device,
        seed=args.seed,
        synthetic_spine_path=args.synthetic_spine,
        profile=args.profile,
        profile_output=args.profile_output,
        debug_write_aux_targets=args.debug_write_aux_targets,
        oracle_structured_table_path=args.oracle_structured_table,
    )


if __name__ == "__main__":
    main()
