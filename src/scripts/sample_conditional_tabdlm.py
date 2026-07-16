#!/usr/bin/env python3
"""Sample attributes from a Conditional TABDLM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.sample import sample_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Conditional TABDLM attributes.")
    parser.add_argument("--config", default="configs/attribute_generation/conditional_tabdlm_rel_amazon_exp1.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--synthetic-spine", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-rows", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--text-top-k", type=int, default=None)
    parser.add_argument("--sampling-steps", default=None, help="Inference denoising steps, or 'full' for the trained diffusion horizon.")
    parser.add_argument("--timestep-spacing", choices=["uniform", "quadratic", "leading", "trailing"], default=None)
    parser.add_argument("--inference-dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-output", default=None)
    parser.add_argument("--length-mode", choices=["normal", "empirical_length", "oracle_length"], default=None)
    parser.add_argument("--oracle-real-table", default=None, help="Real table used only to read row-level text lengths in oracle_length diagnostics.")
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
        sample_batch_size=args.sample_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        text_top_k=args.text_top_k,
        sampling_steps=args.sampling_steps,
        timestep_spacing=args.timestep_spacing,
        inference_dtype=args.inference_dtype,
        compile_model=args.compile_model,
        profile=args.profile,
        profile_output=args.profile_output,
        length_mode=args.length_mode,
        oracle_real_table_path=args.oracle_real_table,
        device=args.device,
        seed=args.seed,
        synthetic_spine_path=args.synthetic_spine,
        debug_write_aux_targets=args.debug_write_aux_targets,
        disable_length_calibration=args.disable_length_calibration,
        repair_invalid_categoricals=args.repair_invalid_categoricals,
    )


if __name__ == "__main__":
    main()
