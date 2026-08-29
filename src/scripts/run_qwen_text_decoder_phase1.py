#!/usr/bin/env python3
"""Run only the frozen Qwen3-0.6B Phase-1 diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.qwen_text_decoder.phase1 import QwenPhase1Experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/qwen_text_decoder_06b_phase1.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("audit", "compare", "generate", "evaluate", "diffusion", "report", "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    experiment = QwenPhase1Experiment(Path(args.config))
    if args.stage in {"audit", "all"}:
        experiment.audit()
    if args.stage in {"compare", "all"}:
        experiment.same_subset_comparison(args.device)
    if args.stage in {"generate", "all"}:
        experiment.generate(args.device, skip_existing=not args.no_skip_existing)
    if args.stage in {"evaluate", "all"}:
        experiment.evaluate(args.device)
    if args.stage in {"diffusion", "all"}:
        experiment.evaluate_diffusion_oracles(args.device)
    if args.stage in {"report", "all"}:
        experiment.report()


if __name__ == "__main__":
    main()

