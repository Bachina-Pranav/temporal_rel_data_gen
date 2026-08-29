#!/usr/bin/env python3
"""Run inference-only Qwen3-0.6B follow-up diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.qwen_text_decoder.followup import QwenFollowupExperiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/qwen_text_decoder_06b_followup.yaml")
    parser.add_argument("--stage", choices=("preflight", "generate", "evaluate", "diffusion", "report", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    experiment = QwenFollowupExperiment(Path(args.config))
    if args.stage in {"preflight", "all"}:
        experiment.preflight()
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

