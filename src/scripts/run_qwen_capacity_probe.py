#!/usr/bin/env python3
"""Run the controlled Qwen3 capacity probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.qwen_text_decoder.capacity_probe import QwenCapacityProbe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/qwen_capacity_probe.yaml")
    parser.add_argument("--stage", choices=("prepare", "preflight", "train", "generate", "evaluate", "report", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    experiment = QwenCapacityProbe(Path(args.config))
    if args.stage in {"prepare", "all"}: experiment.prepare()
    if args.stage in {"preflight", "all"}: experiment.preflight()
    if args.stage in {"train", "all"}: experiment.train(args.device, skip_existing=args.skip_existing)
    if args.stage in {"generate", "all"}: experiment.generate(args.device, skip_existing=args.skip_existing)
    if args.stage in {"evaluate", "all"}: experiment.evaluate(args.device)
    if args.stage in {"report", "all"}: experiment.report()


if __name__ == "__main__":
    main()
