#!/usr/bin/env python3
"""Train and confirm the selected relational prefix on the full train split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.qwen_text_decoder.relational_prefix import QwenRelationalPrefixExperiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/qwen_relational_prefix_full.yaml")
    parser.add_argument("--stage", choices=("preflight", "context", "train", "confirm", "report", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    experiment = QwenRelationalPrefixExperiment(Path(args.config))
    if args.stage in {"preflight", "all"}: experiment.preflight()
    if args.stage in {"context", "all"} and not (
        args.skip_existing and experiment._context_path("validation").is_file()
    ): experiment.build_context_cache(args.device)
    if args.stage in {"train", "all"} and not (
        args.skip_existing and (experiment.output / "training/best_prefix.pt").is_file()
    ): experiment.train(args.device, max_epochs=args.max_epochs)
    if args.stage in {"confirm", "all"}: experiment.confirm(args.device, force=True)
    if args.stage in {"report", "all"}: experiment.report()


if __name__ == "__main__":
    main()
