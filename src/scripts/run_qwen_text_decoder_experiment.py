#!/usr/bin/env python3
"""Run the focused Qwen3-0.6B pretrained text-decoder experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.qwen_text_decoder import QwenTextExperiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/qwen_text_decoder_06b.yaml")
    parser.add_argument("--stage", choices=("preflight", "train", "oracle", "generated", "evaluate", "report", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-model-download", action="store_true", help="Preflight-only offline inspection; training still requires a pinned local snapshot.")
    args = parser.parse_args()
    experiment = QwenTextExperiment(Path(args.config))
    if args.stage in {"preflight", "all"}: experiment.preflight(resolve_model=not args.skip_model_download)
    if args.stage in {"train", "all"}: experiment.train(args.device)
    if args.stage in {"oracle", "all"}: experiment.generate("oracle_structured", args.device)
    if args.stage == "generated": experiment.generate("generated_structured", args.device)
    if args.stage == "all":
        try:
            experiment.generate("generated_structured", args.device)
        except RuntimeError as exc:
            print(f"[generated_structured] STOPPED: {exc}")
            print("[oracle_structured] Continuing with the valid decoder-isolation diagnostic.")
    if args.stage in {"evaluate", "all"}:
        for mode in ("oracle_structured", "generated_structured"):
            if (experiment.output_dir / mode / "synthetic_text.csv").is_file():
                experiment.evaluate(mode, args.device)
            else:
                print(f"[{mode}] evaluation skipped: synthetic_text.csv is absent")
    if args.stage in {"report", "all"}: experiment.report()


if __name__ == "__main__":
    main()
