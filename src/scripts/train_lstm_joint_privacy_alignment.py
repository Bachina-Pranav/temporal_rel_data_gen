#!/usr/bin/env python3
"""Train v5.1 joint LSTM privacy/alignment experiment."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.dataset import prepare_rel_amazon_data  # noqa: E402
from attribute_generation.conditional_tabdlm.lstm_joint import train_lstm_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp5_1_lstm_privacy_alignment.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v5.1 LSTM privacy/alignment generator.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-prepare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    schema_path = config.data_dir / "schema.json"
    if args.force_prepare or not schema_path.exists():
        prepare_rel_amazon_data(config)
    best = train_lstm_from_config(config, device=args.device)
    logs_dir = config.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    root_log = config.output_dir / "train_log.jsonl"
    if root_log.exists():
        shutil.copyfile(root_log, logs_dir / "train_metrics.jsonl")
    print(best)


if __name__ == "__main__":
    main()
