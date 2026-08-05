#!/usr/bin/env python3
"""Run negative and positive controls for the paper-grade C2ST."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.paper_metrics.c2st_sanity import c2st_integrity_audit  # noqa: E402
from evaluation.paper_metrics.utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-table", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows-per-side", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chance-tolerance", type=float, default=0.15)
    parser.add_argument("--corruption-auc-minimum", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    real_path = Path(args.real_table or config["real_table_path"])
    result = c2st_integrity_audit(
        pd.read_csv(real_path),
        config,
        max_rows_per_side=args.max_rows_per_side,
        seed=args.seed,
        chance_tolerance=args.chance_tolerance,
        corruption_auc_minimum=args.corruption_auc_minimum,
    )
    write_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_controls_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
