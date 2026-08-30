#!/usr/bin/env python3
"""Run the pinned official ClavaDDPM baseline on MovieLens-100K."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.clavaddpm import ClavaDDPMMovieLensRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/clavaddpm_movielens.yaml")
    parser.add_argument("--stage", choices=("preflight", "smoke", "full", "all"), default="preflight")
    parser.add_argument("--dataset", choices=("movielens_100k",), default="movielens_100k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--official-python", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    runner = ClavaDDPMMovieLensRunner(Path(args.config), official_python=args.official_python)
    if args.stage in {"preflight", "all"}:
        runner.preflight()
    if args.stage in {"smoke", "all"}:
        runner.run("smoke", skip_existing=args.skip_existing)
    if args.stage in {"full", "all"}:
        runner.run("full", skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()

