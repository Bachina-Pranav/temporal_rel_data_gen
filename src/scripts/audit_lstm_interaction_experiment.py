#!/usr/bin/env python3
"""Run strict preflight validation for a schema-driven LSTM experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.experiment_audit import (  # noqa: E402
    audit_interaction_experiment,
    write_audit_outputs,
)
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a prepared temporal interaction table before LSTM training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.evaluation_config).open(encoding="utf-8") as handle:
        evaluation_config = yaml.safe_load(handle)
    report, roles = audit_interaction_experiment(
        load_config(args.config),
        evaluation_config,
    )
    write_audit_outputs(report, roles, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
