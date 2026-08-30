#!/usr/bin/env python3
"""Evaluate finished Qwen text artifacts that lack canonical metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.qwen_text_decoder.experiment import QwenTextExperiment, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/overnight_8h_session/pending_canonical_evaluations.json",
    )
    args = parser.parse_args()
    configs = [Path("configs/experiments/qwen_text_decoder_06b.yaml")]
    configs.extend(sorted(Path("outputs/qwen_capacity_probe").glob("*/config_resolved.yaml")))
    records = []
    for config in configs:
        if not config.is_file():
            continue
        experiment = QwenTextExperiment(config)
        for mode in ("oracle_structured", "generated_structured"):
            synthetic = experiment.output_dir / mode / "synthetic_text.csv"
            metrics = experiment.output_dir / mode / "canonical_text_c2st.json"
            record = {
                "config": str(config), "mode": mode,
                "synthetic": str(synthetic), "metrics": str(metrics),
            }
            if not synthetic.is_file():
                record["status"] = "not_generated"
            elif args.skip_existing and metrics.is_file():
                record["status"] = "reused"
            else:
                experiment.evaluate(mode, args.device)
                record["status"] = "completed"
            records.append(record)
    result = {
        "scope": "finished Qwen artifacts only; no generation or retraining",
        "records": records,
        "completed_or_reused": sum(row["status"] in {"completed", "reused"} for row in records),
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

