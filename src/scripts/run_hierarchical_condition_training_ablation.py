#!/usr/bin/env python3
"""Train clean, corrupted, and mixed-condition hierarchical diffusion variants."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.diffusion_diagnostics import (  # noqa: E402
    current_git_commit,
    file_sha256,
    unique_run_root,
    write_json,
)
from attribute_generation.conditional_tabdlm.hierarchical_train import (  # noqa: E402
    train_hierarchical_from_config,
)
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from scripts.hierarchical_training_ablation_utils import (  # noqa: E402
    run_checkpoint_diagnostics,
    write_ablation_comparison,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "clean": {
        "text_conditioning": {
            "clean_probability": 1.0,
            "corrupted_probability": 0.0,
            "generated_probability": 0.0,
        }
    },
    "corrupted": {
        "text_conditioning": {
            "clean_probability": 0.0,
            "corrupted_probability": 1.0,
            "generated_probability": 0.0,
        }
    },
    "mixed": {
        "text_conditioning": {
            "clean_probability": 0.50,
            "corrupted_probability": 0.25,
            "generated_probability": 0.25,
        }
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument(
        "--diagnostic-experiment-config",
        default=None,
        help="When provided, resample and evaluate O1/O4 after every training run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.config)
    with base_path.open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    base_output = Path(base["paths"]["output_dir"])
    prepared_data_dir = base_output / "data"
    if not prepared_data_dir.exists():
        raise FileNotFoundError(
            f"Missing shared prepared data/tokenizer: {prepared_data_dir}. "
            "Train or prepare the base configuration first."
        )
    run_root = unique_run_root(
        args.output_root, "hierarchical_condition_training_ablation"
    )
    experiment_manifest = {
        "git_commit": current_git_commit(),
        "base_config": str(base_path),
        "base_config_sha256": file_sha256(base_path),
        "prepared_data_dir": str(prepared_data_dir),
        "variants": list(args.variants),
        "seeds": list(args.seeds),
        "runs": [],
    }
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = run_root / f"{variant}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
            resolved = copy.deepcopy(base)
            resolved.setdefault("paths", {})["output_dir"] = str(run_dir)
            resolved["paths"]["prepared_data_dir"] = str(prepared_data_dir)
            training = resolved.setdefault("training", {})
            training.update(copy.deepcopy(VARIANTS[variant]))
            training["seed"] = int(seed)
            training["mask_padding_in_attention"] = True
            corruption = training["text_conditioning"].setdefault(
                "corruption", {}
            )
            corruption.setdefault(
                "replacement_probability", 0.80
            )
            corruption.setdefault(
                "mask_probability", 0.10
            )
            corruption.setdefault(
                "length_bucket_step_probability", 0.50
            )
            training.setdefault("modality_gradient_audit_interval", 200)
            if args.epochs is not None:
                training["epochs"] = int(args.epochs)
            if args.max_train_batches is not None:
                training["max_train_batches"] = int(args.max_train_batches)
            if args.max_valid_batches is not None:
                training["max_valid_batches"] = int(args.max_valid_batches)
            resolved.setdefault("loss_group_weights", {
                "structured": 1.0,
                "summary": 1.0,
                "review": 1.0,
                "auxiliary": 1.0,
            })
            resolved.setdefault(
                "loss_groups",
                {
                    "field_groups": infer_field_loss_groups(resolved),
                },
            )
            config_path = run_dir / "training_config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(resolved, handle, sort_keys=False)
            record = {
                "variant": variant,
                "seed": int(seed),
                "config_path": str(config_path),
                "config_sha256": file_sha256(config_path),
                "status": "planned" if args.dry_run else "running",
            }
            experiment_manifest["runs"].append(record)
            write_json(run_root / "experiment_manifest.json", experiment_manifest)
            if args.dry_run:
                continue
            started = time.perf_counter()
            checkpoint = train_hierarchical_from_config(
                load_config(config_path), device=args.device
            )
            record.update(
                {
                    "status": "completed",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "training_seconds": float(time.perf_counter() - started),
                }
            )
            if args.diagnostic_experiment_config:
                diagnostics_root = run_checkpoint_diagnostics(
                    template_path=Path(
                        args.diagnostic_experiment_config
                    ),
                    model_config_path=config_path,
                    checkpoint_path=Path(checkpoint),
                    output_dir=run_dir / "diagnostics",
                    seed=int(seed),
                    device=args.device,
                )
                record["diagnostics_root"] = str(diagnostics_root)
            write_json(run_root / "experiment_manifest.json", experiment_manifest)
    if not args.dry_run:
        write_ablation_comparison(experiment_manifest["runs"], run_root)
    print(json.dumps(experiment_manifest, indent=2, sort_keys=True))


def infer_field_loss_groups(config: dict[str, Any]) -> dict[str, str]:
    targets = (config.get("columns") or {}).get("target") or {}
    auxiliary = set(
        ((config.get("auxiliary_targets") or {}).get("categorical") or [])
    )
    text_fields = list(targets.get("text") or [])
    field_groups: dict[str, str] = {}
    for field in targets.get("categorical") or []:
        field_groups[str(field)] = "structured"
    for field in auxiliary:
        field_groups[str(field)] = "auxiliary"
    for index, field in enumerate(text_fields):
        field_groups[str(field)] = (
            "summary" if index == 0 else "review"
        )
    return field_groups


if __name__ == "__main__":
    main()
