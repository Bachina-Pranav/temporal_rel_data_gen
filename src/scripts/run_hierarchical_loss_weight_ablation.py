#!/usr/bin/env python3
"""Train and evaluate configurable hierarchical diffusion loss-weight variants."""

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
from scripts.run_hierarchical_condition_training_ablation import (  # noqa: E402
    infer_field_loss_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument("--diagnostic-experiment-config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.config)
    variants_path = Path(args.variants_config)
    base = load_yaml_mapping(base_path)
    variants_document = load_yaml_mapping(variants_path)
    variants = variants_document.get("variants") or {}
    if not isinstance(variants, dict) or not variants:
        raise ValueError("Loss-weight variants config must define a nonempty variants mapping")
    prepared_data_dir = Path(base["paths"]["output_dir"]) / "data"
    if not prepared_data_dir.exists():
        raise FileNotFoundError(
            f"Missing shared prepared data/tokenizer: {prepared_data_dir}"
        )
    run_root = unique_run_root(
        args.output_root, "hierarchical_loss_weight_ablation"
    )
    manifest: dict[str, Any] = {
        "git_commit": current_git_commit(),
        "base_config": str(base_path),
        "base_config_sha256": file_sha256(base_path),
        "variants_config": str(variants_path),
        "variants_config_sha256": file_sha256(variants_path),
        "prepared_data_dir": str(prepared_data_dir),
        "seeds": list(args.seeds),
        "runs": [],
    }
    for variant, variant_config in variants.items():
        if not isinstance(variant_config, dict):
            raise ValueError(f"Loss variant {variant!r} must be a mapping")
        for seed in args.seeds:
            run_dir = run_root / f"{variant}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
            resolved = copy.deepcopy(base)
            resolved.setdefault("paths", {})["output_dir"] = str(run_dir)
            resolved["paths"]["prepared_data_dir"] = str(prepared_data_dir)
            training = resolved.setdefault("training", {})
            training["seed"] = int(seed)
            training["mask_padding_in_attention"] = True
            training.setdefault("modality_gradient_audit_interval", 200)
            if args.epochs is not None:
                training["epochs"] = int(args.epochs)
            if args.max_train_batches is not None:
                training["max_train_batches"] = int(args.max_train_batches)
            if args.max_valid_batches is not None:
                training["max_valid_batches"] = int(args.max_valid_batches)
            resolved["loss_group_weights"] = dict(
                variant_config.get("loss_group_weights") or {}
            )
            resolved.setdefault("loss_groups", {})["field_groups"] = dict(
                variant_config.get("field_groups")
                or infer_field_loss_groups(resolved)
            )
            config_path = run_dir / "training_config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(resolved, handle, sort_keys=False)
            record: dict[str, Any] = {
                "variant": str(variant),
                "seed": int(seed),
                "config_path": str(config_path),
                "config_sha256": file_sha256(config_path),
                "status": "planned" if args.dry_run else "running",
            }
            manifest["runs"].append(record)
            write_json(run_root / "experiment_manifest.json", manifest)
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
                    "training_seconds": float(
                        time.perf_counter() - started
                    ),
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
            write_json(run_root / "experiment_manifest.json", manifest)
    if not args.dry_run:
        write_ablation_comparison(manifest["runs"], run_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


if __name__ == "__main__":
    main()
