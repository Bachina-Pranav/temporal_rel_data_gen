#!/usr/bin/env python3
"""Gated M1-M4 numerical-head experiments built on the existing LSTM runner."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default=(
            "configs/experiments/"
            "lstm_numerical_heads_hm_10k.yaml"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=[
            "write-configs",
            "calibration",
            "baseline-eval",
            "smoke",
            "one-seed",
            "full",
            "summarize",
        ],
        required=True,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Defaults to all M1-M4 variants.",
    )
    parser.add_argument(
        "--promote-variants",
        nargs="+",
        default=None,
        help="Required for --stage full after reviewing one-seed results.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-batch-size", default="8192")
    parser.add_argument("--smoke-rows", type=int, default=256)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--rebuild-precomputed",
        action="store_true",
        help=(
            "Rebuild stale shared pretokenized and neighbor caches. "
            "The first variant rebuilds them; later variants reuse them."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--real-numerical-oracle-c2st",
        type=float,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_yaml(args.experiment_config)
    base = load_yaml(matrix["base_config"])
    output_root = Path(matrix["output_root"])
    config_dir = output_root / "variant_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    variants = resolve_variants(matrix, args.variants)
    config_paths = write_variant_configs(
        matrix,
        base,
        variants,
        config_dir,
    )
    if args.stage == "write-configs":
        print_paths(config_paths)
        return
    if args.stage == "calibration":
        calibration_root = matrix.get(
            "calibration_baseline_experiment_root",
            matrix["baseline_experiment_root"],
        )
        command = [
            python(),
            "src/scripts/run_lstm_numerical_calibration_diagnostics.py",
            "--experiment-root",
            str(calibration_root),
            "--output-dir",
            str(output_root / "calibration_q0_q4"),
            "--seeds",
            *[str(seed) for seed in matrix["seeds"]],
        ]
        if args.real_numerical_oracle_c2st is not None:
            command.extend(
                [
                    "--real-numerical-oracle-c2st",
                    str(args.real_numerical_oracle_c2st),
                ]
            )
        run(command, args)
        return
    if args.stage == "baseline-eval":
        refresh_baseline_evaluation(matrix, args)
        return
    if args.stage == "summarize":
        run(
            [
                python(),
                "src/scripts/"
                "summarize_lstm_numerical_head_experiments.py",
                "--experiment-config",
                str(args.experiment_config),
            ],
            args,
        )
        return
    if args.stage == "full":
        if not args.promote_variants:
            raise SystemExit(
                "--stage full requires --promote-variants after the "
                "smoke and one-seed decision gate"
            )
        variants = resolve_variants(
            matrix,
            args.promote_variants,
        )
    seeds = (
        [int(matrix["first_seed"])]
        if args.stage == "one-seed"
        else [int(seed) for seed in matrix["seeds"]]
    )
    for name in variants:
        variant_root = output_root / name
        command = [
            python(),
            "src/scripts/run_lstm_multiseed_experiment.py",
            "--config",
            str(config_paths[name]),
            "--evaluation-config",
            str(matrix["evaluation_config"]),
            "--output-root",
            str(variant_root),
            "--pretokenized-dir",
            str(matrix["pretokenized_dir"]),
            "--neighbor-cache-dir",
            str(matrix["neighbor_cache_dir"]),
            "--device",
            args.device,
            "--sample-batch-size",
            str(args.sample_batch_size),
            "--smoke-rows",
            str(args.smoke_rows),
            "--seeds",
            *[str(seed) for seed in seeds],
        ]
        if args.stage == "smoke":
            command.append("--smoke-only")
        else:
            command.append("--skip-smoke")
        if args.skip_existing:
            command.append("--skip-existing")
        if args.rebuild_precomputed:
            command.append("--rebuild-precomputed")
        if args.dry_run:
            command.append("--dry-run")
        run(command, args)
        if args.stage != "smoke" and not args.dry_run:
            for seed in seeds:
                run_context_diagnostic(
                    name,
                    variant_root,
                    seed,
                    matrix,
                    args,
                )
    write_manifest(
        matrix,
        variants,
        args.stage,
        output_root,
    )
    if args.stage in {"one-seed", "full"} and not args.dry_run:
        run(
            [
                python(),
                "src/scripts/"
                "summarize_lstm_numerical_head_experiments.py",
                "--experiment-config",
                str(args.experiment_config),
            ],
            args,
        )


def write_variant_configs(
    matrix: dict[str, Any],
    base: dict[str, Any],
    variants: list[str],
    config_dir: Path,
) -> dict[str, Path]:
    shared = matrix.get("shared_numerical_head") or {}
    paths: dict[str, Path] = {}
    for name in variants:
        definition = matrix["variants"][name]
        resolved = copy.deepcopy(base)
        numerical_heads = copy.deepcopy(shared)
        deep_update(
            numerical_heads,
            definition.get("numerical_heads") or {},
        )
        resolved["numerical_heads"] = numerical_heads
        resolved["experiment_name"] = name
        resolved.setdefault("experiment_metadata", {}).update(
            {
                "numerical_head_variant": name,
                "description": definition.get("description"),
                "baseline_architecture_changed": True,
                "test_data_used_for_head_or_prior": False,
            }
        )
        path = config_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved, handle, sort_keys=False)
        paths[name] = path
    return paths


def run_context_diagnostic(
    name: str,
    root: Path,
    seed: int,
    matrix: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    run_root = root / "runs" / f"seed_{seed}"
    run(
        [
            python(),
            "src/scripts/diagnose_lstm_numerical_context_usage.py",
            "--checkpoint",
            str(run_root / "checkpoints" / "best.pt"),
            "--synthetic-spine",
            str(root / "shared" / "spines" / "test_spine.csv"),
            "--graph-history-prefix",
            str(
                root
                / "shared"
                / "spines"
                / "history_prefix_spine.csv"
            ),
            "--evaluation-real",
            str(root / "shared" / "spines" / "test_real.csv"),
            "--output",
            str(
                run_root
                / "evaluation"
                / "numerical_context_usage.json"
            ),
            "--num-rows",
            "2048",
            "--device",
            args.device,
            "--seed",
            str(seed),
        ],
        args,
    )


def refresh_baseline_evaluation(
    matrix: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    root = Path(matrix["baseline_experiment_root"])
    shared = root / "shared" / "spines"
    for seed in matrix["seeds"]:
        run_root = root / "runs" / f"seed_{int(seed)}"
        config = run_root / "config_resolved.yaml"
        evaluation_config = (
            run_root / "evaluation_config_resolved.yaml"
        )
        run(
            [
                python(),
                "src/scripts/evaluate_lstm_attribute_diagnostics.py",
                "--config",
                str(config),
                "--train-real",
                str(shared / "train_real.csv"),
                "--evaluation-real",
                str(shared / "test_real.csv"),
                "--synthetic",
                str(
                    run_root
                    / "samples"
                    / "synthetic_interactions.csv"
                ),
                "--graph-history-prefix",
                str(shared / "history_prefix_spine.csv"),
                "--evaluation-config",
                str(evaluation_config),
                "--output",
                str(
                    run_root
                    / "evaluation"
                    / "attribute_diagnostics_numerical_head_comparison.json"
                ),
                "--seed",
                str(int(seed)),
            ],
            args,
        )
        run_context_diagnostic(
            "M0_original_lstm_v53",
            root,
            int(seed),
            matrix,
            args,
        )
    if not args.dry_run:
        run(
            [
                python(),
                "src/scripts/"
                "summarize_lstm_numerical_head_experiments.py",
                "--experiment-config",
                str(args.experiment_config),
            ],
            args,
        )


def resolve_variants(
    matrix: dict[str, Any],
    requested: list[str] | None,
) -> list[str]:
    available = list(matrix.get("variants") or {})
    selected = list(requested or available)
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(
            f"Unknown variants {unknown}; available={available}"
        )
    return selected


def deep_update(
    destination: dict[str, Any],
    update: dict[str, Any],
) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(
            destination.get(key),
            dict,
        ):
            deep_update(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)


def run(command: list[str], args: argparse.Namespace) -> None:
    print("$ " + " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def write_manifest(
    matrix: dict[str, Any],
    variants: list[str],
    stage: str,
    output_root: Path,
) -> None:
    manifest = {
        "stage_completed": stage,
        "variants": variants,
        "baseline_experiment_root": matrix[
            "baseline_experiment_root"
        ],
        "full_three_seed_runs_require_explicit_promotion": True,
    }
    path = output_root / f"{stage}_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(path)


def print_paths(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        print(f"{name}: {path}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def python() -> str:
    return sys.executable


if __name__ == "__main__":
    main()
