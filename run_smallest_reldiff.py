#!/usr/bin/env python3
"""Run RelDiff on the smallest dataset from the paper.

The smallest dataset in the paper's table is Walmart by total row count.
This wrapper orchestrates the repository scripts so you can run:

    python run_smallest_reldiff.py --epochs 100

For a paper-scale run, use --epochs 10000.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = "walmart"
CONFIG = "src/reldiff/configs/reldiff_config.toml"
DATASET_CONFIG = "src/reldiff/configs/data/walmart.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download data, train RelDiff, and sample Walmart synthetic databases."
    )
    parser.add_argument("--dataset", default=DATASET, help="Dataset directory name.")
    parser.add_argument("--config-path", default=CONFIG)
    parser.add_argument("--dataset-config-path", default=DATASET_CONFIG)
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs. Use 10000 for the paper-scale setting.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--sampling-batch-size", type=int, default=20000)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--run-id", default="_smallest")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--sampling-device", default="cuda")
    parser.add_argument(
        "--structure",
        choices=["original", "generated", "2k"],
        default="original",
        help=(
            "original samples attributes on the real graph. generated/2k require "
            "precomputed structure files under data/structure."
        ),
    )
    parser.add_argument(
        "--make-structure",
        action="store_true",
        help=(
            "Create the requested generated/2k structure before sampling. "
            "The generated option requires graph-tool in the current environment."
        ),
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable online wandb logging.")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help=(
            "Do not stop early when CUDA is unavailable. The current upstream "
            "train/sample scripts still call .cuda(), so CPU runs may fail."
        ),
    )
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80, flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else src_path + os.pathsep + env["PYTHONPATH"]
    )
    return env


def ensure_cuda_or_explain(args: argparse.Namespace) -> None:
    if args.allow_no_cuda:
        return
    try:
        import torch
    except ImportError:
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. The upstream train/sample scripts currently "
            "instantiate the model with .cuda(), so use a GPU system or patch the "
            "scripts for CPU before running."
        )


def maybe_download(args: argparse.Namespace, env: dict[str, str]) -> None:
    metadata = ROOT / "data" / "original" / args.dataset / "metadata.json"
    if metadata.exists():
        print(f"Found dataset metadata at {metadata.relative_to(ROOT)}")
        return
    if args.skip_download:
        raise SystemExit(
            f"Missing {metadata.relative_to(ROOT)} and --skip-download was passed."
        )
    run([sys.executable, "reproducibility/download_data.py"], env)
    if not metadata.exists():
        raise SystemExit(
            f"Download finished, but {metadata.relative_to(ROOT)} was not found. "
            "Check the extracted dataset names under data/original."
        )


def structure_path(dataset: str, structure: str) -> Path:
    if structure == "generated":
        return ROOT / "data" / "structure" / f"{dataset}_graph_gen.pkl"
    if structure == "2k":
        return ROOT / "data" / "structure" / f"{dataset}_graph_2k.pkl"
    raise ValueError(f"Unsupported generated structure mode: {structure}")


def maybe_make_structure(args: argparse.Namespace, env: dict[str, str]) -> None:
    if args.structure == "original":
        return
    expected = structure_path(args.dataset, args.structure)
    if expected.exists():
        print(f"Found structure at {expected.relative_to(ROOT)}")
        return
    if not args.make_structure:
        raise SystemExit(
            f"Missing {expected.relative_to(ROOT)}. Re-run with --make-structure, "
            "or generate the structure separately."
        )

    run(
        [
            sys.executable,
            "src/structure/to_networkx.py",
            "--dataset_name",
            args.dataset,
            "--data-path",
            "data",
        ],
        env,
    )

    if args.structure == "generated":
        run(
            [
                sys.executable,
                "src/structure/generate_d2k_plus_sbm.py",
                "--dataset_name",
                args.dataset,
                "--data-dir",
                "data",
            ],
            env,
        )
        upstream_path = ROOT / "data" / "structure" / f"{args.dataset}_graph__gen.pkl"
        if upstream_path.exists() and not expected.exists():
            shutil.move(upstream_path, expected)
    else:
        run(
            [
                sys.executable,
                "src/structure/generate_bjdd.py",
                "--dataset_name",
                args.dataset,
                "--data-dir",
                "data",
            ],
            env,
        )

    if not expected.exists():
        raise SystemExit(f"Structure generation did not create {expected.relative_to(ROOT)}")


def train(args: argparse.Namespace, env: dict[str, str]) -> None:
    cmd = [
        sys.executable,
        "src/scripts/train_joint_diffusion.py",
        args.dataset,
        "--num-epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--sampling-batch-size",
        str(args.sampling_batch_size),
        "--run-id",
        args.run_id,
        "--config-path",
        args.config_path,
        "--dataset-config-path",
        args.dataset_config_path,
        "--device",
        args.device,
        "--sampling-device",
        args.sampling_device,
    ]
    if not args.wandb:
        cmd.append("--no-wandb")
    if args.compile_model:
        cmd.append("--compile-model")
    if args.use_ema:
        cmd.append("--use-ema")
    if args.mixed_precision:
        cmd.append("--mixed-precision")
    run(cmd, env)


def sample(args: argparse.Namespace, env: dict[str, str]) -> None:
    cmd = [
        sys.executable,
        "src/scripts/sample_joint_diffusion.py",
        args.dataset,
        "--num-samples",
        str(args.num_samples),
        "--sampling-batch-size",
        str(args.sampling_batch_size),
        "--run-id",
        args.run_id,
        "--config-path",
        args.config_path,
        "--dataset-config-path",
        args.dataset_config_path,
        "--structure",
        args.structure,
        "--device",
        args.device,
        "--sampling-device",
        args.sampling_device,
    ]
    if args.compile_model:
        cmd.append("--compile-model")
    if args.use_ema:
        cmd.append("--use-ema")
    run(cmd, env)


def main() -> None:
    args = parse_args()
    env = python_env()
    ensure_cuda_or_explain(args)
    maybe_download(args, env)

    if not args.skip_train:
        train(args, env)

    if not args.skip_sample:
        maybe_make_structure(args, env)
        sample(args, env)


if __name__ == "__main__":
    main()
