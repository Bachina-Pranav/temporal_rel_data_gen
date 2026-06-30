#!/usr/bin/env python3
"""Run RelDiff on the smallest dataset from the paper.

The smallest dataset in the paper's table is Walmart by total row count.
This wrapper orchestrates the repository scripts so you can run:

    python run_smallest_reldiff.py --epochs 100

For a paper-scale run, use --epochs 10000.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
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
    parser.add_argument(
        "--skip-metadata-fix",
        action="store_true",
        help=(
            "Do not patch downloaded SDV metadata. Newer SDV versions reject "
            "integer ID columns that still carry regex_format constraints."
        ),
    )
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
    except ImportError as exc:
        raise SystemExit(
            "PyTorch failed to import before training started:\n"
            f"    {exc}\n\n"
            "For the common libtorch_cpu.so iJIT_NotifyEvent error, run:\n"
            "    conda install -y 'mkl<2024.1' 'intel-openmp<2024.1'\n"
            "Then re-run this script."
        ) from exc
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. The upstream train/sample scripts currently "
            "instantiate the model with .cuda(), so use a GPU system or patch the "
            "scripts for CPU before running."
        )


def _pyg_wheel_url(torch_module) -> str:
    torch_version = torch_module.__version__.split("+", maxsplit=1)[0]
    cuda_version = torch_module.version.cuda
    cuda_tag = "cpu"
    if cuda_version:
        cuda_tag = "cu" + "".join(cuda_version.split(".")[:2])
    return f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"


def ensure_pyg_sampler_backend_or_explain() -> None:
    errors: list[str] = []
    for module_name in ("pyg_lib", "torch_sparse"):
        try:
            importlib.import_module(module_name)
            return
        except Exception as exc:  # ImportError, ABI mismatch, or missing CUDA symbols.
            errors.append(f"{module_name}: {exc}")

    import torch

    wheel_url = _pyg_wheel_url(torch)
    raise SystemExit(
        "PyG NeighborLoader requires either pyg_lib or torch_sparse, but neither "
        "backend could be imported.\n\n"
        "Install the PyG extension wheels that match your current Torch/CUDA build:\n"
        f"    pip install --no-index pyg_lib torch_scatter torch_sparse -f {wheel_url}\n\n"
        "Then verify and re-run:\n"
        "    python -c \"import pyg_lib, torch_sparse; print('PyG sampling backend ok')\"\n"
        "    python run_smallest_reldiff.py --epochs 100 --num-samples 1\n\n"
        "Import attempts:\n"
        + "\n".join(f"    {error}" for error in errors)
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
    try:
        import gdown  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "The downloader requires gdown. Install the reproducibility extras with:\n"
            "    pip install -r reproducibility/requirements.txt\n"
            "or just:\n"
            "    pip install gdown\n"
            "Then re-run this command."
        ) from exc
    run([sys.executable, "reproducibility/download_data.py"], env)
    if not metadata.exists():
        raise SystemExit(
            f"Download finished, but {metadata.relative_to(ROOT)} was not found. "
            "Check the extracted dataset names under data/original."
        )


def _is_integer_like_csv_column(csv_path: Path, column: str) -> bool:
    seen_value = False
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            return False
        for row in reader:
            value = row.get(column, "").strip()
            if value == "":
                continue
            seen_value = True
            if value.startswith(("+", "-")):
                value = value[1:]
            if not value.isdigit():
                return False
    return seen_value


def fix_integer_id_metadata_regexes(args: argparse.Namespace) -> None:
    """Remove stale regex constraints that newer SDV rejects for integer IDs."""
    if args.skip_metadata_fix:
        return

    dataset_dir = ROOT / "data" / "original" / args.dataset
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return

    with metadata_path.open() as handle:
        metadata = json.load(handle)

    changed: list[str] = []
    for table_name, table_metadata in metadata.get("tables", {}).items():
        csv_path = dataset_dir / f"{table_name}.csv"
        if not csv_path.exists():
            continue
        for column_name, column_metadata in table_metadata.get("columns", {}).items():
            if column_metadata.get("sdtype") != "id":
                continue
            regex_keys = [
                key for key in ("regex_format", "regex") if key in column_metadata
            ]
            if not regex_keys:
                continue
            if not _is_integer_like_csv_column(csv_path, column_name):
                continue
            for key in regex_keys:
                column_metadata.pop(key, None)
            changed.append(f"{table_name}.{column_name}")

    if not changed:
        return

    backup_path = metadata_path.with_suffix(".json.bak")
    if not backup_path.exists():
        shutil.copyfile(metadata_path, backup_path)
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=4)
        handle.write("\n")
    print(
        "Removed SDV regex constraints from integer ID columns: "
        + ", ".join(changed)
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
    ensure_pyg_sampler_backend_or_explain()
    maybe_download(args, env)
    fix_integer_id_metadata_regexes(args)

    if not args.skip_train:
        train(args, env)

    if not args.skip_sample:
        maybe_make_structure(args, env)
        sample(args, env)


if __name__ == "__main__":
    main()
