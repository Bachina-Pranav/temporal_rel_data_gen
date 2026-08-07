#!/usr/bin/env python3
"""Audit whether completed LSTM runs can be reused in a controlled comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


CONFIG_SECTIONS = (
    "columns",
    "schema",
    "generated_attributes",
    "continuous_numerical_fields",
    "count_numerical_fields",
    "numerical",
    "tokenizer",
    "id_encoding",
    "datetime_encoding",
    "model",
    "graph_conditioning",
    "text_decoder",
    "review_text_decoder",
    "summary_decoder",
    "text_length_prediction",
    "loss_weights",
)
TRAINING_FIELDS = (
    "epoch_mode",
    "max_steps",
    "epochs",
    "steps_per_eval",
    "steps_per_checkpoint",
    "validation_max_batches",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "sampling_mode",
    "train_row_sampling",
    "physical_batch_size",
    "batch_size",
    "target_effective_batch_size",
    "gradient_accumulation_steps",
    "lr",
    "learning_rate",
    "weight_decay",
    "mixed_precision",
    "amp_dtype",
    "gradient_clip_norm",
)
REQUIRED_SPLIT_FILES = (
    "train_real.csv",
    "validation_real.csv",
    "test_real.csv",
    "test_spine.csv",
    "history_prefix_spine.csv",
)
REQUIRED_RUN_FILES = (
    "checkpoints/best.pt",
    "samples/synthetic_interactions.csv",
    "evaluation/paper_grade/metrics.json",
    "evaluation/attribute_diagnostics.json",
    "training_metadata.json",
    "config_resolved.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument(
        "--c2st-source",
        default="src/evaluation/paper_metrics/c2st.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_comparability(
        Path(args.legacy_root),
        Path(args.candidate_root),
        seeds=[int(seed) for seed in args.seeds],
        c2st_source=Path(args.c2st_source),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(json.dumps(report, indent=2, sort_keys=True))


def audit_comparability(
    legacy_root: Path,
    candidate_root: Path,
    *,
    seeds: list[int],
    c2st_source: Path,
) -> dict[str, Any]:
    legacy = root_manifest(legacy_root, seeds, c2st_source)
    candidate = root_manifest(candidate_root, seeds, c2st_source)
    differing: dict[str, Any] = {}
    unavailable: list[str] = []

    compare_mapping(
        "split_fingerprints",
        legacy["split_fingerprints"],
        candidate["split_fingerprints"],
        differing,
        unavailable,
    )
    compare_mapping(
        "precomputed_split_fingerprints",
        legacy["precomputed_split_fingerprints"],
        candidate["precomputed_split_fingerprints"],
        differing,
        unavailable,
    )
    compare_mapping(
        "configuration",
        legacy["configuration"],
        candidate["configuration"],
        differing,
        unavailable,
    )
    compare_mapping(
        "c2st",
        legacy["c2st"],
        candidate["c2st"],
        differing,
        unavailable,
    )
    compare_mapping(
        "controlled_source_fingerprints",
        legacy["controlled_source_fingerprints"],
        candidate["controlled_source_fingerprints"],
        differing,
        unavailable,
    )
    for side, manifest in (
        ("legacy", legacy),
        ("candidate", candidate),
    ):
        if "source_sha256" not in manifest["c2st"]:
            unavailable.append(f"{side}.c2st.source_sha256")
        if not manifest["precomputed_split_fingerprints"]:
            unavailable.append(
                f"{side}.precomputed_split_fingerprints"
            )
        if not manifest["controlled_source_fingerprints"]:
            unavailable.append(
                f"{side}.controlled_source_fingerprints"
            )

    reusable_legacy = [
        int(seed)
        for seed in seeds
        if legacy["runs"][str(seed)]["complete"]
    ]
    reusable_candidate = [
        int(seed)
        for seed in seeds
        if candidate["runs"][str(seed)]["complete"]
    ]
    comparable = not differing and not unavailable
    reused_runs = (
        sorted(set(reusable_legacy + reusable_candidate))
        if comparable
        else reusable_candidate
    )
    required = sorted(set(seeds).difference(reused_runs))
    reasons = []
    if differing:
        reasons.append(
            "One or more required fingerprints or controlled settings differ."
        )
    if unavailable:
        reasons.append(
            "Full comparability cannot be established because required "
            "historical fingerprints are unavailable."
        )
    if not reasons:
        reasons.append(
            "All required fingerprints and controlled settings match."
        )
    return {
        "comparable": bool(comparable),
        "reasons": reasons,
        "differing_fields": differing,
        "unavailable_fields": sorted(set(unavailable)),
        "reused_runs": reused_runs,
        "required_new_runs": required,
        "legacy_root": str(legacy_root),
        "candidate_root": str(candidate_root),
        "legacy_manifest": legacy,
        "candidate_manifest": candidate,
        "policy": (
            "A missing required fingerprint is treated as not comparable; "
            "the audit never assumes equality from mutable paths."
        ),
    }


def root_manifest(
    root: Path,
    seeds: list[int],
    c2st_source: Path,
) -> dict[str, Any]:
    shared = resolve_spines(root)
    split_fingerprints = {
        name: file_sha256(shared / name)
        for name in REQUIRED_SPLIT_FILES
        if (shared / name).exists()
    }
    run_manifests = {
        str(seed): run_manifest(root / "runs" / f"seed_{seed}")
        for seed in seeds
    }
    representative = next(
        (
            root / "runs" / f"seed_{seed}" / "config_resolved.yaml"
            for seed in seeds
            if (
                root / "runs" / f"seed_{seed}" / "config_resolved.yaml"
            ).exists()
        ),
        None,
    )
    config = load_yaml(representative) if representative else {}
    persisted = load_json_optional(
        root / "shared" / "comparability_manifest.json"
    )
    evaluation_hash = representative_evaluation_hash(root, seeds)
    c2st = {}
    if persisted.get("c2st_source_sha256"):
        c2st["source_sha256"] = persisted["c2st_source_sha256"]
    if evaluation_hash:
        c2st["evaluation_config_sha256"] = evaluation_hash
    return {
        "root": str(root),
        "split_directory": str(shared),
        "split_fingerprints": split_fingerprints,
        "precomputed_split_fingerprints": (
            persisted.get("precomputed_split_fingerprints", {})
            if persisted
            else {}
        ),
        "configuration": controlled_configuration(config),
        "c2st": c2st,
        "controlled_source_fingerprints": (
            persisted.get("controlled_source_fingerprints", {})
            if persisted
            else {}
        ),
        "git_commit": (
            persisted.get("git_commit")
            if persisted
            else repository_commit(root)
        ),
        "runs": run_manifests,
        "persisted_comparability_manifest": bool(persisted),
    }


def controlled_configuration(config: dict[str, Any]) -> dict[str, Any]:
    training = dict(config.get("training") or {})
    return {
        **{
            section: config.get(section)
            for section in CONFIG_SECTIONS
            if section in config
        },
        "training": {
            field: training.get(field)
            for field in TRAINING_FIELDS
            if field in training
        },
        "numerical_metadata": config.get("_numerical_metadata"),
        "numerical_head_metadata": config.get(
            "_numerical_head_metadata"
        ),
    }


def run_manifest(run_root: Path) -> dict[str, Any]:
    missing = [
        str(run_root / relative)
        for relative in REQUIRED_RUN_FILES
        if not (run_root / relative).exists()
    ]
    return {
        "complete": not missing,
        "missing": missing,
        "checkpoint_sha256": file_sha256(
            run_root / "checkpoints" / "best.pt"
        ),
        "synthetic_sha256": file_sha256(
            run_root / "samples" / "synthetic_interactions.csv"
        ),
    }


def compare_mapping(
    name: str,
    left: dict[str, Any],
    right: dict[str, Any],
    differing: dict[str, Any],
    unavailable: list[str],
) -> None:
    if not left or not right:
        unavailable.append(name)
        return
    keys = sorted(set(left) | set(right))
    for key in keys:
        path = f"{name}.{key}"
        if key not in left or key not in right:
            unavailable.append(path)
        elif canonical(left[key]) != canonical(right[key]):
            differing[path] = {
                "legacy": left[key],
                "candidate": right[key],
            }


def resolve_spines(root: Path) -> Path:
    preferred = root / "shared" / "spines"
    return preferred if preferred.exists() else root / "shared"


def representative_evaluation_hash(
    root: Path,
    seeds: list[int],
) -> str | None:
    for seed in seeds:
        path = (
            root
            / "runs"
            / f"seed_{seed}"
            / "evaluation_config_resolved.yaml"
        )
        if path.exists():
            controlled = strip_runtime_fields(load_yaml(path))
            return hashlib.sha256(
                canonical(controlled).encode("utf-8")
            ).hexdigest()
    return None


def strip_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_runtime_fields(item)
            for key, item in value.items()
            if key not in {"seed", "output_dir", "output_path"}
            and not key.endswith("_path")
            and not key.endswith("_table")
        }
    if isinstance(value, list):
        return [strip_runtime_fields(item) for item in value]
    return value


def repository_commit(root: Path) -> str | None:
    inventory = load_json_optional(
        root / "shared" / "repository_inventory.json"
    )
    return inventory.get("git_commit") if inventory else None


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def load_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
