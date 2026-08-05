"""Shared orchestration for hierarchical diffusion training ablations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from attribute_generation.conditional_tabdlm.diffusion_diagnostics import (
    write_json,
)
from scripts.run_hierarchical_diffusion_diagnostics import (
    run_diagnostic_experiment,
)


def run_checkpoint_diagnostics(
    *,
    template_path: Path,
    model_config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
    modes: tuple[str, ...] = ("O1", "O4"),
) -> Path:
    with template_path.open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    if not isinstance(experiment, dict):
        raise ValueError(
            f"Expected a diagnostic experiment mapping in {template_path}"
        )
    resolved = copy.deepcopy(experiment)
    resolved["experiment_name"] = "checkpoint_condition_diagnostics"
    resolved["output_root"] = str(output_dir)
    resolved.setdefault("model", {})["config"] = str(model_config_path)
    resolved["model"]["checkpoint"] = str(checkpoint_path)
    resolved["seeds"] = [int(seed)]
    resolved["enabled_matrices"] = ["progressive_conditioning"]
    resolved.setdefault("matrices", {})[
        "progressive_conditioning"
    ] = list(modes)
    resolved_path = output_dir / "diagnostic_experiment.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    return run_diagnostic_experiment(
        resolved,
        experiment_config_path=resolved_path,
        device_override=device,
        seeds_override=[int(seed)],
        matrices_override=["progressive_conditioning"],
    )


def write_ablation_comparison(
    records: list[dict[str, Any]],
    output_root: Path,
) -> None:
    completed: list[pd.DataFrame] = []
    for record in records:
        diagnostics_root = record.get("diagnostics_root")
        if not diagnostics_root:
            continue
        path = Path(diagnostics_root) / "consolidated_results.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "training_variant", str(record["variant"]))
        frame["training_seed"] = int(record["seed"])
        frame["training_seconds"] = record.get("training_seconds")
        completed.append(frame)
    if not completed:
        return
    comparison = pd.concat(completed, ignore_index=True, sort=False)
    comparison.to_csv(output_root / "ablation_results.csv", index=False)
    write_json(
        output_root / "ablation_results.json",
        comparison.where(pd.notna(comparison), None).to_dict(
            orient="records"
        ),
    )
    numeric = [
        column
        for column in comparison.select_dtypes(include="number").columns
        if column not in {"seed", "training_seed"}
    ]
    rows: list[dict[str, Any]] = []
    for (variant, label), group in comparison.groupby(
        ["training_variant", "label"], dropna=False
    ):
        row: dict[str, Any] = {
            "training_variant": variant,
            "label": label,
            "num_seeds": int(group["training_seed"].nunique()),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_mean"] = (
                float(values.mean()) if len(values) else None
            )
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(
        output_root / "ablation_aggregate_mean_std.csv", index=False
    )
    report = [
        "# Hierarchical Training Ablation",
        "",
        "O1 is an oracle diagnostic upper bound; O4 is the valid generated pipeline.",
        "",
        dataframe_markdown(aggregate),
        "",
        "Select a setting using the complete fidelity, constraint, robustness, and runtime profile rather than one metric.",
    ]
    (output_root / "ablation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def dataframe_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No completed diagnostic evaluations._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"
