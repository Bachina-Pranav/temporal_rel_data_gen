#!/usr/bin/env python3
"""Build one GPT-ready report from compact architecture-finalization outputs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = "outputs/architecture_finalization_compact"
DEFAULT_CONFIG = (
    "configs/experiments/lstm_architecture_finalization_compact.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=DEFAULT_ROOT)
    parser.add_argument("--experiment-config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        help="Defaults to <experiment-root>/gpt_analysis_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_root)
    config_path = Path(args.experiment_config)
    output = (
        Path(args.output)
        if args.output
        else root / "gpt_analysis_report.md"
    )
    report = build_report(root, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(output)


def build_report(root: Path, config_path: Path) -> str:
    required = {
        "final decision": root / "final_architecture.json",
        "architecture lock": root / "architecture_lock.json",
        "training plan": root / "training_plan.json",
        "categorical selection": root / "categorical_selection.json",
        "temporal selection": root / "temporal_selection.json",
        "validation table": root / "compact_validation_table.md",
        "cross-dataset results": root / "final_cross_dataset_results.md",
        "validity audit": root / "validity_audit.md",
        "experiment config": config_path,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot build GPT report; required artifacts are missing:\n- "
            + "\n- ".join(missing)
        )

    decision = read_json(required["final decision"])
    lock = read_json(required["architecture lock"])
    plan = read_json(required["training plan"])
    categorical = read_json(required["categorical selection"])
    temporal = read_json(required["temporal selection"])
    config = yaml.safe_load(required["experiment config"].read_text(encoding="utf-8"))
    executed = plan.get("executed_new_training_runs") or []
    freeze = bool(decision.get("freeze"))
    support = config.get("support_head") or {}
    global_prior = support.get("global_prior") or {}
    temporal_prior = global_prior.get("temporal_prior") or {}

    lines = [
        "# GPT Analysis Packet: Final LSTM Attribute-Generator Architecture",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Repository commit: `{git_commit()}`",
        f"Experiment root: `{root}`",
        "",
        "## Requested Analysis",
        "",
        "Analyze whether the architecture-freeze decision is supported by the reported evidence. "
        "Pay particular attention to the temporal-prior rejection, cross-dataset transfer, "
        "remaining Rel-HM full-row discrimination, metric tradeoffs, and limitations of the "
        "single-generator-seed compact finalization. Do not treat NA metrics as zero or as wins.",
        "",
        "## Executive Decision",
        "",
        f"- Architecture frozen: **{'YES' if freeze else 'NO'}**",
        f"- Blocking reason: {decision.get('blocking_reason') or 'none'}",
        f"- Numerical architecture: **{decision.get('numerical_architecture')}**",
        f"- Support prior alpha: `{decision.get('support_prior_alpha')}`",
        f"- Residual weight gamma: `{decision.get('support_residual_weight')}`",
        f"- Numerical sampling temperature: `{decision.get('support_sampling_temperature')}`",
        f"- Categorical architecture: **{decision.get('categorical_architecture')}**",
        f"- Temporal-prior lambda: **{decision.get('temporal_prior_lambda')}**",
        f"- Primary remaining weakness: {decision.get('primary_remaining_weakness')}",
        "",
        "## Experiment Scope",
        "",
        f"- Generator seed: `{decision.get('seed')}`; evaluator seed: `{config.get('evaluator_seed')}`.",
        "- This was a compact, decision-only finalization, not a new multi-seed benchmark.",
        "- Validation evidence selected the architecture; test metrics were not used for selection.",
        f"- Maximum authorized new training runs: `{config.get('maximum_new_training_runs')}`.",
        f"- Actual new training runs: `{len(executed)}`.",
    ]
    lines.extend(f"  - {item}" for item in executed)
    lines.extend(
        [
            "- Compatible Rel-HM controls and Amazon/MovieLens transfer results were reused without retraining.",
            f"- Tested temporal lambdas: `{config.get('temporal_lambdas')}`.",
            "- The final architecture uses past-only relational context and contains no dataset-specific generation logic.",
            "",
            "## Architecture Tested",
            "",
            "The support head combines a smoothed empirical support prior with a learned residual:",
            "",
            "```text",
            "logit_k(x,t) = log(p_mix(v_k | b) + epsilon) + gamma * delta_k(x,t)",
            "```",
            "",
            f"- Routing mode: `{support.get('mode')}`",
            f"- Direct-support maximum: `{support.get('direct_support_max_values')}` values",
            f"- Hierarchical support bins: `{support.get('hierarchical_num_bins')}`",
            f"- Prior alpha: `{global_prior.get('alpha')}`",
            f"- Residual gamma: `{global_prior.get('residual_weight')}`",
            f"- Residual temperature: `{global_prior.get('residual_temperature')}`",
            f"- Temporal buckets: `{temporal_prior.get('num_time_buckets')}`",
            f"- Temporal backoff strength: `{temporal_prior.get('backoff_strength')}`",
            "- Final categorical head: original learned categorical head, without categorical-prior anchoring.",
            "- Final temporal prior: disabled because neither temporal candidate passed every validation gate.",
            "",
            "## Metric Interpretation",
            "",
            "- `full_row_c2st`, `numerical_only_c2st`, `categorical_only_c2st`, and "
            "`text_embedding_c2st`: lower is better; 0 means chance-level discrimination under "
            "the repository's normalized C2ST-error definition.",
            "- `shape_error`, `trend_error`, `support_tv`, `support_js`, conditional errors, and "
            "invalid rates: lower is better; 0 is ideal.",
            "- `constraint_violation`: lower is better; 0 is ideal.",
            "- `fk_similarity`: higher is better; 1 is ideal.",
            "- `rows_per_second`: higher is better. Training and sampling seconds are lower-is-better runtime measures.",
            "- `NA` means the metric is not applicable or unavailable and must not be interpreted as zero.",
            "",
            "## Validation Selection Evidence",
            "",
            strip_leading_title(required["validation table"].read_text(encoding="utf-8")),
            "",
            "## Final Cross-Dataset Test Results",
            "",
            strip_leading_title(required["cross-dataset results"].read_text(encoding="utf-8")),
            "",
            "## Freeze Checks",
            "",
        ]
    )
    checks = decision.get("checks") or {}
    if checks:
        for name, passed in checks.items():
            lines.append(f"- `{name}`: **{'PASS' if passed else 'FAIL'}**")
    else:
        lines.append("- No individual freeze checks were recorded.")
    lines.extend(
        [
            "",
            "## Validity Audit",
            "",
            strip_leading_title(required["validity audit"].read_text(encoding="utf-8")),
            "",
            "## Selection Rationale",
            "",
            f"- Categorical decision: {categorical.get('reason')}",
            f"- Temporal decision: {temporal.get('reason')}",
            "- The temporal decision defaults to lambda 0 unless a candidate improves trend by the configured "
            "minimum while remaining inside full-row and numerical C2ST tolerances and preserving support/shape.",
            "- Since lambda 0 remained selected, no conditional MovieLens temporal-prior retraining was needed.",
            "",
            "## Important Limitations",
            "",
            "- The compact finalization used one generator seed to minimize runtime. It resolves the architecture "
            "decision using prior completed evidence, but it does not estimate fresh multi-seed uncertainty.",
            "- Some final rows are reused compatible runs, so runtime fields may describe their original executions.",
            "- Amazon-Toy FINAL equals M2 because the selected architecture matched the compatible reused original-head run.",
            "- Rel-HM full-row C2ST improved strongly but remains the largest reported final error.",
            "",
            "## Machine-Readable Decision Evidence",
            "",
            "### Architecture Lock",
            "",
            json_block(lock),
            "",
            "### Training Plan",
            "",
            json_block(plan),
            "",
            "### Categorical Selection",
            "",
            json_block(categorical),
            "",
            "### Temporal Selection",
            "",
            json_block(temporal),
            "",
            "### Final Decision",
            "",
            json_block(decision),
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for name, path in required.items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines).rstrip() + "\n"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_leading_title(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(sanitize(value), indent=2, sort_keys=True) + "\n```"


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    main()
