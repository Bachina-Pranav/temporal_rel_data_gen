from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_compact_finalization_gpt_report import build_report  # noqa: E402


def test_build_report_combines_decisions_and_sanitizes_nonfinite(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "evaluator_seed": 42,
                "maximum_new_training_runs": 3,
                "temporal_lambdas": [0.1, 0.25],
                "support_head": {
                    "mode": "support_prior",
                    "global_prior": {
                        "alpha": 1.0,
                        "residual_weight": 0.25,
                        "residual_temperature": 1.0,
                        "temporal_prior": {},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    payloads = {
        "final_architecture.json": {
            "freeze": True,
            "seed": 42,
            "numerical_architecture": "prior-residual support",
            "categorical_architecture": "original",
            "temporal_prior_lambda": 0.0,
            "checks": {"validity": True},
            "results": [{"full_row_c2st": float("nan")}],
        },
        "architecture_lock.json": {"status": "compact_validation_locked"},
        "training_plan.json": {
            "executed_new_training_runs": ["Rel-HM TEMP_010 seed 42"]
        },
        "categorical_selection.json": {"reason": "Original was simpler."},
        "temporal_selection.json": {"reason": "No candidate passed."},
    }
    for name, value in payloads.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    for name in (
        "compact_validation_table.md",
        "final_cross_dataset_results.md",
        "validity_audit.md",
    ):
        (root / name).write_text(f"# {name}\n\nEvidence\n", encoding="utf-8")

    report = build_report(root, config)

    assert "Architecture frozen: **YES**" in report
    assert "Rel-HM TEMP_010 seed 42" in report
    assert '"full_row_c2st": null' in report
    assert "## Final Cross-Dataset Test Results" in report
