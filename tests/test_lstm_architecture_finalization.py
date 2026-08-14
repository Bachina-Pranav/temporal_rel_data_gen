from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.categorical_head import (  # noqa: E402
    PriorAnchoredCategoricalHead,
    fit_categorical_head_metadata,
)
from attribute_generation.conditional_tabdlm.support_calibration import (  # noqa: E402
    support_calibration_metrics,
    support_probability_table,
)
from scripts.run_lstm_architecture_finalization import (  # noqa: E402
    evaluator_fingerprint,
    promote_schema_numeric_ordinals,
    require_lock,
    select_validation_winner,
    validation_comparability,
)
from scripts.summarize_lstm_architecture_finalization import (  # noqa: E402
    evaluator_comparability,
)
from scripts.summarize_lstm_architecture_for_llm import (  # noqa: E402
    build_summary,
    render_markdown,
)


class Vocab:
    size = 3

    def encode(self, value):
        return {"a": 0, "b": 1, "c": 2}.get(value, 2)


def test_categorical_prior_head_starts_at_training_distribution():
    head = PriorAnchoredCategoricalHead(
        4,
        3,
        {
            "enabled": True,
            "probabilities": [0.8, 0.15, 0.05],
            "alpha": 1.0,
            "residual_weight": 0.5,
            "residual_init_scale": 0.0,
        },
    )
    probability = torch.softmax(head(torch.zeros(5, 4)), dim=1)

    assert torch.allclose(
        probability,
        torch.tensor([[0.8, 0.15, 0.05]]).expand(5, -1),
        atol=1e-6,
    )


def test_categorical_prior_masks_categories_absent_from_training():
    head = PriorAnchoredCategoricalHead(
        2,
        3,
        {
            "enabled": True,
            "probabilities": [0.8, 0.2, 0.0],
            "residual_weight": 0.0,
        },
    )

    probability = torch.softmax(head(torch.zeros(1, 2)), dim=1)

    assert probability[0, 2].item() == 0.0


def test_categorical_prior_metadata_uses_only_generated_targets():
    raw = {
        "categorical_heads": {"prior": {"enabled": True}},
    }
    schema = SimpleNamespace(
        categorical_targets=("channel",),
        model_categorical_targets=("channel", "auxiliary"),
    )
    config = SimpleNamespace(raw=raw, schema=schema)
    frame = pd.DataFrame({"channel": ["a", "a", "b"]})

    metadata = fit_categorical_head_metadata(
        config,
        {"channel": Vocab(), "auxiliary": Vocab()},
        train_frame=frame,
        train_dataset=None,
    )

    assert metadata["columns"]["channel"]["enabled"] is True
    assert metadata["columns"]["auxiliary"]["enabled"] is False


def test_numeric_ordinal_promotion_is_schema_driven():
    raw = {
        "generated_attributes": {
            "score": {
                "semantic_type": "ordinal_categorical",
                "valid_domain": [0.5, 1.0, 1.5],
            },
            "label": {
                "semantic_type": "categorical",
                "valid_domain": ["low", "high"],
            },
        },
        "columns": {
            "target": {
                "categorical": ["score", "label"],
                "numerical": [],
                "text": [],
            }
        },
    }

    resolved = promote_schema_numeric_ordinals(raw)

    assert resolved["columns"]["target"]["numerical"] == ["score"]
    assert resolved["columns"]["target"]["categorical"] == ["label"]
    assert raw["columns"]["target"]["categorical"] == ["score", "label"]


def test_support_diagnostics_report_reverse_kl_and_invalid_support():
    table = support_probability_table(
        pd.Series([1.0, 1.0, 2.0]),
        pd.Series([1.0, 2.0]),
        pd.Series([1.0, 1.5, np.nan]),
    )

    metrics = support_calibration_metrics(table)

    assert metrics["kl_generated_to_train"] >= 0
    assert metrics["invalid_support_rate"] == pytest.approx(1 / 3)
    assert metrics["invalid_or_missing_rate"] == pytest.approx(2 / 3)


def test_test_stage_requires_fully_validation_frozen_lock(tmp_path: Path):
    with pytest.raises(RuntimeError, match="not validation-locked"):
        require_lock(tmp_path, require_fully_frozen=True)


def test_evaluator_hash_ignores_only_run_specific_table_paths(tmp_path: Path):
    source = ROOT / "configs/evaluation/single_event_table_paper_metrics_hm_10k_customers.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["real_table_path"] = "run/validation_real.csv"
    raw["synthetic_table_path"] = "run/synthetic.csv"
    resolved = tmp_path / "resolved.yaml"
    resolved.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert (
        evaluator_fingerprint(source)["evaluator_hash"]
        == evaluator_fingerprint(resolved)["evaluator_hash"]
    )
    raw["evaluation"]["max_rows_for_c2st"] = 123
    resolved.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert (
        evaluator_fingerprint(source)["evaluator_hash"]
        != evaluator_fingerprint(resolved)["evaluator_hash"]
    )


def test_validation_comparability_rejects_split_mismatch(tmp_path: Path):
    variants = {"A": {}, "B": {}}
    matrix = {"variants": variants}
    common_config = {
        "dataset": {"name": "example"},
        "event_spine": {"timestamp": "time"},
        "columns": {"target": {"numerical": ["value"]}},
        "schema": {"fields": {"value": {"type": "numerical"}}},
    }
    for model in variants:
        config_path = tmp_path / "resolved_configs/rel_hm" / f"{model}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(common_config), encoding="utf-8")
        manifest_path = (
            tmp_path
            / "rel_hm/validation"
            / model
            / "shared/comparability_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "evaluation_config_sha256": "eval",
            "c2st_source_sha256": "source",
            "split_fingerprints": {"train_real.csv": "same"},
            "precomputed_split_fingerprints": {},
            "pretokenized_metadata_sha256": "pretok",
            "neighbor_cache_metadata_sha256": "neighbor",
        }
        if model == "B":
            manifest["split_fingerprints"]["train_real.csv"] = "different"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validation_comparability(matrix, tmp_path)

    assert result["comparable"] is False
    assert result["mismatches"] == {"B": ["split_fingerprints"]}


def test_validation_selection_prefers_simplicity_within_equivalence_band():
    candidates = [
        {
            "model": "simple",
            "full_row_c2st": 0.50,
            "numerical_only_c2st": 0.40,
            "support_tv": 0.05,
            "seed_std": 0.02,
            "complexity_rank": 1,
        },
        {
            "model": "complex",
            "full_row_c2st": 0.495,
            "numerical_only_c2st": 0.395,
            "support_tv": 0.045,
            "seed_std": 0.01,
            "complexity_rank": 4,
        },
    ]
    policy = {
        "full_c2st_equivalence_tolerance": 0.01,
        "numerical_c2st_equivalence_tolerance": 0.02,
        "support_tv_equivalence_tolerance": 0.01,
    }

    winner, trace = select_validation_winner(candidates, policy)

    assert winner["model"] == "simple"
    assert len(trace["stages"]) == 3


def test_llm_summary_keeps_decision_evidence_and_removes_nan(tmp_path: Path):
    decision = {
        "final_model_name": "candidate",
        "freeze_architecture": False,
        "freeze_recommendation": "DO NOT FREEZE",
        "selection_policy": {
            "architecture_selected_on": "Rel-HM validation only",
            "test_data_used_for_selection": False,
            "validation_selection": {
                "selected_model": "candidate_base",
                "selected_metrics": {
                    "full_row_c2st": 0.25,
                    "numerical_only_c2st": 0.20,
                    "support_tv": 0.10,
                },
                "eligible_candidates": [
                    {
                        "model": "candidate_base",
                        "full_row_c2st": 0.25,
                        "numerical_only_c2st": 0.20,
                        "support_tv": 0.10,
                        "ignored_nan": float("nan"),
                    }
                ],
                "selection_split": "validation",
                "test_metrics_consulted": False,
            },
            "temperature_selection": {"selected_temperature": 1.0},
            "categorical_selection": {"adopted": True},
        },
        "final_architecture": {
            "numerical_routing": "training-only auto router",
            "continuous_head": "Gaussian location/scale",
            "support_head_equation": "prior + residual",
            "temporal_relational_context": "past-only",
            "text_architecture_changed": False,
        },
        "chosen_hyperparameters": {"support_sampling_temperature": 1.0},
        "acceptance_checks": {
            "validity": {"passed": True, "observed": 0.0},
            "transfer": {
                "passed": False,
                "observed": {"amazon_toy": 0.03, "unused": float("nan")},
            },
        },
        "aggregate_metrics": [
            {
                "dataset": "rel_hm",
                "split": "test",
                "model": "candidate",
                "num_seeds": 3,
                "full_row_c2st_mean": 0.30,
                "full_row_c2st_std": 0.01,
                "text_embedding_c2st_mean": float("nan"),
                "rows_per_second_mean": 5000.0,
            },
            {
                "dataset": "amazon_toy",
                "split": "test",
                "model": "M2_global_support",
                "num_seeds": 1,
                "full_row_c2st_mean": 0.58,
            },
            {
                "dataset": "amazon_toy",
                "split": "test",
                "model": "final",
                "num_seeds": 1,
                "full_row_c2st_mean": 0.61,
            },
        ],
        "paired_deltas": [
            {
                "baseline": "M2_global_support",
                "metric": "full_row_c2st",
                "seed": 17,
                "candidate_minus_baseline": -0.1,
            }
        ],
        "evaluator_audit": {
            "status": "passed",
            "fixed_evaluator_seed": 42,
            "hash_mismatches": [],
        },
        "remaining_weaknesses": ["full-row discrimination remains above chance"],
    }

    summary = build_summary(decision, tmp_path / "decision.json")
    rendered_json = json.dumps(summary, allow_nan=False)
    rendered_markdown = render_markdown(summary)

    assert "ignored_nan" not in rendered_json
    assert "text_embedding_c2st" not in rendered_json
    assert summary["decision"]["failed_checks"] == ["transfer"]
    assert summary["transfer_deltas_final_minus_m2"][0][
        "final_minus_m2"
    ] == pytest.approx(0.03)
    assert "DO NOT FREEZE" in rendered_markdown
    assert "**FAIL**" in rendered_markdown


def test_report_evaluator_comparability_allows_resolved_domains(
    tmp_path: Path,
):
    base = {
        "real_table_path": "real.csv",
        "synthetic_table_path": "synthetic.csv",
        "table": {
            "columns": {
                "rating": {
                    "type": "categorical",
                    "dtype": "int",
                    "valid_values": [1, 2, 3, 4, 5, "1", "2"],
                },
                "price": {
                    "type": "numerical",
                    "support": {"min": 0.0},
                },
            }
        },
        "evaluation": {
            "random_seed": 42,
            "c2st": {"n_splits": 5},
        },
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    rows = []
    for model in ("M0", "M2"):
        run = tmp_path / model / "runs/seed_42"
        metrics = run / "evaluation/paper_grade/metrics.json"
        metrics.parent.mkdir(parents=True)
        metrics.write_text("{}\n", encoding="utf-8")
        resolved = {
            **base,
            "real_table_path": "resolved_real.csv",
            "synthetic_table_path": f"{model}_synthetic.csv",
        }
        resolved["table"] = {
            "columns": {
                "rating": {
                    "type": "categorical",
                    "dtype": "int",
                    "valid_values": [1.0, 2.0, 3.0, 4.0, 5.0],
                },
                "price": {
                    "type": "numerical",
                    "support": {"min": 0.1, "max": 9.9},
                },
            }
        }
        (run / "evaluation_config_resolved.yaml").write_text(
            yaml.safe_dump(resolved),
            encoding="utf-8",
        )
        rows.append(
            {
                "dataset": "rel_hm",
                "model": model,
                "seed": 42,
                "metrics_path": str(metrics),
            }
        )
    matrix = {
        "evaluator_seed": 42,
        "rel_hm": {"evaluation_config": str(base_path)},
        "transfer": {"datasets": {}},
    }

    result = evaluator_comparability(
        tmp_path,
        matrix,
        pd.DataFrame(rows),
    )

    assert result["status"] == "passed"
    assert not result["hash_mismatches"]
    assert not result["within_dataset_hash_mismatches"]


def test_report_evaluator_comparability_rejects_method_drift(
    tmp_path: Path,
):
    base = {
        "table": {"columns": {"rating": {"type": "categorical"}}},
        "evaluation": {"random_seed": 42, "c2st": {"n_splits": 5}},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    run = tmp_path / "M0/runs/seed_42"
    metrics = run / "evaluation/paper_grade/metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("{}\n", encoding="utf-8")
    changed = {
        **base,
        "evaluation": {"random_seed": 42, "c2st": {"n_splits": 3}},
    }
    (run / "evaluation_config_resolved.yaml").write_text(
        yaml.safe_dump(changed),
        encoding="utf-8",
    )
    matrix = {
        "evaluator_seed": 42,
        "rel_hm": {"evaluation_config": str(base_path)},
        "transfer": {"datasets": {}},
    }

    result = evaluator_comparability(
        tmp_path,
        matrix,
        pd.DataFrame(
            [{
                "dataset": "rel_hm",
                "model": "M0",
                "seed": 42,
                "metrics_path": str(metrics),
            }]
        ),
    )

    assert result["status"] == "failed"
    assert result["hash_mismatches"]
