from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.support_calibration import (  # noqa: E402
    corrected_logit_bias,
    support_calibration_metrics,
    support_probability_table,
)


def load_script(name: str):
    path = ROOT / "src" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_support_calibration_reports_flattening_and_finite_correction():
    train = pd.Series(np.repeat([1.0, 2.0, 3.0], [80, 15, 5]))
    validation = pd.Series(np.repeat([1.0, 2.0, 3.0], [40, 8, 2]))
    generated = pd.Series(np.repeat([1.0, 2.0, 3.0], [20, 15, 15]))

    table = support_probability_table(train, validation, generated)
    metrics = support_calibration_metrics(table)
    bias = corrected_logit_bias(table, strength=0.5)

    assert metrics["total_variation_train_vs_generated"] > 0
    assert metrics["diagnosis"]["underproduces_dominant_values"] is True
    assert metrics["diagnosis"]["overproduces_rare_values"] is True
    assert metrics["diagnosis"]["flattens_support_distribution"] is True
    assert np.isfinite(bias).all()


def test_acceptance_checks_use_not_evaluable_for_missing_values():
    module = load_script("summarize_lstm_numerical_head_experiments")

    checks = module.acceptance_checks(None)

    assert checks
    assert all(
        check["status"] == "not_evaluable"
        for check in checks.values()
    )
    assert all(check["passed"] is None for check in checks.values())


def test_acceptance_checks_fail_only_observed_bad_values():
    module = load_script("summarize_lstm_numerical_head_experiments")
    values = pd.Series(
        {
            "constraint_violation": 0.1,
            "fk_similarity": 1.0,
            "full_row_c2st": 0.4,
            "numerical_only_c2st": 0.3,
            "rows_per_second": 2000.0,
        }
    )

    checks = module.acceptance_checks(values)

    assert checks["constraint_zero"]["status"] == "failed"
    assert checks["fk_similarity_one"]["status"] == "passed"
    assert (
        checks["destination_numerical_mae_target"]["status"]
        == "not_evaluable"
    )


def test_comparability_audit_does_not_reuse_unfingerprinted_legacy_run(
    tmp_path: Path,
):
    module = load_script("audit_lstm_run_comparability")
    legacy = tmp_path / "legacy"
    candidate = tmp_path / "candidate"
    for root in (legacy, candidate):
        spines = root / "shared" / "spines"
        spines.mkdir(parents=True)
        for name in module.REQUIRED_SPLIT_FILES:
            (spines / name).write_text("id,value\n1,2\n", encoding="utf-8")
        run = root / "runs" / "seed_42"
        for relative in module.REQUIRED_RUN_FILES:
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".yaml":
                path.write_text(
                    "model: {row_hidden_dim: 8}\n"
                    "training: {max_steps: 2}\n",
                    encoding="utf-8",
                )
            elif path.suffix == ".json":
                path.write_text("{}\n", encoding="utf-8")
            else:
                path.write_bytes(b"x")
        (run / "evaluation_config_resolved.yaml").write_text(
            "c2st: {max_rows: 10}\n",
            encoding="utf-8",
        )
    candidate_manifest = {
        "precomputed_split_fingerprints": {
            "train_indices.npy": "candidate"
        },
        "c2st_source_sha256": "same",
    }
    (candidate / "shared" / "comparability_manifest.json").write_text(
        json.dumps(candidate_manifest),
        encoding="utf-8",
    )

    report = module.audit_comparability(
        legacy,
        candidate,
        seeds=[17, 42, 73],
        c2st_source=ROOT / "src/evaluation/paper_metrics/c2st.py",
    )

    assert report["comparable"] is False
    assert report["reused_runs"] == [42]
    assert report["required_new_runs"] == [17, 73]
    assert any(
        "legacy.precomputed_split_fingerprints" == field
        for field in report["unavailable_fields"]
    )


def test_numerical_router_has_no_dataset_or_column_name_special_cases():
    sources = [
        ROOT
        / "src/attribute_generation/conditional_tabdlm/numerical_head.py",
        ROOT
        / "src/attribute_generation/conditional_tabdlm/numerical_type.py",
    ]
    forbidden = (
        "hm_10k",
        "sales_channel_id",
        "article_id",
        "customer_id",
    )

    combined = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in sources
    )

    assert all(token not in combined for token in forbidden)


def test_movielens_and_amazon_rating_remain_schema_categorical():
    module = load_script("audit_lstm_numerical_router_regressions")
    matrix = yaml.safe_load(
        (
            ROOT
            / "configs/experiments/"
            "lstm_numerical_router_regressions.yaml"
        ).read_text(encoding="utf-8")
    )

    for dataset in ("movielens_100k", "amazon_toy"):
        raw = yaml.safe_load(
            (ROOT / matrix["datasets"][dataset]["config"]).read_text(
                encoding="utf-8"
            )
        )
        targets = module.declared_targets(raw)

        assert targets["numerical"] == []
        assert "rating" in targets["categorical"]


def test_freeze_summary_handles_missing_experiments_as_not_evaluable():
    module = load_script("summarize_lstm_numerical_architecture_freeze")
    frame = pd.DataFrame()

    assert module.model_means(frame, "M2_global_support") is None
    assert (
        module.best_model(frame, ["M2P_R0_global_prior"], [17, 42, 73])[
            "status"
        ]
        == "not_evaluable"
    )
