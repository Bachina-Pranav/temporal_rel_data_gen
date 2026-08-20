from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.text_c2st_audit import (  # noqa: E402
    EmbeddingStore,
    TextC2STProtocol,
    c2st_error,
    compare_text_frames,
    evaluate_prepared_embeddings,
    implied_auc,
    length_prefixed_sha256,
    prepare_protocol_embeddings,
)


def test_normalized_c2st_error_and_implied_auc_are_inverse():
    assert c2st_error(0.5) == 0.0
    assert c2st_error(0.87) == 0.74
    assert implied_auc(0.74) == 0.87


def test_canonical_text_hash_ignores_null_and_whitespace_formatting():
    left = pd.Series([None, "  good   item ", "cafe\u0301"])
    right = pd.Series(["", "good item", "caf\u00e9"])

    assert length_prefixed_sha256(left, canonical=True) == length_prefixed_sha256(
        right, canonical=True
    )
    assert length_prefixed_sha256(left, canonical=False) != length_prefixed_sha256(
        right, canonical=False
    )


def test_text_frame_comparison_distinguishes_order_and_formatting():
    left = pd.DataFrame(
        {
            "summary": [" one ", "two"],
            "review_text": ["alpha", "beta"],
        }
    )
    formatting = pd.DataFrame(
        {
            "summary": ["one", "two"],
            "review_text": ["alpha", "beta"],
        }
    )
    reordered = formatting.iloc[::-1].reset_index(drop=True)

    format_result = compare_text_frames(left, formatting)
    order_result = compare_text_frames(formatting, reordered)

    assert format_result["fields"]["summary"]["changed_rows_canonical"] == 0
    assert format_result["fields"]["summary"]["format_or_null_only_rows"] == 1
    assert order_result["fields"]["summary"]["row_order_only_difference"]


def test_hash_protocol_reuses_embeddings_and_is_same_seed_reproducible(tmp_path):
    real = pd.DataFrame(
        {
            "summary": [f"real summary {index}" for index in range(40)],
            "review_text": [f"high quality real review {index}" for index in range(40)],
        }
    )
    synthetic = pd.DataFrame(
        {
            "summary": [f"synthetic title {index}" for index in range(40)],
            "review_text": [f"generated synthetic content {index}" for index in range(40)],
        }
    )
    protocol = TextC2STProtocol(
        name="test",
        embedding_backend="deterministic_hash",
        embedding_model="requested-minilm-label",
        preprocessing="historical_hash",
        classifiers=("logistic_regression",),
        max_rows=32,
        seed=42,
        n_splits=4,
    )
    store = EmbeddingStore(tmp_path / "cache", device="cpu")
    prepared = prepare_protocol_embeddings(
        real, synthetic, protocol, store, label="audit_test"
    )

    first = evaluate_prepared_embeddings(
        prepared,
        protocol.classifiers,
        seed=42,
        n_splits=4,
        protocol=protocol,
    )
    second = evaluate_prepared_embeddings(
        prepared,
        protocol.classifiers,
        seed=42,
        n_splits=4,
        protocol=protocol,
    )

    assert first["macro_error"] == second["macro_error"]
    assert first["combined"] is not None
    assert first["combined"]["num_real"] == 32
    assert np.isclose(
        first["macro_error"],
        np.mean(
            [
                first["per_field"]["summary"]["error"],
                first["per_field"]["review_text"]["error"],
            ]
        ),
    )


def test_aggregation_records_difference_when_aucs_cross_chance(monkeypatch):
    import evaluation.text_c2st_audit as audit

    results = iter(
        [
            {"auc": 0.8, "error": 0.6},
            {"auc": 0.4, "error": 0.2},
            {"auc": 0.7, "error": 0.4},
        ]
    )
    monkeypatch.setattr(
        audit,
        "evaluate_embedding_pair",
        lambda *args, **kwargs: next(results),
    )
    prepared = {
        "num_real": 2,
        "num_synthetic": 2,
        "arrays": {
            "summary": {"real": np.zeros((2, 2)), "synthetic": np.ones((2, 2))},
            "review_text": {
                "real": np.zeros((2, 2)),
                "synthetic": np.ones((2, 2)),
            },
        },
    }

    result = audit.evaluate_prepared_embeddings(
        prepared,
        ("logistic_regression",),
        seed=42,
        n_splits=2,
    )

    assert np.isclose(result["macro_error"], 0.4)
    assert np.isclose(result["macro_error_from_macro_auc"], 0.2)
    assert np.isclose(result["aggregation_identity_gap"], 0.2)


def test_audit_classifier_reproduces_project_classifier_protocol():
    from evaluation.paper_metrics.c2st import run_binary_classifiers
    from evaluation.text_c2st_audit import evaluate_embedding_pair

    rng = np.random.RandomState(42)
    real = rng.normal(0.0, 1.0, size=(30, 8))
    synthetic = rng.normal(0.25, 1.0, size=(30, 8))
    classifiers = (
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    )
    x = np.vstack([real, synthetic])
    y = np.array([1] * len(real) + [0] * len(synthetic), dtype=int)

    expected = run_binary_classifiers(x, y, list(classifiers), seed=42)
    actual = evaluate_embedding_pair(
        real,
        synthetic,
        classifiers,
        seed=42,
        n_splits=5,
    )

    for classifier in classifiers:
        assert np.isclose(
            actual["per_classifier"][classifier]["auc"],
            expected[classifier]["auc"],
        )
        assert np.isclose(
            actual["per_classifier"][classifier]["error"],
            expected[classifier]["error"],
        )
