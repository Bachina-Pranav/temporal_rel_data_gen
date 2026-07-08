from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_joint import candidate_train_batch_sizes, runtime_metadata  # noqa: E402


def test_lstm_runtime_metadata_has_projection_fields():
    metadata = runtime_metadata(
        total_seconds=10.0,
        rows=1000,
        batch_size=128,
        device="cuda",
        mixed_precision=True,
        lengths={"summary": [2, 4], "review_text": [10, 20, 30]},
    )
    assert metadata["total_sampling_seconds"] == 10.0
    assert metadata["rows_per_second"] == 100.0
    assert metadata["projected_hours_for_10m_rows"] > 0
    assert metadata["p95_generated_review_text_tokens"] is not None


def test_lstm_train_batch_size_candidates_halve_to_floor():
    assert candidate_train_batch_sizes(256, 32) == [256, 128, 64, 32]
    assert candidate_train_batch_sizes(64, 16) == [64, 32, 16]
    assert candidate_train_batch_sizes(16, 32) == [16]
