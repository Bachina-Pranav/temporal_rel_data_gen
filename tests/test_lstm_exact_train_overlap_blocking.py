from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    FastSamplerOptions,
    block_exact_train_overlaps,
    ensure_privacy_counters,
    normalize_privacy_text,
)


def test_exact_train_overlap_blocking_marks_and_changes_candidate():
    options = FastSamplerOptions(
        exact_train_overlap_blocking_enabled=True,
        max_summary_resample_attempts=2,
        train_text_sets={"summary": {normalize_privacy_text("great product")}},
    )
    ensure_privacy_counters(options)

    output = block_exact_train_overlaps(["great product"], "summary", options)

    assert output[0] != "great product"
    assert options.privacy_counters["summary_exact_overlap_candidates"] == 1
    assert options.privacy_counters["summary_exact_overlap_blocked"] == 1
    assert options.privacy_counters["summary_exact_overlap_unresolved"] == 0
