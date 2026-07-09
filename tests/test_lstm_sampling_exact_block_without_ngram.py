from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    FastSamplerOptions,
    block_exact_train_overlaps,
    no_repeat_ngram_size_for_column,
    normalize_privacy_text,
    privacy_summary_fields,
)


def test_exact_blocking_can_run_without_no_repeat_ngram():
    options = FastSamplerOptions(
        use_config_privacy_controls=False,
        exact_train_overlap_blocking_enabled=True,
        summary_exact_blocking_enabled=True,
        review_text_exact_blocking_enabled=True,
        no_repeat_ngram_enabled=False,
        max_summary_resample_attempts=2,
        train_text_sets={"summary": {normalize_privacy_text("great product")}},
    )

    output = block_exact_train_overlaps(["great product"], "summary", options)
    fields = privacy_summary_fields(options)

    assert output[0] != "great product"
    assert no_repeat_ngram_size_for_column(options, "summary") == 0
    assert fields["no_repeat_ngram_enabled"] is False
    assert fields["summary_no_repeat_ngram_enabled"] is False
    assert fields["summary_exact_blocking_enabled"] is True
    assert fields["summary_exact_overlap_blocked"] == 1
