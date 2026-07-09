from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    FastSamplerOptions,
    no_repeat_ngram_size_for_column,
    privacy_summary_fields,
)


def test_summary_no_repeat_ngram_can_be_enabled_without_review_text_ngram():
    options = FastSamplerOptions(
        use_config_privacy_controls=False,
        no_repeat_ngram_enabled=True,
        summary_no_repeat_ngram_enabled=True,
        review_text_no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_size=3,
        review_text_no_repeat_ngram_size=4,
    )

    fields = privacy_summary_fields(options)

    assert no_repeat_ngram_size_for_column(options, "summary") == 3
    assert no_repeat_ngram_size_for_column(options, "review_text") == 0
    assert fields["no_repeat_ngram_enabled"] is True
    assert fields["summary_no_repeat_ngram_enabled"] is True
    assert fields["review_text_no_repeat_ngram_enabled"] is False
