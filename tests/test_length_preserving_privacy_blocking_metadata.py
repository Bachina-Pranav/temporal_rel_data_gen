from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    FastSamplerOptions,
    initialize_privacy_counters,
    privacy_summary_fields,
    update_length_bucket_preservation_counts,
)
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import SimpleTextTokenizer  # noqa: E402


def test_length_preserving_privacy_metadata_contains_per_field_counts():
    options = FastSamplerOptions(
        length_preserving_exact_blocking_enabled=True,
        dependency_aware_text_decoding_enabled=True,
        exact_train_overlap_blocking_enabled=True,
        text_field_names=["title", "body"],
        field_exact_blocking_enabled={"title": True, "body": True},
    )
    counters = initialize_privacy_counters(options, ["title", "body"])
    counters["title_exact_overlap_candidates"] = 2
    counters["title_resample_attempts_total"] = 4
    counters["title_length_bucket_preserved_count"] = 10

    fields = privacy_summary_fields(options)

    assert fields["length_preserving_exact_blocking_enabled"] is True
    assert fields["dependency_aware_text_decoding_enabled"] is True
    assert fields["text_fields_with_privacy_blocking"] == ["title", "body"]
    assert fields["title_exact_overlap_candidates"] == 2
    assert fields["title_resample_attempts_mean"] == 2.0
    assert fields["body_length_bucket_changed_count"] == 0


def test_length_bucket_preservation_counter_stays_zero_changed_for_same_bucket_text():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("user_id",),
        datetime_columns=("event_time",),
        categorical_targets=(),
        auxiliary_categorical_targets=("summary_length_bucket",),
        text_targets=("title",),
        text_max_lengths={"title": 8},
        summary_length_buckets={"short": (1, 3)},
    )
    tokenizer = SimpleTextTokenizer().fit(["one two"])
    ids, _ = tokenizer.encode("one two", max_length=8)
    options = FastSamplerOptions(text_field_names=["title"])

    update_length_bucket_preservation_counts(
        options,
        "title",
        torch.tensor([ids]),
        ["short"],
        tokenizer,
        schema,
        min_content_tokens=1,
    )
    fields = privacy_summary_fields(options)

    assert fields["title_length_bucket_preserved_count"] == 1
    assert fields["title_length_bucket_changed_count"] == 0
