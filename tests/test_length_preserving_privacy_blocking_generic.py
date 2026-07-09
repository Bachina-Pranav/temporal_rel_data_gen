from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import text_field_policies_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402


def test_length_preserving_privacy_policy_supports_generic_text_fields():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("user_id", "item_id"),
        datetime_columns=("event_time",),
        categorical_targets=("title_length_bucket", "body_length_bucket"),
        text_targets=("title", "body"),
        text_max_lengths={"title": 8, "body": 16},
        summary_length_buckets={"short": (1, 3)},
        review_text_length_buckets={"long": (4, 12)},
    )
    raw = {
        "paths": {"train_data_path": "unused.csv", "synthetic_spine_path": "unused.csv", "output_dir": "unused"},
        "columns": {
            "condition": {"foreign_keys": ["user_id", "item_id"], "datetimes": ["event_time"]},
            "target": {"categorical": ["title_length_bucket", "body_length_bucket"], "text": ["title", "body"]},
        },
        "text_fields": [
            {
                "name": "title",
                "target_column": "title",
                "length_bucket_column": "title_length_bucket",
                "privacy": {"exact_train_overlap_blocking": True, "max_resample_attempts": 4},
            },
            {
                "name": "body",
                "target_column": "body",
                "length_bucket_column": "body_length_bucket",
                "privacy": {"exact_train_overlap_blocking": True, "max_resample_attempts": 2},
                "dependencies": {"conditions_on": ["title"]},
            },
        ],
    }
    config = ConditionalTABDLMConfig(raw=raw, schema=schema, config_path=None)

    policies = text_field_policies_from_config(config)

    assert [policy.target_column for policy in policies] == ["title", "body"]
    assert policies[0].length_bucket_column == "title_length_bucket"
    assert policies[1].conditions_on == ("title",)
    assert policies[1].max_resample_attempts == 2
