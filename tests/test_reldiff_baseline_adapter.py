from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.reldiff.adapter import (  # noqa: E402
    decode_generated_timestamp,
    encode_training_timestamp,
    generation_validity,
    postprocess_generated_interactions,
    repeated_pair_summary,
    restrict_entity_tables_to_interactions,
    resolve_split_labels,
)
from baselines.reldiff.schema import RelDiffDatasetConfig, load_dataset_config  # noqa: E402
from scripts.preflight_reldiff_baseline import annotate_audit_frame  # noqa: E402
from scripts.run_reldiff_baseline import ROOT as RUNNER_ROOT  # noqa: E402
from scripts.run_reldiff_baseline import repository_command_path  # noqa: E402


def test_declared_datasets_have_requested_roles_and_no_text_surrogates():
    amazon = load_dataset_config(
        ROOT / "configs/baselines/reldiff/amazon_toy.yaml"
    )
    movie = load_dataset_config(
        ROOT / "configs/baselines/reldiff/movielens_100k.yaml"
    )
    hm = load_dataset_config(ROOT / "configs/baselines/reldiff/rel_hm.yaml")

    assert amazon.structured_attributes == ("rating", "verified")
    assert amazon.ignored_attributes == ("summary", "review_text")
    assert movie.structured_attributes == ("rating",)
    assert movie.numerical_attributes == ()
    assert hm.structured_attributes == ("price", "sales_channel_id")
    assert all("text_length" not in str(config.to_manifest()) for config in (amazon, movie, hm))


def test_split_resolution_prefers_explicit_labels_and_legacy_policy_is_chronological():
    explicit = pd.DataFrame(
        {
            "event_time": pd.date_range("2020-01-01", periods=3, tz="UTC"),
            "split": ["train", "val", "test"],
        }
    )
    labels, source = resolve_split_labels(explicit, "event_time")
    assert labels.tolist() == ["train", "validation", "test"]
    assert source == "explicit_split_column"

    chronological = pd.DataFrame(
        {"event_time": pd.date_range("2020-01-01", periods=20, tz="UTC")[::-1]}
    )
    labels, source = resolve_split_labels(chronological, "event_time")
    assert source == "legacy_time_aware_90_5_5"
    assert labels.value_counts().to_dict() == {"train": 18, "validation": 1, "test": 1}
    train_max = chronological.loc[labels == "train", "event_time"].max()
    validation_time = chronological.loc[labels == "validation", "event_time"].iloc[0]
    test_time = chronological.loc[labels == "test", "event_time"].iloc[0]
    assert train_max < validation_time < test_time


def test_timestamp_encoding_uses_supplied_training_origin_and_round_trips():
    config = temporary_config(Path("unused"))
    origin = pd.Timestamp("2021-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "event_time": [
                "2021-01-01T00:00:00Z",
                "2021-01-01T00:01:30Z",
            ]
        }
    )
    encoded = encode_training_timestamp(frame, config, origin)
    assert encoded["event_time"].tolist() == [0.0, 90.0]
    decoded = decode_generated_timestamp(encoded["event_time"], origin)
    assert decoded.tolist() == list(pd.to_datetime(frame["event_time"], utc=True))


def test_smoke_entity_restriction_keeps_only_referenced_entities_in_original_order():
    config = temporary_config(Path("unused"))
    source = pd.DataFrame({"user_id": ["u3", "u1", "u2"]})
    destination = pd.DataFrame({"movie_id": ["m2", "m3", "m1"]})
    interactions = pd.DataFrame(
        {"user_id": ["u2", "u1"], "movie_id": ["m1", "m2"]}
    )
    kept_source, kept_destination = restrict_entity_tables_to_interactions(
        source, destination, interactions, config
    )
    assert kept_source["user_id"].tolist() == ["u1", "u2"]
    assert kept_destination["movie_id"].tolist() == ["m2", "m1"]


def test_repeated_pair_statistics_count_rows_not_only_extra_duplicates():
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u2"],
            "movie_id": ["i1", "i1", "i1", "i2", "i1"],
            "value": list("ABCDE"),
        }
    )
    summary, top, examples = repeated_pair_summary(frame, "user_id", "movie_id")
    assert summary["num_rows"] == 5
    assert summary["num_unique_pairs"] == 3
    assert summary["num_repeated_pair_rows"] == 3
    assert summary["fraction_rows_in_repeated_pairs"] == 0.6
    assert summary["max_pair_multiplicity"] == 3
    assert top.iloc[0]["multiplicity"] == 3
    assert set(examples["value"]) == {"A", "B", "C"}


def test_audit_annotation_replaces_existing_split_provenance_columns():
    frame = pd.DataFrame(
        {
            "split": ["train"],
            "dataset": ["source_value"],
            "record_type": ["source_value"],
            "multiplicity": [3],
        }
    )
    annotated = annotate_audit_frame(
        frame,
        dataset="movielens_100k",
        split="complete",
        record_type="repeated_pair_event_example",
    )
    assert annotated.columns.tolist()[:3] == ["dataset", "split", "record_type"]
    assert annotated.loc[0, "dataset"] == "movielens_100k"
    assert annotated.loc[0, "split"] == "complete"
    assert annotated.loc[0, "record_type"] == "repeated_pair_event_example"


def test_repository_command_path_accepts_relative_and_absolute_repo_paths():
    relative = Path("outputs/baselines/reldiff/runtime.json")
    absolute = RUNNER_ROOT / relative
    assert repository_command_path(relative) == str(relative)
    assert repository_command_path(absolute) == str(relative)


def test_generated_postprocessing_restores_entity_labels_timestamp_and_validity(tmp_path: Path):
    source = pd.DataFrame({"user_id": ["u1", "u2"]})
    destination = pd.DataFrame({"movie_id": ["m1", "m2"]})
    interaction = pd.DataFrame(
        {
            "event_id": [0, 1],
            "user_id": ["u1", "u2"],
            "movie_id": ["m1", "m2"],
            "event_time": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
            "rating": [1.0, 5.0],
            "split": ["train", "test"],
        }
    )
    source_path = tmp_path / "users.csv"
    destination_path = tmp_path / "movies.csv"
    interaction_path = tmp_path / "interactions.csv"
    source.to_csv(source_path, index=False)
    destination.to_csv(destination_path, index=False)
    interaction.to_csv(interaction_path, index=False)
    config = temporary_config(
        interaction_path,
        source_path=source_path,
        destination_path=destination_path,
    )

    provenance = tmp_path / "config"
    mappings = provenance / "id_mappings"
    splits = provenance / "splits"
    mappings.mkdir(parents=True)
    splits.mkdir(parents=True)
    pd.DataFrame(
        {"internal_id": [0, 1], "original_id": ["u1", "u2"]}
    ).to_csv(mappings / "users.csv", index=False)
    pd.DataFrame(
        {"internal_id": [0, 1], "original_id": ["m1", "m2"]}
    ).to_csv(mappings / "movies.csv", index=False)
    train_path = splits / "train.csv"
    interaction.drop(columns="split").to_csv(train_path, index=False)
    manifest = {
        "training_time_origin": "2020-01-01T00:00:00+00:00",
        "training_rows": 1,
        "splits": {
            "train": {
                "path": str(train_path),
                "rows": 1,
                "timestamp_max": "2020-01-01T00:00:00+00:00",
            }
        },
    }
    manifest_path = provenance / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    raw = tmp_path / "raw_generated.csv"
    pd.DataFrame(
        {
            "event_id": [99],
            "user_id": [1],
            "movie_id": [0],
            "event_time": [3600.0],
            "rating": [5.0],
        }
    ).to_csv(raw, index=False)
    output = tmp_path / "generated/synthetic_interactions.csv"
    report = postprocess_generated_interactions(
        config,
        generated_table=raw,
        data_manifest=manifest_path,
        output=output,
    )
    generated = pd.read_csv(output)
    assert generated.loc[0, "user_id"] == "u2"
    assert generated.loc[0, "movie_id"] == "m1"
    assert pd.Timestamp(generated.loc[0, "event_time"]) == pd.Timestamp(
        "2020-01-01T01:00:00+00:00"
    )
    assert report["timestamp_clipped"] is False
    validity = generation_validity(config, output, manifest_path)
    assert validity["valid"] is True


def test_upstream_entry_points_only_add_adapter_execution_flags():
    train = (ROOT / "src/scripts/train_joint_diffusion.py").read_text()
    sample = (ROOT / "src/scripts/sample_joint_diffusion.py").read_text()
    assert "--preserve-explicit-table-nodes" in train
    assert "--preserve-explicit-table-nodes" in sample
    assert "transform_fk_tables=not args.preserve_explicit_table_nodes" in train
    assert "transform_fk_tables=not args.preserve_explicit_table_nodes" in sample
    assert "--seed" in train and "--seed" in sample


def test_zero_feature_projection_is_bias_only_and_avoids_linear_zero_width():
    joint = (ROOT / "src/reldiff/models/joint.py").read_text()
    assert "class ZeroFeatureLinear" in joint
    assert "ZeroFeatureLinear(dim_t) if d_in == 0" in joint
    assert "self.bias.unsqueeze(0).expand(input.shape[0], -1)" in joint


def temporary_config(
    interaction_path: Path,
    *,
    source_path: Path = Path("users.csv"),
    destination_path: Path = Path("movies.csv"),
) -> RelDiffDatasetConfig:
    return RelDiffDatasetConfig(
        key="temporary",
        display_name="Temporary",
        interaction_path=interaction_path,
        source_entity_path=source_path,
        destination_entity_path=destination_path,
        source_table="users",
        destination_table="movies",
        interaction_table="interactions",
        source_pk="user_id",
        destination_pk="movie_id",
        source_fk="user_id",
        destination_fk="movie_id",
        event_id="event_id",
        timestamp="event_time",
        numerical_attributes=(),
        categorical_attributes=("rating",),
        ignored_attributes=(),
        evaluation_config=Path("unused.yaml"),
    )
