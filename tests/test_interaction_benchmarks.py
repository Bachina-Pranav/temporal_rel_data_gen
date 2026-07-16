from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.dataset import (  # noqa: E402
    ConditionalTABDLMDataset,
    load_category_vocabs,
    load_numerical_metadata,
    load_prepared_tables,
    load_text_tokenizer,
)
from attribute_generation.conditional_tabdlm.lstm_joint import build_lstm_model, lstm_joint_loss, make_lstm_collate_fn  # noqa: E402
from attribute_generation.conditional_tabdlm.numerical import inverse_transform_numerical, sample_gaussian_params  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from data_preprocessing.interaction_datasets.base import safe_extract_archive  # noqa: E402
from data_preprocessing.interaction_datasets.hm import HMAdapter  # noqa: E402
from data_preprocessing.interaction_datasets.movielens import MovieLensAdapter  # noqa: E402
from data_preprocessing.interaction_datasets.retailrocket import RetailRocketAdapter  # noqa: E402
from data_preprocessing.interaction_datasets.subset import build_interaction_subset, select_source_entities  # noqa: E402
from data_preprocessing.interaction_datasets.yelp import YelpAdapter  # noqa: E402


def test_selection_preserves_complete_histories_and_is_deterministic():
    counts = {"u1": 2, "u2": 3, "u3": 4, "u4": 1}
    a = select_source_entities(counts, target_interactions=5, seed=7)
    b = select_source_entities(counts, target_interactions=5, seed=7)
    assert a.selected_ids == b.selected_ids
    assert sum(counts[source_id] for source_id in a.selected_ids) == a.final_count
    assert a.absolute_deviation == 0


def test_movielens_subset_builds_complete_user_histories(tmp_path: Path):
    raw = tmp_path / "raw" / "movielens" / "ml-25m"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 2, 3],
            "movieId": [10, 11, 10, 12, 13, 13],
            "rating": [4.0, 5.0, 3.5, 2.0, 4.5, 1.0],
            "timestamp": [1_000, 2_000, 3_000, 4_000, 5_000, 6_000],
        }
    ).to_csv(raw / "ratings.csv", index=False)
    pd.DataFrame({"movieId": [10, 11, 12, 13, 99], "title": ["a", "b", "c", "d", "z"]}).to_csv(raw / "movies.csv", index=False)

    manifest = build_interaction_subset(
        MovieLensAdapter(),
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        target_interactions=5,
        seed=3,
        chunk_size=2,
    )
    out_dir = tmp_path / "processed" / "movielens_100k"
    interactions = pd.read_csv(out_dir / "interactions.csv")
    movies = pd.read_csv(out_dir / "movies.csv")
    assert manifest["complete_source_histories"] is True
    assert manifest["foreign_key_valid"] is True
    assert set(interactions["split"]) == {"train", "validation", "test"}
    assert set(interactions["movie_id"].astype(str)).issubset(set(movies["movie_id"].astype(str)))
    assert pd.to_datetime(interactions["event_time"], utc=True).min() == pd.Timestamp(1_000, unit="s", tz="UTC")


def test_yelp_json_streaming_preserves_text_and_counts(tmp_path: Path):
    raw = tmp_path / "raw" / "yelp"
    raw.mkdir(parents=True)
    reviews = [
        {"review_id": "r1", "user_id": "u1", "business_id": "b1", "date": "2020-01-01", "stars": 5, "useful": 1, "funny": 0, "cool": 2, "text": "great \"quoted\" food"},
        {"review_id": "r2", "user_id": "u1", "business_id": "b2", "date": "2020-01-02", "stars": 4, "useful": 0, "funny": 0, "cool": 0, "text": "nice"},
        {"review_id": "r3", "user_id": "u2", "business_id": "b1", "date": "2020-01-03", "stars": 1, "useful": 3, "funny": 1, "cool": 0, "text": "bad"},
    ]
    write_jsonl(raw / "yelp_academic_dataset_review.json", reviews)
    write_jsonl(raw / "yelp_academic_dataset_user.json", [{"user_id": "u1"}, {"user_id": "u2"}])
    write_jsonl(raw / "yelp_academic_dataset_business.json", [{"business_id": "b1"}, {"business_id": "b2"}])

    manifest = build_interaction_subset(YelpAdapter(), raw_root=tmp_path / "raw", processed_root=tmp_path / "processed", target_interactions=2, seed=1, chunk_size=1)
    interactions = pd.read_csv(tmp_path / "processed" / "yelp_100k" / "interactions.csv")
    assert manifest["complete_source_histories"] is True
    assert "great \"quoted\" food" in set(interactions["review_text"])
    assert (interactions[["useful", "funny", "cool"]] >= 0).all().all()


def test_retailrocket_timestamp_event_mapping_and_audit(tmp_path: Path):
    raw = tmp_path / "raw" / "retailrocket"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": [1_000, 2_000, 3_000],
            "visitorid": [1, 1, 2],
            "event": ["view", "addtocart", "transaction"],
            "itemid": [10, 11, 10],
            "transactionid": ["", "", "tx1"],
        }
    ).to_csv(raw / "events.csv", index=False)
    pd.DataFrame({"itemid": [10], "property": ["categoryid"], "value": ["x"], "timestamp": [1]}).to_csv(raw / "item_properties_part1.csv", index=False)
    pd.DataFrame({"itemid": [11], "property": ["categoryid"], "value": ["y"], "timestamp": [2]}).to_csv(raw / "item_properties_part2.csv", index=False)
    pd.DataFrame({"categoryid": ["x"], "parentid": [""]}).to_csv(raw / "category_tree.csv", index=False)

    build_interaction_subset(RetailRocketAdapter(), raw_root=tmp_path / "raw", processed_root=tmp_path / "processed", target_interactions=2, seed=4, chunk_size=1)
    out = tmp_path / "processed" / "retailrocket_100k"
    interactions = pd.read_csv(out / "interactions.csv")
    assert set(interactions["event_type"]).issubset({"view", "addtocart", "transaction"})
    assert "transactionid" not in interactions.columns
    assert (out / "events_audit.csv").exists()
    assert pd.to_datetime(interactions["event_time"], utc=True).min() == pd.Timestamp(1_000, unit="ms", tz="UTC")


def test_hm_date_and_price_validation(tmp_path: Path):
    raw = tmp_path / "raw" / "hm"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "t_dat": ["2020-09-01", "2020-09-02", "2020-09-03"],
            "customer_id": ["c1", "c1", "c2"],
            "article_id": ["a1", "a2", "a1"],
            "price": [0.1, 0.2, 0.3],
            "sales_channel_id": [1, 2, 1],
        }
    ).to_csv(raw / "transactions_train.csv", index=False)
    pd.DataFrame({"customer_id": ["c1", "c2"]}).to_csv(raw / "customers.csv", index=False)
    pd.DataFrame({"article_id": ["a1", "a2"]}).to_csv(raw / "articles.csv", index=False)

    manifest = build_interaction_subset(HMAdapter(), raw_root=tmp_path / "raw", processed_root=tmp_path / "processed", target_interactions=2, seed=4, chunk_size=1)
    interactions = pd.read_csv(tmp_path / "processed" / "hm_100k" / "interactions.csv")
    assert manifest["foreign_key_valid"] is True
    assert pd.to_numeric(interactions["price"]).ge(0).all()
    assert pd.to_datetime(interactions["event_time"], utc=True).min() == pd.Timestamp("2020-09-01", tz="UTC")


def test_unsafe_archive_path_is_rejected(tmp_path: Path):
    archive = tmp_path / "bad.tar"
    payload = b"nope"
    info = tarfile.TarInfo("../escape.txt")
    info.size = len(payload)
    with tarfile.open(archive, "w") as tf:
        tf.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError):
        safe_extract_archive(archive, tmp_path / "out")


def test_lstm_numerical_heads_loss_and_inverse_transform(tmp_path: Path):
    table = tmp_path / "interactions.csv"
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3", "u3", "u4"],
            "item_id": ["i1", "i2", "i1", "i2", "i3", "i1"],
            "event_time": pd.date_range("2021-01-01", periods=6, freq="D"),
            "kind": ["a", "b", "a", "b", "a", "b"],
            "price": [1.0, 2.0, 1.5, 2.5, 3.0, 4.0],
            "count": [0, 1, 2, 0, 3, 4],
        }
    ).to_csv(table, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(tiny_lstm_config(table, tmp_path / "out")), encoding="utf-8")
    config = load_config(config_path)
    train, _, _ = load_prepared_tables(config)
    vocabs = load_category_vocabs(config)
    tokenizer = load_text_tokenizer(config)
    metadata = load_numerical_metadata(config)
    dataset = ConditionalTABDLMDataset(train, config.schema, vocabs, tokenizer, 64, numerical_metadata=metadata)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=make_lstm_collate_fn)))
    model = build_lstm_model(config, vocabs, tokenizer)
    logits = model(batch["foreign_key_ids"], batch["datetime_values"], batch["categorical_ids"], batch["text_ids"])
    loss, component = lstm_joint_loss(logits, batch, config.schema, {"price": 1.0, "count": 1.0, "kind": 1.0}, tokenizer, {}, config=config)
    assert torch.isfinite(loss)
    assert "price" in component and "count" in component
    sampled = sample_gaussian_params(logits["numerical"]["count"])
    restored = inverse_transform_numerical(sampled, metadata["count"])
    assert torch.all(restored >= 0)
    assert torch.allclose(restored, torch.round(restored))


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def tiny_lstm_config(table: Path, output_dir: Path) -> dict:
    return {
        "experiment_name": "tiny_lstm_numeric",
        "paths": {"train_data_path": str(table), "synthetic_spine_path": str(table), "output_dir": str(output_dir)},
        "columns": {
            "condition": {"foreign_keys": ["user_id", "item_id"], "datetimes": ["event_time"]},
            "target": {"categorical": ["kind"], "numerical": ["price", "count"], "text": []},
        },
        "schema": {
            "fields": {
                "kind": {"role": "generated_attribute", "semantic_type": "categorical"},
                "price": {"role": "generated_attribute", "semantic_type": "continuous_numerical", "preprocessing": "standardize"},
                "count": {"role": "generated_attribute", "semantic_type": "count_numerical", "preprocessing": "log1p_standardize"},
            }
        },
        "text": {"max_length": {}},
        "tokenizer": {"max_vocab_size": 100, "min_frequency": 1, "lowercase": True},
        "id_encoding": {"num_buckets": 64, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
        "model": {"row_hidden_dim": 32, "latent_noise_dim": 8, "categorical_context_dim": 4, "dropout": 0.0, "use_graph_context": False},
        "text_decoder": {"embedding_dim": 8, "hidden_dim": 16, "num_layers": 1, "dropout": 0.0},
        "training": {"batch_size": 2, "epochs": 1, "num_workers": 0, "mixed_precision": False},
    }
