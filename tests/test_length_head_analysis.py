from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.length_head_analysis import (  # noqa: E402
    TextDatasetSpec,
    add_length_columns,
    analyze_text_dataset,
    history_feature_frame,
)
from attribute_generation.conditional_tabdlm.tokenization import (  # noqa: E402
    SimpleTextTokenizer,
)


def test_history_features_are_strictly_past_for_timestamp_ties():
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "event_time": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"],
                utc=True,
            ),
            "token_length": [2.0, 100.0, 200.0, 4.0],
        }
    )

    history = history_feature_frame(
        frame,
        entity_column="user_id",
        timestamp_column="event_time",
        length_column="token_length",
    )

    assert history["prior_count"].tolist() == [0.0, 1.0, 1.0, 3.0]
    assert pd.isna(history.loc[0, "past_mean_length"])
    assert history.loc[1, "past_mean_length"] == 2.0
    assert history.loc[2, "past_mean_length"] == 2.0
    assert history.loc[3, "past_mean_length"] == (2.0 + 100.0 + 200.0) / 3.0
    assert history.loc[1, "previous_timestamp"] < frame.loc[1, "event_time"]
    assert history.loc[2, "previous_timestamp"] < frame.loc[2, "event_time"]


def test_generator_tokenizer_length_and_true_empty_indicator_are_separate():
    frame = pd.DataFrame({"review_text": ["", "two words", "punctuation!"]})

    result = add_length_columns(
        frame, "review_text", SimpleTextTokenizer(lowercase=True)
    )

    assert result["token_length"].tolist() == [1.0, 2.0, 2.0]
    assert result["word_count"].tolist() == [0.0, 2.0, 1.0]
    assert result["_text_empty"].tolist() == [True, False, False]


def test_end_to_end_length_analysis_uses_chronological_splits(
    tmp_path, monkeypatch
):
    import evaluation.length_head_analysis as module

    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        yaml.safe_dump({"tokenizer": {"lowercase": True}}),
        encoding="utf-8",
    )
    start = pd.Timestamp("2020-01-01", tz="UTC")
    split_paths = {}
    offset = 0
    for split, size in (("train", 20), ("validation", 6), ("test", 6)):
        rows = []
        for index in range(size):
            rows.append(
                {
                    "user_id": f"u{index % 3}",
                    "item_id": f"i{index % 4}",
                    "event_time": start + pd.Timedelta(days=offset + index),
                    "rating": 1 + index % 5,
                    "review_text": "word " * (1 + index % 7),
                }
            )
        offset += size
        path = tmp_path / f"{split}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        split_paths[split] = path

    def fake_fit(train_x, train_y, test_x, *, seed):
        return pd.Series(float(pd.Series(train_y).median()), index=test_x.index).to_numpy(), "test predictor"

    monkeypatch.setattr(module, "fit_predictor", fake_fit)
    monkeypatch.setattr(module, "save_length_figures", lambda *args, **kwargs: None)
    spec = TextDatasetSpec(
        name="Toy Text",
        config_path=config_path,
        train_path=split_paths["train"],
        validation_path=split_paths["validation"],
        test_path=split_paths["test"],
        source_column="user_id",
        destination_column="item_id",
        timestamp_column="event_time",
        text_columns=("review_text",),
        structured_columns=("rating",),
        table_columns={
            "rating": {
                "type": "categorical",
                "semantic_type": "ordinal_categorical",
            }
        },
    )

    result = analyze_text_dataset(spec, tmp_path / "analysis", seed=42)

    assert len(result["summary"]) == 1
    assert {row["model"] for row in result["predictive"]} == set(
        module.MODEL_FEATURE_FAMILIES
    )
    assert result["leakage"][0]["passed"] is True
    assert result["leakage"][0]["checks"]["source_history_strictly_past"] is True
