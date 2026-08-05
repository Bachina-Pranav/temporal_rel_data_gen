from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.dataset import (  # noqa: E402
    ConditionalTABDLMDataset,
    collate_and_mask,
)
from attribute_generation.conditional_tabdlm.diffusion_diagnostics import (  # noqa: E402
    PROGRESSIVE_CONDITION_SPECS,
    text_generation_diagnostics,
    unique_run_root,
)
from attribute_generation.conditional_tabdlm.hierarchical_sample import (  # noqa: E402
    align_oracle_frame,
    sample_content_logits,
    zero_history_side,
)
from attribute_generation.conditional_tabdlm.hierarchical_train import (  # noqa: E402
    corrupt_categorical_values,
)
from attribute_generation.conditional_tabdlm.sample import (  # noqa: E402
    sample_categorical_logits,
)
from attribute_generation.conditional_tabdlm.neighbor_sampling import (  # noqa: E402
    TemporalHistoryIndex,
)
from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMSchema,
)
from attribute_generation.conditional_tabdlm.tokenization import (  # noqa: E402
    CategoryVocab,
    SimpleTextTokenizer,
)
from attribute_generation.conditional_tabdlm.train import denoising_loss  # noqa: E402
from evaluation.paper_metrics.c2st_sanity import c2st_integrity_audit  # noqa: E402
from scripts.compare_structured_attribute_generators import (  # noqa: E402
    empirical_conditional_baseline,
    foreign_key_conditioned_target_error,
)
from scripts.prepare_hierarchical_diffusion_benchmark import (  # noqa: E402
    select_evaluation_rows,
)
from scripts.run_hierarchical_diffusion_diagnostics import (  # noqa: E402
    build_run_specifications,
)
from scripts.hierarchical_training_ablation_utils import (  # noqa: E402
    write_ablation_comparison,
)


def diagnostic_schema() -> ConditionalTABDLMSchema:
    return ConditionalTABDLMSchema(
        foreign_key_columns=("user_id", "item_id"),
        datetime_columns=("event_time",),
        categorical_targets=("score",),
        auxiliary_categorical_targets=("body_length_bucket",),
        text_targets=("body",),
        text_max_lengths={"body": 8},
        review_text_length_buckets={"short": (0, 3), "long": (4, 6)},
    )


def test_progressive_modes_distinguish_oracle_upper_bounds():
    assert set(PROGRESSIVE_CONDITION_SPECS) == {"O1", "O2", "O3", "O4", "O5"}
    assert PROGRESSIVE_CONDITION_SPECS["O1"].oracle_structured
    assert PROGRESSIVE_CONDITION_SPECS["O2"].oracle_lengths
    assert not PROGRESSIVE_CONDITION_SPECS["O3"].oracle_lengths
    assert PROGRESSIVE_CONDITION_SPECS["O4"].valid_generative_baseline
    assert PROGRESSIVE_CONDITION_SPECS["O5"].graph_mode == "no_graph"


def test_oracle_alignment_uses_event_keys_and_duplicate_occurrence():
    spine = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["i1", "i1", "i2"],
            "event_time": ["2020-01-01", "2020-01-01", "2020-01-02"],
        }
    )
    source = spine.copy()
    source["score"] = ["first", "second", "third"]
    source = source.iloc[[2, 0, 1]].reset_index(drop=True)
    aligned, metadata = align_oracle_frame(
        spine,
        source,
        condition_columns=["user_id", "item_id", "event_time"],
    )
    assert aligned["score"].tolist() == ["first", "second", "third"]
    assert metadata["all_rows_matched"] is True


def test_oracle_alignment_rejects_invalid_timestamp_and_missing_event():
    spine = pd.DataFrame(
        {"user_id": ["u"], "item_id": ["i"], "event_time": ["not-a-time"]}
    )
    source = spine.assign(score="x")
    with pytest.raises(ValueError, match="invalid values"):
        align_oracle_frame(
            spine,
            source,
            condition_columns=["user_id", "item_id", "event_time"],
        )
    valid = spine.assign(event_time="2020-01-01")
    other = source.assign(event_time="2020-01-02")
    with pytest.raises(ValueError, match="does not contain"):
        align_oracle_frame(
            valid,
            other,
            condition_columns=["user_id", "item_id", "event_time"],
        )


@pytest.mark.parametrize(
    "policy", ["current", "greedy", "top_k", "nucleus", "temperature"]
)
def test_content_sampling_never_returns_special_tokens(policy):
    tokenizer = SimpleTextTokenizer().fit(["alpha beta gamma delta"])
    logits = torch.full((100, tokenizer.vocab_size), -20.0)
    logits[:, list(tokenizer.special_ids)] = 100.0
    logits[:, tokenizer.vocab["alpha"]] = 1.0
    sampled = sample_content_logits(
        logits,
        tokenizer,
        decoding_policy=policy,
        temperature=1.0,
        top_p=0.9,
        top_k=2,
    )
    assert not set(sampled.tolist()).intersection(tokenizer.special_ids)
    assert tokenizer.vocab[tokenizer.empty_token] not in sampled.tolist()


def test_padding_can_be_excluded_consistently_from_attention_and_loss():
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "event_time": ["2020-01-01", "2020-01-02"],
            "score": ["1", "2"],
            "body": ["short", "also short"],
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("user_id", "item_id"),
        datetime_columns=("event_time",),
        categorical_targets=("score",),
        text_targets=("body",),
        text_max_lengths={"body": 8},
    )
    vocabs = {"score": CategoryVocab.from_values("score", frame["score"])}
    tokenizer = SimpleTextTokenizer().fit(frame["body"])
    dataset = ConditionalTABDLMDataset(frame, schema, vocabs, tokenizer, 32)
    batch = collate_and_mask(
        [dataset[0], dataset[1]],
        schema,
        vocabs,
        tokenizer,
        min_mask_prob=1.0,
        max_mask_prob=1.0,
        mask_padding_in_attention=True,
    )
    clean = torch.stack([dataset[0]["text_ids"]["body"], dataset[1]["text_ids"]["body"]])
    padding = clean == tokenizer.pad_id
    assert (batch["text_attention"]["body"][padding] == 0).all()
    assert (batch["text_labels"]["body"][padding] == -100).all()


def test_condition_corruption_stays_in_vocab_or_explicit_mask():
    schema = diagnostic_schema()
    vocabs = {
        "score": CategoryVocab.from_values("score", ["1", "2", "3"]),
        "body_length_bucket": CategoryVocab.from_values(
            "body_length_bucket", ["short", "long"]
        ),
    }
    clean = torch.tensor([[0, 0], [1, 1], [2, 0]], dtype=torch.long)
    replaced = corrupt_categorical_values(
        clean,
        vocabs,
        schema,
        {"replacement_probability": 1.0, "mask_probability": 0.0},
    )
    assert (replaced != clean).all()
    for index, column in enumerate(schema.model_categorical_targets):
        assert int(replaced[:, index].min()) >= 0
        assert int(replaced[:, index].max()) < vocabs[column].size
    masked = corrupt_categorical_values(
        clean,
        vocabs,
        schema,
        {"replacement_probability": 0.0, "mask_probability": 1.0},
    )
    for index, column in enumerate(schema.model_categorical_targets):
        assert (masked[:, index] == vocabs[column].mask_id).all()


def test_categorical_sampling_has_zero_support_for_missing_token():
    vocab = CategoryVocab.from_values("score", ["one", "two"])
    missing_id = vocab.token_to_id["<missing>"]
    logits = torch.full((200, vocab.size), -1000.0)
    logits[:, missing_id] = 1000.0
    sampled = sample_categorical_logits(
        logits, "score", vocab, temperature=1.0
    )
    assert missing_id not in sampled.tolist()


def test_loss_groups_are_normalized_then_weighted_once():
    schema = diagnostic_schema()
    tokenizer = SimpleTextTokenizer().fit(["alpha beta"])
    score_vocab = CategoryVocab.from_values("score", ["1", "2"])
    length_vocab = CategoryVocab.from_values(
        "body_length_bucket", ["short", "long"]
    )
    logits = {
        "categorical": {
            "score": torch.zeros((2, score_vocab.size)),
            "body_length_bucket": torch.zeros((2, length_vocab.size)),
        },
        "text": {
            "body": torch.zeros((2, 2, tokenizer.vocab_size)),
        },
    }
    batch = {
        "foreign_key_ids": torch.zeros((2, 2), dtype=torch.long),
        "categorical_labels": torch.tensor([[0, 0], [1, 1]]),
        "text_labels": {
            "body": torch.tensor(
                [
                    [tokenizer.vocab["alpha"], tokenizer.vocab["beta"]],
                    [tokenizer.vocab["beta"], tokenizer.vocab["alpha"]],
                ]
            )
        },
    }
    loss, components = denoising_loss(
        logits,
        batch,
        schema,
        text_tokenizer=tokenizer,
        loss_group_weights={
            "structured": 2.0,
            "auxiliary": 3.0,
            "review": 4.0,
        },
        field_loss_groups={
            "score": "structured",
            "body_length_bucket": "auxiliary",
            "body": "review",
        },
    )
    expected = sum(
        components[column]["weighted_mean_loss"]
        for column in ("score", "body_length_bucket", "body")
    )
    assert float(loss) == pytest.approx(expected)
    assert components["body"]["loss_group"] == "review"


def test_zero_history_side_does_not_remove_other_side():
    batch = {
        "customer_history_mask": torch.ones((2, 3), dtype=torch.bool),
        "customer_history_row_index": torch.ones((2, 3), dtype=torch.long),
        "customer_history_time": torch.ones((2, 3)),
        "product_history_mask": torch.ones((2, 4), dtype=torch.bool),
    }
    zero_history_side(batch, "customer")
    assert not batch["customer_history_mask"].any()
    assert (batch["customer_history_row_index"] == -1).all()
    assert batch["product_history_mask"].all()


def test_history_coverage_frame_labels_cold_partial_and_warm_rows():
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u1"],
            "item_id": ["i1", "i2", "i1", "i1"],
            "event_time": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
            ],
        }
    )
    index = TemporalHistoryIndex(
        frame,
        "user_id",
        "item_id",
        "event_time",
        32,
    )
    coverage = index.coverage_frame_for_rows([0, 1, 2, 3])
    assert coverage["history_group"].tolist() == [
        "cold",
        "partial",
        "partial",
        "warm",
    ]
    assert coverage.loc[3, "customer_history_count"] == 2
    assert coverage.loc[3, "product_history_count"] == 2


def test_text_diagnostics_flags_leakage_and_duplication():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("user_id",),
        datetime_columns=("event_time",),
        categorical_targets=("score",),
        text_targets=("body",),
        text_max_lengths={"body": 8},
    )
    real = pd.DataFrame({"body": ["alpha beta", "gamma delta"]})
    synthetic = pd.DataFrame({"body": ["alpha beta", "[PAD] alpha"]})
    metrics = text_generation_diagnostics(
        real, synthetic, schema=schema, tokenizer=SimpleTextTokenizer()
    )
    body = metrics["per_column"]["body"]
    assert body["special_token_leakage_rate"] == 0.5
    assert body["padding_token_leakage_rate"] == 0.5
    assert body["exact_training_row_duplication_rate"] == 0.5


def test_c2st_integrity_controls_pass_on_iid_fixture():
    rng = np.random.RandomState(3)
    real = pd.DataFrame(
        {
            "value": rng.normal(size=600),
            "label": rng.choice(["a", "b", "c"], size=600),
            "text": rng.choice(["alpha beta", "gamma delta"], size=600),
        }
    )
    config = {
        "table": {
            "columns": {
                "value": {"type": "numerical"},
                "label": {"type": "categorical"},
                "text": {"type": "text"},
            }
        },
        "evaluation": {
            "c2st": {
                "enabled": True,
                "classifiers": ["logistic_regression"],
                "n_splits": 3,
            }
        },
    }
    audit = c2st_integrity_audit(
        real,
        config,
        max_rows_per_side=200,
        chance_tolerance=0.25,
        corruption_auc_minimum=0.75,
    )
    assert audit["all_controls_passed"]


def test_benchmark_selection_and_run_matrix_are_explicit(tmp_path):
    frame = pd.DataFrame({"value": range(10)})
    assert len(select_evaluation_rows(frame, "all")) == 10
    assert select_evaluation_rows(frame, 3)["value"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        select_evaluation_rows(frame, 11)
    specifications = build_run_specifications(
        {
            "matrices": {
                "progressive_conditioning": ["O1", "O5"],
                "graph_context": [
                    {"name": "graph_shuffled", "mode": "shuffled"}
                ],
            }
        },
        {"progressive_conditioning", "graph_context"},
    )
    assert [item["label"] for item in specifications] == [
        "O1",
        "O5",
        "graph_shuffled",
    ]
    first = unique_run_root(tmp_path, "experiment")
    second = unique_run_root(tmp_path, "experiment")
    assert first != second
    assert first.exists() and second.exists()


def test_empirical_baseline_samples_joint_target_rows():
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["i1", "i1", "i2"],
            "event_time": ["2020-01-01", "2020-01-02", "2020-02-01"],
            "score": ["1", "2", "3"],
            "flag": ["x", "y", "z"],
        }
    )
    spine = train[["user_id", "item_id", "event_time"]].copy()
    sampled = empirical_conditional_baseline(
        train,
        spine,
        ("user_id", "item_id"),
        ("event_time",),
        ["user_id", "item_id", "event_time"],
        ["score", "flag"],
        np.random.RandomState(1),
    )
    valid_pairs = set(zip(train["score"], train["flag"]))
    assert set(zip(sampled["score"], sampled["flag"])).issubset(valid_pairs)


def test_foreign_key_conditioned_error_detects_entity_collapse():
    real = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "rating": ["1", "1", "5", "5"],
            "price": [1.0, 1.0, 5.0, 5.0],
        }
    )
    matching = real.copy()
    collapsed = real.assign(rating="1", price=1.0)
    assert foreign_key_conditioned_target_error(
        real,
        matching,
        foreign_key="user_id",
        target="rating",
        categorical=True,
    ) == pytest.approx(0.0)
    assert foreign_key_conditioned_target_error(
        real,
        collapsed,
        foreign_key="user_id",
        target="rating",
        categorical=True,
    ) > 0.0
    assert foreign_key_conditioned_target_error(
        real,
        collapsed,
        foreign_key="user_id",
        target="price",
        categorical=False,
    ) > 0.0


def test_training_ablation_results_are_consolidated(tmp_path):
    diagnostic_root = tmp_path / "diagnostic"
    diagnostic_root.mkdir()
    pd.DataFrame(
        [
            {
                "matrix": "progressive_conditioning",
                "label": "O1",
                "seed": 42,
                "status": "completed",
                "text_embedding_c2st_error": 0.2,
            }
        ]
    ).to_csv(diagnostic_root / "consolidated_results.csv", index=False)
    write_ablation_comparison(
        [
            {
                "variant": "baseline",
                "seed": 42,
                "training_seconds": 12.0,
                "diagnostics_root": str(diagnostic_root),
            }
        ],
        tmp_path,
    )
    comparison = pd.read_csv(tmp_path / "ablation_results.csv")
    assert comparison.loc[0, "training_variant"] == "baseline"
    assert (tmp_path / "ablation_aggregate_mean_std.csv").exists()
    assert (tmp_path / "ablation_report.md").exists()
