from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import FastSamplerOptions, sample_lstm_fast_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.lstm_joint import save_lstm_checkpoint  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


def test_lstm_privacy_fast_sampler_schema(tmp_path):
    frame, config, vocabs, tokenizer, model = make_lstm_fast_fixture()
    raw = copy.deepcopy(config.raw)
    raw["graph_conditioning"] = {"enabled": False}
    raw["paths"] = {
        "train_data_path": str(tmp_path / "real.csv"),
        "synthetic_spine_path": str(tmp_path / "spine.csv"),
        "output_dir": str(tmp_path / "out"),
    }
    raw["sampling"] = {
        "temperature": {"categorical": 0.9, "summary": 1.05, "review_text": 0.95},
        "top_p": {"categorical": 0.95, "summary": 0.97, "review_text": 0.95},
        "no_repeat_ngram": {"enabled": True, "summary_ngram_size": 3, "review_text_ngram_size": 4},
        "exact_train_overlap_blocking": {"enabled": True, "max_resample_attempts": {"summary": 1, "review_text": 1}},
    }
    tiny_config = ConditionalTABDLMConfig(raw=raw, schema=config.schema, config_path=None)
    frame.to_csv(raw["paths"]["synthetic_spine_path"], index=False)
    frame.to_csv(raw["paths"]["train_data_path"], index=False)
    checkpoint = tmp_path / "out" / "checkpoints" / "best.pt"
    save_lstm_checkpoint(checkpoint, model, tiny_config, vocabs, tokenizer, epoch=1, valid_metrics={})

    output = sample_lstm_fast_from_config(
        tiny_config,
        checkpoint_path=checkpoint,
        output_path=tmp_path / "out" / "synthetic_review_attrs_fast.csv",
        num_rows=2,
        batch_size=2,
        device="cpu",
        options=FastSamplerOptions(
            profile=True,
            max_batch_size=2,
            write_chunk_size=2,
            no_repeat_ngram_enabled=True,
            summary_no_repeat_ngram_size=3,
            review_text_no_repeat_ngram_size=4,
            exact_train_overlap_blocking_enabled=True,
            max_summary_resample_attempts=1,
            max_review_text_resample_attempts=1,
        ),
    )
    generated = pd.read_csv(output)

    assert list(generated.columns) == ["customer_id", "product_id", "review_time", "rating", "verified", "summary", "review_text"]
