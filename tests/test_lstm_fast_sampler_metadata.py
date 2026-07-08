from __future__ import annotations

import sys
import copy
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    FastSamplerOptions,
    fast_sampler_metadata,
    sample_lstm_fast_from_config,
)
from attribute_generation.conditional_tabdlm.lstm_joint import save_lstm_checkpoint  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


def test_lstm_fast_sampler_metadata_contains_optimization_flags():
    _, config, vocabs, _, _ = make_lstm_fast_fixture()
    metadata = fast_sampler_metadata(
        Path("best.pt"),
        Path("spine.csv"),
        Path("synthetic.csv"),
        rows=12,
        batch_size=4,
        temperature=0.9,
        top_p=0.95,
        seed=42,
        config=config,
        vocabs=vocabs,
        options=FastSamplerOptions(profile=True),
        mixed_precision_used=True,
        torch_compile_used=False,
        total_seconds=1.5,
    )

    assert metadata["optimized_sampler"] is True
    assert metadata["decode_mode"] == "bucketed"
    assert metadata["graph_context_cached"] is True
    assert metadata["condition_embeddings_cached"] is True
    assert metadata["active_row_masking"] is True
    assert metadata["length_bucketed_decoding"] is True
    assert metadata["joint_generation"] is True
    assert metadata["review_text_generated_jointly"] is True
    assert metadata["review_text_separate_stage"] is False


def test_lstm_fast_sampler_tiny_sampling_writes_metadata(tmp_path):
    frame, config, vocabs, tokenizer, model = make_lstm_fast_fixture()
    raw = copy.deepcopy(config.raw)
    raw["graph_conditioning"] = {"enabled": False}
    raw["paths"] = {
        "train_data_path": str(tmp_path / "real.csv"),
        "synthetic_spine_path": str(tmp_path / "spine.csv"),
        "output_dir": str(tmp_path / "out"),
    }
    tiny_config = ConditionalTABDLMConfig(raw=raw, schema=config.schema, config_path=None)
    frame.to_csv(raw["paths"]["synthetic_spine_path"], index=False)
    checkpoint = tmp_path / "out" / "checkpoints" / "best.pt"
    save_lstm_checkpoint(checkpoint, model, tiny_config, vocabs, tokenizer, epoch=1, valid_metrics={})

    output = sample_lstm_fast_from_config(
        tiny_config,
        checkpoint_path=checkpoint,
        output_path=tmp_path / "out" / "synthetic_review_attrs_fast.csv",
        num_rows=2,
        batch_size=2,
        device="cpu",
        options=FastSamplerOptions(profile=True, max_batch_size=2, write_chunk_size=2),
    )

    generated = pd.read_csv(output)
    metadata_path = output.parent / "metadata" / "fast_sampler_metadata.json"
    runtime_path = output.parent / "metadata" / "runtime_sampling_fast.json"
    assert list(generated.columns) == ["customer_id", "product_id", "review_time", "rating", "verified", "summary", "review_text"]
    assert metadata_path.exists()
    assert runtime_path.exists()
