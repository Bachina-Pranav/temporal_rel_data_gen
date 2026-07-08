from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import FastSamplerOptions, fast_sampler_metadata  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


def test_lstm_fast_sampler_graph_safety_metadata():
    _, config, vocabs, _, _ = make_lstm_fast_fixture()
    metadata = fast_sampler_metadata(
        Path("best.pt"),
        Path("spine.csv"),
        Path("synthetic.csv"),
        rows=10,
        batch_size=5,
        temperature=0.9,
        top_p=0.95,
        seed=42,
        config=config,
        vocabs=vocabs,
        options=FastSamplerOptions(),
        mixed_precision_used=False,
        torch_compile_used=False,
        total_seconds=1.0,
    )

    assert metadata["graph_conditioning_mode"] == "structure_only_temporal"
    assert metadata["temporal_filter_mode"] == "past_only"
    assert metadata["graph_uses_future_events"] is False
    assert metadata["graph_uses_target_attributes"] is False
    assert metadata["real_graph_used_at_sampling"] is False
