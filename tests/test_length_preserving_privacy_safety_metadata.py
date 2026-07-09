from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402
from run_lstm_length_preserving_privacy_ablation import VARIANTS, sampling_config_payload  # noqa: E402


def test_length_preserving_privacy_safety_metadata_preserves_no_leakage_contract():
    _, config, _, _, _ = make_lstm_fast_fixture()
    payload = sampling_config_payload(
        VARIANTS["v51_length_preserving_exact_block"],
        {
            "synthetic_spine": "spine.csv",
            "real_reviews": "real.csv",
            "num_rows": 5000,
            "seed": 42,
        },
        config,
    )

    assert payload["joint_generation"] is True
    assert payload["review_text_generated_jointly"] is True
    assert payload["review_text_separate_stage"] is False
    assert payload["graph_conditioning_mode"] == "structure_only_temporal"
    assert payload["temporal_filter_mode"] == "past_only"
    assert payload["graph_uses_future_events"] is False
    assert payload["graph_uses_target_attributes"] is False
    assert payload["real_graph_used_at_sampling"] is False
