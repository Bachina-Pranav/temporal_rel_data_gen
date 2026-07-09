from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.lstm_joint import write_lstm_model_metadata  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


def test_lstm_privacy_alignment_metadata(tmp_path):
    _, config, _, _, _ = make_lstm_fast_fixture()
    raw = copy.deepcopy(config.raw)
    raw["experiment_name"] = "conditional_tabdlm_exp5_1_lstm_privacy_alignment"
    raw["base_experiment"] = "conditional_tabdlm_exp5_lstm_joint_full_review_text"
    raw["review_text_decoder"] = {
        "condition_on_summary": True,
        "summary_condition_type": "final_hidden_plus_mean_pool",
    }
    raw["loss"] = {"text_label_smoothing": {"enabled": True, "summary": 0.05, "review_text": 0.03}}
    raw["training_regularization"] = {"decoder_input_token_dropout": {"enabled": True, "summary": 0.1, "review_text": 0.05}}
    config = type(config)(raw=raw, schema=config.schema, config_path=None)

    write_lstm_model_metadata(config, tmp_path)
    metadata = json.loads((tmp_path / "model_metadata.json").read_text())

    assert metadata["experiment_name"] == "conditional_tabdlm_exp5_1_lstm_privacy_alignment"
    assert metadata["review_text_conditioned_on_summary"] is True
    assert metadata["joint_generation"] is True
    assert metadata["review_text_separate_stage"] is False
    assert metadata["loss_weighting"]["mgda_enabled"] is False
