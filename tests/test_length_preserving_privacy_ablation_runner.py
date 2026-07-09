from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402
from run_lstm_length_preserving_privacy_ablation import (  # noqa: E402
    VARIANTS,
    fast_options_for_variant,
    sampling_config_payload,
    variant_paths,
    write_sampling_config,
)


def test_length_preserving_ablation_runner_paths_and_options(tmp_path):
    paths = variant_paths(tmp_path, "v51_length_preserving_exact_block")
    args = argparse.Namespace(seed=42)
    options = fast_options_for_variant(VARIANTS["v51_length_preserving_exact_block"], args, paths)

    assert paths.output_csv == tmp_path / "runs" / "v51_length_preserving_exact_block" / "synthetic_review_attrs_fast.csv"
    assert paths.runtime_json == tmp_path / "runs" / "v51_length_preserving_exact_block" / "metadata" / "runtime_sampling_fast.json"
    assert paths.eval_json == (
        tmp_path
        / "runs"
        / "v51_length_preserving_exact_block"
        / "evaluation"
        / "eval_metrics_fast_sampler_fixed_decode_normalized.json"
    )
    assert options.length_preserving_exact_blocking_enabled is True
    assert options.no_repeat_ngram_enabled is False
    assert options.review_text_no_repeat_ngram_enabled is False


def test_length_preserving_ablation_runner_writes_sampling_config(tmp_path):
    _, config, _, _, _ = make_lstm_fast_fixture()
    args = argparse.Namespace(
        synthetic_spine="spine.csv",
        real_reviews="real.csv",
        num_rows=5000,
        seed=42,
    )
    path = tmp_path / "runs" / "v51_length_preserving_exact_block" / "metadata" / "sampling_config.json"

    write_sampling_config(VARIANTS["v51_length_preserving_exact_block"], args, config, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == sampling_config_payload(VARIANTS["v51_length_preserving_exact_block"], args, config)
    assert payload["ablation_name"] == "v5.3_lstm_length_preserving_privacy_sampler"
    assert payload["sampler_defaults"]["length_preserving_exact_blocking"] is True
