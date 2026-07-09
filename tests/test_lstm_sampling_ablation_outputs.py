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
from run_lstm_sampling_privacy_ablation import (  # noqa: E402
    VARIANTS,
    sampling_config_payload,
    variant_paths,
    write_sampling_config,
)


def test_sampling_ablation_variant_paths_match_required_layout(tmp_path):
    paths = variant_paths(tmp_path, "v5_exact_block_only")

    assert paths.output_csv == tmp_path / "runs" / "v5_exact_block_only" / "synthetic_review_attrs_fast.csv"
    assert paths.runtime_json == tmp_path / "runs" / "v5_exact_block_only" / "metadata" / "runtime_sampling_fast.json"
    assert paths.sampling_config_json == tmp_path / "runs" / "v5_exact_block_only" / "metadata" / "sampling_config.json"
    assert paths.eval_json == (
        tmp_path
        / "runs"
        / "v5_exact_block_only"
        / "evaluation"
        / "eval_metrics_fast_sampler_fixed_decode_normalized.json"
    )


def test_sampling_ablation_writes_sampling_config(tmp_path):
    _, config, _, _, _ = make_lstm_fast_fixture()
    args = argparse.Namespace(
        synthetic_spine="spine.csv",
        real_reviews="real.csv",
        num_rows=5000,
        seed=42,
    )
    path = tmp_path / "runs" / "v5_exact_block_only" / "metadata" / "sampling_config.json"

    write_sampling_config(VARIANTS["v5_exact_block_only"], args, config, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == sampling_config_payload(VARIANTS["v5_exact_block_only"], args, config)
    assert payload["ablation_name"] == "v5.2_lstm_sampling_privacy_ablation"
    assert payload["variant"] == "v5_exact_block_only"
    assert payload["sampler_defaults"]["decode_mode"] == "bucketed"
