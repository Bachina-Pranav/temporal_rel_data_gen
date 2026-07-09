from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from run_lstm_sampling_privacy_ablation import DEFAULT_VARIANTS, VARIANTS  # noqa: E402


def test_sampling_ablation_variants_have_correct_checkpoint_sources():
    assert DEFAULT_VARIANTS == [
        "v5_exact_block_only",
        "v5_exact_block_summary_ngram_only",
        "v51_exact_block_only_no_ngram",
        "v51_no_block_no_ngram",
        "v5_summary_exact_block_only",
    ]
    assert VARIANTS["v5_exact_block_only"].checkpoint_source == "v5"
    assert VARIANTS["v5_exact_block_only"].config_path.endswith("exp5_lstm_joint_full_review_text.yaml")
    assert VARIANTS["v51_exact_block_only_no_ngram"].checkpoint_source == "v5.1"
    assert VARIANTS["v51_exact_block_only_no_ngram"].config_path.endswith("exp5_1_lstm_privacy_alignment.yaml")
    assert VARIANTS["v51_no_block_no_ngram"].exact_train_overlap_blocking_enabled is False
    assert VARIANTS["v5_summary_exact_block_only"].summary_exact_blocking_enabled is True
    assert VARIANTS["v5_summary_exact_block_only"].review_text_exact_blocking_enabled is False
