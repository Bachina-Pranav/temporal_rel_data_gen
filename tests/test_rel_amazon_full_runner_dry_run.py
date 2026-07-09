from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from run_rel_amazon_best_model_full_sampling_and_eval import run_rel_amazon_full  # noqa: E402


def test_rel_amazon_full_runner_refuses_short_spine(tmp_path):
    real = tmp_path / "review.csv"
    spine = tmp_path / "synthetic_review.csv"
    pd.DataFrame({"x": range(5)}).to_csv(real, index=False)
    pd.DataFrame({"x": range(4)}).to_csv(spine, index=False)
    args = argparse.Namespace(
        real_table=str(real),
        synthetic_spine=str(spine),
        sampler_config="config.yaml",
        checkpoint="best.pt",
        eval_config="eval.yaml",
        sample_output=str(tmp_path / "sample.csv"),
        runtime_output=str(tmp_path / "runtime.json"),
        eval_output_dir=str(tmp_path / "eval"),
        num_rows="auto",
        batch_size=None,
        device=None,
        seed=None,
        write_chunk_size=100000,
        dry_run=True,
        length_preserving_exact_blocking=True,
        disable_review_text_ngram_blocking=True,
        auto_batch_size=True,
        mixed_precision=True,
    )

    with pytest.raises(SystemExit):
        run_rel_amazon_full(args)
