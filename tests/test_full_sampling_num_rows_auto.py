from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from run_sample_best_model_full_event_table import run_full_sampling  # noqa: E402


class DummyOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_full_sampling_num_rows_auto_passes_real_row_count(tmp_path):
    real = pd.DataFrame({"x": range(7)})
    spine = pd.DataFrame({"x": range(10)})
    real_path = tmp_path / "real.csv"
    spine_path = tmp_path / "spine.csv"
    output_path = tmp_path / "synthetic.csv"
    profile_path = tmp_path / "runtime.json"
    real.to_csv(real_path, index=False)
    spine.to_csv(spine_path, index=False)
    captured = {}

    def fake_sampler(config, **kwargs):
        captured["num_rows"] = kwargs["num_rows"]
        pd.DataFrame({"x": range(kwargs["num_rows"])}).to_csv(kwargs["output_path"], index=False)
        return Path(kwargs["output_path"])

    args = argparse.Namespace(
        real_table=str(real_path),
        synthetic_spine=str(spine_path),
        config="dummy.yaml",
        checkpoint="best.pt",
        output=str(output_path),
        profile_output=str(profile_path),
        num_rows="auto",
        batch_size=None,
        device=None,
        seed=42,
        length_preserving_exact_blocking=True,
        disable_review_text_ngram_blocking=True,
        auto_batch_size=True,
        mixed_precision=True,
        write_chunk_size=10000,
    )

    run_full_sampling(args, sampler_fn=fake_sampler, load_config_fn=lambda path: {"path": path}, options_cls=DummyOptions)

    assert captured["num_rows"] == 7
    metadata = pd.read_json(profile_path, typ="series")
    assert int(metadata["num_requested_rows"]) == 7
    assert int(metadata["num_generated_rows"]) == 7
