from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from attribute_generation.conditional_tabdlm.utils import save_json  # noqa: E402
from train_lstm_joint_full_review_text import load_config_with_overrides, write_training_metadata  # noqa: E402


def test_fixed_step_training_metadata_records_scaling_fields(tmp_path):
    real = tmp_path / "real.csv"
    real.write_text(
        "review_time,customer_id,product_id,rating,verified,summary,review_text\n"
        "2020-01-01,1,10,5,True,good,good product\n"
        "2020-01-02,2,20,1,False,bad,bad product\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config_path.write_text(
        f"""
dataset_name: rel_amazon
model_type: conditional_tabdlm_lstm_joint_full_text
model_family: conditional_tabdlm_lstm_joint_full_text
paths:
  train_data_path: {real}
  synthetic_spine_path: {real}
  output_dir: {output_dir}
columns:
  condition:
    foreign_keys: [customer_id, product_id]
    datetimes: [review_time]
  target:
    categorical: [rating, verified]
    numerical: []
    text: [summary, review_text]
text:
  max_length:
    summary: 8
    review_text: 8
training:
  epoch_mode: false
  max_steps: 10
  physical_batch_size: 2
  gradient_accumulation_steps: 4
  mixed_precision: false
""",
        encoding="utf-8",
    )
    args = Namespace(
        config=str(config_path),
        real_table=None,
        synthetic_spine=None,
        output_dir=None,
        mixed_precision=None,
        auto_batch_size=None,
        num_workers=None,
        max_train_rows=None,
        train_row_sampling=None,
        max_steps=None,
        steps_per_eval=None,
        steps_per_checkpoint=None,
        epoch_mode=None,
        sampling_mode=None,
        effective_batch_size=None,
        target_effective_batch_size=None,
        gradient_accumulation_steps=None,
        physical_batch_size=None,
        profile_steps=None,
        warmup_profile_steps=None,
        pretokenized_dir=None,
        neighbor_cache_dir=None,
        amp_dtype=None,
        profile=False,
    )
    config = load_config_with_overrides(args)
    runtime_dir = output_dir / "metadata"
    runtime_dir.mkdir(parents=True)
    save_json(
        {
            "train_mode": "fixed_step",
            "epoch_mode": False,
            "max_steps": 10,
            "physical_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "train_rows_available": 2,
            "train_rows_seen_approx": 80,
            "full_epoch_equivalent_fraction": 40.0,
            "architecture_changed": False,
        },
        runtime_dir / "training_runtime.json",
    )

    write_training_metadata(config, output_dir / "checkpoints" / "best.pt", 1.5)
    metadata = json.loads((output_dir / "training_metadata.json").read_text())

    assert metadata["train_mode"] == "fixed_step"
    assert metadata["epoch_mode"] is False
    assert metadata["max_steps"] == 10
    assert metadata["effective_batch_size"] == 8
    assert metadata["architecture_changed"] is False
