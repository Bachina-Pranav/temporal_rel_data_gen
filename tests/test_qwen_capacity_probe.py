from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from attribute_generation.qwen_text_decoder.capacity_probe import QwenCapacityProbe


def test_capacity_probe_uses_identical_fixed_rows_and_one_epoch(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    frame = pd.DataFrame(
        {
            "customer_id": range(30),
            "product_id": range(100, 130),
            "review_time": ["2020-01-01"] * 30,
            "rating": [5] * 30,
            "verified": [True] * 30,
            "summary": ["summary"] * 30,
            "review_text": ["review"] * 30,
        }
    )
    frame.to_csv(benchmark / "train_real.csv", index=False)
    frame.head(10).to_csv(benchmark / "validation_real.csv", index=False)
    base = yaml.safe_load(Path("configs/experiments/qwen_text_decoder_06b.yaml").read_text())
    base["data"]["benchmark_dir"] = str(benchmark)
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    config = {
        "seed": 42,
        "base_experiment_config": str(base_path),
        "output_dir": str(tmp_path / "output"),
        "data": {"train_rows": 20, "validation_rows": 8},
        "models": {
            "qwen3_06b": {"model_id": "Qwen/Qwen3-0.6B-Base"},
            "qwen3_17b": {"model_id": "Qwen/Qwen3-1.7B-Base"},
        },
        "training": {
            "epochs": 1,
            "train_batch_size": 8,
            "eval_batch_size": 8,
            "gradient_accumulation_steps": 8,
        },
        "generation": {},
        "evaluation": {},
    }
    config_path = tmp_path / "capacity.yaml"
    config_path.write_text(yaml.safe_dump(config))
    probe = QwenCapacityProbe(config_path)
    manifest = probe.prepare()
    assert manifest["train_rows"] == 20
    assert manifest["validation_rows"] == 8
    resolved = [
        yaml.safe_load((tmp_path / f"output/{label}/config_resolved.yaml").read_text())
        for label in ("qwen3_06b", "qwen3_17b")
    ]
    assert all(item["training"]["epochs"] == 1 for item in resolved)
    assert resolved[0]["data"]["benchmark_dir"] == resolved[1]["data"]["benchmark_dir"]

