from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.dataset import (  # noqa: E402
    load_category_vocabs,
    load_text_tokenizer,
)
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.train import (  # noqa: E402
    build_model,
    save_checkpoint,
)
from scripts.prepare_hierarchical_diffusion_benchmark import (  # noqa: E402
    prepare_fixed_benchmark,
)
from scripts.run_hierarchical_diffusion_diagnostics import (  # noqa: E402
    run_diagnostic_experiment,
)


def test_small_diagnostic_pipeline_runs_end_to_end(tmp_path):
    data_path = tmp_path / "events.csv"
    rows = []
    for index in range(40):
        rows.append(
            {
                "user_id": f"u{index % 5}",
                "item_id": f"i{index % 7}",
                "event_time": f"2020-01-{1 + index % 28:02d} "
                f"{index // 28:02d}:00:00",
                "score": str(1 + index % 3),
                "body": "alpha beta" if index % 2 else "gamma delta epsilon",
            }
        )
    pd.DataFrame(rows).to_csv(data_path, index=False)
    model_output = tmp_path / "model"
    model_config_path = tmp_path / "model.yaml"
    model_config = {
        "experiment_name": "tiny_hierarchical_diagnostic",
        "dataset_name": "tiny",
        "paths": {
            "train_data_path": str(data_path),
            "synthetic_spine_path": str(tmp_path / "unused.csv"),
            "output_dir": str(model_output),
        },
        "columns": {
            "condition": {
                "foreign_keys": ["user_id", "item_id"],
                "datetimes": ["event_time"],
            },
            "target": {
                "categorical": ["score"],
                "numerical": [],
                "text": ["body"],
            },
        },
        "auxiliary_targets": {
            "categorical": ["summary_length_bucket"]
        },
        "schema": {
            "fields": {
                "score": {
                    "type": "categorical",
                    "generation_role": "structured",
                },
                "summary_length_bucket": {
                    "type": "categorical",
                    "generation_role": "structured",
                },
                "body": {
                    "type": "text",
                    "generation_role": "text",
                    "length_field": "summary_length_bucket",
                },
            }
        },
        "generation": {
            "factorization": "structured_then_text",
            "stages": [
                {
                    "name": "structured",
                    "fields": ["score", "summary_length_bucket"],
                    "condition_on": ["event_context"],
                },
                {
                    "name": "text",
                    "fields": ["body"],
                    "condition_on": ["structured", "event_context"],
                },
            ],
        },
        "text": {"max_length": {"body": 8}},
        "summary_length": {
            "enabled": True,
            "use_length_bucket_in_sampling": True,
            "force_pad_after_eos": True,
            "buckets": {"short": [1, 3], "long": [4, 6]},
        },
        "tokenizer": {"max_vocab_size": 100, "min_frequency": 1},
        "id_encoding": {"num_buckets": 32, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
        "model": {
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "condition_dim": 8,
            "use_graph_context": False,
        },
        "diffusion": {
            "timesteps": 2,
            "sampling_steps": 1,
            "mask_schedule": "linear",
            "min_mask_prob": 0.1,
            "max_mask_prob": 0.9,
        },
        "sampling": {"minimum_text_content_tokens": 1},
    }
    model_config_path.write_text(
        yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8"
    )
    benchmark_dir = tmp_path / "benchmark"
    manifest = prepare_fixed_benchmark(
        config_path=model_config_path,
        output_dir=benchmark_dir,
        num_evaluation_rows="all",
        seed=42,
    )
    config = load_config(model_config_path)
    vocabs = load_category_vocabs(config)
    tokenizer = load_text_tokenizer(config)
    model = build_model(config, vocabs, tokenizer)
    checkpoint_path = model_output / "checkpoints" / "best.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        config,
        vocabs,
        tokenizer,
        epoch=0,
        valid_metrics={"total_loss": 0.0},
    )
    evaluation_config_path = tmp_path / "evaluation.yaml"
    evaluation_config = {
        "dataset_name": "tiny",
        "table": {
            "columns": {
                "user_id": {"type": "foreign_key", "nullable": False},
                "item_id": {"type": "foreign_key", "nullable": False},
                "event_time": {"type": "datetime", "nullable": False},
                "score": {
                    "type": "categorical",
                    "valid_values": ["1", "2", "3"],
                    "nullable": False,
                },
                "body": {"type": "text", "nullable": False},
            }
        },
        "evaluation": {
            "random_seed": 42,
            "text": {
                "embedding_model": "dummy",
                "cache_embeddings": False,
                "max_text_rows": 10,
                "text_columns": ["body"],
            },
            "c2st": {"enabled": False},
        },
    }
    evaluation_config_path.write_text(
        yaml.safe_dump(evaluation_config, sort_keys=False),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment = {
        "experiment_name": "tiny_diagnostic",
        "output_root": str(tmp_path / "runs"),
        "benchmark": {
            "manifest": str(benchmark_dir / "benchmark_manifest.json")
        },
        "model": {
            "config": str(model_config_path),
            "checkpoint": str(checkpoint_path),
        },
        "evaluation": {"config": str(evaluation_config_path)},
        "seeds": [42],
        "enabled_matrices": ["progressive_conditioning"],
        "sampling": {
            "batch_size": 2,
            "structured_steps": 1,
            "text_steps": 1,
            "inference_dtype": "float32",
            "text_top_k": 4,
            "top_p": 1.0,
            "temperature": 1.0,
            "minimum_text_content_tokens": 1,
            "device": "cpu",
        },
        "matrices": {"progressive_conditioning": ["O5"]},
    }
    experiment_path.write_text(
        yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8"
    )
    run_root = run_diagnostic_experiment(
        experiment,
        experiment_config_path=experiment_path,
        device_override="cpu",
    )
    results = pd.read_csv(run_root / "consolidated_results.csv")
    assert results["label"].tolist() == ["O5"]
    assert results["status"].tolist() == ["completed"]
    with (run_root / "diagnosis_and_recommendation.json").open() as handle:
        diagnosis = json.load(handle)
    assert diagnosis["recommendation"]["status"] == "not_yet_determined"
    assert manifest["row_counts"]["evaluation_real"] > 0
