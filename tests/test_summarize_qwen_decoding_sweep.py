from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.summarize_qwen_decoding_sweep import build_markdown, build_summary


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_decoding_sweep_summary_is_compact_and_includes_confirmation(tmp_path):
    root = tmp_path / "qwen" / "decoding_sweep"
    root.mkdir(parents=True)
    rows = []
    for index, name in enumerate(
        ("D0_t090_p095_r105", "D1_t105_p095_r105", "D2_t115_p098_r110")
    ):
        rows.append(
            {
                "configuration": name,
                "temperature": [0.9, 1.05, 1.15][index],
                "top_p": [0.95, 0.95, 0.98][index],
                "repetition_penalty": [1.05, 1.05, 1.1][index],
                "summary_c2st": 0.5 - index * 0.01,
                "review_c2st": 0.7 - index * 0.01,
                "macro_c2st": 0.6 - index * 0.01,
                "review_distinct_2": 0.3 + index * 0.01,
                "review_repeated_ngram_rate": 0.7 - index * 0.01,
                "summary_exact_train_overlap": 0.4,
                "review_exact_train_overlap": 0.05,
                "rating_balanced_accuracy": 0.4,
                "rating_macro_f1": 0.4,
            }
        )
        policy = root / name
        policy.mkdir()
        (policy / "synthetic_text.csv").write_text("summary,review_text\na,b\n")
        write_json(policy / "diversity_metrics.json", {})
        write_json(policy / "memorization_metrics.json", {})
        write_json(policy / "conditioning_metrics.json", {})
        write_json(policy / "generation_metrics.json", {})
    pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
    write_json(
        root / "decoding_decision.json",
        {
            "selected_configuration": "D1_t105_p095_r105",
            "decoding_bottleneck": "STRONGLY SUPPORTED",
            "clearly_preferable_to_D0": True,
        },
    )
    write_json(
        root / "test_confirmation/confirmation_decision.json",
        {
            "run": True,
            "frozen_validation_configuration": "D1_t105_p095_r105",
            "summary_c2st": 0.4,
            "review_c2st": 0.6,
            "macro_c2st": 0.5,
            "macro_improvement_vs_fixed_baseline": 0.1,
        },
    )
    write_json(root / "real_validation_diversity_metrics.json", {})
    write_json(root / "fixed_probe_metrics.json", {})
    summary = build_summary(root)
    markdown = build_markdown(summary)
    assert summary["selection_decision"]["selected_configuration"].startswith("D1")
    assert summary["heldout_test_confirmation"]["run"] is True
    assert "Questions To Brainstorm" in markdown
    assert "D1_t105_p095_r105" in markdown
