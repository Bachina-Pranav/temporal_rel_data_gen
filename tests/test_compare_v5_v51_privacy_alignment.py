from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compare_v5_v51_privacy_alignment_writes_outputs(tmp_path):
    v5 = tmp_path / "v5.json"
    v51 = tmp_path / "v51.json"
    v5_runtime = tmp_path / "v5_runtime.json"
    v51_runtime = tmp_path / "v51_runtime.json"
    payload = {
        "marginal_categorical": {"rating_distribution_l1": 0.14, "verified_distribution_l1": 0.07},
        "joint": {"rating_verified_joint_l1": 0.16},
        "length_diagnostics": {"summary_length_ks": 0.05, "review_text_length_ks": 0.11},
        "text_privacy": {
            "summary_exact_train_overlap_rate": 0.29,
            "review_text_exact_train_overlap_rate": 0.008,
            "summary_nearest_neighbor_rougeL_mean": 0.2,
            "review_text_nearest_neighbor_rougeL_mean": 0.1,
        },
        "text_consistency": {
            "real_summary_review_text_rougeL_mean": 0.07,
            "synthetic_summary_review_text_rougeL_mean": 0.03,
            "real_summary_review_text_token_jaccard_mean": 0.05,
            "synthetic_summary_review_text_token_jaccard_mean": 0.02,
            "rating_text_consistency_accuracy": 0.71,
            "rating_review_text_consistency_accuracy": 0.70,
            "verified_review_text_predictor_auc": 0.84,
        },
    }
    v5.write_text(json.dumps(payload))
    improved = json.loads(json.dumps(payload))
    improved["text_privacy"]["summary_exact_train_overlap_rate"] = 0.12
    improved["text_consistency"]["synthetic_summary_review_text_rougeL_mean"] = 0.05
    v51.write_text(json.dumps(improved))
    v5_runtime.write_text(json.dumps({"total_sampling_seconds": 475, "rows_per_second": 105, "projected_hours_for_10m_rows": 26}))
    v51_runtime.write_text(json.dumps({"total_sampling_seconds": 500, "rows_per_second": 100, "projected_hours_for_10m_rows": 27}))
    output_dir = tmp_path / "comparison"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/scripts/compare_v5_v51_privacy_alignment.py"),
            "--v5",
            str(v5),
            "--v51",
            str(v51),
            "--v5-runtime",
            str(v5_runtime),
            "--v51-runtime",
            str(v51_runtime),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "comparison.md").exists()
