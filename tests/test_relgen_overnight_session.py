from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from overnight_session import OvernightSession


def write_script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def base_config(tmp_path: Path, a_script: Path, b_script: Path) -> Path:
    empty = tmp_path / "missing.py"
    config = {
        "session": {
            "output_dir": str(tmp_path / "session"),
            "duration_hours": 1.0,
            "monitor_interval_seconds": 0.05,
            "heartbeat_interval_seconds": 0.05,
            "process_poll_seconds": 0.02,
            "minimum_free_disk_gb": 0,
        },
        "jobs": {
            "relational_prefix_probe": {
                "command": [sys.executable, str(a_script)],
                "timeout_minutes": 0.05,
                "estimated_minutes": 0,
                "decision_files": [str(tmp_path / "decision.json")],
            },
            "relational_prefix_full": {
                "command": [sys.executable, str(b_script), "B1"],
                "timeout_minutes": 0.05,
                "estimated_minutes": 0,
            },
            "qwen_capacity_probe": {
                "command": [sys.executable, str(b_script), "B2"],
                "timeout_minutes": 0.05,
                "estimated_minutes": 0,
            },
            "clavaddpm": {
                "entrypoints": [str(empty)],
                "common_arguments": [],
                "preflight": {"arguments": [], "timeout_minutes": 1, "estimated_minutes": 0},
                "smoke": {"arguments": [], "timeout_minutes": 1, "estimated_minutes": 0},
                "full": {"arguments": [], "timeout_minutes": 1, "estimated_minutes": 0},
            },
            "pending_canonical_evaluation": {"entrypoints": [str(empty)]},
            "repeated_seed_c2st": {"entrypoints": [str(empty)]},
        },
        "artifact_roots": [],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_failure_is_isolated_and_failure_branch_runs(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    marker = tmp_path / "branch.txt"
    write_script(a, "import sys\nprint('failure', file=sys.stderr)\nraise SystemExit(7)\n")
    write_script(b, f"import pathlib,sys\npathlib.Path({str(marker)!r}).write_text(sys.argv[1])\n")
    config = base_config(tmp_path, a, b)
    session = OvernightSession(
        config,
        root=tmp_path,
        wait_for_deadline=False,
        allow_heartbeat_only=True,
    )
    session.run()
    assert session.records["A_relational_prefix_probe"].status == "failed"
    assert session.records["B2_qwen_capacity_probe"].status == "completed"
    assert marker.read_text() == "B2"
    assert (tmp_path / "session/final_overnight_report.md").is_file()


def test_supported_decision_selects_full_prefix_branch(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    decision = tmp_path / "decision.json"
    marker = tmp_path / "branch.txt"
    write_script(a, f"import pathlib\npathlib.Path({str(decision)!r}).write_text('{{\"classification\": \"strongly_supported\"}}')\n")
    write_script(b, f"import pathlib,sys\npathlib.Path({str(marker)!r}).write_text(sys.argv[1])\n")
    config = base_config(tmp_path, a, b)
    session = OvernightSession(
        config,
        root=tmp_path,
        wait_for_deadline=False,
        allow_heartbeat_only=True,
    )
    session.run()
    assert session.records["B1_full_relational_prefix"].status == "completed"
    assert marker.read_text() == "B1"
    payload = json.loads((tmp_path / "session/job_a_decision.json").read_text())
    assert payload["classification"] == "strongly_supported"


def test_timeout_is_recorded_without_killing_session(tmp_path):
    slow = tmp_path / "slow.py"
    b = tmp_path / "b.py"
    write_script(slow, "import time\ntime.sleep(5)\n")
    write_script(b, "pass\n")
    config = base_config(tmp_path, slow, b)
    value = yaml.safe_load(config.read_text())
    value["jobs"]["relational_prefix_probe"]["timeout_minutes"] = 0.001
    config.write_text(yaml.safe_dump(value))
    session = OvernightSession(
        config,
        root=tmp_path,
        wait_for_deadline=False,
        allow_heartbeat_only=True,
    )
    session.run()
    assert session.records["A_relational_prefix_probe"].status == "timed_out"
    assert session.records["B2_qwen_capacity_probe"].status == "completed"


def test_preflight_rejects_heartbeat_only_launch(tmp_path):
    missing = tmp_path / "missing.py"
    config = base_config(tmp_path, missing, missing)
    value = yaml.safe_load(config.read_text())
    value["jobs"]["relational_prefix_probe"].pop("command")
    value["jobs"]["relational_prefix_probe"]["entrypoints"] = [str(missing)]
    value["jobs"]["qwen_capacity_probe"].pop("command")
    value["jobs"]["qwen_capacity_probe"]["entrypoints"] = [str(missing)]
    config.write_text(yaml.safe_dump(value))
    session = OvernightSession(config, root=tmp_path, wait_for_deadline=False)
    report = session.preflight()
    assert report["launch_recommended"] is False
    assert "only produce diagnostics" in " ".join(report["warnings"])


def test_transient_failure_gets_exactly_one_unchanged_retry(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    marker = tmp_path / "attempt.txt"
    write_script(
        a,
        "import pathlib,sys\n"
        f"p=pathlib.Path({str(marker)!r})\n"
        "if not p.exists():\n"
        " p.write_text('first')\n"
        " print('temporary model-loading issue', file=sys.stderr)\n"
        " raise SystemExit(3)\n",
    )
    write_script(b, "pass\n")
    config = base_config(tmp_path, a, b)
    session = OvernightSession(
        config,
        root=tmp_path,
        wait_for_deadline=False,
        allow_heartbeat_only=True,
    )
    session.run()
    record = session.records["A_relational_prefix_probe"]
    assert record.status == "completed"
    assert record.retries == 1
