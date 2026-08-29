"""Deadline-aware, failure-isolated orchestration for RelGen overnight jobs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


FINAL_STATUSES = {
    "completed",
    "failed",
    "skipped",
    "blocked",
    "timed_out",
    "partially_completed",
}
SUPPORTED_A_DECISIONS = {
    "strongly_supported",
    "moderately_supported",
    "weakly_supported",
    "rejected",
    "unresolved",
    "job_failed",
}
TRANSIENT_PATTERNS = (
    "cuda out of memory",
    "cudnn_status_alloc_failed",
    "dataloader worker",
    "worker exited unexpectedly",
    "connection reset",
    "temporary failure",
    "temporarily unavailable",
    "model-loading issue",
    "model loading timeout",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_run(command: Sequence[str], timeout: float = 15) -> str | None:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


@dataclass
class JobRecord:
    job_name: str
    status: str = "skipped"
    start_time: str | None = None
    end_time: str | None = None
    exit_code: int | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    runtime_seconds: float = 0.0
    failure_reason: str | None = None
    command: list[str] | None = None
    retries: int = 0
    branch: str | None = None
    result: dict[str, Any] = field(default_factory=dict)


class OvernightSession:
    """Run independent scientific jobs until an absolute deadline."""

    def __init__(
        self,
        config_path: Path,
        *,
        root: Path,
        device: str = "cuda",
        output_dir: Path | None = None,
        duration_hours: float | None = None,
        wait_for_deadline: bool = True,
        force_new_session: bool = False,
        allow_heartbeat_only: bool = False,
        preflight_only: bool = False,
    ) -> None:
        self.root = root.resolve()
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        session_cfg = self.config.get("session", {})
        configured_output = Path(session_cfg.get("output_dir", "outputs/overnight_8h_session"))
        self.output_dir = (output_dir or configured_output)
        if not self.output_dir.is_absolute():
            self.output_dir = self.root / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.duration_hours = float(duration_hours or session_cfg.get("duration_hours", 8.0))
        self.monitor_interval = float(session_cfg.get("monitor_interval_seconds", 300))
        self.heartbeat_interval = float(session_cfg.get("heartbeat_interval_seconds", 60))
        self.poll_seconds = min(float(session_cfg.get("process_poll_seconds", 15)), 120)
        self.minimum_free_disk_gb = float(session_cfg.get("minimum_free_disk_gb", 5))
        self.retry_limit = int(session_cfg.get("transient_retry_limit", 1))
        self.wait_for_deadline = wait_for_deadline
        self.force_new_session = force_new_session
        self.allow_heartbeat_only = allow_heartbeat_only
        self.preflight_only = preflight_only
        self.manifest_path = self.output_dir / "session_manifest.json"
        self.status_path = self.output_dir / "job_status.json"
        self.log_path = self.output_dir / "session_log.txt"
        self.monitor_path = self.output_dir / "resource_monitor.csv"
        self.heartbeat_path = self.output_dir / "heartbeat.log"
        self.report_path = self.output_dir / "final_overnight_report.md"
        self.decision_path = self.output_dir / "job_a_decision.json"
        self.stop_event = threading.Event()
        self.active_job = "initializing"
        self._active_lock = threading.Lock()
        self.monitor_thread: threading.Thread | None = None
        self.records: dict[str, JobRecord] = {}
        self.start_time: datetime
        self.deadline: datetime
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        if self.manifest_path.exists() and not self.force_new_session:
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("session_state") == "preflight_only" and not self.preflight_only:
                self._initialize_new_session()
                return
            self.start_time = datetime.fromisoformat(manifest["session_start_time"])
            self.deadline = datetime.fromisoformat(manifest["target_end_time"])
            if self.status_path.exists():
                status = json.loads(self.status_path.read_text())
                for value in status.get("jobs", []):
                    record = JobRecord(**value)
                    if record.status == "running":
                        record.status = "partially_completed"
                        record.failure_reason = "Session exited while this job was active."
                    self.records[record.job_name] = record
            self.log("Resuming existing session state.")
            self._write_status()
            return
        self._initialize_new_session()

    def _initialize_new_session(self) -> None:
        self.records = {}
        self.start_time = utc_now()
        self.deadline = self.start_time + timedelta(hours=self.duration_hours)
        atomic_json(self.manifest_path, self._environment_manifest())
        self.log("Initialized overnight session.")
        self._write_status()

    def _environment_manifest(self) -> dict[str, Any]:
        gpu = safe_run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
        cuda = safe_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        git_commit = safe_run(["git", "rev-parse", "HEAD"])
        config_hash = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        return {
            "version": 1,
            "session_state": "preflight_only" if self.preflight_only else "initialized",
            "session_start_time": iso(self.start_time),
            "target_end_time": iso(self.deadline),
            "target_duration_hours": self.duration_hours,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "gpu": gpu,
            "cuda_or_driver_version": cuda,
            "git_commit": git_commit,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.executable,
            "config_path": self.display_path(self.config_path),
            "config_sha256": config_hash,
            "device": self.device,
            "scientific_hyperparameters_may_be_changed_by_watchdog": False,
        }

    def display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path.resolve())

    def log(self, message: str) -> None:
        line = f"[{iso(utc_now())}] {message}"
        with self.log_path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def elapsed_hours(self) -> float:
        return max(0.0, (utc_now() - self.start_time).total_seconds() / 3600)

    def remaining_seconds(self) -> float:
        return max(0.0, (self.deadline - utc_now()).total_seconds())

    def _write_status(self) -> None:
        atomic_json(
            self.status_path,
            {
                "session_status": "running" if not self.stop_event.is_set() else "stopping",
                "active_job": self.active_job,
                "updated_at": iso(utc_now()),
                "seconds_remaining": self.remaining_seconds(),
                "jobs": [asdict(value) for value in self.records.values()],
            },
        )

    def set_active(self, name: str) -> None:
        with self._active_lock:
            self.active_job = name
        self._write_status()

    def start_monitor(self) -> None:
        if not self.monitor_path.exists():
            with self.monitor_path.open("w", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        "timestamp",
                        "active_job",
                        "gpu_utilization_percent",
                        "gpu_memory_used_mb",
                        "gpu_memory_total_mb",
                        "free_disk_gb",
                    ]
                )
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self) -> None:
        while not self.stop_event.is_set():
            query = safe_run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
            gpu_values: list[str | None] = [None, None, None]
            if query:
                gpu_values = [part.strip() for part in query.splitlines()[0].split(",")][:3]
                gpu_values += [None] * (3 - len(gpu_values))
            free_gb = shutil.disk_usage(self.root).free / 1024**3
            with self._active_lock:
                active = self.active_job
            with self.monitor_path.open("a", newline="") as handle:
                csv.writer(handle).writerow([iso(utc_now()), active, *gpu_values, f"{free_gb:.3f}"])
            self.stop_event.wait(self.monitor_interval)

    def resolve_command(self, cfg: Mapping[str, Any], arguments: Sequence[str] | None = None) -> list[str] | None:
        if cfg.get("command"):
            command = [self._substitute(str(value)) for value in cfg["command"]]
            if arguments is not None:
                command += [self._substitute(str(value)) for value in arguments]
            return command
        for entrypoint in cfg.get("entrypoints", []):
            path = Path(entrypoint)
            if not path.is_absolute():
                path = self.root / path
            if path.is_file():
                selected_args = list(arguments if arguments is not None else cfg.get("arguments", []))
                return [sys.executable, self.display_path(path)] + [self._substitute(str(value)) for value in selected_args]
        return None

    def _substitute(self, value: str) -> str:
        return value.format(device=self.device, root=str(self.root), output_dir=str(self.output_dir))

    def preflight(self) -> dict[str, Any]:
        jobs = self.config.get("jobs", {})
        report: dict[str, Any] = {
            "checked_at": iso(utc_now()),
            "ready": {},
            "warnings": [],
            "launch_recommended": True,
        }
        simple_names = (
            "relational_prefix_probe",
            "relational_prefix_full",
            "qwen_capacity_probe",
            "pending_canonical_evaluation",
            "repeated_seed_c2st",
        )
        for name in simple_names:
            cfg = jobs.get(name, {})
            command = self.resolve_command(cfg)
            reusable = self._completion_artifacts_exist(cfg)
            report["ready"][name] = {
                "ready": command is not None or reusable,
                "command": command,
                "reusable_completed_artifacts": reusable,
                "searched_entrypoints": cfg.get("entrypoints", []),
            }
        clava_cfg = jobs.get("clavaddpm", {})
        clava_command = self.resolve_command(clava_cfg, [])
        report["ready"]["clavaddpm"] = {
            "ready": clava_command is not None,
            "command_prefix": clava_command,
            "searched_entrypoints": clava_cfg.get("entrypoints", []),
        }
        a_ready = report["ready"]["relational_prefix_probe"]
        if a_ready["reusable_completed_artifacts"]:
            prior_decision, _ = self._find_job_a_decision()
        else:
            prior_decision = "job_failed"
        selected_followup = (
            "relational_prefix_full"
            if prior_decision in {"strongly_supported", "moderately_supported"}
            else "qwen_capacity_probe"
        )
        scientific_ready = bool(
            a_ready.get("command")
            or report["ready"][selected_followup].get("command")
            or report["ready"]["clavaddpm"].get("command_prefix")
        )
        if not scientific_ready:
            report["launch_recommended"] = False
            report["warnings"].append(
                "No primary scientific runner is present. Launching would only produce diagnostics and heartbeat time."
            )
        if not report["ready"]["relational_prefix_probe"]["ready"]:
            report["warnings"].append(
                "Job A is blocked: the requested existing relational-prefix implementation is absent from this checkout."
            )
        if not report["ready"]["qwen_capacity_probe"]["ready"]:
            report["warnings"].append("Job B2 is blocked: no capacity-probe runner was found.")
        if not report["ready"]["clavaddpm"]["ready"]:
            report["warnings"].append(
                "Job C is blocked: no official ClavaDDPM adapter/runner was found; the watchdog will not invent one."
            )
        free_gb = shutil.disk_usage(self.root).free / 1024**3
        report["free_disk_gb"] = free_gb
        report["minimum_free_disk_gb"] = self.minimum_free_disk_gb
        if free_gb < self.minimum_free_disk_gb:
            report["launch_recommended"] = False
            report["warnings"].append("Free disk is below the configured safety threshold.")
        atomic_json(self.output_dir / "preflight.json", report)
        self._write_preflight_markdown(report)
        return report

    def _write_preflight_markdown(self, report: Mapping[str, Any]) -> None:
        lines = ["# RelGen Overnight Preflight", "", f"Launch recommended: **{report['launch_recommended']}**", ""]
        lines += ["| Job | Ready |", "|---|---:|"]
        for name, value in report["ready"].items():
            lines.append(f"| {name} | {value['ready']} |")
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in report.get("warnings", [])] or ["- None"]
        (self.output_dir / "preflight.md").write_text("\n".join(lines) + "\n")

    def run(self) -> None:
        preflight = self.preflight()
        if not preflight["launch_recommended"] and not self.allow_heartbeat_only:
            raise RuntimeError(
                "Overnight launch refused because no primary scientific job is runnable or disk safety failed. "
                "Inspect preflight.md; pass --allow-heartbeat-only only when that behavior is intentional."
            )
        self._install_signal_handlers()
        self.start_monitor()
        try:
            self.run_job_a()
            decision = self.classify_job_a()
            self.run_job_b(decision)
            self.run_job_c()
            self.run_job_d()
            self.write_report(final=False)
            if self.wait_for_deadline and not self.stop_event.is_set():
                self.run_heartbeat()
        finally:
            self.set_active("final_summary")
            self.write_report(final=True)
            self.print_final_console()
            self.stop_event.set()
            if self.monitor_thread:
                self.monitor_thread.join(timeout=min(self.monitor_interval + 1, 5))
            self.active_job = "complete"
            atomic_json(
                self.status_path,
                {
                    "session_status": "completed",
                    "active_job": "complete",
                    "updated_at": iso(utc_now()),
                    "seconds_remaining": self.remaining_seconds(),
                    "jobs": [asdict(value) for value in self.records.values()],
                },
            )

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def stop(signum: int, _frame: Any) -> None:
            self.log(f"Received signal {signum}; requesting graceful shutdown.")
            self.stop_event.set()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

    def run_job_a(self) -> JobRecord:
        cfg = self.config["jobs"]["relational_prefix_probe"]
        return self.run_configured_job("A_relational_prefix_probe", cfg)

    def classify_job_a(self) -> str:
        record = self.records.get("A_relational_prefix_probe")
        if not record or record.status not in {"completed", "partially_completed"}:
            decision = "job_failed"
            source = record.failure_reason if record else "Job A did not run."
        else:
            decision, source = self._find_job_a_decision()
        payload = {
            "classification": decision,
            "classified_at": iso(utc_now()),
            "validation_only_decision": True,
            "source": source,
            "job_a_status": record.status if record else "missing",
        }
        atomic_json(self.decision_path, payload)
        self.log(f"Job A classification: {decision} ({source})")
        return decision

    def _find_job_a_decision(self) -> tuple[str, str]:
        cfg = self.config["jobs"]["relational_prefix_probe"]
        for candidate in cfg.get("decision_files", []):
            path = self.root / candidate
            if not path.is_file():
                continue
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            found = find_supported_decision(value)
            if found:
                return found, self.display_path(path)
        return "unresolved", "No explicit validation decision artifact was found."

    def run_job_b(self, decision: str) -> JobRecord:
        if decision in {"strongly_supported", "moderately_supported"}:
            name = "B1_full_relational_prefix"
            cfg = dict(self.config["jobs"]["relational_prefix_full"])
            branch = "FULL RELATIONAL PREFIX"
            remaining_minutes = self.remaining_seconds() / 60
            full_estimate = float(cfg.get("estimated_minutes", 210))
            if remaining_minutes < full_estimate + 10 and cfg.get("one_epoch_arguments"):
                cfg["arguments"] = cfg["one_epoch_arguments"]
                cfg["estimated_minutes"] = cfg.get("one_epoch_estimated_minutes", 110)
                cfg["selected_epoch_budget"] = 1
        else:
            name = "B2_qwen_capacity_probe"
            cfg = dict(self.config["jobs"]["qwen_capacity_probe"])
            branch = "QWEN CAPACITY PROBE"
        return self.run_configured_job(name, cfg, branch=branch)

    def run_job_c(self) -> None:
        cfg = self.config["jobs"]["clavaddpm"]
        common = list(cfg.get("common_arguments", []))
        preflight_cfg = dict(cfg)
        preflight_cfg.update(cfg.get("preflight", {}))
        preflight_cfg["arguments"] = list(cfg.get("preflight", {}).get("arguments", [])) + common
        preflight = self.run_configured_job("C1_clavaddpm_preflight", preflight_cfg)
        if preflight.status != "completed":
            self.record_without_run(
                "C2_C3_clavaddpm_smoke", "blocked", "ClavaDDPM preflight did not complete."
            )
            self.record_without_run("C4_clavaddpm_full", "blocked", "ClavaDDPM smoke was not run.")
            return
        smoke_cfg = dict(cfg)
        smoke_cfg.update(cfg.get("smoke", {}))
        smoke_cfg["arguments"] = list(cfg.get("smoke", {}).get("arguments", [])) + common
        smoke = self.run_configured_job("C2_C3_clavaddpm_smoke", smoke_cfg)
        if smoke.status != "completed":
            self.record_without_run("C4_clavaddpm_full", "blocked", "ClavaDDPM smoke did not pass.")
            return
        if self.remaining_seconds() < 3600:
            self.record_without_run(
                "C4_clavaddpm_full", "skipped", "Fewer than 60 minutes remained after smoke."
            )
            return
        full_cfg = dict(cfg)
        full_cfg.update(cfg.get("full", {}))
        full_cfg["arguments"] = list(cfg.get("full", {}).get("arguments", [])) + common
        self.run_configured_job("C4_clavaddpm_full", full_cfg)

    def run_job_d(self) -> None:
        if self.remaining_seconds() >= 20 * 60:
            self.run_configured_job(
                "D1_pending_canonical_evaluation",
                self.config["jobs"]["pending_canonical_evaluation"],
                allow_late_evaluation=True,
            )
        else:
            self.record_without_run("D1_pending_canonical_evaluation", "skipped", "Less than 20 minutes remained.")
        if self.remaining_seconds() >= 20 * 60:
            self.run_configured_job(
                "D2_repeated_seed_c2st",
                self.config["jobs"]["repeated_seed_c2st"],
                allow_late_evaluation=True,
            )
        else:
            self.record_without_run("D2_repeated_seed_c2st", "skipped", "Less than 20 minutes remained.")
        started = utc_now()
        self.set_active("D3_D4_internal_diagnostics")
        try:
            self.write_runtime_comparison()
            self.write_artifact_manifest()
            status, reason = "completed", None
        except Exception as error:  # diagnostics must not kill the session
            status, reason = "failed", f"{type(error).__name__}: {error}"
        self.records["D3_D4_internal_diagnostics"] = JobRecord(
            job_name="D3_D4_internal_diagnostics",
            status=status,
            start_time=iso(started),
            end_time=iso(utc_now()),
            runtime_seconds=(utc_now() - started).total_seconds(),
            failure_reason=reason,
        )
        self._write_status()

    def run_configured_job(
        self,
        name: str,
        cfg: Mapping[str, Any],
        *,
        branch: str | None = None,
        allow_late_evaluation: bool = False,
    ) -> JobRecord:
        existing = self.records.get(name)
        if existing and existing.status == "completed":
            self.log(f"[{name}] reusing completed session record.")
            return existing
        if self._completion_artifacts_exist(cfg):
            now = utc_now()
            record = JobRecord(
                job_name=name,
                status="completed",
                start_time=iso(now),
                end_time=iso(now),
                branch=branch,
                result={
                    "reused_completed_artifacts": True,
                    "completion_files": list(cfg.get("completion_files", [])),
                },
            )
            self.records[name] = record
            self._write_status()
            self.log(f"[{name}] reusing completed scientific artifacts.")
            return record
        command = self.resolve_command(cfg)
        if command is None:
            return self.record_without_run(
                name,
                "blocked",
                "No compatible entrypoint exists. Searched: " + ", ".join(cfg.get("entrypoints", [])),
                branch=branch,
            )
        estimate = float(cfg.get("estimated_minutes", 0))
        checkpointable = bool(cfg.get("checkpointable", False))
        gate_reason = self.start_gate(estimate, checkpointable, allow_late_evaluation)
        if gate_reason:
            return self.record_without_run(name, "skipped", gate_reason, branch=branch, command=command)
        free_gb = shutil.disk_usage(self.root).free / 1024**3
        if free_gb < self.minimum_free_disk_gb:
            return self.record_without_run(
                name,
                "blocked",
                f"Disk safety gate: {free_gb:.2f} GiB free, {self.minimum_free_disk_gb:.2f} GiB required. No outputs were deleted.",
                branch=branch,
                command=command,
            )
        timeout_seconds = min(
            float(cfg.get("timeout_minutes", 60)) * 60,
            max(1.0, self.remaining_seconds()),
        )
        record = self._execute(name, command, timeout_seconds, branch=branch)
        if record.status == "failed" and self._is_transient(record) and self.retry_limit > 0:
            retry_command = cfg.get("retry_command")
            if retry_command:
                resolved_retry = [self._substitute(str(value)) for value in retry_command]
                self.log(f"[{name}] transient failure detected; executing one configured retry.")
                retried = self._execute(name, resolved_retry, timeout_seconds, branch=branch, retry=1)
                return retried
            if not self._is_oom(record):
                self.log(f"[{name}] transient failure detected; retrying the unchanged command once.")
                return self._execute(name, command, timeout_seconds, branch=branch, retry=1)
            record.failure_reason = (record.failure_reason or "") + (
                " OOM signature detected, but no explicit retry_command is configured; batch settings were not changed silently."
            )
            self.records[name] = record
            self._write_status()
        return record

    def _completion_artifacts_exist(self, cfg: Mapping[str, Any]) -> bool:
        files = list(cfg.get("completion_files", []))
        if not files:
            return False
        return all(
            (Path(path) if Path(path).is_absolute() else self.root / path).is_file()
            for path in files
        )

    def start_gate(self, estimate_minutes: float, checkpointable: bool, allow_late_evaluation: bool) -> str | None:
        if self.stop_event.is_set() or self.remaining_seconds() <= 0:
            return "The global deadline was reached."
        elapsed = self.elapsed_hours()
        if not allow_late_evaluation and elapsed >= 7.25:
            return "T+7.25h generation/training launch gate is active."
        if not allow_late_evaluation and elapsed >= 6.5 and estimate_minutes > 75:
            return "T+6.5h gate forbids new training jobs expected to exceed 75 minutes."
        if estimate_minutes * 60 + 5 * 60 > self.remaining_seconds() and not checkpointable:
            return "Estimated runtime does not fit before the absolute session deadline."
        return None

    def _execute(
        self,
        name: str,
        command: Sequence[str],
        timeout_seconds: float,
        *,
        branch: str | None,
        retry: int = 0,
    ) -> JobRecord:
        logs = self.output_dir / "jobs"
        logs.mkdir(parents=True, exist_ok=True)
        suffix = f".retry{retry}" if retry else ""
        stdout_path = logs / f"{name}{suffix}.stdout.log"
        stderr_path = logs / f"{name}{suffix}.stderr.log"
        started = utc_now()
        record = JobRecord(
            job_name=name,
            status="running",
            start_time=iso(started),
            stdout_path=self.display_path(stdout_path),
            stderr_path=self.display_path(stderr_path),
            command=list(command),
            retries=retry,
            branch=branch,
        )
        self.records[name] = record
        self.set_active(name)
        self.log(f"[{name}] starting: {' '.join(command)}")
        process: subprocess.Popen[Any] | None = None
        timed_out = False
        try:
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                process = subprocess.Popen(list(command), cwd=self.root, stdout=stdout, stderr=stderr)
                while process.poll() is None:
                    elapsed = (utc_now() - started).total_seconds()
                    if self.stop_event.is_set() or elapsed >= timeout_seconds or utc_now() >= self.deadline:
                        timed_out = True
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                        break
                    time.sleep(min(self.poll_seconds, max(0.1, timeout_seconds - elapsed)))
            exit_code = process.returncode if process is not None else None
            if timed_out:
                status = "timed_out"
                reason = "Per-job timeout, global deadline, or graceful-stop request was reached."
            elif exit_code == 0:
                status, reason = "completed", None
            else:
                status = "failed"
                reason = self._failure_tail(stderr_path, stdout_path, exit_code)
        except Exception as error:
            status, exit_code = "failed", None
            reason = f"{type(error).__name__}: {error}"
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        ended = utc_now()
        record.status = status
        record.exit_code = exit_code
        record.end_time = iso(ended)
        record.runtime_seconds = (ended - started).total_seconds()
        record.failure_reason = reason
        self.records[name] = record
        self._write_status()
        self.log(f"[{name}] {status} after {record.runtime_seconds:.1f}s" + (f": {reason}" if reason else ""))
        return record

    def _failure_tail(self, stderr_path: Path, stdout_path: Path, exit_code: int | None) -> str:
        chunks = []
        for path in (stderr_path, stdout_path):
            try:
                chunks.append(path.read_text(errors="replace")[-4000:])
            except OSError:
                pass
        tail = "\n".join(chunks).strip()
        if len(tail) > 1200:
            tail = tail[-1200:]
        return f"Exit code {exit_code}. Tail: {tail}" if tail else f"Exit code {exit_code}; no error text was captured."

    def _is_transient(self, record: JobRecord) -> bool:
        reason = (record.failure_reason or "").lower()
        return any(pattern in reason for pattern in TRANSIENT_PATTERNS)

    def _is_oom(self, record: JobRecord) -> bool:
        reason = (record.failure_reason or "").lower()
        return "out of memory" in reason or "cudnn_status_alloc_failed" in reason

    def record_without_run(
        self,
        name: str,
        status: str,
        reason: str,
        *,
        branch: str | None = None,
        command: Sequence[str] | None = None,
    ) -> JobRecord:
        if status not in FINAL_STATUSES:
            raise ValueError(f"Unsupported terminal status: {status}")
        now = utc_now()
        record = JobRecord(
            job_name=name,
            status=status,
            start_time=iso(now),
            end_time=iso(now),
            failure_reason=reason,
            branch=branch,
            command=list(command) if command else None,
        )
        self.records[name] = record
        self._write_status()
        self.log(f"[{name}] {status}: {reason}")
        return record

    def run_heartbeat(self) -> None:
        self.set_active("E_heartbeat")
        started = utc_now()
        self.log("Heartbeat started; all currently runnable scientific work is finished.")
        while not self.stop_event.is_set() and utc_now() < self.deadline:
            with self.heartbeat_path.open("a") as handle:
                handle.write(f"{iso(utc_now())}\n")
            sleep_for = min(self.heartbeat_interval, self.remaining_seconds())
            if sleep_for <= 0:
                break
            self.stop_event.wait(sleep_for)
            if self.elapsed_hours() >= 7.8 and not self.report_path.exists():
                self.write_report(final=False)
        ended = utc_now()
        self.records["E_heartbeat"] = JobRecord(
            job_name="E_heartbeat",
            status="completed",
            start_time=iso(started),
            end_time=iso(ended),
            runtime_seconds=(ended - started).total_seconds(),
            result={"scientific_runtime": False},
        )
        self._write_status()

    def write_runtime_comparison(self) -> None:
        roots = [self.root / path for path in self.config.get("artifact_roots", [])]
        roots.insert(0, self.root / "outputs/qwen_text_decoder_06b")
        rows: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            patterns = ("*efficiency*.json", "*runtime*.json", "*parameter*.json")
            for pattern in patterns:
                for path in root.rglob(pattern):
                    if path in seen or path.stat().st_size > 10 * 1024**2:
                        continue
                    seen.add(path)
                    try:
                        value = json.loads(path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    flat = flatten_scalars(value)
                    selected = {
                        key: flat.get(key)
                        for key in (
                            "model_id",
                            "total_parameters",
                            "trainable_parameters",
                            "training_seconds",
                            "generation_seconds",
                            "peak_gpu_memory_bytes",
                        )
                    }
                    if any(item is not None for item in selected.values()):
                        rows.append({"artifact": self.display_path(path), **selected})
        output = self.output_dir / "runtime_parameter_comparison.csv"
        columns = [
            "artifact",
            "model_id",
            "total_parameters",
            "trainable_parameters",
            "training_seconds",
            "generation_seconds",
            "peak_gpu_memory_bytes",
        ]
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def write_artifact_manifest(self) -> None:
        entries: list[dict[str, Any]] = []
        for configured in self.config.get("artifact_roots", []):
            root = self.root / configured
            if not root.exists():
                entries.append({"root": configured, "status": "missing"})
                continue
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                try:
                    digest = sha256_file(path)
                    entries.append(
                        {
                            "path": self.display_path(path),
                            "size_bytes": path.stat().st_size,
                            "sha256": digest,
                            "status": "verified",
                        }
                    )
                except OSError as error:
                    entries.append(
                        {"path": self.display_path(path), "status": "unreadable", "error": str(error)}
                    )
        atomic_json(
            self.output_dir / "artifact_hash_manifest.json",
            {"generated_at": iso(utc_now()), "artifacts": entries},
        )

    def productive_gpu_hours(self) -> float:
        if not self.monitor_path.exists():
            return 0.0
        try:
            rows = list(csv.DictReader(self.monitor_path.open()))
        except OSError:
            return 0.0
        total = 0.0
        for left, right in zip(rows, rows[1:]):
            try:
                util = float(left["gpu_utilization_percent"])
                start = datetime.fromisoformat(left["timestamp"])
                end = datetime.fromisoformat(right["timestamp"])
            except (TypeError, ValueError, KeyError):
                continue
            active = left.get("active_job", "")
            if util > 5 and active not in {"E_heartbeat", "initializing", "complete", "final_summary"}:
                total += max(0.0, (end - start).total_seconds())
        return total / 3600

    def write_report(self, *, final: bool) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        decision = json.loads(self.decision_path.read_text()) if self.decision_path.exists() else {}
        records = list(self.records.values())
        lines = ["# RelGen 8-Hour Overnight Experiment Session", "", "## 1. Timeline", ""]
        lines += ["| Job | Start | End | Runtime | Status |", "|---|---|---|---:|---|"]
        for record in records:
            lines.append(
                f"| {record.job_name} | {record.start_time or ''} | {record.end_time or ''} | "
                f"{record.runtime_seconds / 60:.1f} min | {record.status} |"
            )
        lines += [
            "",
            "## 2. Temporal-Relational Prefix Probe",
            "",
            f"- Status: {self._status_of('A_relational_prefix_probe')}",
            f"- Classification: {decision.get('classification', 'unresolved')}",
            f"- Decision source: {decision.get('source', 'not available')}",
            "",
            "## 3. Conditional Follow-Up",
            "",
        ]
        b = next((r for r in records if r.job_name.startswith("B")), None)
        lines += [
            f"- Branch: {b.branch if b else 'SKIPPED'}",
            f"- Status: {b.status if b else 'skipped'}",
            f"- Reason: {b.failure_reason if b and b.failure_reason else 'See branch output artifacts.'}",
            "",
            "## 4. ClavaDDPM MovieLens Baseline",
            "",
            f"- Preflight: {self._status_of('C1_clavaddpm_preflight')}",
            f"- Smoke: {self._status_of('C2_C3_clavaddpm_smoke')}",
            f"- Full run: {self._status_of('C4_clavaddpm_full')}",
            "",
            "## 5. Additional Evaluations",
            "",
            f"- Pending canonical evaluations: {self._status_of('D1_pending_canonical_evaluation')}",
            f"- Repeated-seed cached C2ST: {self._status_of('D2_repeated_seed_c2st')}",
            f"- Runtime and artifact diagnostics: {self._status_of('D3_D4_internal_diagnostics')}",
            "",
            "## 6. Failures",
            "",
        ]
        failures = [r for r in records if r.status in {"failed", "blocked", "timed_out", "partially_completed"}]
        if failures:
            for record in failures:
                command = " ".join(record.command or []) or "not launched"
                lines += [
                    f"### {record.job_name}",
                    "",
                    f"- Status: {record.status}",
                    f"- Command: `{command}`",
                    f"- Error: {record.failure_reason}",
                    f"- Retried: {record.retries > 0}",
                    "- Fallback: the watchdog continued to the next eligible job.",
                    "",
                ]
        else:
            lines += ["No failures were recorded.", ""]
        wall_hours = (utc_now() - self.start_time).total_seconds() / 3600
        heartbeat = self.records.get("E_heartbeat")
        lines += [
            "## 7. GPU Utilization",
            "",
            f"- Approximate productive GPU hours: {self.productive_gpu_hours():.3f}",
            f"- Heartbeat hours (not scientific runtime): {(heartbeat.runtime_seconds / 3600) if heartbeat else 0:.3f}",
            "- Training, generation, and evaluation are identified by active-job labels in `resource_monitor.csv`.",
            "",
            "## 8. Key New Scientific Results",
            "",
        ]
        completed_science = [
            r for r in records if r.status == "completed" and r.job_name[:1] in {"A", "B", "C"}
        ]
        if completed_science:
            lines += [f"- {record.job_name}: completed; inspect its frozen output directory." for record in completed_science]
        else:
            lines.append("- No new primary scientific result completed; blocked jobs are not presented as results.")
        lines += [
            "",
            "## 9. Recommended Next Experiment",
            "",
            self.recommended_next_experiment(decision.get("classification", "unresolved")),
            "",
            "## Session Metadata",
            "",
            f"- Start: {manifest['session_start_time']}",
            f"- Target end: {manifest['target_end_time']}",
            f"- Report state: {'final' if final else 'interim'}",
            f"- Wall clock so far: {wall_hours:.3f} hours",
        ]
        self.report_path.write_text("\n".join(lines) + "\n")

    def _status_of(self, name: str) -> str:
        record = self.records.get(name)
        return record.status if record else "not_run"

    def recommended_next_experiment(self, decision: str) -> str:
        if self._status_of("A_relational_prefix_probe") == "blocked":
            return "Restore or implement the frozen leakage-safe relational-prefix probe runner, then run Job A without changing its specified design."
        if decision in {"strongly_supported", "moderately_supported"} and self._status_of("B1_full_relational_prefix") != "completed":
            return "Run the frozen full-data relational-prefix confirmation selected by Job A."
        if decision not in {"strongly_supported", "moderately_supported"} and self._status_of("B2_qwen_capacity_probe") != "completed":
            return "Run the controlled Qwen3-0.6B versus Qwen3-1.7B one-epoch capacity probe."
        if self._status_of("C4_clavaddpm_full") != "completed":
            return "Complete the official ClavaDDPM MovieLens seed-42 baseline after its smoke test passes."
        return "Run one held-out confirmation of the selected architecture; do not open a new hyperparameter sweep."

    def print_final_console(self) -> None:
        decision = json.loads(self.decision_path.read_text()) if self.decision_path.exists() else {}
        b = next((r for r in self.records.values() if r.job_name.startswith("B")), None)
        failures = [r for r in self.records.values() if r.status in {"failed", "blocked", "timed_out"}]
        heartbeat = self.records.get("E_heartbeat")
        wall = (utc_now() - self.start_time).total_seconds() / 3600
        print("\n============================================================")
        print("RELGEN 8-HOUR OVERNIGHT SESSION COMPLETE")
        print("============================================================")
        print(f"\nTOTAL WALL CLOCK:\n{wall:.3f} hours")
        print(f"\nPRODUCTIVE GPU HOURS:\n{self.productive_gpu_hours():.3f}")
        print("\n------------------------------------------------------------")
        print("JOB A — RELATIONAL PREFIX")
        print("------------------------------------------------------------")
        print(f"\nSTATUS:\n{self._status_of('A_relational_prefix_probe')}")
        print(f"\nRESULT:\n{decision.get('classification', 'unresolved')}")
        print("\n------------------------------------------------------------")
        print("JOB B — CONDITIONAL FOLLOW-UP")
        print("------------------------------------------------------------")
        print(f"\nBRANCH:\n{b.branch if b else 'SKIPPED'}")
        print(f"\nSTATUS:\n{b.status if b else 'skipped'}")
        print("\n------------------------------------------------------------")
        print("JOB C — CLAVADDPM MOVIELENS")
        print("------------------------------------------------------------")
        print(f"\nPREFLIGHT:\n{self._status_of('C1_clavaddpm_preflight')}")
        print(f"\nSMOKE:\n{self._status_of('C2_C3_clavaddpm_smoke')}")
        print(f"\nFULL RUN:\n{self._status_of('C4_clavaddpm_full')}")
        print("\n------------------------------------------------------------")
        print("FAILURES")
        print("------------------------------------------------------------")
        print("\n" + ("\n".join(f"{r.job_name}: {r.status}" for r in failures) if failures else "None"))
        print("\n------------------------------------------------------------")
        print("HEARTBEAT TIME")
        print("------------------------------------------------------------")
        print(f"\n{(heartbeat.runtime_seconds / 3600) if heartbeat else 0:.3f} hours")
        print("\n------------------------------------------------------------")
        print("MOST IMPORTANT NEW RESULT")
        print("------------------------------------------------------------")
        print(f"\nJob A classification: {decision.get('classification', 'unresolved')}")
        print("\n------------------------------------------------------------")
        print("NEXT EXPERIMENT")
        print("------------------------------------------------------------")
        print(f"\n{self.recommended_next_experiment(decision.get('classification', 'unresolved'))}")
        print("\n============================================================", flush=True)


def find_supported_decision(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return normalized if normalized in SUPPORTED_A_DECISIONS else None
    if isinstance(value, Mapping):
        preferred = ("classification", "relational_conditioning", "decision", "verdict")
        for key in preferred:
            if key in value:
                found = find_supported_decision(value[key])
                if found:
                    return found
        for child in value.values():
            found = find_supported_decision(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = find_supported_decision(child)
            if found:
                return found
    return None


def flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_scalars(child, child_prefix))
            if not isinstance(child, (Mapping, list)):
                result.setdefault(str(key), child)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(flatten_scalars(child, f"{prefix}.{index}"))
    elif prefix:
        result[prefix] = value
    return result


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        chunk = handle.read(chunk_size)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(chunk_size)
    return digest.hexdigest()
