"""Strict adapter around the official ClavaDDPM implementation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ClavaDDPMMovieLensRunner:
    config_path: Path
    official_python: str | None = None

    def __post_init__(self) -> None:
        self.config = yaml.safe_load(self.config_path.read_text())
        self.root = Path.cwd().resolve()
        self.output = self._resolve(self.config["output_dir"])
        self.checkout = self._resolve(self.config["official"]["checkout_dir"])
        configured_python = self.config["official"].get("python_executable", "auto")
        self.python = self.official_python or (
            sys.executable if configured_python == "auto" else str(configured_python)
        )
        self.seed = int(self.config.get("seed", 42))

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def ensure_official_checkout(self) -> dict[str, Any]:
        repository = str(self.config["official"]["repository"])
        commit = str(self.config["official"]["commit"])
        if not (self.checkout / ".git").is_dir():
            self.checkout.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", repository, str(self.checkout)], check=True
            )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=self.checkout,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if status:
            raise RuntimeError(
                f"Official ClavaDDPM checkout is modified; refusing to overwrite it: {self.checkout}"
            )
        has_commit = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=self.checkout,
            check=False, capture_output=True,
        ).returncode == 0
        if not has_commit:
            subprocess.run(["git", "fetch", "origin", commit], cwd=self.checkout, check=True)
        subprocess.run(["git", "checkout", "--detach", commit], cwd=self.checkout, check=True)
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.checkout,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if actual != commit:
            raise RuntimeError(f"ClavaDDPM commit pin failed: expected {commit}, found {actual}")
        return {"repository": repository, "requested_commit": commit, "actual_commit": actual}

    @property
    def adapted_data(self) -> Path:
        return self.output / "adapted_train"

    def prepare_data(self, limit_rows: int | None = None) -> dict[str, Any]:
        paths = {name: self._resolve(path) for name, path in self.config["data"].items() if name in {"interactions", "users", "movies"}}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("MovieLens benchmark files are missing:\n- " + "\n- ".join(missing))
        interactions = pd.read_csv(paths["interactions"], low_memory=False)
        required = {"event_id", "user_id", "movie_id", "event_time", "rating", "split"}
        absent = sorted(required - set(interactions.columns))
        if absent:
            raise RuntimeError(f"MovieLens interactions are missing required columns: {absent}")
        split = str(self.config["data"].get("split", "train"))
        interactions = interactions.loc[
            interactions["split"].astype(str).str.lower() == split
        ].copy()
        interactions["_parsed_time"] = pd.to_datetime(
            interactions["event_time"], errors="coerce", utc=True
        )
        if interactions["_parsed_time"].isna().any():
            raise RuntimeError("MovieLens training split contains invalid timestamps")
        interactions = interactions.sort_values(
            ["_parsed_time", "event_id"], kind="mergesort"
        ).reset_index(drop=True)
        if limit_rows is not None:
            interactions = interactions.head(int(limit_rows)).copy()
        users = pd.DataFrame({
            "user_id": sorted(interactions["user_id"].unique()),
        })
        movies = pd.DataFrame({
            "movie_id": sorted(interactions["movie_id"].unique()),
        })
        # ClavaDDPM trains one unconditional model per root table and therefore
        # requires a non-ID feature even when a benchmark parent is ID-only.
        users["clava_placeholder"] = 0.0
        movies["clava_placeholder"] = 0.0
        event_time_seconds = interactions["_parsed_time"].array.asi8 / 1e9
        events = pd.DataFrame({
            "rating_event_id": interactions["event_id"],
            "user_id": interactions["user_id"],
            "movie_id": interactions["movie_id"],
            "event_time": event_time_seconds.astype(float),
            "rating": interactions["rating"].astype(str),
        })
        self.adapted_data.mkdir(parents=True, exist_ok=True)
        users.to_csv(self.adapted_data / "user.csv", index=False)
        movies.to_csv(self.adapted_data / "movie.csv", index=False)
        events.to_csv(self.adapted_data / "rating_event.csv", index=False)
        interactions.drop(columns=["_parsed_time"]).to_csv(
            self.adapted_data / "evaluation_real.csv", index=False
        )
        metadata = {
            "relation_order": [
                [None, "user"], [None, "movie"],
                ["user", "rating_event"], ["movie", "rating_event"],
            ],
            "tables": {
                "user": {"parents": [], "children": ["rating_event"]},
                "movie": {"parents": [], "children": ["rating_event"]},
                "rating_event": {"parents": ["user", "movie"], "children": []},
            },
        }
        domains = {
            "user": {"clava_placeholder": {"size": 1, "type": "continuous"}},
            "movie": {"clava_placeholder": {"size": 1, "type": "continuous"}},
            "rating_event": {
                "event_time": {"size": int(events["event_time"].nunique()), "type": "continuous"},
                "rating": {"size": int(events["rating"].nunique()), "type": "discrete"},
            },
        }
        write_json(self.adapted_data / "dataset_meta.json", metadata)
        for table, domain in domains.items():
            write_json(self.adapted_data / f"{table}_domain.json", domain)
        repeated = events.groupby(["user_id", "movie_id"], dropna=False).size()
        manifest = {
            "source_split": split,
            "chronological_training_only": True,
            "num_users": len(users),
            "num_movies": len(movies),
            "num_rating_events": len(events),
            "timestamp_min": interactions["_parsed_time"].min().isoformat(),
            "timestamp_max": interactions["_parsed_time"].max().isoformat(),
            "rating_support": sorted(interactions["rating"].astype(float).unique().tolist()),
            "repeated_pair_rows": int(repeated[repeated > 1].sum()),
            "max_pair_multiplicity": int(repeated.max()),
            "two_parent_interaction_table": True,
            "adapter_only_placeholder_removed_from_final_outputs": True,
            "file_sha256": {
                path.name: sha256(path)
                for path in sorted(self.adapted_data.iterdir())
                if path.is_file() and path.name != "adapter_manifest.json"
            },
        }
        write_json(self.adapted_data / "adapter_manifest.json", manifest)
        return manifest

    def dependency_audit(self) -> dict[str, Any]:
        code = (
            "import json, sys; "
            f"sys.path.insert(0, {str(self.checkout)!r}); "
            "import torch, pandas, numpy, sklearn, complex_pipeline; "
            "print(json.dumps({'python': sys.version, 'torch': torch.__version__, "
            "'pandas': pandas.__version__, 'numpy': numpy.__version__, "
            "'sklearn': sklearn.__version__, 'cuda': torch.cuda.is_available()}))"
        )
        process = subprocess.run(
            [self.python, "-c", code], cwd=self.checkout,
            check=False, capture_output=True, text=True,
        )
        parsed = None
        if process.returncode == 0:
            try:
                parsed = json.loads(process.stdout.strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                pass
        return {
            "python_executable": self.python,
            "import_exit_code": process.returncode,
            "versions": parsed,
            "stdout": process.stdout[-4000:],
            "stderr": process.stderr[-8000:],
            "passed": process.returncode == 0 and parsed is not None,
        }

    def preflight(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        checkout = self.ensure_official_checkout()
        data = self.prepare_data()
        dependency = self.dependency_audit()
        source = (self.checkout / "pipeline_utils.py").read_text()
        pipeline = (self.checkout / "complex_pipeline.py").read_text()
        capability = {
            "two_parent_interaction_table": "handle_multi_parent(" in pipeline,
            "foreign_key_relationships": "relation_order" in pipeline,
            "explicit_event_rows": data["num_rating_events"] > 0,
            "repeated_interaction_representation_supported": (
                "rating_event_id" in pd.read_csv(
                    self.adapted_data / "rating_event.csv", nrows=1
                ).columns
            ),
            "numerical_timestamp": True,
            "categorical_ordinal_rating": True,
            "batched_matching_available": "def match_tables" in source,
        }
        passed = dependency["passed"] and all(capability.values())
        result = {
            "status": "passed" if passed else "blocked",
            "official_checkout": checkout,
            "dependency_audit": dependency,
            "capability_audit": capability,
            "data_adapter": data,
            "repeated_pairs_observed_in_this_train_split": data["repeated_pair_rows"] > 0,
            "core_algorithm_modified": False,
            "timestamp_source": "ClavaDDPM-generated numerical attribute",
            "relgen_event_spine_used": False,
            "matching_implementation": "official batched FAISS one-to-one matching",
            "host": platform.node(),
        }
        write_json(self.output / "preflight.json", result)
        lines = [
            "# ClavaDDPM MovieLens Preflight", "",
            f"Status: **{result['status']}**", "",
            f"- Official commit: `{checkout['actual_commit']}`",
            f"- Training interactions: {data['num_rating_events']:,}",
            "- Schema: `user -> rating_event <- movie`",
            "- Timestamp is generated by ClavaDDPM as a numerical attribute.",
            "- RelGen timestamps and event spine are not supplied.",
            "- Repeated interaction rows remain explicit event rows.",
            "- Core ClavaDDPM source is unmodified.",
            f"- Dependency import: {'passed' if dependency['passed'] else 'failed'}",
        ]
        if not dependency["passed"]:
            lines += ["", "## Dependency Error", "", "```", dependency["stderr"], "```"]
        (self.output / "preflight.md").write_text("\n".join(lines) + "\n")
        if not passed:
            raise RuntimeError(
                "ClavaDDPM preflight is blocked. See " + str(self.output / "preflight.md")
            )
        return result

    def _official_config(self, stage: str) -> tuple[dict[str, Any], Path]:
        if stage not in {"smoke", "full"}:
            raise ValueError(stage)
        values = json.loads(json.dumps(self.config["clava"]))
        stage_root = self.output / stage
        if stage == "smoke":
            smoke = self.config["smoke"]
            values["clustering"]["num_clusters"] = int(smoke["num_clusters"])
            values["diffusion"].update({
                "iterations": int(smoke["diffusion_iterations"]),
                "num_timesteps": int(smoke["num_timesteps"]),
                "d_layers": list(smoke["diffusion_layers"]),
                "batch_size": int(smoke["batch_size"]),
            })
            values["classifier"].update({
                "iterations": int(smoke["classifier_iterations"]),
                "d_layers": list(smoke["classifier_layers"]),
                "batch_size": int(smoke["batch_size"]),
            })
            self.prepare_data(limit_rows=int(smoke["train_rows"]))
            debug = {"sample_scale": float(smoke["sample_scale"])}
        else:
            self.prepare_data()
            debug = None
        official = {
            "general": {
                "data_dir": str(self.adapted_data),
                "exp_name": f"movielens_{stage}_seed_{self.seed}",
                "workspace_dir": str(stage_root / "official_workspace"),
                "sample_prefix": "",
                "test_data_dir": None,
            },
            **values,
        }
        if debug is not None:
            official["debug"] = debug
        path = stage_root / "official_config.json"
        write_json(path, official)
        return official, path

    def _write_seed_launcher(self, stage: str, config_path: Path) -> Path:
        """Inject the requested seed without editing the pinned checkout."""

        path = self.output / stage / "official_seed_launcher.py"
        source = f'''import random
import runpy
import sys

import numpy as np
import torch

SEED = {self.seed}
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import pipeline_modules

_original_get_T_dict = pipeline_modules.get_T_dict
def _seeded_get_T_dict():
    value = _original_get_T_dict()
    value["seed"] = SEED
    return value
pipeline_modules.get_T_dict = _seeded_get_T_dict

sys.argv = ["complex_pipeline.py", "--config_path", {str(config_path)!r}]
runpy.run_path("complex_pipeline.py", run_name="__main__")
'''
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return path

    def run(self, stage: str, skip_existing: bool = True) -> dict[str, Any]:
        completion = self.output / stage / "run_summary.json"
        if skip_existing and completion.is_file():
            existing = json.loads(completion.read_text())
            if existing.get("valid"):
                return existing
        preflight_path = self.output / "preflight.json"
        if not preflight_path.is_file() or json.loads(preflight_path.read_text()).get("status") != "passed":
            self.preflight()
        _, config_path = self._official_config(stage)
        launcher_path = self._write_seed_launcher(stage, config_path)
        stage_root = self.output / stage
        log_path = stage_root / "official_pipeline.log"
        resource_path = stage_root / "official_resource.time.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "/usr/bin/time", "-v", "-o", str(resource_path),
            self.python, str(launcher_path),
        ]
        environment = os.environ.copy()
        environment.update({"PYTHONHASHSEED": str(self.seed), "CLAVA_SEED": str(self.seed)})
        started = time.perf_counter()
        with log_path.open("w") as log:
            process = subprocess.run(
                command, cwd=self.checkout, env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        elapsed = time.perf_counter() - started
        synthetic = self._canonicalize_generated(stage)
        validity = self._validity(stage, synthetic)
        result = {
            "stage": stage,
            "official_exit_code": process.returncode,
            "official_pipeline_completed_without_error": process.returncode == 0,
            "generation_present_despite_exit_code": synthetic.is_file(),
            "runtime_seconds": elapsed,
            "command": command,
            "official_commit": self.config["official"]["commit"],
            "requested_and_effective_seed": self.seed,
            "seed_plumbing": "wrapper seeds Python/NumPy/Torch and overrides only ClavaDDPM's hard-coded T_dict seed",
            "valid": validity["valid"],
            "validity": validity,
            "synthetic_interactions": str(synthetic),
            "official_log": str(log_path),
        }
        write_json(completion, result)
        if not validity["valid"]:
            raise RuntimeError(f"ClavaDDPM {stage} did not produce a valid synthetic database")
        if stage == "full":
            self.evaluate_full(synthetic)
        return result

    def _generated_path(self, stage: str, table: str) -> Path:
        return self.output / stage / "official_workspace" / table / "_final" / f"{table}_synthetic.csv"

    def _canonicalize_generated(self, stage: str) -> Path:
        event_path = self._generated_path(stage, "rating_event")
        destination = self.output / stage / "generated/synthetic_interactions.csv"
        if not event_path.is_file():
            return destination
        events = pd.read_csv(event_path, low_memory=False)
        required = {"rating_event_id", "user_id", "movie_id", "event_time", "rating"}
        if not required.issubset(events.columns):
            return destination
        output = events.rename(columns={"rating_event_id": "event_id"}).copy()
        for column in ("event_id", "user_id", "movie_id"):
            output[column] = pd.to_numeric(output[column], errors="coerce").round().astype("Int64")
        seconds = pd.to_numeric(output["event_time"], errors="coerce")
        output["event_time"] = pd.to_datetime(seconds, unit="s", errors="coerce", utc=True).astype(str)
        output["rating"] = pd.to_numeric(output["rating"], errors="coerce")
        destination.parent.mkdir(parents=True, exist_ok=True)
        output[["event_id", "user_id", "movie_id", "event_time", "rating"]].to_csv(destination, index=False)
        for table in ("user", "movie"):
            source = self._generated_path(stage, table)
            if source.is_file():
                frame = pd.read_csv(source).drop(columns=["clava_placeholder"], errors="ignore")
                frame.to_csv(destination.parent / f"synthetic_{table}s.csv", index=False)
        return destination

    def _validity(self, stage: str, synthetic: Path) -> dict[str, Any]:
        if not synthetic.is_file():
            result = {"valid": False, "reason": "synthetic interaction table missing"}
            write_json(self.output / stage / "generated/generation_validity.json", result)
            return result
        events = pd.read_csv(synthetic, low_memory=False)
        users_path = synthetic.parent / "synthetic_users.csv"
        movies_path = synthetic.parent / "synthetic_movies.csv"
        users = pd.read_csv(users_path) if users_path.is_file() else pd.DataFrame(columns=["user_id"])
        movies = pd.read_csv(movies_path) if movies_path.is_file() else pd.DataFrame(columns=["movie_id"])
        ratings = set(pd.to_numeric(events["rating"], errors="coerce").dropna())
        expected_ratings = set(np.arange(0.5, 5.01, 0.5))
        report = {
            "row_count": len(events),
            "source_fk_valid": bool(events["user_id"].isin(users["user_id"]).all()),
            "destination_fk_valid": bool(events["movie_id"].isin(movies["movie_id"]).all()),
            "timestamp_parse_error_rate": float(pd.to_datetime(events["event_time"], errors="coerce", utc=True).isna().mean()),
            "rating_support_valid": ratings.issubset(expected_ratings),
            "event_id_unique": bool(events["event_id"].is_unique),
        }
        report["valid"] = bool(
            report["row_count"] > 0
            and report["source_fk_valid"]
            and report["destination_fk_valid"]
            and report["timestamp_parse_error_rate"] == 0
            and report["rating_support_valid"]
            and report["event_id_unique"]
        )
        write_json(self.output / stage / "generated/generation_validity.json", report)
        return report

    def evaluate_full(self, synthetic: Path) -> None:
        evaluation = yaml.safe_load(self._resolve(self.config["data"]["evaluation_config"]).read_text())
        evaluation["real_table_path"] = str(self.adapted_data / "evaluation_real.csv")
        evaluation["synthetic_table_path"] = str(synthetic)
        evaluation["table"]["columns"]["user_id"]["parent_table_path"] = str(synthetic.parent / "synthetic_users.csv")
        evaluation["table"]["columns"]["movie_id"]["parent_table_path"] = str(synthetic.parent / "synthetic_movies.csv")
        config_path = self.output / "full/evaluation_config_resolved.yaml"
        config_path.write_text(yaml.safe_dump(evaluation, sort_keys=False))
        subprocess.run(
            [
                sys.executable, "src/scripts/evaluate_single_event_table_paper_metrics.py",
                "--config", str(config_path),
                "--real-table", str(self.adapted_data / "evaluation_real.csv"),
                "--synthetic-table", str(synthetic),
                "--output-dir", str(self.output / "full/evaluation/paper_grade"),
                "--seed", str(self.seed),
            ],
            cwd=self.root, check=True,
        )
