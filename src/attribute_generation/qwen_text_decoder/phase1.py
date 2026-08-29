"""Frozen-checkpoint Phase-1 diagnostics for the Qwen text decoder."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

from attribute_generation.conditional_tabdlm.diffusion_diagnostics import (
    PROGRESSIVE_CONDITION_SPECS,
)
from attribute_generation.conditional_tabdlm.hierarchical_sample import (
    hierarchical_sample_from_config,
)
from attribute_generation.conditional_tabdlm.schema import load_config
from evaluation.text_c2st_audit import (
    EmbeddingStore,
    TextC2STProtocol,
    canonical_text,
    evaluate_protocol,
    file_sha256,
)

from .experiment import (
    QwenTextExperiment,
    alignment_audit,
    distribution_comparison,
    nested_c2st,
    validate_runtime_dependencies,
    write_json,
)
from .followup import (
    ALIGNMENT_COLUMNS,
    QwenFollowupExperiment,
    dataframe_sha256,
    directory_fingerprint,
    discover_diffusion_artifacts,
    hardlink_or_copy,
)


EXPECTED_MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
STABLE_EVENT_ID_CANDIDATES = (
    "event_id",
    "review_id",
    "interaction_id",
    "transaction_id",
)


def _canonical_key_series(series: pd.Series, *, timestamp: bool = False) -> pd.Series:
    if timestamp:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.map(lambda value: "<NA>" if pd.isna(value) else value.isoformat())

    def normalize(value: Any) -> str:
        if pd.isna(value):
            return "<NA>"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return canonical_text(value)

    return series.map(normalize)


def _key_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: _canonical_key_series(
                frame[column], timestamp=column == "review_time"
            )
            for column in columns
        }
    )


def _key_counts(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keys = _key_frame(frame, columns)
    return keys.value_counts(sort=False, dropna=False).rename("count").reset_index()


def _counts_equal(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> bool:
    left_counts = _key_counts(left, columns).rename(columns={"count": "left_count"})
    right_counts = _key_counts(right, columns).rename(columns={"count": "right_count"})
    merged = left_counts.merge(right_counts, on=columns, how="outer").fillna(0)
    return bool((merged["left_count"] == merged["right_count"]).all())


def align_exact_population(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    full_reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Align by an exact event-key multiset, never by an unverified row slice."""
    missing = sorted(set(ALIGNMENT_COLUMNS).difference(reference.columns) | set(ALIGNMENT_COLUMNS).difference(candidate.columns))
    stable_ids = [
        column
        for column in STABLE_EVENT_ID_CANDIDATES
        if column in reference.columns and column in candidate.columns
    ]
    keys = [*ALIGNMENT_COLUMNS, *stable_ids]
    report: dict[str, Any] = {
        "required_alignment_columns": list(ALIGNMENT_COLUMNS),
        "stable_event_identifiers_used": stable_ids,
        "alignment_keys": keys,
        "reference_rows": int(len(reference)),
        "candidate_rows": int(len(candidate)),
        "missing_columns": missing,
        "positional_slice_used": False,
    }
    if missing:
        report.update({"aligned": False, "reason": "required event keys are missing"})
        return None, report

    reference_keys = _key_frame(reference, keys)
    duplicate_mask = reference_keys.duplicated(keys, keep=False)
    report["reference_rows_in_duplicate_key_groups"] = int(duplicate_mask.sum())
    report["reference_duplicate_key_groups"] = int(
        reference_keys.loc[duplicate_mask].drop_duplicates().shape[0]
    )

    source = candidate
    method = "exact_candidate_multiset"
    if len(candidate) != len(reference):
        if full_reference is None or len(candidate) != len(full_reference):
            report.update(
                {
                    "aligned": False,
                    "reason": "candidate is not the held-out population and no same-size full real table was available",
                }
            )
            return None, report
        full_missing = sorted(set(keys).difference(full_reference.columns))
        if full_missing:
            report.update(
                {
                    "aligned": False,
                    "reason": f"full real table lacks alignment keys: {full_missing}",
                }
            )
            return None, report
        full_left = _key_frame(full_reference, keys)
        full_right = _key_frame(candidate, keys)
        ordered_full_match = bool(full_left.equals(full_right))
        report["full_table_ordered_key_match"] = ordered_full_match
        report["full_reference_rows"] = int(len(full_reference))
        if not ordered_full_match:
            report.update(
                {
                    "aligned": False,
                    "reason": "full synthetic artifact does not have an exact ordered event-key match to the full real table",
                }
            )
            return None, report

        identity_columns = [
            column for column in reference.columns if column in full_reference.columns
        ]
        identity_reference = _key_frame(reference, identity_columns)
        identity_full = _key_frame(full_reference, identity_columns)
        full_lookup = identity_full.copy()
        full_lookup["_occurrence"] = full_lookup.groupby(
            identity_columns, sort=False
        ).cumcount()
        full_lookup["_candidate_row"] = np.arange(len(full_lookup), dtype=np.int64)
        requested = identity_reference.copy()
        requested["_occurrence"] = requested.groupby(
            identity_columns, sort=False
        ).cumcount()
        matched = requested.merge(
            full_lookup,
            on=[*identity_columns, "_occurrence"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if matched["_candidate_row"].isna().any():
            report.update(
                {
                    "aligned": False,
                    "reason": "held-out event-key multiset is not contained in the verified full table",
                }
            )
            return None, report
        source = candidate.iloc[matched["_candidate_row"].astype(int).to_numpy()].reset_index(drop=True)
        method = "verified_full_table_order_then_exact_event_key_lookup"
        report["full_real_identity_columns_used_for_row_location"] = identity_columns
        report["real_target_content_used_only_for_source_row_location"] = bool(
            set(identity_columns).difference(keys)
        )

    if len(source) == len(reference) and not _counts_equal(reference, source, keys):
        report.update({"aligned": False, "reason": "event-key multisets differ"})
        return None, report

    source_keys = _key_frame(source, keys)
    source_lookup = source_keys.copy()
    source_lookup["_occurrence"] = source_lookup.groupby(keys, sort=False).cumcount()
    source_lookup["_candidate_row"] = np.arange(len(source_lookup), dtype=np.int64)
    requested = reference_keys.copy()
    requested["_occurrence"] = requested.groupby(keys, sort=False).cumcount()
    matched = requested.merge(
        source_lookup,
        on=[*keys, "_occurrence"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if matched["_candidate_row"].isna().any():
        report.update({"aligned": False, "reason": "one-to-one event-key matching failed"})
        return None, report
    aligned = source.iloc[matched["_candidate_row"].astype(int).to_numpy()].reset_index(drop=True)
    ordered = alignment_audit(reference, aligned)
    report.update(
        {
            "aligned": bool(ordered["aligned"]),
            "reason": ordered["reason"],
            "method": method,
            "ordered_alignment_audit": ordered,
            "aligned_dataframe_sha256": dataframe_sha256(aligned),
            "duplicate_groups_matched_by_occurrence_within_identical_event_keys": bool(
                report["reference_duplicate_key_groups"]
            ),
        }
    )
    return (aligned if report["aligned"] else None), report


def exact_minilm_snapshot(model_id: str, revision: str) -> dict[str, str]:
    """Resolve one exact cached commit; never choose another revision or fallback."""
    from huggingface_hub import scan_cache_dir

    matches = []
    for repo in scan_cache_dir().repos:
        if repo.repo_id != model_id:
            continue
        matches.extend(
            candidate
            for candidate in repo.revisions
            if candidate.commit_hash == revision and candidate.snapshot_path.is_dir()
        )
    if not matches:
        raise RuntimeError(
            f"Exact local MiniLM snapshot {model_id}@{revision} is unavailable; "
            "Phase-1 refuses a network, hash-embedding, or revision fallback."
        )
    snapshot = matches[0].snapshot_path.resolve()
    return {"model_id": model_id, "revision": revision, "local_snapshot": str(snapshot)}


def matched_memorization_metrics(
    train: pd.DataFrame,
    real_heldout: pd.DataFrame,
    qwen: pd.DataFrame,
    *,
    training_rows: int = 20000,
    max_features: int = 50000,
) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    result: dict[str, Any] = {
        "comparison": "real heldout and Qwen measured against the same real training corpus",
        "privacy_claim": "No differential privacy claim; this is an obvious-memorization diagnostic only.",
        "nearest_neighbor_training_policy": "lexicographically sorted unique non-empty canonical training strings",
        "nearest_neighbor_training_rows_max": int(training_rows),
    }
    for field in ("summary", "review_text"):
        train_values = train[field].map(canonical_text)
        train_set = set(train_values) - {""}
        train_sample = sorted(train_set)[: int(training_rows)]
        field_result: dict[str, Any] = {}
        vectorizer = None
        nearest = None
        if train_sample:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, max_features=int(max_features)
            )
            train_matrix = vectorizer.fit_transform(train_sample)
            nearest = NearestNeighbors(
                n_neighbors=1, metric="cosine", algorithm="brute"
            ).fit(train_matrix)
        for label, frame in (("real_heldout", real_heldout), ("qwen", qwen)):
            values = frame[field].map(canonical_text)
            overlap = values.isin(train_set)
            nn_summary = None
            if vectorizer is not None and nearest is not None and len(values):
                matrix = vectorizer.transform(values.tolist())
                distances, _ = nearest.kneighbors(matrix)
                similarities = 1.0 - distances[:, 0]
                nn_summary = {
                    "sample_rows": int(len(values)),
                    "mean": float(np.mean(similarities)),
                    "median": float(np.median(similarities)),
                    "p90": float(np.quantile(similarities, 0.90)),
                    "p95": float(np.quantile(similarities, 0.95)),
                    "max": float(np.max(similarities)),
                    "metric": "nearest-train TF-IDF unigram+bigram cosine",
                }
            field_result[label] = {
                "rows": int(len(values)),
                "exact_train_overlap_count": int(overlap.sum()),
                "exact_train_overlap_rate": float(overlap.mean()),
                "nearest_neighbor": nn_summary,
            }
        result[field] = field_result
    return result


def _lower_is_better_answer(left: Any, right: Any) -> str:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return "UNRESOLVED (matched artifact unavailable)"
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return "UNRESOLVED (matched artifact unavailable)"
    return "YES" if left_value < right_value else "NO"


@dataclass
class QwenPhase1Experiment(QwenFollowupExperiment):
    """Minimal Phase-1 orchestration over existing frozen diagnostic code."""

    config_path: Path

    def __post_init__(self) -> None:
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = Path(self.config["output_dir"])
        self.base_output = Path(self.config["base_output_dir"])
        self.base_config_path = Path(self.config["base_experiment_config"])
        self.base = QwenTextExperiment(self.base_config_path, output_dir=self.base_output)
        self.seed = int(self.config.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @property
    def subset_path(self) -> Path:
        return self.output / "evaluation_population.csv"

    def _required_base_artifacts(self) -> list[Path]:
        return [
            self.base_output / "training/model_source.json",
            self.base_output / "training/training_efficiency.json",
            self.base_output / "training/best_adapter/adapter_config.json",
            self.base_output / "oracle_structured/synthetic_text.csv",
            self.base_output / "oracle_structured/canonical_text_c2st.json",
            self.base.benchmark / "benchmark_manifest.json",
            self.base.benchmark / "train_real.csv",
            self.base.benchmark / "test_real.csv",
        ]

    def audit(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        missing = [str(path) for path in self._required_base_artifacts() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required frozen Phase-1 inputs are missing:\n- " + "\n- ".join(missing))
        validate_runtime_dependencies()
        real = pd.read_csv(self.base.benchmark / "test_real.csv", low_memory=False)
        expected_rows = int(self.config["evaluation_population"]["expected_rows"])
        if len(real) != expected_rows:
            raise RuntimeError(
                f"Frozen test population has {len(real):,} rows; expected exactly {expected_rows:,}."
            )
        real.to_csv(self.subset_path, index=False)
        qwen = pd.read_csv(
            self.base_output / "oracle_structured/synthetic_text.csv", low_memory=False
        )
        qwen_audit = alignment_audit(real, qwen)
        if not qwen_audit["aligned"]:
            raise RuntimeError("Existing Qwen B0 output is not aligned to the frozen test population")

        source = exact_minilm_snapshot(
            self.config["evaluation"]["embedding_model"],
            self.config["evaluation"]["embedding_revision"],
        )
        write_json(self.output / "evaluation_model_source.json", source)
        structured = self.config["structured_diffusion"]
        structured_checkpoint = Path(structured["checkpoint"])
        structured_config = Path(structured["config"])
        for path, label in (
            (structured_checkpoint, "frozen structured checkpoint"),
            (structured_config, "structured model config"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")

        rows_per_second = float(
            self.config["compute_budget"]["measured_qwen_rows_per_second"]
        )
        generations = int(self.config["compute_budget"]["fresh_qwen_generations"])
        projected_seconds = len(real) * generations / rows_per_second
        projected_hours = projected_seconds / 3600.0
        overhead_hours = float(
            self.config["compute_budget"]["estimated_non_generation_gpu_hours"]
        )
        projected_total_hours = projected_hours + overhead_hours
        hard_hours = float(self.config["compute_budget"]["hard_gpu_hours"])
        budget = {
            "population_rows": int(len(real)),
            "fresh_qwen_generations": generations,
            "reused_qwen_generations": ["B0 normal/oracle-structured"],
            "measured_qwen_rows_per_second": rows_per_second,
            "projected_qwen_generation_seconds": projected_seconds,
            "projected_qwen_generation_hours": projected_hours,
            "estimated_non_generation_gpu_hours": overhead_hours,
            "projected_total_gpu_hours": projected_total_hours,
            "target_total_gpu_hours": float(self.config["compute_budget"]["target_gpu_hours"]),
            "hard_total_gpu_hours": hard_hours,
            "within_hard_total_budget": projected_total_hours <= hard_hours,
        }
        write_json(self.output / "compute_budget.json", budget)
        if projected_total_hours > hard_hours:
            raise RuntimeError(
                f"Phase-1 projects to {projected_total_hours:.2f} GPU hours "
                f"({projected_hours:.2f} generation + {overhead_hours:.2f} overhead), "
                f"above the {hard_hours:.2f}-hour hard guidance."
            )

        report = {
            "status": "passed",
            "training_permitted": False,
            "evaluation_population": {
                "split": "test",
                "rows": int(len(real)),
                "path": str(self.subset_path),
                "sha256": file_sha256(self.subset_path),
            },
            "existing_qwen_b0_alignment": qwen_audit,
            "qwen_adapter": str(self.base_output / "training/best_adapter"),
            "qwen_adapter_sha256": directory_fingerprint(
                self.base_output / "training/best_adapter"
            ),
            "canonical_evaluator": source,
            "structured_diffusion": {
                "config": str(structured_config),
                "config_sha256": file_sha256(structured_config),
                "checkpoint": str(structured_checkpoint),
                "checkpoint_sha256": file_sha256(structured_checkpoint),
                "structured_only_patch": True,
            },
            "compute_budget": budget,
        }
        write_json(self.output / "code_audit.json", report)
        (self.output / "code_audit.md").write_text(
            self._code_audit_markdown(report), encoding="utf-8"
        )
        return report

    def _code_audit_markdown(self, report: dict[str, Any]) -> str:
        return "\n".join(
            [
                "# Qwen3-0.6B Phase-1 Code Audit",
                "",
                "## Existing Code Reused",
                "",
                "- Main experiment split: frozen `test_real.csv`; no new split or subset sampling.",
                "- Qwen adapter/tokenizer loading and generation bounds: existing `QwenFollowupExperiment` implementation.",
                "- Prompt policies and parsing: existing `build_prompt` and `parse_policy_continuation` paths.",
                "- Canonical Text C2ST: existing MiniLM embedding store and logistic-regression protocol.",
                "- Historical O1-O5 discovery: existing diffusion run manifests and conditioning specifications.",
                "- B0 normal output: exact existing oracle-structured Qwen output; no regeneration.",
                "",
                "## Minimal Patches",
                "",
                "- Added a Phase-1-only orchestrator; broad conditioning, decoding, and capacity experiments are unreachable from it.",
                "- Added exact event-key multiset matching for same-population LSTM/diffusion extraction.",
                "- Added structured-only hierarchical sampling, preserving the frozen structured denoiser while skipping text forwards.",
                "- Added matched real-heldout/Qwen memorization controls with the same training retrieval corpus.",
                "- Added an exact MiniLM commit requirement with no embedding fallback.",
                "",
                "## Leakage And Reuse Guards",
                "",
                "- Experiment C samples rating and verified from the frozen structured model on the exact test event spine.",
                "- Experiment C never passes true rating, true verified, true summary, or true review to Qwen.",
                "- B1 receives true summary but never true review or review length.",
                "- B2 receives neither generated nor true summary.",
                "- Existing full-table outputs are used only after complete ordered event-key verification against the full real table.",
                "",
                "## Budget",
                "",
                f"- Fresh Qwen generations: {report['compute_budget']['fresh_qwen_generations']}.",
                f"- Projected Qwen generation time: {report['compute_budget']['projected_qwen_generation_hours']:.3f} GPU hours.",
                "- No checkpoint is trained or modified.",
                "",
            ]
        )

    def _ensure_audit(self) -> dict[str, Any]:
        path = self.output / "code_audit.json"
        return json.loads(path.read_text()) if path.is_file() else self.audit()

    def _evaluation_context(
        self, device: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, EmbeddingStore, TextC2STProtocol, str]:
        self._ensure_audit()
        train = pd.read_csv(self.base.benchmark / "train_real.csv", low_memory=False)
        real = pd.read_csv(self.subset_path, low_memory=False)
        source = json.loads((self.output / "evaluation_model_source.json").read_text())
        expected = self.config["evaluation"]["embedding_revision"]
        if source.get("revision") != expected or not Path(source["local_snapshot"]).is_dir():
            raise RuntimeError("Canonical MiniLM snapshot is missing or is not the exact required commit")
        evaluation = self.config["evaluation"]
        protocol = TextC2STProtocol(
            name="canonical_paper_text_c2st_v1",
            embedding_backend="minilm",
            embedding_model=source["local_snapshot"],
            preprocessing="canonical",
            classifiers=("logistic_regression",),
            max_rows=int(evaluation["max_rows_per_class"]),
            seed=int(evaluation["seed"]),
            n_splits=int(evaluation["folds"]),
        )
        store = EmbeddingStore(self.output / "embedding_cache", device=device)
        return train, real, store, protocol, source["local_snapshot"]

    def _full_real(self) -> pd.DataFrame | None:
        path = Path(self.config["evaluation_population"]["full_real_table"])
        return pd.read_csv(path, low_memory=False) if path.is_file() else None

    def _align_first_candidate(
        self, label: str, candidates: Iterable[str], real: pd.DataFrame
    ) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
        audits = []
        full_real = self._full_real()
        selected = None
        for value in candidates:
            path = Path(value)
            if not path.is_file():
                audits.append({"path": str(path), "status": "missing", "aligned": False})
                continue
            frame = pd.read_csv(path, low_memory=False)
            aligned, audit = align_exact_population(
                real, frame, full_reference=full_real
            )
            audit.update(
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "required_text_columns_present": bool(
                        {"summary", "review_text"}.issubset(frame.columns)
                    ),
                }
            )
            audits.append(audit)
            if aligned is not None and audit["required_text_columns_present"]:
                selected = aligned
                break
        for audit in audits:
            audit["model"] = label
            audit["selected"] = bool(
                selected is not None and audit.get("aligned") and audit.get("required_text_columns_present")
            )
            if audit["selected"]:
                break
        return selected, audits

    def same_subset_comparison(self, device: str = "cuda") -> dict[str, Any]:
        self._ensure_audit()
        _, real, store, protocol, _ = self._evaluation_context(device)
        qwen = pd.read_csv(
            self.base_output / "oracle_structured/synthetic_text.csv", low_memory=False
        )
        qwen_aligned, qwen_audit = align_exact_population(real, qwen)
        if qwen_aligned is None:
            raise RuntimeError("Existing Qwen oracle output failed exact population alignment")
        frames: dict[str, pd.DataFrame] = {"Qwen oracle": qwen_aligned}
        audits: dict[str, Any] = {"Qwen oracle": qwen_audit}
        for label, key in (("LSTM", "lstm"), ("Masked diffusion", "masked_diffusion")):
            aligned, candidate_audits = self._align_first_candidate(
                label, self.config["same_subset_candidates"][key], real
            )
            audits[label] = candidate_audits
            if aligned is not None:
                frames[label] = aligned

        out = self.output / "same_subset_comparison"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        detailed = {}
        for label in ("LSTM", "Masked diffusion", "Qwen oracle"):
            frame = frames.get(label)
            if frame is None:
                rows.append(
                    {
                        "model": label,
                        "summary_c2st": None,
                        "review_c2st": None,
                        "macro_c2st": None,
                        "status": "exactly aligned artifact unavailable",
                    }
                )
                continue
            result = evaluate_protocol(
                real,
                frame,
                protocol,
                store,
                label=f"phase1_same_subset_{label.lower().replace(' ', '_')}",
            )
            detailed[label] = result
            summary, review, macro = nested_c2st(result)
            rows.append(
                {
                    "model": label,
                    "summary_c2st": summary,
                    "review_c2st": review,
                    "macro_c2st": macro,
                    "status": "ok",
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "canonical_results.csv", index=False)
        write_json(out / "alignment_audit.json", audits)
        write_json(out / "canonical_results_detailed.json", detailed)
        lines = [
            "# Controlled Same-Subset Decoder Comparison",
            "",
            "All reported rows use the exact frozen 3,982-row held-out event population.",
            "Lower C2ST error is better. No uncertainty or significance claim is made.",
            "",
            frame.to_string(index=False),
            "",
        ]
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return {"rows": rows, "audits": audits}

    def _prepare_structured_conditions(self, device: str, skip_existing: bool) -> pd.DataFrame:
        _, real, _, _, _ = self._evaluation_context("cpu")
        out = self.output / "generated_structured_qwen"
        out.mkdir(parents=True, exist_ok=True)
        spine_path = out / "evaluation_spine.csv"
        structured_path = out / "generated_structured.csv"
        audit_path = out / "alignment_audit.json"
        real.loc[:, list(ALIGNMENT_COLUMNS)].to_csv(spine_path, index=False)
        if not (skip_existing and structured_path.is_file() and audit_path.is_file()):
            cfg = self.config["structured_diffusion"]
            hierarchical_sample_from_config(
                load_config(cfg["config"]),
                checkpoint_path=cfg["checkpoint"],
                output_path=structured_path,
                num_rows="all",
                sample_batch_size=int(cfg["batch_size"]),
                structured_steps=int(cfg["structured_steps"]),
                timestep_spacing=str(cfg["timestep_spacing"]),
                inference_dtype=str(cfg["inference_dtype"]),
                temperature=float(cfg["temperature"]),
                graph_mode_override=str(cfg["graph_mode"]),
                device=device,
                seed=self.seed,
                synthetic_spine_path=spine_path,
                profile=True,
                profile_output=out / "structured_sampling_runtime.json",
                structured_only=True,
            )
            runtime_path = out / "structured_sampling_runtime.json"
            if runtime_path.is_file():
                runtime = json.loads(runtime_path.read_text())
                budget_path = self.output / "compute_budget.json"
                budget = json.loads(budget_path.read_text())
                budget["structured_sampling_wall_clock_seconds"] = float(
                    runtime.get("total_sampling_seconds", 0.0)
                )
                write_json(budget_path, budget)
        structured = pd.read_csv(structured_path, low_memory=False)
        audit = alignment_audit(real, structured)
        audit.update(
            {
                "generated_columns": list(structured.columns),
                "required_generated_attributes": ["rating", "verified"],
                "required_generated_attributes_present": bool(
                    {"rating", "verified"}.issubset(structured.columns)
                ),
                "true_structured_attributes_supplied_to_sampler": False,
                "fixed_real_event_spine": True,
            }
        )
        audit["passed"] = bool(
            audit["aligned"] and audit["required_generated_attributes_present"]
        )
        write_json(audit_path, audit)
        if not audit["passed"]:
            raise RuntimeError("Generated structured attributes failed strict event-spine alignment")
        return structured

    def generate(self, device: str = "cuda", *, skip_existing: bool = True) -> dict[str, Any]:
        self._ensure_audit()
        validate_runtime_dependencies()
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        real = pd.read_csv(self.subset_path, low_memory=False)
        structured = self._prepare_structured_conditions(device, skip_existing)
        normal_destination = self.output / "oracle_summary/normal.csv"
        hardlink_or_copy(
            self.base_output / "oracle_structured/synthetic_text.csv", normal_destination
        )
        base_metrics = self.base_output / "oracle_structured/generation_metrics.json"
        if base_metrics.is_file():
            hardlink_or_copy(base_metrics, normal_destination.parent / "generation_metrics.json")
            normal_metrics = json.loads(base_metrics.read_text())
            normal_metrics.update(
                {
                    "policy": "normal",
                    "conditioning": "rating_verified",
                    "summary_mode": "generated",
                    "reused_from": str(
                        self.base_output / "oracle_structured/synthetic_text.csv"
                    ),
                    "output_sha256": file_sha256(normal_destination),
                }
            )
            write_json(
                self.output / "generation_metrics/normal.json", normal_metrics
            )

        adapter = self.base_output / "training/best_adapter"
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter, local_files_only=True, torch_dtype=dtype
        ).to(device).eval()
        jobs = [
            (
                "oracle_summary",
                real,
                self.output / "oracle_summary/oracle_summary.csv",
            ),
            (
                "no_summary",
                real,
                self.output / "oracle_summary/no_summary.csv",
            ),
            (
                "generated_structured",
                structured.assign(summary=""),
                self.output / "generated_structured_qwen/synthetic_text.csv",
            ),
        ]
        records: dict[str, Any] = {}
        started = time.perf_counter()
        for name, conditions, destination in jobs:
            metrics_path = self.output / "generation_metrics" / f"{name}.json"
            if skip_existing and destination.is_file() and metrics_path.is_file():
                records[name] = json.loads(metrics_path.read_text())
                print(f"[phase1] reuse {name}: {destination}", flush=True)
                continue
            records[name] = self._generate_policy(
                model,
                tokenizer,
                conditions,
                name,
                self.config["generation"]["policies"][name],
                destination,
                int(self.config["generation"]["batch_size"]),
                device,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        budget = json.loads((self.output / "compute_budget.json").read_text())
        budget.update(
            {
                "generation_status": "complete",
                "generation_wall_clock_seconds_this_invocation": float(
                    time.perf_counter() - started
                ),
                "fresh_generation_metrics": records,
                "fresh_generation_gpu_hours_sum": float(
                    sum(float(item.get("seconds", 0.0)) for item in records.values())
                    / 3600.0
                ),
            }
        )
        write_json(self.output / "compute_budget.json", budget)
        return budget

    def evaluate(self, device: str = "cuda") -> dict[str, Any]:
        started = time.perf_counter()
        train, real, store, protocol, _ = self._evaluation_context(device)
        artifacts = {
            "normal": self.output / "oracle_summary/normal.csv",
            "oracle_summary": self.output / "oracle_summary/oracle_summary.csv",
            "no_summary": self.output / "oracle_summary/no_summary.csv",
            "generated_structured": self.output
            / "generated_structured_qwen/synthetic_text.csv",
        }
        missing = [str(path) for path in artifacts.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Phase-1 generation artifacts are missing:\n- " + "\n- ".join(missing))
        frames = {name: pd.read_csv(path, low_memory=False) for name, path in artifacts.items()}
        for name, frame in frames.items():
            audit = alignment_audit(real, frame)
            if not audit["aligned"]:
                raise RuntimeError(f"{name} output is not aligned to the frozen population")

        c2st = {}
        distributions = {}
        for name, frame in frames.items():
            fields = ("review_text",) if name in {"oracle_summary", "no_summary"} else ("summary", "review_text")
            c2st[name] = evaluate_protocol(
                real,
                frame,
                protocol,
                store,
                fields=fields,
                label=f"phase1_{name}",
            )
            distributions[name] = distribution_comparison(real, frame)
        review_scores = {name: nested_c2st(c2st[name])[1] for name in ("normal", "oracle_summary", "no_summary")}
        gain = float(review_scores["normal"] - review_scores["oracle_summary"])
        generation_metrics = {}
        for name in ("normal", "oracle_summary", "no_summary"):
            path = self.output / "generation_metrics" / f"{name}.json"
            if name == "normal" and not path.is_file():
                base_metrics = self.base_output / "oracle_structured/generation_metrics.json"
                normal_csv = self.output / "oracle_summary/normal.csv"
                if base_metrics.is_file() and normal_csv.is_file():
                    repaired = json.loads(base_metrics.read_text())
                    repaired.update(
                        {
                            "policy": "normal",
                            "conditioning": "rating_verified",
                            "summary_mode": "generated",
                            "reused_from": str(
                                self.base_output
                                / "oracle_structured/synthetic_text.csv"
                            ),
                            "output_sha256": file_sha256(normal_csv),
                            "metadata_reconstructed_without_regeneration": True,
                        }
                    )
                    write_json(path, repaired)
            generation_metrics[name] = (
                json.loads(path.read_text()) if path.is_file() else {}
            )
        oracle_summary_result = {
            "review_c2st": review_scores,
            "oracle_summary_gain": gain,
            "lower_c2st_is_better": True,
            "review_distribution": {
                name: distributions[name]["review_text"]
                for name in ("normal", "oracle_summary", "no_summary")
            },
            "generation_efficiency": generation_metrics,
        }
        write_json(self.output / "oracle_summary/metrics.json", oracle_summary_result)
        (self.output / "oracle_summary/report.md").write_text(
            "# Oracle-Summary Diagnostic\n\n"
            + "\n".join(
                f"- {name}: review C2ST `{value:.6f}`"
                for name, value in review_scores.items()
            )
            + f"\n- Oracle-summary gain: `{gain:.6f}`\n\nLower is better; these are descriptive diagnostics.\n",
            encoding="utf-8",
        )

        oracle_macro = nested_c2st(c2st["normal"])[2]
        generated_summary, generated_review, generated_macro = nested_c2st(
            c2st["generated_structured"]
        )
        degradation = float(generated_macro - oracle_macro)
        generated_result = {
            "oracle_structured_qwen": {
                "summary_c2st": nested_c2st(c2st["normal"])[0],
                "review_c2st": nested_c2st(c2st["normal"])[1],
                "macro_c2st": oracle_macro,
            },
            "generated_structured_qwen": {
                "summary_c2st": generated_summary,
                "review_c2st": generated_review,
                "macro_c2st": generated_macro,
            },
            "upstream_degradation": degradation,
            "interpretation_scope": "fixed-real-event-spine attribute-generator -> text-generator diagnostic",
            "not_full_relgen_database_generation": True,
        }
        write_json(
            self.output / "generated_structured_qwen/metrics.json", generated_result
        )
        (self.output / "generated_structured_qwen/report.md").write_text(
            "# Generated-Structured -> Qwen\n\n"
            f"- Oracle-structured macro C2ST: `{oracle_macro:.6f}`\n"
            f"- Generated-structured macro C2ST: `{generated_macro:.6f}`\n"
            f"- Upstream degradation: `{degradation:.6f}`\n\n"
            "This is a fixed-real-event-spine attribute-generator -> text-generator "
            "diagnostic, not full RelGen database generation.\n",
            encoding="utf-8",
        )

        memorization = matched_memorization_metrics(
            train,
            real,
            frames["normal"],
            training_rows=int(self.config["memorization"]["nearest_neighbor_training_rows"]),
            max_features=int(self.config["memorization"]["max_features"]),
        )
        memorization_out = self.output / "memorization_control"
        memorization_out.mkdir(parents=True, exist_ok=True)
        write_json(memorization_out / "metrics.json", memorization)
        (memorization_out / "report.md").write_text(
            self._memorization_markdown(memorization), encoding="utf-8"
        )
        write_json(self.output / "phase1_canonical_text_c2st.json", c2st)
        write_json(self.output / "phase1_distribution_metrics.json", distributions)
        self._record_phase_time("phase1_evaluation", time.perf_counter() - started)
        return {
            "oracle_summary": oracle_summary_result,
            "generated_structured": generated_result,
            "memorization": memorization,
        }

    def _memorization_markdown(self, metrics: dict[str, Any]) -> str:
        lines = [
            "# Memorization Control",
            "",
            "Real held-out and Qwen outputs are compared against the same real training corpus.",
            "This is not a differential privacy analysis.",
            "",
        ]
        for field in ("summary", "review_text"):
            lines.append(f"## {field}")
            for label in ("real_heldout", "qwen"):
                item = metrics[field][label]
                nn = item["nearest_neighbor"]
                lines.append(
                    f"- {label}: exact={item['exact_train_overlap_rate']:.6f}; "
                    f"NN mean={nn['mean']:.6f}, median={nn['median']:.6f}, "
                    f"p90={nn['p90']:.6f}, p95={nn['p95']:.6f}, max={nn['max']:.6f}"
                )
            lines.append("")
        return "\n".join(lines)

    def evaluate_diffusion_oracles(self, device: str = "cuda") -> dict[str, Any]:
        started = time.perf_counter()
        self._ensure_audit()
        required = self.config["diffusion_oracle"]["required_labels"]
        selected_root, artifacts = discover_diffusion_artifacts(
            self.config["diffusion_oracle"]["roots"], required
        )
        if selected_root is None:
            raise FileNotFoundError("No complete existing O1-O4 diffusion diagnostic root was found")
        optional = set(self.config["diffusion_oracle"].get("optional_labels", []))
        for manifest_path in selected_root.rglob("run_manifest.json"):
            manifest = json.loads(manifest_path.read_text())
            label = str(manifest.get("label", "")).upper()
            synthetic = manifest_path.parent / "synthetic_table.csv"
            if label not in optional or manifest.get("status") != "completed" or not synthetic.is_file():
                continue
            artifacts.append(
                {
                    "label": label,
                    "seed": int(manifest.get("seed", -1)),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": file_sha256(manifest_path),
                    "synthetic_path": str(synthetic),
                    "synthetic_sha256": file_sha256(synthetic),
                    "conditioning": PROGRESSIVE_CONDITION_SPECS[label].to_dict(),
                }
            )

        _, qwen_real, store, protocol, _ = self._evaluation_context(device)
        benchmark_manifest = json.loads(
            (self.base.benchmark / "benchmark_manifest.json").read_text()
        )
        record = benchmark_manifest["files"]["evaluation_real"]
        historical_real_path = Path(record["path"])
        if not historical_real_path.is_file() or file_sha256(historical_real_path) != record["sha256"]:
            raise RuntimeError("Historical diffusion evaluation population is missing or changed")
        historical_real = pd.read_csv(historical_real_path, low_memory=False)
        population_audit = alignment_audit(qwen_real, historical_real)
        if not population_audit["aligned"]:
            raise RuntimeError("Diffusion oracle and Qwen oracle evaluation populations differ")

        rows = []
        detailed = {}
        for artifact in sorted(artifacts, key=lambda row: (row["label"], row["seed"])):
            synthetic = pd.read_csv(artifact["synthetic_path"], low_memory=False)
            synthetic_audit = alignment_audit(historical_real, synthetic)
            if not synthetic_audit["aligned"]:
                raise RuntimeError(
                    f"Historical {artifact['label']} seed {artifact['seed']} output is not aligned"
                )
            key = f"{artifact['label']}_seed{artifact['seed']}"
            result = evaluate_protocol(
                historical_real,
                synthetic,
                protocol,
                store,
                label=f"phase1_diffusion_{key}",
            )
            detailed[key] = result
            summary, review, macro = nested_c2st(result)
            rows.append(
                {
                    "label": artifact["label"],
                    "seed": artifact["seed"],
                    "summary_c2st": summary,
                    "review_c2st": review,
                    "macro_c2st": macro,
                    "conditioning": artifact["conditioning"]["description"],
                    "valid_generative_baseline": artifact["conditioning"]["valid_generative_baseline"],
                    "synthetic_path": artifact["synthetic_path"],
                }
            )
        per_seed = pd.DataFrame(rows)
        aggregate = per_seed.groupby("label", as_index=False).agg(
            summary_c2st=("summary_c2st", "mean"),
            review_c2st=("review_c2st", "mean"),
            macro_c2st=("macro_c2st", "mean"),
            macro_c2st_std=("macro_c2st", "std"),
            num_seeds=("seed", "nunique"),
            conditioning=("conditioning", "first"),
            valid_generative_baseline=("valid_generative_baseline", "first"),
        )
        out = self.output / "diffusion_oracle_canonical"
        out.mkdir(parents=True, exist_ok=True)
        per_seed.to_csv(out / "canonical_results_per_seed.csv", index=False)
        aggregate.to_csv(out / "canonical_results.csv", index=False)
        write_json(out / "canonical_results_detailed.json", detailed)
        write_json(
            out / "alignment_audit.json",
            {
                "qwen_vs_diffusion_real_population": population_audit,
                "selected_historical_root": str(selected_root),
                "artifacts": artifacts,
            },
        )
        phase1_c2st_path = self.output / "phase1_canonical_text_c2st.json"
        qwen = (
            json.loads(phase1_c2st_path.read_text())["normal"]
            if phase1_c2st_path.is_file()
            else json.loads(
                (self.base_output / "oracle_structured/canonical_text_c2st.json").read_text()
            )
        )
        qwen_macro = nested_c2st(qwen)[2]
        oracle_rows = aggregate.loc[
            ~aggregate["valid_generative_baseline"].astype(bool)
        ]
        if oracle_rows.empty:
            raise RuntimeError("No oracle-conditioned O1-O3 diffusion result was available")
        best = oracle_rows.loc[oracle_rows["macro_c2st"].idxmin()].to_dict()
        (out / "report.md").write_text(
            "# Canonical Diffusion Oracle Re-evaluation\n\n"
            f"- Matched evaluation rows: {len(historical_real):,}\n"
            f"- Best oracle diffusion: {best['label']} (macro `{best['macro_c2st']:.6f}`)\n"
            f"- Qwen oracle macro: `{qwen_macro:.6f}`\n\n"
            "All variants use the exact pinned MiniLM evaluator; lower is better.\n",
            encoding="utf-8",
        )
        self._record_phase_time("diffusion_oracle_evaluation", time.perf_counter() - started)
        return {"per_seed": rows, "aggregate": aggregate.to_dict("records")}

    def _record_phase_time(self, name: str, seconds: float) -> None:
        path = self.output / "compute_budget.json"
        budget = json.loads(path.read_text()) if path.is_file() else {}
        budget.setdefault("evaluation_stages", {})[name] = {
            "wall_clock_seconds": float(seconds)
        }
        generation_seconds = sum(
            float(item.get("seconds", 0.0))
            for item in budget.get("fresh_generation_metrics", {}).values()
        )
        budget["measured_phase1_wall_clock_seconds"] = float(
            generation_seconds
            + float(budget.get("structured_sampling_wall_clock_seconds", 0.0))
            + sum(
                float(item["wall_clock_seconds"])
                for item in budget["evaluation_stages"].values()
            )
        )
        write_json(path, budget)

    def report(self) -> dict[str, Any]:
        required = [
            self.output / "same_subset_comparison/canonical_results.csv",
            self.output / "oracle_summary/metrics.json",
            self.output / "generated_structured_qwen/metrics.json",
            self.output / "memorization_control/metrics.json",
            self.output / "diffusion_oracle_canonical/canonical_results.csv",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Phase-1 report inputs are missing:\n- " + "\n- ".join(missing))
        same = pd.read_csv(required[0])
        oracle = json.loads(required[1].read_text())
        generated = json.loads(required[2].read_text())
        memorization = json.loads(required[3].read_text())
        diffusion = pd.read_csv(required[4])
        budget = json.loads((self.output / "compute_budget.json").read_text())

        gain = float(oracle["oracle_summary_gain"])
        propagation = "STRONG" if gain >= 0.10 else "MODERATE" if gain >= 0.03 else "WEAK"
        degradation = float(generated["upstream_degradation"])
        upstream = "YES" if degradation >= 0.05 else "NO" if degradation <= 0.02 else "UNCLEAR"
        summary_excess = memorization["summary"]["qwen"]["exact_train_overlap_rate"] - memorization["summary"]["real_heldout"]["exact_train_overlap_rate"]
        review_excess = memorization["review_text"]["qwen"]["exact_train_overlap_rate"] - memorization["review_text"]["real_heldout"]["exact_train_overlap_rate"]
        max_excess = max(summary_excess, review_excess)
        excess_memorization = "YES" if max_excess >= 0.05 else "NO" if max_excess <= 0.02 else "UNCLEAR"
        qwen_macro = float(generated["oracle_structured_qwen"]["macro_c2st"])
        oracle_candidates = diffusion.loc[
            ~diffusion["valid_generative_baseline"].astype(bool)
        ]
        oracle_diffusion = oracle_candidates.loc[
            oracle_candidates["macro_c2st"].idxmin()
        ]
        advantage = float(oracle_diffusion["macro_c2st"] - qwen_macro)
        pretraining = "STRONG" if advantage >= 0.10 else "MODERATE" if advantage >= 0.03 else "WEAK" if advantage > 0 else "UNCLEAR"

        oracle_review = float(oracle["review_c2st"]["oracle_summary"])
        if propagation == "STRONG":
            dominant = "summary propagation"
            next_experiment = "D. independent summary/review factorization"
        elif excess_memorization == "YES":
            dominant = "low diversity / mode concentration"
            next_experiment = "A. decoding sweep"
        elif upstream == "YES":
            dominant = "upstream structured error"
            next_experiment = "F. temporal-relational conditioning"
        else:
            dominant = "decoder capacity"
            next_experiment = "C. Qwen3-1.7B capacity probe"
        diagnosis = {
            "decoder capacity": "strongly supported" if oracle_review >= 0.60 else "moderately supported",
            "summary propagation": "strongly supported" if propagation == "STRONG" else "moderately supported" if propagation == "MODERATE" else "weakly supported",
            "low diversity / mode concentration": "strongly supported" if excess_memorization == "YES" else "unresolved" if excess_memorization == "UNCLEAR" else "weakly supported",
            "upstream structured error": "strongly supported" if upstream == "YES" else "unresolved" if upstream == "UNCLEAR" else "weakly supported",
            "memorization": "moderately supported" if excess_memorization == "YES" else "unresolved" if excess_memorization == "UNCLEAR" else "weakly supported",
            "evaluator mismatch": "rejected",
        }
        decision = {
            "summary_propagation": propagation,
            "upstream_error_material": upstream,
            "excess_memorization": excess_memorization,
            "pretraining_advantage": pretraining,
            "dominant_remaining_bottleneck": dominant,
            "phase2_recommendation": next_experiment,
            "failure_diagnosis": diagnosis,
        }
        model_scores = same.set_index("model").to_dict("index")
        qwen_same = model_scores.get("Qwen oracle", {}).get("macro_c2st")
        decision["same_subset_answers"] = {
            "qwen_beats_lstm": _lower_is_better_answer(
                qwen_same, model_scores.get("LSTM", {}).get("macro_c2st")
            ),
            "qwen_beats_masked_diffusion": _lower_is_better_answer(
                qwen_same,
                model_scores.get("Masked diffusion", {}).get("macro_c2st"),
            ),
        }
        write_json(self.output / "phase1_decision.json", decision)
        report = self._final_report(
            same,
            oracle,
            generated,
            memorization,
            diffusion,
            budget,
            decision,
        )
        (self.output / "phase1_report.md").write_text(report, encoding="utf-8")
        self._print_final(same, oracle, generated, memorization, oracle_diffusion, budget, decision)
        return decision

    def _final_report(
        self,
        same: pd.DataFrame,
        oracle: dict[str, Any],
        generated: dict[str, Any],
        memorization: dict[str, Any],
        diffusion: pd.DataFrame,
        budget: dict[str, Any],
        decision: dict[str, Any],
    ) -> str:
        same_table = same.to_string(index=False)
        diffusion_table = diffusion.to_string(index=False)
        diagnosis = "\n".join(
            f"- {name}: **{value}**" for name, value in decision["failure_diagnosis"].items()
        )
        return f"""# Qwen3-0.6B Phase-1 Diagnostics

## 1. Executive Summary

The exact-population results, oracle-summary intervention, generated-structured run,
matched memorization control, and canonical diffusion-oracle re-evaluation are reported
below. Lower Text C2ST error is better. All conclusions are descriptive.

1. Qwen beats LSTM on the matched population: **{decision['same_subset_answers']['qwen_beats_lstm']}**.
2. Qwen beats masked diffusion on the matched population: **{decision['same_subset_answers']['qwen_beats_masked_diffusion']}**.
3. Oracle-summary effect: **{decision['summary_propagation']}**.
4. Removing summary is reported directly below; no significance claim is made.
5. Generated structured conditioning is material: **{decision['upstream_error_material']}**.
6. Excess memorization beyond natural duplication: **{decision['excess_memorization']}**.
7. Pretrained-Qwen advantage over oracle masked diffusion: **{decision['pretraining_advantage']}**.
8. Dominant remaining bottleneck: **{decision['dominant_remaining_bottleneck']}**.

## 2. Controlled Same-Subset Comparison

```text
{same_table}
```

## 3. Oracle-Summary Diagnostic

- Normal review C2ST: {oracle['review_c2st']['normal']:.6f}
- Oracle-summary review C2ST: {oracle['review_c2st']['oracle_summary']:.6f}
- No-summary review C2ST: {oracle['review_c2st']['no_summary']:.6f}
- Oracle-summary gain: {oracle['oracle_summary_gain']:.6f}
- Summary propagation: **{decision['summary_propagation']}**

## 4. Generated-Structured Qwen

- Oracle decoder macro C2ST: {generated['oracle_structured_qwen']['macro_c2st']:.6f}
- Generated-structured macro C2ST: {generated['generated_structured_qwen']['macro_c2st']:.6f}
- Upstream degradation: {generated['upstream_degradation']:.6f}
- Upstream error material: **{decision['upstream_error_material']}**

This is fixed-real-spine attribute-generator -> text-generator quality. It is not full
end-to-end RelGen database generation because the event spine remains real and fixed.

## 5. Memorization Control

- Summary exact overlap, real heldout: {memorization['summary']['real_heldout']['exact_train_overlap_rate']:.6f}
- Summary exact overlap, Qwen: {memorization['summary']['qwen']['exact_train_overlap_rate']:.6f}
- Review exact overlap, real heldout: {memorization['review_text']['real_heldout']['exact_train_overlap_rate']:.6f}
- Review exact overlap, Qwen: {memorization['review_text']['qwen']['exact_train_overlap_rate']:.6f}
- Excess memorization: **{decision['excess_memorization']}**

This diagnostic does not support any differential privacy claim.

## 6. Canonical Diffusion Oracle Re-Evaluation

```text
{diffusion_table}
```

- Pretraining advantage: **{decision['pretraining_advantage']}**

## 7. Failure Diagnosis

{diagnosis}

Dominant remaining bottleneck: **{decision['dominant_remaining_bottleneck']}**.

## 8. Phase-2 Recommendation

**{decision['phase2_recommendation']}**

Only this one next experiment is recommended from the supplied Phase-2 list.

Total measured Phase-1 wall clock recorded by the runner: {budget.get('measured_phase1_wall_clock_seconds')} seconds.
"""

    def _print_final(
        self,
        same: pd.DataFrame,
        oracle: dict[str, Any],
        generated: dict[str, Any],
        memorization: dict[str, Any],
        oracle_diffusion: pd.Series,
        budget: dict[str, Any],
        decision: dict[str, Any],
    ) -> None:
        print("\n============================================================")
        print("QWEN3-0.6B PHASE-1 DIAGNOSTICS")
        print("============================================================\n")
        print("CONTROLLED SAME-SUBSET C2ST\n------------------------------------------------------------")
        print(same.to_string(index=False))
        print("\n------------------------------------------------------------\nORACLE SUMMARY\n------------------------------------------------------------")
        print(f"Normal review: {oracle['review_c2st']['normal']}")
        print(f"Oracle-summary review: {oracle['review_c2st']['oracle_summary']}")
        print(f"No-summary review: {oracle['review_c2st']['no_summary']}")
        print(f"oracle_summary_gain: {oracle['oracle_summary_gain']}")
        print(f"SUMMARY PROPAGATION: {decision['summary_propagation']}")
        print("\n------------------------------------------------------------\nGENERATED STRUCTURED -> QWEN\n------------------------------------------------------------")
        print(f"Oracle Qwen: {generated['oracle_structured_qwen']['macro_c2st']}")
        print(f"Generated-structured Qwen: {generated['generated_structured_qwen']['macro_c2st']}")
        print(f"upstream_degradation: {generated['upstream_degradation']}")
        print(f"UPSTREAM ERROR MATERIAL: {decision['upstream_error_material']}")
        print("\n------------------------------------------------------------\nMEMORIZATION CONTROL\n------------------------------------------------------------")
        print("                         REAL HELDOUT    QWEN")
        print(f"Summary exact overlap    {memorization['summary']['real_heldout']['exact_train_overlap_rate']:.6f}       {memorization['summary']['qwen']['exact_train_overlap_rate']:.6f}")
        print(f"Review exact overlap     {memorization['review_text']['real_heldout']['exact_train_overlap_rate']:.6f}       {memorization['review_text']['qwen']['exact_train_overlap_rate']:.6f}")
        print(f"EXCESS MEMORIZATION: {decision['excess_memorization']}")
        print("\n------------------------------------------------------------\nDIFFUSION ORACLE\n------------------------------------------------------------")
        print(f"Best oracle diffusion: {oracle_diffusion['label']} {oracle_diffusion['macro_c2st']}")
        print(f"Qwen oracle: {generated['oracle_structured_qwen']['macro_c2st']}")
        print(f"PRETRAINING ADVANTAGE: {decision['pretraining_advantage']}")
        print("\n------------------------------------------------------------\nTOTAL WALL CLOCK\n------------------------------------------------------------")
        print(budget.get("measured_phase1_wall_clock_seconds"))
        print("\n------------------------------------------------------------\nDOMINANT REMAINING BOTTLENECK\n------------------------------------------------------------")
        print(decision["dominant_remaining_bottleneck"])
        print("\n------------------------------------------------------------\nNEXT EXPERIMENT\n------------------------------------------------------------")
        print(decision["phase2_recommendation"])
        print("\n============================================================")
