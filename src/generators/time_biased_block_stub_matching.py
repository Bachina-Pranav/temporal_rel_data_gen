"""Time-biased exact block-stub matching event-spine generator."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .fast_event_spine_metrics import evaluate_fast_event_spine
from .fast_lowrank_temporal_event import load_entity_blocks
from .fast_temporal_activity import FastTemporalActivityModel, canonical_time_bucket, resolve_auto_alpha
from .lowrank_time_gated_affinity import LowRankTimeGatedAffinity
from .stub_dynamic_pairing import reorder_products_within_cells_by_dynamic_affinity
from .stub_time_assignment import assign_stubs_to_slots_by_time
from .temporal_kernel_bandwidth import estimate_temporal_kernel_bandwidths
from .temporal_shrinkage_estimator import DEFAULT_CANDIDATE_ALPHAS, estimate_temporal_shrinkage_alpha


METHOD_NAME = "time_biased_block_stub_matching"
METHOD_ALIAS = "temporal_stub_matching_event"


class TimeBiasedBlockStubMatchingGenerator:
    """Exact degree/block-time temporal event-spine generator.

    TimeBiasedBlockStubMatchingGenerator exactly matches entity degree stubs
    to block-pair-time slots. Desired times for entity stubs are sampled using
    local temporal kernel smoothing around each entity's observed event times,
    with bandwidths estimated from real inter-event gaps and no synthetic-
    metric tuning. This preserves entity lifecycle identity while adding
    stochastic temporal variation. Products are then paired to customers within
    each temporal cell using a fixed penalized low-rank dynamic affinity score
    F_{u,i,t} = (z_u * g_t)^T z_i.
    """

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        structure_debug_dir: Optional[str | Path] = None,
        time_granularity: str = "day",
        time_gate_granularity: str = "month",
        rank: int = 32,
        alpha_customer_time: Any = "auto",
        alpha_product_time: Any = "auto",
        temporal_shrinkage_mode: str = "median_degree",
        desired_time_sampling_mode: str = "local_kernel",
        alpha_time_gate: Any = "auto",
        block_time_smoothing: float = 5.0,
        kernel_bandwidth_mode: str = "auto_block_iqr",
        kernel_bandwidth_scale: float = 0.25,
        kernel_min_bandwidth_days: float = 1.0,
        kernel_max_bandwidth_days: Optional[float] = None,
        kernel_fixed_bandwidth_days: float = 7.0,
        kernel_type: str = "discrete_laplace",
        pairing_mode: str = "dynamic_exact_penalized",
        max_exact_affinity_cell_size: int = 128,
        large_cell_pairing: str = "projection_sort",
        desired_time_jitter: float = 1e-3,
        enable_fast_overlap_repair: bool = False,
        overlap_resample_prob: float = 0.0,
        lambda_duplicate_pair: float = 1.0,
        lambda_real_pair_overlap: float = 1.0,
        lambda_exact_event_overlap: float = 3.0,
        seed: int = 42,
    ):
        if time_granularity != "day":
            raise ValueError("TimeBiasedBlockStubMatchingGenerator currently supports day event buckets only")
        if temporal_shrinkage_mode not in {"median_degree", "empirical_bayes", "fixed"}:
            raise ValueError("temporal_shrinkage_mode must be median_degree, empirical_bayes, or fixed")
        if desired_time_sampling_mode not in {"mixture_shrinkage", "empirical_bayes", "local_kernel", "empirical_exact"}:
            raise ValueError("desired_time_sampling_mode must be mixture_shrinkage, empirical_bayes, local_kernel, or empirical_exact")
        if kernel_bandwidth_mode not in {"auto_block_iqr", "auto_global_iqr", "fixed"}:
            raise ValueError("kernel_bandwidth_mode must be auto_block_iqr, auto_global_iqr, or fixed")
        if kernel_type not in {"discrete_laplace", "discrete_gaussian", "none"}:
            raise ValueError("kernel_type must be discrete_laplace, discrete_gaussian, or none")
        if pairing_mode not in {"random", "static_projection", "dynamic_projection", "dynamic_exact_small", "dynamic_exact_penalized"}:
            raise ValueError("unsupported pairing_mode")
        if large_cell_pairing not in {"projection_sort", "exact_greedy"}:
            raise ValueError("large_cell_pairing must be projection_sort or exact_greedy")
        self.customer_id_col = customer_id_col
        self.product_id_col = product_id_col
        self.timestamp_col = timestamp_col
        self.structure_debug_dir = Path(structure_debug_dir) if structure_debug_dir else None
        self.time_granularity = time_granularity
        self.time_gate_granularity = time_gate_granularity
        self.rank = int(rank)
        self.alpha_customer_time = alpha_customer_time
        self.alpha_product_time = alpha_product_time
        self.temporal_shrinkage_mode = temporal_shrinkage_mode
        self.desired_time_sampling_mode = desired_time_sampling_mode
        self.alpha_time_gate = alpha_time_gate
        self.block_time_smoothing = float(block_time_smoothing)
        self.kernel_bandwidth_mode = kernel_bandwidth_mode
        self.kernel_bandwidth_scale = float(kernel_bandwidth_scale)
        self.kernel_min_bandwidth_days = float(kernel_min_bandwidth_days)
        self.kernel_max_bandwidth_days = None if kernel_max_bandwidth_days is None else float(kernel_max_bandwidth_days)
        self.kernel_fixed_bandwidth_days = float(kernel_fixed_bandwidth_days)
        self.kernel_type = kernel_type
        self.pairing_mode = pairing_mode
        self.max_exact_affinity_cell_size = int(max_exact_affinity_cell_size)
        self.large_cell_pairing = large_cell_pairing
        self.desired_time_jitter = float(desired_time_jitter)
        self.enable_fast_overlap_repair = bool(enable_fast_overlap_repair)
        self.overlap_resample_prob = float(overlap_resample_prob)
        self.lambda_duplicate_pair = float(lambda_duplicate_pair)
        self.lambda_real_pair_overlap = float(lambda_real_pair_overlap)
        self.lambda_exact_event_overlap = float(lambda_exact_event_overlap)
        self.seed = int(seed)
        self.real_df: Optional[pd.DataFrame] = None
        self.synthetic_df: Optional[pd.DataFrame] = None
        self.customer_blocks: Dict[Any, int] = {}
        self.product_blocks: Dict[Any, int] = {}
        self.block_warnings: List[str] = []
        self.block_pair_time_counts = pd.DataFrame()
        self.customer_degrees: Dict[Any, int] = {}
        self.product_degrees: Dict[Any, int] = {}
        self.real_event_set: set[tuple[Any, Any, str]] = set()
        self.customer_activity: Optional[FastTemporalActivityModel] = None
        self.product_activity: Optional[FastTemporalActivityModel] = None
        self.affinity_model: Optional[LowRankTimeGatedAffinity] = None
        self.real_pair_keys: set[int] = set()
        self.real_event_keys: set[int] = set()
        self.day_to_code: Dict[str, int] = {}
        self.code_to_day: List[str] = []
        self.gate_to_code: Dict[str, int] = {}
        self.code_to_gate: List[str] = []
        self.slot_id = np.asarray([], dtype=int)
        self.slot_customer_block = np.asarray([], dtype=int)
        self.slot_product_block = np.asarray([], dtype=int)
        self.slot_time_code = np.asarray([], dtype=int)
        self.slot_time_bucket = np.asarray([], dtype=object)
        self.slot_time_gate_code = np.asarray([], dtype=int)
        self.slot_customer_idx = np.asarray([], dtype=np.int64)
        self.slot_product_idx = np.asarray([], dtype=np.int64)
        self.slot_customer_id = np.asarray([], dtype=object)
        self.slot_product_id = np.asarray([], dtype=object)
        self.slot_summary: Dict[str, Any] = {}
        self.customer_assignment_summary: Dict[str, Any] = {}
        self.product_assignment_summary: Dict[str, Any] = {}
        self.dynamic_pairing_summary: Dict[str, Any] = {}
        self.runtime_metadata: Dict[str, Any] = {}
        self.customer_alpha_result: Dict[str, Any] = {}
        self.product_alpha_result: Dict[str, Any] = {}
        self.alpha_customer_time_selected: Optional[float] = None
        self.alpha_product_time_selected: Optional[float] = None
        self.customer_kernel_bandwidth_result: Dict[str, Any] = {}
        self.product_kernel_bandwidth_result: Dict[str, Any] = {}
        self.customer_local_kernel_state: Dict[str, Any] = {}
        self.product_local_kernel_state: Dict[str, Any] = {}

    def fit(self, real_df: pd.DataFrame) -> "TimeBiasedBlockStubMatchingGenerator":
        fit_start = time.time()
        print("[fit] loading data", flush=True)
        required = [self.customer_id_col, self.product_id_col, self.timestamp_col]
        missing = [col for col in required if col not in real_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        frame = real_df[required].copy()
        frame["_time_bucket"] = canonical_time_bucket(frame[self.timestamp_col], self.time_granularity)
        frame[self.timestamp_col] = frame["_time_bucket"]
        self.code_to_day = sorted(frame["_time_bucket"].unique().tolist())
        self.day_to_code = {day: idx for idx, day in enumerate(self.code_to_day)}
        frame["_time_code"] = frame["_time_bucket"].map(self.day_to_code).astype(int)
        frame["_time_gate_bucket"] = canonical_time_bucket(frame["_time_bucket"], self.time_gate_granularity)
        self.code_to_gate = sorted(frame["_time_gate_bucket"].unique().tolist())
        self.gate_to_code = {gate: idx for idx, gate in enumerate(self.code_to_gate)}
        frame["_time_gate_code"] = frame["_time_gate_bucket"].map(self.gate_to_code).astype(int)
        self.customer_blocks, self.product_blocks, self.block_warnings = self._load_or_fallback_blocks(frame)
        frame["_customer_block"] = frame[self.customer_id_col].map(self.customer_blocks).fillna(0).astype(int)
        frame["_product_block"] = frame[self.product_id_col].map(self.product_blocks).fillna(0).astype(int)
        self.real_df = frame
        self.customer_degrees = frame[self.customer_id_col].value_counts().astype(int).to_dict()
        self.product_degrees = frame[self.product_id_col].value_counts().astype(int).to_dict()
        self.real_event_set = set(
            zip(
                frame[self.customer_id_col].to_numpy(dtype=object),
                frame[self.product_id_col].to_numpy(dtype=object),
                frame["_time_bucket"].to_numpy(dtype=object),
            )
        )
        self.block_pair_time_counts = self._build_block_pair_time_counts(frame)
        self._select_temporal_shrinkage_alphas(frame)
        print("[fit] fitting temporal activity models", flush=True)
        self.customer_activity = FastTemporalActivityModel(
            alpha=self.alpha_customer_time_selected,
            block_time_smoothing=self.block_time_smoothing,
            granularity=self.time_granularity,
            entity_kind="customer",
        ).fit(frame, self.customer_id_col, "_time_bucket", self.customer_blocks)
        self.product_activity = FastTemporalActivityModel(
            alpha=self.alpha_product_time_selected,
            block_time_smoothing=self.block_time_smoothing,
            granularity=self.time_granularity,
            entity_kind="product",
        ).fit(frame, self.product_id_col, "_time_bucket", self.product_blocks)
        self._fit_temporal_kernel_bandwidths()
        self._build_real_overlap_key_sets(frame)
        print("[fit] fitting low-rank time-gated affinity", flush=True)
        self.affinity_model = LowRankTimeGatedAffinity(
            rank=self.rank,
            alpha_time_gate=self.alpha_time_gate,
            time_gate_granularity=self.time_gate_granularity,
            seed=self.seed,
        ).fit(frame, self.customer_id_col, self.product_id_col, "_time_bucket")
        self.runtime_metadata["fit_seconds"] = float(time.time() - fit_start)
        print(f"[fit] done in {self.runtime_metadata['fit_seconds']:.2f}s", flush=True)
        return self

    def sample(self, seed: Optional[int] = None) -> pd.DataFrame:
        if self.real_df is None or self.customer_activity is None or self.product_activity is None or self.affinity_model is None:
            raise RuntimeError("Call fit before sample.")
        sample_start = time.time()
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        self._build_slots()
        print("[customer-stubs] assigning exact customer stubs by time-biased sort", flush=True)
        self.slot_customer_idx, self.customer_assignment_summary = assign_stubs_to_slots_by_time(
            self.customer_activity.entity_ids,
            self.customer_degrees,
            self.customer_blocks,
            self.slot_customer_block,
            self.slot_time_code,
            self.customer_activity,
            rng,
            jitter=self.desired_time_jitter,
            log_label="customer-stubs",
            return_entity_indices=True,
            desired_time_sampling_mode=self.desired_time_sampling_mode,
            local_kernel_state=self.customer_local_kernel_state,
        )
        print(f"[customer-stubs] done in {self.customer_assignment_summary['assignment_seconds']:.2f}s", flush=True)
        print("[product-stubs] assigning exact product stubs by time-biased sort", flush=True)
        self.slot_product_idx, self.product_assignment_summary = assign_stubs_to_slots_by_time(
            self.product_activity.entity_ids,
            self.product_degrees,
            self.product_blocks,
            self.slot_product_block,
            self.slot_time_code,
            self.product_activity,
            rng,
            jitter=self.desired_time_jitter,
            log_label="product-stubs",
            return_entity_indices=True,
            desired_time_sampling_mode=self.desired_time_sampling_mode,
            local_kernel_state=self.product_local_kernel_state,
        )
        print(f"[product-stubs] done in {self.product_assignment_summary['assignment_seconds']:.2f}s", flush=True)
        self.slot_customer_id = self.customer_activity.entity_ids[self.slot_customer_idx]
        self.slot_product_id = self.product_activity.entity_ids[self.slot_product_idx]
        print(f"[pairing] {self.pairing_mode} over {len(self.block_pair_time_counts):,} cells", flush=True)
        self.slot_product_id, self.dynamic_pairing_summary = reorder_products_within_cells_by_dynamic_affinity(
            self.slot_customer_id,
            self.slot_product_id,
            self.slot_customer_block,
            self.slot_product_block,
            self.slot_time_code,
            self.slot_time_gate_code,
            self.affinity_model,
            self.pairing_mode,
            self.max_exact_affinity_cell_size,
            rng,
            large_cell_pairing=self.large_cell_pairing,
            code_to_time_gate=self.code_to_gate,
            code_to_time_bucket=self.code_to_day,
            enable_fast_overlap_repair=self.enable_fast_overlap_repair,
            real_event_set=self.real_event_set,
            slot_customer_idx=self.slot_customer_idx,
            slot_product_idx=self.slot_product_idx,
            real_pair_keys=self.real_pair_keys,
            real_event_keys=self.real_event_keys,
            num_products=len(self.product_activity.entity_ids),
            num_time_codes=len(self.code_to_day),
            lambda_duplicate_pair=self.lambda_duplicate_pair,
            lambda_real_pair_overlap=self.lambda_real_pair_overlap,
            lambda_exact_event_overlap=self.lambda_exact_event_overlap,
        )
        print(f"[pairing] done in {self.dynamic_pairing_summary['dynamic_pairing_seconds']:.2f}s", flush=True)
        synthetic = pd.DataFrame(
            {
                self.customer_id_col: self.slot_customer_id,
                self.product_id_col: self.slot_product_id,
                self.timestamp_col: self.slot_time_bucket,
            }
        ).sort_values([self.timestamp_col, self.customer_id_col, self.product_id_col]).reset_index(drop=True)
        self.synthetic_df = synthetic
        self.runtime_metadata.update(
            {
                "customer_stub_assignment_seconds": float(self.customer_assignment_summary["assignment_seconds"]),
                "product_stub_assignment_seconds": float(self.product_assignment_summary["assignment_seconds"]),
                "customer_stub_construction_seconds": float(self.customer_assignment_summary["stub_construction_seconds"]),
                "customer_desired_time_sampling_seconds": float(self.customer_assignment_summary["desired_time_sampling_seconds"]),
                "customer_sorting_seconds": float(self.customer_assignment_summary["sorting_seconds"]),
                "customer_slot_assignment_seconds": float(self.customer_assignment_summary["slot_assignment_seconds"]),
                "product_stub_construction_seconds": float(self.product_assignment_summary["stub_construction_seconds"]),
                "product_desired_time_sampling_seconds": float(self.product_assignment_summary["desired_time_sampling_seconds"]),
                "product_sorting_seconds": float(self.product_assignment_summary["sorting_seconds"]),
                "product_slot_assignment_seconds": float(self.product_assignment_summary["slot_assignment_seconds"]),
                "dynamic_pairing_seconds": float(self.dynamic_pairing_summary["dynamic_pairing_seconds"]),
            }
        )
        verify_start = time.time()
        self._verify_exact_constraints(synthetic)
        self.runtime_metadata["verification_seconds"] = float(time.time() - verify_start)
        sample_seconds = float(time.time() - sample_start)
        self.runtime_metadata.update(
            {
                "sample_seconds": sample_seconds,
                "total_seconds": sample_seconds + float(self.runtime_metadata.get("fit_seconds", 0.0)),
                "events_per_second": float(len(synthetic) / max(sample_seconds, 1e-9)),
            }
        )
        print("[verify] all exact constraints passed", flush=True)
        print(
            f"[done] total_seconds={self.runtime_metadata['total_seconds']:.2f}, "
            f"events_per_second={self.runtime_metadata['events_per_second']:.1f}",
            flush=True,
        )
        return synthetic

    def save_outputs(self, output_dir: str | Path) -> None:
        if self.synthetic_df is None:
            raise RuntimeError("Call sample before save_outputs.")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.synthetic_df.to_csv(output / "synthetic_review.csv", index=False)
        self.save_debug(output / "debug")
        self.save_metadata(output / "metadata.json")

    def save_debug(self, debug_dir: str | Path) -> None:
        if self.real_df is None:
            raise RuntimeError("Call fit before save_debug.")
        debug = Path(debug_dir)
        debug.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({self.customer_id_col: list(self.customer_blocks), "customer_block": list(self.customer_blocks.values())}).to_csv(
            debug / "customer_blocks.csv",
            index=False,
        )
        pd.DataFrame({self.product_id_col: list(self.product_blocks), "product_block": list(self.product_blocks.values())}).to_csv(
            debug / "product_blocks.csv",
            index=False,
        )
        self.block_pair_time_counts.to_csv(debug / "block_pair_time_counts.csv", index=False)
        self.customer_activity.save_summary(debug / "customer_time_activity_summary.json")
        self.product_activity.save_summary(debug / "product_time_activity_summary.json")
        self.affinity_model.save_summary(debug / "lowrank_time_gated_affinity_summary.json")
        write_json(self.slot_summary, debug / "slot_summary.json")
        write_json(self.customer_assignment_summary, debug / "customer_stub_assignment_summary.json")
        write_json(self.product_assignment_summary, debug / "product_stub_assignment_summary.json")
        write_json(self.dynamic_pairing_summary, debug / "dynamic_pairing_summary.json")
        write_json(self.customer_alpha_result, debug / "customer_temporal_shrinkage_estimation.json")
        write_json(self.product_alpha_result, debug / "product_temporal_shrinkage_estimation.json")
        write_json(self._kernel_diagnostics(self.customer_kernel_bandwidth_result), debug / "customer_temporal_kernel_bandwidths.json")
        write_json(self._kernel_diagnostics(self.product_kernel_bandwidth_result), debug / "product_temporal_kernel_bandwidths.json")
        write_json(self.array_dtype_summary(), debug / "array_dtype_summary.json")

    def save_metadata(self, path: str | Path) -> None:
        write_json(self.metadata(), path)

    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, compute_c2st: bool = False) -> Dict[str, Any]:
        metrics = evaluate_fast_event_spine(
            real_df,
            synthetic_df,
            customer_col=self.customer_id_col,
            product_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            compute_c2st=compute_c2st,
            metadata=self.metadata(),
        )
        real_bpt = self._bpt_counts(real_df)
        syn_bpt = self._bpt_counts(synthetic_df)
        cells = sorted(set(real_bpt.index).union(set(syn_bpt.index)))
        bpt_l1 = float(sum(abs(real_bpt.get(cell, 0) - syn_bpt.get(cell, 0)) for cell in cells) / max(len(real_df), 1))
        metrics["block_pair_time_count_l1"] = bpt_l1
        metrics["block_pair_time_exact_match"] = bool(bpt_l1 == 0.0)
        metrics.update(self.dynamic_affinity_diagnostics(real_df, synthetic_df))
        return metrics

    def metadata(self) -> Dict[str, Any]:
        if self.real_df is None:
            raise RuntimeError("Call fit before metadata.")
        affinity_summary = self.affinity_model.summary() if self.affinity_model is not None else {}
        metadata = {
            "method": METHOD_NAME,
            "alias": METHOD_ALIAS,
            "class_name": "TimeBiasedBlockStubMatchingGenerator",
            "generates_joint_events": True,
            "event_tuple": [self.customer_id_col, self.product_id_col, self.timestamp_col],
            "preserves_total_events_exactly": True,
            "preserves_customer_degrees_exactly": True,
            "preserves_product_degrees_exactly": True,
            "preserves_block_pair_time_counts_exactly": True,
            "preserves_daily_counts_exactly": True,
            "preserves_blocks_exactly": True,
            "uses_time_biased_stub_assignment": True,
            "uses_customer_time_activity": True,
            "uses_product_time_activity": True,
            "uses_time_dependent_pairing_score": self.pairing_mode != "random",
            "uses_dense_F_u_i_t": False,
            "dynamic_affinity_type": "time_gated_low_rank",
            "dynamic_affinity_formula": "(z_u * g_t)^T z_i",
            "no_degree_repair": True,
            "no_quota_rejection_sampling": True,
            "no_per_event_candidate_pool_scoring": True,
            "no_cell_level_repair_loops": True,
            "desired_time_sampling_mode": self.desired_time_sampling_mode,
            "kernel_type": self.kernel_type,
            "kernel_bandwidth_mode": self.kernel_bandwidth_mode,
            "kernel_bandwidth_scale": float(self.kernel_bandwidth_scale),
            "kernel_min_bandwidth_days": float(self.kernel_min_bandwidth_days),
            "kernel_max_bandwidth_days": self.kernel_max_bandwidth_days,
            "kernel_fixed_bandwidth_days": float(self.kernel_fixed_bandwidth_days),
            "temporal_alpha_used": self.desired_time_sampling_mode in {"mixture_shrinkage", "empirical_bayes"},
            "empirical_bayes_used": self.desired_time_sampling_mode == "empirical_bayes",
            "bandwidth_selection_uses_synthetic_metrics": False,
            "temporal_shrinkage_mode": self.temporal_shrinkage_mode,
            "alpha_customer_time_requested": self.alpha_customer_time,
            "alpha_product_time_requested": self.alpha_product_time,
            "alpha_customer_time_selected": float(self.alpha_customer_time_selected),
            "alpha_product_time_selected": float(self.alpha_product_time_selected),
            "alpha_selection_objective": self._alpha_selection_objective(),
            "alpha_selection_uses_synthetic_metrics": False,
            "alpha_candidate_grid": list(DEFAULT_CANDIDATE_ALPHAS),
            "customer_alpha_best": self._alpha_result_value(self.customer_alpha_result, "best_alpha"),
            "customer_alpha_num_holdout_events": int(self.customer_alpha_result.get("num_holdout_events", 0)),
            "customer_alpha_fallback_used": bool(self.customer_alpha_result.get("fallback_used", False)),
            "customer_alpha_candidate_results": list(self.customer_alpha_result.get("candidate_results", [])),
            "product_alpha_best": self._alpha_result_value(self.product_alpha_result, "best_alpha"),
            "product_alpha_num_holdout_events": int(self.product_alpha_result.get("num_holdout_events", 0)),
            "product_alpha_fallback_used": bool(self.product_alpha_result.get("fallback_used", False)),
            "product_alpha_candidate_results": list(self.product_alpha_result.get("candidate_results", [])),
            "customer_global_bandwidth": self._kernel_global_bandwidth(self.customer_kernel_bandwidth_result),
            "product_global_bandwidth": self._kernel_global_bandwidth(self.product_kernel_bandwidth_result),
            "customer_block_bandwidths": self._kernel_block_bandwidths(self.customer_kernel_bandwidth_result),
            "product_block_bandwidths": self._kernel_block_bandwidths(self.product_kernel_bandwidth_result),
            "customer_entity_bandwidth_summary": self._kernel_entity_summary(self.customer_kernel_bandwidth_result),
            "product_entity_bandwidth_summary": self._kernel_entity_summary(self.product_kernel_bandwidth_result),
            "pairing_mode": self.pairing_mode,
            "large_cell_pairing": self.large_cell_pairing,
            "max_exact_affinity_cell_size": int(self.max_exact_affinity_cell_size),
            "lambda_duplicate_pair": float(self.lambda_duplicate_pair),
            "lambda_real_pair_overlap": float(self.lambda_real_pair_overlap),
            "lambda_exact_event_overlap": float(self.lambda_exact_event_overlap),
            "pairing_penalties_fixed_defaults": self._uses_default_pairing_penalties(),
            "uses_penalized_dynamic_pairing": self.pairing_mode == "dynamic_exact_penalized",
            "desired_time_jitter": float(self.desired_time_jitter),
            "enable_fast_overlap_repair": bool(self.enable_fast_overlap_repair),
            "overlap_resample_prob": float(self.overlap_resample_prob),
            "time_granularity": self.time_granularity,
            "time_gate_granularity": self.time_gate_granularity,
            "rank": int(self.rank),
            "alpha_customer_time": float(self.customer_activity.alpha_resolved),
            "alpha_product_time": float(self.product_activity.alpha_resolved),
            "alpha_time_gate": float(affinity_summary.get("alpha_time_gate", 0.0)),
            "num_customers": int(len(self.customer_degrees)),
            "num_products": int(len(self.product_degrees)),
            "num_events": int(len(self.real_df)),
            "num_time_buckets": int(len(self.code_to_day)),
            "num_customer_blocks": int(len(set(self.customer_blocks.values()))),
            "num_product_blocks": int(len(set(self.product_blocks.values()))),
            "num_block_pair_time_cells": int(len(self.block_pair_time_counts)),
            "block_warnings": list(self.block_warnings),
            "lowrank_affinity": affinity_summary,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        metadata.update(self.slot_summary)
        metadata.update(self.runtime_metadata)
        return metadata

    def dynamic_affinity_diagnostics(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        real = real_df[[self.customer_id_col, self.product_id_col, self.timestamp_col]].copy()
        syn = synthetic_df[[self.customer_id_col, self.product_id_col, self.timestamp_col]].copy()
        real[self.timestamp_col] = canonical_time_bucket(real[self.timestamp_col], self.time_granularity)
        syn[self.timestamp_col] = canonical_time_bucket(syn[self.timestamp_col], self.time_granularity)
        real_scores = self._score_pairs_by_time(real)
        syn_scores = self._score_pairs_by_time(syn)
        return {
            "mean_dynamic_affinity_real": float(np.mean(real_scores)) if len(real_scores) else 0.0,
            "mean_dynamic_affinity_synthetic": float(np.mean(syn_scores)) if len(syn_scores) else 0.0,
            "median_dynamic_affinity_real": float(np.median(real_scores)) if len(real_scores) else 0.0,
            "median_dynamic_affinity_synthetic": float(np.median(syn_scores)) if len(syn_scores) else 0.0,
            "dynamic_affinity_distribution_ks": ks_stat(real_scores, syn_scores),
        }

    def _score_pairs_by_time(self, frame: pd.DataFrame) -> np.ndarray:
        if self.affinity_model is None or len(frame) == 0:
            return np.asarray([], dtype=float)
        scores = []
        for time_bucket, group in frame.groupby(self.timestamp_col, sort=False):
            scores.append(
                self.affinity_model.score_pairs(
                    group[self.customer_id_col].to_numpy(dtype=object),
                    group[self.product_id_col].to_numpy(dtype=object),
                    time_bucket,
                )
            )
        return np.concatenate(scores) if scores else np.asarray([], dtype=float)

    def _load_or_fallback_blocks(self, frame: pd.DataFrame) -> tuple[Dict[Any, int], Dict[Any, int], List[str]]:
        warnings = []
        customer_blocks = load_entity_blocks(self.structure_debug_dir, "customer_blocks.csv", self.customer_id_col, "customer_block")
        product_blocks = load_entity_blocks(self.structure_debug_dir, "product_blocks.csv", self.product_id_col, "product_block")
        if not customer_blocks:
            warnings.append("customer blocks missing; fell back to one customer block")
            customer_blocks = {entity: 0 for entity in frame[self.customer_id_col].unique()}
        if not product_blocks:
            warnings.append("product blocks missing; fell back to one product block")
            product_blocks = {entity: 0 for entity in frame[self.product_id_col].unique()}
        return customer_blocks, product_blocks, warnings

    def _select_temporal_shrinkage_alphas(self, frame: pd.DataFrame) -> None:
        """Choose customer/product alpha without looking at synthetic metrics."""

        if self.desired_time_sampling_mode in {"local_kernel", "empirical_exact"}:
            self.alpha_customer_time_selected = 0.0
            self.alpha_product_time_selected = 0.0
            objective = f"{self.desired_time_sampling_mode}_alpha_not_used"
            self.customer_alpha_result = self._manual_alpha_result(0.0, len(self.customer_degrees), objective, False)
            self.product_alpha_result = self._manual_alpha_result(0.0, len(self.product_degrees), objective, False)
            return

        if self.desired_time_sampling_mode == "empirical_bayes":
            print("[fit] selecting temporal shrinkage alphas by held-out likelihood", flush=True)
            self.customer_alpha_result = estimate_temporal_shrinkage_alpha(
                df=frame,
                entity_col=self.customer_id_col,
                time_col="_time_bucket",
                block_map=self.customer_blocks,
                seed=self.seed,
            )
            self.product_alpha_result = estimate_temporal_shrinkage_alpha(
                df=frame,
                entity_col=self.product_id_col,
                time_col="_time_bucket",
                block_map=self.product_blocks,
                seed=self.seed,
            )
            self.alpha_customer_time_selected = float(self.customer_alpha_result["best_alpha"])
            self.alpha_product_time_selected = float(self.product_alpha_result["best_alpha"])
            return

        if self.desired_time_sampling_mode == "mixture_shrinkage" and self.temporal_shrinkage_mode != "fixed":
            customer_alpha = resolve_auto_alpha("auto", self.customer_degrees.values())
            product_alpha = resolve_auto_alpha("auto", self.product_degrees.values())
            objective = "median_entity_degree"
            customer_fallback = False
            product_fallback = False
        else:
            customer_alpha = resolve_auto_alpha(self.alpha_customer_time, self.customer_degrees.values())
            product_alpha = resolve_auto_alpha(self.alpha_product_time, self.product_degrees.values())
            objective = "fixed_manual_alpha"
            customer_fallback = str(self.alpha_customer_time).lower() == "auto"
            product_fallback = str(self.alpha_product_time).lower() == "auto"

        self.alpha_customer_time_selected = float(customer_alpha)
        self.alpha_product_time_selected = float(product_alpha)
        self.customer_alpha_result = self._manual_alpha_result(customer_alpha, len(self.customer_degrees), objective, customer_fallback)
        self.product_alpha_result = self._manual_alpha_result(product_alpha, len(self.product_degrees), objective, product_fallback)

    def _fit_temporal_kernel_bandwidths(self) -> None:
        if self.customer_activity is None or self.product_activity is None:
            return
        if self.desired_time_sampling_mode not in {"local_kernel", "empirical_exact"}:
            self.customer_kernel_bandwidth_result = {}
            self.product_kernel_bandwidth_result = {}
            self.customer_local_kernel_state = {}
            self.product_local_kernel_state = {}
            return
        print("[fit] estimating temporal kernel bandwidths", flush=True)
        self.customer_kernel_bandwidth_result = self._estimate_kernel_bandwidth_for_activity(self.customer_activity)
        self.product_kernel_bandwidth_result = self._estimate_kernel_bandwidth_for_activity(self.product_activity)
        self.customer_local_kernel_state = self._local_kernel_state(self.customer_activity, self.customer_kernel_bandwidth_result)
        self.product_local_kernel_state = self._local_kernel_state(self.product_activity, self.product_kernel_bandwidth_result)

    def _estimate_kernel_bandwidth_for_activity(self, activity: FastTemporalActivityModel) -> Dict[str, Any]:
        state = activity.get_fast_sampling_state()
        return estimate_temporal_kernel_bandwidths(
            entity_offsets=state["empirical_offsets"],
            entity_time_values=state["empirical_time_values"],
            entity_blocks=state["entity_block"],
            num_blocks=max(len(activity.entities_by_block), int(max(activity.entities_by_block.keys())) + 1 if activity.entities_by_block else 1),
            default_bandwidth=self.kernel_fixed_bandwidth_days,
            bandwidth_mode=self.kernel_bandwidth_mode,
            bandwidth_scale=self.kernel_bandwidth_scale,
            min_bandwidth=self.kernel_min_bandwidth_days,
            max_bandwidth=self.kernel_max_bandwidth_days,
        )

    def _local_kernel_state(self, activity: FastTemporalActivityModel, bandwidth_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "num_time_codes": int(len(activity.time_buckets)),
            "bandwidth_mode": self.kernel_bandwidth_mode,
            "bandwidth_scale": float(self.kernel_bandwidth_scale),
            "min_bandwidth": float(self.kernel_min_bandwidth_days),
            "max_bandwidth": self.kernel_max_bandwidth_days,
            "kernel": self.kernel_type,
            "fallback_mode": "block",
            "entity_bandwidths": bandwidth_result.get("entity_bandwidths"),
            "block_bandwidths": bandwidth_result.get("block_bandwidths"),
            "global_bandwidth": float(bandwidth_result.get("global_bandwidth", self.kernel_fixed_bandwidth_days)),
        }

    def _manual_alpha_result(self, alpha: float, num_entities: int, objective: str, fallback_used: bool) -> Dict[str, Any]:
        return {
            "best_alpha": float(alpha),
            "candidate_results": [],
            "num_entities": int(num_entities),
            "num_holdout_events": 0,
            "fallback_used": bool(fallback_used),
            "avg_log_likelihood": None,
            "alpha_candidate_grid": list(DEFAULT_CANDIDATE_ALPHAS),
            "selection_objective": objective,
            "num_likelihood_evaluations": 0,
        }

    def _alpha_selection_objective(self) -> str:
        if self.desired_time_sampling_mode in {"local_kernel", "empirical_exact"}:
            return f"{self.desired_time_sampling_mode}_alpha_not_used"
        if self.desired_time_sampling_mode == "empirical_bayes":
            return "heldout_temporal_log_likelihood"
        if self.temporal_shrinkage_mode == "median_degree":
            return "median_entity_degree"
        return "fixed_manual_alpha"

    def _alpha_result_value(self, result: Dict[str, Any], key: str) -> Optional[float]:
        value = result.get(key)
        return None if value is None else float(value)

    def _uses_default_pairing_penalties(self) -> bool:
        return (
            float(self.lambda_duplicate_pair) == 1.0
            and float(self.lambda_real_pair_overlap) == 1.0
            and float(self.lambda_exact_event_overlap) == 3.0
        )

    def _kernel_diagnostics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return dict(result.get("diagnostics", {})) if result else {}

    def _kernel_global_bandwidth(self, result: Dict[str, Any]) -> Optional[float]:
        return None if not result else float(result.get("global_bandwidth", 0.0))

    def _kernel_block_bandwidths(self, result: Dict[str, Any]) -> Dict[str, float]:
        if not result:
            return {}
        diagnostics = result.get("diagnostics", {})
        values = diagnostics.get("block_bandwidths", {})
        return {str(key): float(value) for key, value in values.items()}

    def _kernel_entity_summary(self, result: Dict[str, Any]) -> Dict[str, float]:
        if not result:
            return {}
        diagnostics = result.get("diagnostics", {})
        summary = diagnostics.get("entity_bandwidth_summary", {})
        return {str(key): float(value) for key, value in summary.items()}

    def _build_real_overlap_key_sets(self, frame: pd.DataFrame) -> None:
        if self.customer_activity is None or self.product_activity is None:
            return
        customer_lookup = self.customer_activity.entity_to_index
        product_lookup = self.product_activity.entity_to_index
        customer_idx = np.fromiter(
            (customer_lookup[value] for value in frame[self.customer_id_col].to_numpy(dtype=object)),
            dtype=np.int64,
            count=len(frame),
        )
        product_idx = np.fromiter(
            (product_lookup[value] for value in frame[self.product_id_col].to_numpy(dtype=object)),
            dtype=np.int64,
            count=len(frame),
        )
        time_codes = frame["_time_code"].to_numpy(dtype=np.int64)
        num_products = int(len(self.product_activity.entity_ids))
        num_time_codes = int(len(self.code_to_day))
        pair_keys = customer_idx * num_products + product_idx
        event_keys = pair_keys * num_time_codes + time_codes
        self.real_pair_keys = set(int(key) for key in pair_keys)
        self.real_event_keys = set(int(key) for key in event_keys)

    def _build_block_pair_time_counts(self, frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame.groupby(["_customer_block", "_product_block", "_time_bucket", "_time_code", "_time_gate_code"])
            .size()
            .reset_index(name="count")
            .rename(
                columns={
                    "_customer_block": "customer_block",
                    "_product_block": "product_block",
                    "_time_bucket": "time_bucket",
                    "_time_code": "time_code",
                    "_time_gate_code": "time_gate_code",
                }
            )
            .sort_values(["time_code", "customer_block", "product_block"])
            .reset_index(drop=True)
        )

    def _build_slots(self) -> None:
        start = time.time()
        counts = self.block_pair_time_counts["count"].to_numpy(dtype=int)
        self.slot_id = np.arange(int(counts.sum()), dtype=int)
        self.slot_customer_block = np.repeat(self.block_pair_time_counts["customer_block"].to_numpy(dtype=int), counts)
        self.slot_product_block = np.repeat(self.block_pair_time_counts["product_block"].to_numpy(dtype=int), counts)
        self.slot_time_code = np.repeat(self.block_pair_time_counts["time_code"].to_numpy(dtype=int), counts)
        self.slot_time_gate_code = np.repeat(self.block_pair_time_counts["time_gate_code"].to_numpy(dtype=int), counts)
        self.slot_time_bucket = np.asarray([self.code_to_day[int(code)] for code in self.slot_time_code], dtype=object)
        self.slot_summary = {
            "num_slots": int(len(self.slot_id)),
            "num_block_pair_time_cells": int(len(self.block_pair_time_counts)),
            "average_cell_size": float(len(self.slot_id) / max(len(self.block_pair_time_counts), 1)),
            "max_cell_size": int(counts.max()) if len(counts) else 0,
            "num_time_buckets": int(len(self.code_to_day)),
            "num_time_gate_buckets": int(len(self.code_to_gate)),
            "slot_build_seconds": float(time.time() - start),
        }
        print(
            f"[slots] built {self.slot_summary['num_slots']:,} slots from "
            f"{self.slot_summary['num_block_pair_time_cells']:,} block-pair-time cells "
            f"in {self.slot_summary['slot_build_seconds']:.2f}s",
            flush=True,
        )

    def _verify_exact_constraints(self, synthetic: pd.DataFrame) -> None:
        real = self.real_df
        if len(synthetic) != len(real):
            raise RuntimeError(f"Row count mismatch: synthetic={len(synthetic)} real={len(real)}")
        customer_target = self._target_degree_array(self.customer_activity, self.customer_degrees)
        product_target = self._target_degree_array(self.product_activity, self.product_degrees)
        customer_counts = np.bincount(self.slot_customer_idx, minlength=len(customer_target))
        if not np.array_equal(customer_counts, customer_target):
            raise RuntimeError("Customer degree sequence mismatch")
        product_lookup = self.product_activity.entity_to_index
        product_idx_after_pairing = np.fromiter(
            (product_lookup[product] for product in self.slot_product_id),
            dtype=np.int64,
            count=len(self.slot_product_id),
        )
        product_counts = np.bincount(product_idx_after_pairing, minlength=len(product_target))
        if not np.array_equal(product_counts, product_target):
            raise RuntimeError("Product degree sequence mismatch")
        expected_time_counts = np.bincount(real["_time_code"].to_numpy(dtype=np.int64), minlength=len(self.code_to_day))
        synthetic_time_counts = np.bincount(self.slot_time_code.astype(np.int64), minlength=len(self.code_to_day))
        if not np.array_equal(expected_time_counts, synthetic_time_counts):
            raise RuntimeError("Daily count sequence mismatch")
        customer_blocks_after = self.customer_activity.get_fast_sampling_state()["entity_block"][self.slot_customer_idx]
        if not np.array_equal(customer_blocks_after.astype(np.int64), self.slot_customer_block.astype(np.int64)):
            raise RuntimeError("Customer block assignment mismatch")
        product_blocks_after = self.product_activity.get_fast_sampling_state()["entity_block"][product_idx_after_pairing]
        if not np.array_equal(product_blocks_after.astype(np.int64), self.slot_product_block.astype(np.int64)):
            raise RuntimeError("Product block assignment mismatch")
        expected_codes = self._cell_codes(
            self.block_pair_time_counts["customer_block"].to_numpy(dtype=np.int64),
            self.block_pair_time_counts["product_block"].to_numpy(dtype=np.int64),
            self.block_pair_time_counts["time_code"].to_numpy(dtype=np.int64),
        )
        generated_codes = self._cell_codes(
            self.slot_customer_block.astype(np.int64),
            self.slot_product_block.astype(np.int64),
            self.slot_time_code.astype(np.int64),
        )
        expected_counts = np.zeros(max(int(expected_codes.max()) if len(expected_codes) else 0, int(generated_codes.max()) if len(generated_codes) else 0) + 1, dtype=np.int64)
        np.add.at(expected_counts, expected_codes, self.block_pair_time_counts["count"].to_numpy(dtype=np.int64))
        generated_counts = np.bincount(generated_codes, minlength=len(expected_counts))
        if len(generated_counts) > len(expected_counts):
            expected_counts = np.pad(expected_counts, (0, len(generated_counts) - len(expected_counts)))
        if not np.array_equal(expected_counts, generated_counts):
            raise RuntimeError("Block-pair-time counts mismatch")

    def _target_degree_array(self, activity_model: FastTemporalActivityModel, degrees: Dict[Any, int]) -> np.ndarray:
        return np.asarray([int(degrees.get(entity, 0)) for entity in activity_model.entity_ids], dtype=np.int64)

    def _cell_codes(self, customer_block: np.ndarray, product_block: np.ndarray, time_code: np.ndarray) -> np.ndarray:
        max_product = max(
            int(self.block_pair_time_counts["product_block"].max()) if len(self.block_pair_time_counts) else 0,
            int(self.slot_product_block.max()) if len(self.slot_product_block) else 0,
        ) + 1
        max_time = max(
            int(self.block_pair_time_counts["time_code"].max()) if len(self.block_pair_time_counts) else 0,
            int(self.slot_time_code.max()) if len(self.slot_time_code) else 0,
        ) + 1
        return (customer_block.astype(np.int64) * max_product + product_block.astype(np.int64)) * max_time + time_code.astype(np.int64)

    def array_dtype_summary(self) -> Dict[str, Any]:
        arrays = {
            "slot_id": self.slot_id,
            "slot_customer_block": self.slot_customer_block,
            "slot_product_block": self.slot_product_block,
            "slot_time_code": self.slot_time_code,
            "slot_time_gate_code": self.slot_time_gate_code,
            "slot_customer_idx": self.slot_customer_idx,
            "slot_product_idx": self.slot_product_idx,
            "slot_customer_id": self.slot_customer_id,
            "slot_product_id": self.slot_product_id,
        }
        summary = {}
        for name, array in arrays.items():
            arr = np.asarray(array)
            summary[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
        if self.customer_activity is not None:
            customer_state = self.customer_activity.get_fast_sampling_state()
            summary["customer_empirical_time_values"] = {
                "dtype": str(customer_state["empirical_time_values"].dtype),
                "shape": list(customer_state["empirical_time_values"].shape),
            }
        if self.product_activity is not None:
            product_state = self.product_activity.get_fast_sampling_state()
            summary["product_empirical_time_values"] = {
                "dtype": str(product_state["empirical_time_values"].dtype),
                "shape": list(product_state["empirical_time_values"].shape),
            }
        return summary

    def _bpt_counts(self, frame: pd.DataFrame) -> pd.Series:
        tmp = frame.copy()
        time_col = "_time_bucket" if "_time_bucket" in tmp.columns else self.timestamp_col
        tmp["_customer_block_eval"] = tmp[self.customer_id_col].map(self.customer_blocks).fillna(-1).astype(int)
        tmp["_product_block_eval"] = tmp[self.product_id_col].map(self.product_blocks).fillna(-1).astype(int)
        tmp["_time_eval"] = canonical_time_bucket(tmp[time_col], self.time_granularity)
        return tmp.groupby(["_customer_block_eval", "_product_block_eval", "_time_eval"]).size().sort_index()


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
