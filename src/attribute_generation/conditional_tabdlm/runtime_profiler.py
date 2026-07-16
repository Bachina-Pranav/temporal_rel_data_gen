"""Runtime profiling helpers for attribute sampling."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


PROFILE_TIMING_FIELDS = (
    "loading_checkpoint_seconds",
    "loading_synthetic_spine_seconds",
    "loading_model_seconds",
    "loading_spine_seconds",
    "graph_history_build_seconds",
    "graph_context_total_seconds",
    "graph_context_cache_build_seconds",
    "graph_context_lookup_seconds",
    "condition_encoding_seconds",
    "initial_noise_seconds",
    "denoising_loop_seconds",
    "denoising_step_seconds",
    "final_forward_seconds",
    "length_enforcement_seconds",
    "categorical_decoding_seconds",
    "text_decoding_seconds",
    "debug_example_seconds",
    "postprocessing_seconds",
    "row_latent_seconds",
    "categorical_sampling_seconds",
    "rating_sampling_seconds",
    "verified_sampling_seconds",
    "summary_length_sampling_seconds",
    "review_text_length_sampling_seconds",
    "summary_decoding_seconds",
    "review_text_decoding_seconds",
    "detokenization_seconds",
    "csv_writing_seconds",
)


class RuntimeProfiler:
    """Small wall-clock profiler with JSON summary support."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.timings = {field: 0.0 for field in PROFILE_TIMING_FIELDS}
        self.events: list[dict[str, Any]] = []
        self._total_start: float | None = None
        self._total_seconds = 0.0

    def start_total(self) -> None:
        self._total_start = time.perf_counter()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass

    def stop_total(self) -> float:
        if self._total_start is not None:
            self._total_seconds = float(time.perf_counter() - self._total_start)
            self._total_start = None
        return self._total_seconds

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = float(time.perf_counter() - start)
            self.add_time(name, elapsed)

    def add_time(self, name: str, elapsed: float) -> None:
        elapsed = float(elapsed)
        self.timings[name] = self.timings.get(name, 0.0) + elapsed
        if self.enabled:
            self.events.append({"name": name, "seconds": elapsed})

    def summary(
        self,
        *,
        rows_generated: int,
        num_batches: int,
        batch_size_requested: int,
        batch_size_used: int,
        auto_batch_size_enabled: bool,
        summary_lengths: list[int],
        review_text_lengths: list[int],
        device: str,
        mixed_precision_used: bool,
        dtype_used: str,
        torch_compile_used: bool,
        graph_context_cache_mode: str = "none",
        graph_context_cache_hit_rate: float = 0.0,
        graph_context_cache_memory_mb: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        total_seconds = self._total_seconds or self.stop_total()
        rows_per_second = float(rows_generated / max(total_seconds, 1e-9))
        projected_seconds = float(10_000_000 / max(rows_per_second, 1e-9))
        nested_timing_fields = {
            "graph_context_lookup_seconds",
            "rating_sampling_seconds",
            "verified_sampling_seconds",
            "summary_length_sampling_seconds",
            "review_text_length_sampling_seconds",
        }
        component_sum = float(
            sum(value for key, value in self.timings.items() if key not in nested_timing_fields)
        )
        cuda_peak_allocated = None
        cuda_peak_reserved = None
        gpu_name = None
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            try:
                gpu_name = torch.cuda.get_device_name(torch.device(device))
            except Exception:
                gpu_name = torch.cuda.get_device_name(0)
            try:
                cuda_peak_allocated = float(torch.cuda.max_memory_allocated() / (1024**2))
                cuda_peak_reserved = float(torch.cuda.max_memory_reserved() / (1024**2))
            except RuntimeError:
                cuda_peak_allocated = None
                cuda_peak_reserved = None
        summary = {
            "total_sampling_seconds": float(total_seconds),
            **{field: float(self.timings.get(field, 0.0)) for field in PROFILE_TIMING_FIELDS},
            "misc_overhead_seconds": float(max(total_seconds - component_sum, 0.0)),
            "rows_generated": int(rows_generated),
            "num_batches": int(num_batches),
            "batch_size_requested": int(batch_size_requested),
            "batch_size_used": int(batch_size_used),
            "auto_batch_size_enabled": bool(auto_batch_size_enabled),
            "total_summary_tokens_generated": int(sum(summary_lengths)),
            "total_review_text_tokens_generated": int(sum(review_text_lengths)),
            "avg_summary_tokens_generated": _mean(summary_lengths),
            "avg_review_text_tokens_generated": _mean(review_text_lengths),
            "p50_review_text_tokens_generated": _quantile(review_text_lengths, 0.50),
            "p90_review_text_tokens_generated": _quantile(review_text_lengths, 0.90),
            "p95_review_text_tokens_generated": _quantile(review_text_lengths, 0.95),
            "p99_review_text_tokens_generated": _quantile(review_text_lengths, 0.99),
            "max_review_text_tokens_generated": int(max(review_text_lengths)) if review_text_lengths else None,
            "rows_per_second": rows_per_second,
            "seconds_per_1000_rows": float(1000.0 / max(rows_per_second, 1e-9)),
            "projected_seconds_for_10m_rows": projected_seconds,
            "projected_hours_for_10m_rows": float(projected_seconds / 3600.0),
            "device": str(device),
            "gpu_name": gpu_name,
            "mixed_precision_used": bool(mixed_precision_used),
            "dtype_used": str(dtype_used),
            "torch_compile_used": bool(torch_compile_used),
            "cuda_memory_peak_allocated_mb": cuda_peak_allocated,
            "cuda_memory_peak_reserved_mb": cuda_peak_reserved,
            "graph_context_cache_mode": str(graph_context_cache_mode),
            "graph_context_cache_hit_rate": float(graph_context_cache_hit_rate),
            "graph_context_cache_memory_mb": float(graph_context_cache_memory_mb),
        }
        if extra:
            summary.update(extra)
        return summary

    def write_summary(self, path: str | Path, summary: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(summary), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def write_detailed(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"events": self.events, "timings": self.timings}, handle, indent=2, sort_keys=True)
            handle.write("\n")


def _mean(values: list[int]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantile(values: list[int], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
