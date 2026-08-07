"""Schema-driven numerical heads, support metadata, and smoothed priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .numerical import transform_numerical_value
from .numerical_type import infer_numerical_types
from .tokenization import stable_hash_bucket


NUMERICAL_HEAD_MODES = (
    "continuous_baseline",
    "discrete_support",
    "hierarchical_support",
    "auto",
)


def numerical_head_feature_enabled(raw: dict[str, Any]) -> bool:
    cfg = raw.get("numerical_heads") or {}
    return bool(
        "mode" in cfg
        or "columns" in cfg
        or "conditioning" in cfg
        or "prior" in cfg
        or "type_inference" in cfg
    )


def resolve_event_role_indices(
    raw: dict[str, Any],
    foreign_key_columns: tuple[str, ...],
    datetime_columns: tuple[str, ...],
) -> dict[str, int]:
    source = resolve_event_role(raw, "source_fk", "source_foreign_key")
    destination = resolve_event_role(
        raw,
        "destination_fk",
        "destination_foreign_key",
    )
    timestamp = resolve_event_role(raw, "timestamp", "timestamp")
    if source not in foreign_key_columns:
        raise ValueError(
            f"Resolved source foreign key {source!r} is not in "
            f"{list(foreign_key_columns)}"
        )
    if destination not in foreign_key_columns:
        raise ValueError(
            f"Resolved destination foreign key {destination!r} is not in "
            f"{list(foreign_key_columns)}"
        )
    if timestamp not in datetime_columns:
        raise ValueError(
            f"Resolved timestamp {timestamp!r} is not in "
            f"{list(datetime_columns)}"
        )
    return {
        "source_fk_index": int(foreign_key_columns.index(source)),
        "destination_fk_index": int(
            foreign_key_columns.index(destination)
        ),
        "timestamp_index": int(datetime_columns.index(timestamp)),
        "source_fk": source,
        "destination_fk": destination,
        "timestamp": timestamp,
    }


def resolve_event_role(
    raw: dict[str, Any],
    event_key: str,
    schema_role: str,
) -> str:
    explicit = (raw.get("event_spine") or {}).get(event_key)
    if explicit:
        return str(explicit)
    fields = (raw.get("schema") or {}).get("fields") or {}
    matches = [
        str(column)
        for column, metadata in fields.items()
        if str((metadata or {}).get("role", "")) == schema_role
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not resolve {event_key!r} from schema role "
            f"{schema_role!r}; matches={matches}"
        )
    return matches[0]


def fit_numerical_head_metadata(
    config: Any,
    *,
    train_frame: pd.DataFrame | None = None,
    train_dataset: Any | None = None,
    numerical_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Fit all numerical-head state from the training split only."""

    if not config.schema.numerical_targets:
        return {}
    roles = resolve_event_role_indices(
        config.raw,
        config.schema.foreign_key_columns,
        config.schema.datetime_columns,
    )
    standardized, foreign_keys, datetimes = training_arrays(
        config,
        train_frame=train_frame,
        train_dataset=train_dataset,
        numerical_metadata=numerical_metadata,
    )
    if train_frame is not None:
        original = pd.DataFrame(
            {
                column: pd.to_numeric(
                    train_frame[column],
                    errors="raise",
                ).reset_index(drop=True)
                for column in config.schema.numerical_targets
            }
        )
    else:
        original = pd.DataFrame(
            {
                column: inverse_numpy(
                    standardized[:, index],
                    numerical_metadata[column],
                )
                for index, column in enumerate(
                    config.schema.numerical_targets
                )
            }
        )
    head_cfg = dict(config.raw.get("numerical_heads") or {})
    seed = int((config.raw.get("training") or {}).get("seed", 42))
    inferred = infer_numerical_types(
        original,
        config.schema.numerical_targets,
        config=head_cfg.get("type_inference"),
        seed=seed,
    )
    columns: dict[str, Any] = {}
    for index, column in enumerate(config.schema.numerical_targets):
        column_cfg = dict(
            (head_cfg.get("columns") or {}).get(column) or {}
        )
        requested = str(
            column_cfg.get("mode", head_cfg.get("mode", "continuous_baseline"))
        ).lower()
        if requested not in NUMERICAL_HEAD_MODES:
            raise ValueError(
                f"Unsupported numerical head mode for {column!r}: "
                f"{requested!r}"
            )
        resolved = (
            str(inferred[column]["recommended_head"])
            if requested == "auto"
            else requested
        )
        report: dict[str, Any] = {
            "column": column,
            "requested_mode": requested,
            "resolved_mode": resolved,
            "inferred_type": inferred[column],
            "training_only": True,
            "original_dtype": str(original[column].dtype),
            "support_output_dtype": support_output_dtype(
                original[column],
                numerical_metadata[column],
            ),
        }
        if resolved in {
            "discrete_support",
            "hierarchical_support",
        }:
            support, counts = np.unique(
                standardized[:, index],
                return_counts=True,
            )
            original_support = original_support_for_standardized(
                standardized[:, index],
                original[column].to_numpy(),
                support,
                numerical_metadata[column],
            )
            report.update(
                support_metadata(
                    standardized[:, index],
                    support,
                    original_support,
                    counts,
                    head_cfg,
                    column_cfg,
                    foreign_keys[:, roles["destination_fk_index"]],
                    datetimes[:, roles["timestamp_index"]],
                    num_hash_buckets=int(
                        (config.raw.get("id_encoding") or {}).get(
                            "num_buckets",
                            262144,
                        )
                    ),
                )
            )
        columns[column] = report
    return {
        "version": 1,
        "training_only": True,
        "event_roles": roles,
        "conditioning": dict(head_cfg.get("conditioning") or {}),
        "prior": dict(head_cfg.get("prior") or {}),
        "objectives": dict(head_cfg.get("objectives") or {}),
        "columns": columns,
    }


def training_arrays(
    config: Any,
    *,
    train_frame: pd.DataFrame | None,
    train_dataset: Any | None,
    numerical_metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_dataset is not None:
        indices = np.asarray(train_dataset.indices, dtype=np.int64)
        numerical = np.asarray(
            train_dataset.numerical_values[indices],
            dtype=np.float32,
        )
        foreign_keys = np.asarray(
            train_dataset.foreign_key_ids[indices],
            dtype=np.int64,
        )
        datetimes = np.asarray(
            train_dataset.datetime_values[indices],
            dtype=np.float32,
        )
        return numerical, foreign_keys, datetimes
    if train_frame is None:
        raise ValueError(
            "Numerical-head metadata needs a training frame or "
            "pretokenized training dataset"
        )
    numerical = np.column_stack(
        [
            [
                transform_numerical_value(
                    value,
                    numerical_metadata[column],
                )
                for value in train_frame[column]
            ]
            for column in config.schema.numerical_targets
        ]
    ).astype(np.float32)
    num_hash_buckets = int(
        (config.raw.get("id_encoding") or {}).get(
            "num_buckets",
            262144,
        )
    )
    foreign_keys = np.column_stack(
        [
            [
                stable_hash_bucket(column, value, num_hash_buckets)
                for value in train_frame[column]
            ]
            for column in config.schema.foreign_key_columns
        ]
    ).astype(np.int64)
    datetimes = np.column_stack(
        [
            pd.to_datetime(
                train_frame[column],
                errors="coerce",
                utc=True,
            ).array.asi8.astype(np.float64)
            / 1e9
            for column in config.schema.datetime_columns
        ]
    ).astype(np.float32)
    return numerical, foreign_keys, datetimes


def support_metadata(
    training_values_standardized: np.ndarray,
    standardized_support: np.ndarray,
    original_support: np.ndarray,
    counts: np.ndarray,
    head_cfg: dict[str, Any],
    column_cfg: dict[str, Any],
    destination_hashes: np.ndarray,
    timestamps: np.ndarray,
    num_hash_buckets: int,
) -> dict[str, Any]:
    support_limit = int(
        column_cfg.get(
            "direct_support_max_values",
            head_cfg.get("direct_support_max_values", 8192),
        )
    )
    mode = str(
        column_cfg.get("mode", head_cfg.get("mode", "auto"))
    )
    if mode == "discrete_support" and len(standardized_support) > support_limit:
        raise ValueError(
            "discrete_support exceeds direct_support_max_values: "
            f"{len(standardized_support)} > {support_limit}"
        )
    num_bins = min(
        max(
            int(
                column_cfg.get(
                    "hierarchical_num_bins",
                    head_cfg.get("hierarchical_num_bins", 64),
                )
            ),
            2,
        ),
        len(standardized_support),
    )
    bin_ids, bin_offsets = equal_mass_support_bins(
        counts,
        num_bins=num_bins,
    )
    imbalance = str(
        column_cfg.get(
            "class_frequency_weighting",
            head_cfg.get("class_frequency_weighting", "inverse_sqrt"),
        )
    )
    weights = support_class_weights(counts, imbalance)
    prior_cfg = dict(head_cfg.get("prior") or {})
    prior = (
        fit_support_priors(
            standardized_support,
            counts,
            training_values_standardized,
            destination_hashes,
            timestamps,
            {
                **prior_cfg,
                "num_hash_buckets": int(num_hash_buckets),
            },
        )
        if bool(prior_cfg.get("enabled", False))
        else None
    )
    return {
        "support_values_standardized": standardized_support.tolist(),
        "support_values_original": original_support.tolist(),
        "support_counts": counts.astype(np.int64).tolist(),
        "support_size": int(len(standardized_support)),
        "class_frequency_weighting": imbalance,
        "class_weights": weights.tolist(),
        "label_smoothing": float(
            column_cfg.get(
                "label_smoothing",
                head_cfg.get("label_smoothing", 0.0),
            )
        ),
        "ordinal_regularization_weight": float(
            column_cfg.get(
                "ordinal_regularization_weight",
                head_cfg.get("ordinal_regularization_weight", 0.0),
            )
        ),
        "global_calibration_weight": float(
            column_cfg.get(
                "global_calibration_weight",
                head_cfg.get("global_calibration_weight", 0.0),
            )
        ),
        "hierarchical_bin_ids": bin_ids.tolist(),
        "hierarchical_bin_offsets": bin_offsets.tolist(),
        "hierarchical_num_bins": int(len(bin_offsets) - 1),
        "prior": prior,
    }


def equal_mass_support_bins(
    counts: np.ndarray,
    *,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.int64)
    if not len(counts):
        return np.asarray([], dtype=np.int64), np.asarray([0], dtype=np.int64)
    requested = min(max(int(num_bins), 1), len(counts))
    cumulative = np.cumsum(counts)
    cuts = [
        int(np.searchsorted(cumulative, cumulative[-1] * i / requested))
        for i in range(1, requested)
    ]
    offsets = np.unique(
        np.asarray([0, *[cut + 1 for cut in cuts], len(counts)])
    )
    offsets[0] = 0
    offsets[-1] = len(counts)
    bin_ids = np.empty(len(counts), dtype=np.int64)
    for bin_id, (start, end) in enumerate(
        zip(offsets[:-1], offsets[1:])
    ):
        bin_ids[start:end] = bin_id
    return bin_ids, offsets


def support_class_weights(
    counts: np.ndarray,
    mode: str,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    mode = str(mode).lower()
    if mode in {"none", "off", "false"}:
        weights = np.ones_like(counts)
    elif mode == "inverse":
        weights = 1.0 / np.maximum(counts, 1.0)
    elif mode == "effective_number":
        beta = 0.999
        weights = (1.0 - beta) / np.maximum(
            1.0 - np.power(beta, counts),
            1e-12,
        )
    elif mode == "inverse_sqrt":
        weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    else:
        raise ValueError(
            f"Unknown class_frequency_weighting mode: {mode!r}"
        )
    weights /= max(float(np.mean(weights)), 1e-12)
    return np.clip(weights, 0.05, 20.0).astype(np.float32)


def fit_support_priors(
    support: np.ndarray,
    global_counts: np.ndarray,
    training_values: np.ndarray,
    destination_hashes: np.ndarray,
    timestamps: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    support = np.asarray(support, dtype=float)
    support_ids = nearest_support_indices_numpy(
        np.asarray(training_values, dtype=float),
        support,
    )
    if len(support_ids) != len(destination_hashes):
        raise ValueError(
            "Prior fitting rows are not aligned with destination hashes"
        )
    destination_hashes = np.asarray(destination_hashes, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=float)
    destination_counts = pd.Series(destination_hashes).value_counts()
    frequency_mapping = fit_hash_frequency_buckets(
        destination_counts,
        int(config.get("num_frequency_buckets", 4)),
    )
    num_hash_buckets = int(config.get("num_hash_buckets", 262144))
    hash_to_frequency = np.full(
        num_hash_buckets,
        -1,
        dtype=np.int16,
    )
    frequency_labels = sorted(set(frequency_mapping.values()))
    frequency_to_id = {
        label: index
        for index, label in enumerate(frequency_labels)
    }
    for destination, label in frequency_mapping.items():
        hash_to_frequency[int(destination)] = frequency_to_id[label]
    frequency_ids = np.asarray(
        [
            hash_to_frequency[value]
            if 0 <= value < len(hash_to_frequency)
            else -1
            for value in destination_hashes
        ],
        dtype=np.int64,
    )
    time_boundaries = fit_timestamp_boundaries(
        timestamps,
        int(config.get("num_time_buckets", 8)),
    )
    time_ids = np.searchsorted(
        time_boundaries,
        timestamps,
        side="right",
    ).astype(np.int64)
    frequency_counts = grouped_support_counts(
        frequency_ids,
        support_ids,
        len(frequency_labels),
        len(support),
    )
    time_counts = grouped_support_counts(
        time_ids,
        support_ids,
        len(time_boundaries) + 1,
        len(support),
    )
    minimum_exact = int(config.get("min_destination_rows", 20))
    eligible = destination_counts[
        destination_counts >= minimum_exact
    ].index.to_numpy(dtype=np.int64)
    max_exact_values = int(config.get("max_exact_support_values", 64))
    exact_hash_to_row = np.full(
        num_hash_buckets,
        -1,
        dtype=np.int32,
    )
    exact_support_ids = np.full(
        (len(eligible), max_exact_values),
        -1,
        dtype=np.int32,
    )
    exact_support_counts = np.zeros(
        (len(eligible), max_exact_values),
        dtype=np.float32,
    )
    exact_totals = np.zeros(len(eligible), dtype=np.float32)
    for row, destination in enumerate(eligible):
        exact_hash_to_row[int(destination)] = row
        selected = support_ids[destination_hashes == destination]
        values, value_counts = np.unique(
            selected,
            return_counts=True,
        )
        order = np.argsort(-value_counts, kind="mergesort")[
            :max_exact_values
        ]
        width = len(order)
        exact_support_ids[row, :width] = values[order]
        exact_support_counts[row, :width] = value_counts[order]
        exact_totals[row] = float(len(selected))
    return {
        "enabled": True,
        "lambda_prior": float(config.get("lambda_prior", 1.0)),
        "epsilon": float(config.get("epsilon", 1e-8)),
        "time_smoothing": float(config.get("time_smoothing", 100.0)),
        "frequency_smoothing": float(
            config.get("frequency_smoothing", 100.0)
        ),
        "destination_smoothing": float(
            config.get("destination_smoothing", 20.0)
        ),
        "global_counts": np.asarray(
            global_counts,
            dtype=np.float32,
        ).tolist(),
        "time_boundaries_seconds": time_boundaries.tolist(),
        "time_counts": time_counts.tolist(),
        "frequency_counts": frequency_counts.tolist(),
        "hash_to_frequency": hash_to_frequency.tolist(),
        "exact_hash_to_row": exact_hash_to_row.tolist(),
        "exact_support_ids": exact_support_ids.tolist(),
        "exact_support_counts": exact_support_counts.tolist(),
        "exact_totals": exact_totals.tolist(),
        "minimum_exact_destination_rows": minimum_exact,
        "exact_prior_truncated_to_top_values": max_exact_values,
        "destination_representation": "stable_hash_bucket",
    }


class SmoothedSupportPrior(nn.Module):
    """Vectorized global/time/frequency/exact-destination support prior."""

    def __init__(self, metadata: dict[str, Any]):
        super().__init__()
        self.enabled = bool(metadata and metadata.get("enabled", False))
        self.lambda_prior = float(metadata.get("lambda_prior", 1.0))
        self.epsilon = float(metadata.get("epsilon", 1e-8))
        self.time_smoothing = float(metadata.get("time_smoothing", 100.0))
        self.frequency_smoothing = float(
            metadata.get("frequency_smoothing", 100.0)
        )
        self.destination_smoothing = float(
            metadata.get("destination_smoothing", 20.0)
        )
        self.register_buffer(
            "global_counts",
            tensor(metadata.get("global_counts", [1.0]), torch.float32),
        )
        self.register_buffer(
            "time_boundaries",
            tensor(
                metadata.get("time_boundaries_seconds", []),
                torch.float32,
            ),
        )
        self.register_buffer(
            "time_counts",
            tensor(metadata.get("time_counts", []), torch.float32),
        )
        self.register_buffer(
            "frequency_counts",
            tensor(
                metadata.get("frequency_counts", []),
                torch.float32,
            ),
        )
        self.register_buffer(
            "hash_to_frequency",
            tensor(
                metadata.get("hash_to_frequency", []),
                torch.long,
            ),
        )
        self.register_buffer(
            "exact_hash_to_row",
            tensor(
                metadata.get("exact_hash_to_row", []),
                torch.long,
            ),
        )
        self.register_buffer(
            "exact_support_ids",
            tensor(
                metadata.get("exact_support_ids", []),
                torch.long,
            ),
        )
        self.register_buffer(
            "exact_support_counts",
            tensor(
                metadata.get("exact_support_counts", []),
                torch.float32,
            ),
        )
        self.register_buffer(
            "exact_totals",
            tensor(metadata.get("exact_totals", []), torch.float32),
        )

    def forward(
        self,
        destination_ids: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        global_probability = (
            self.global_counts + self.epsilon
        ) / (
            self.global_counts.sum()
            + self.epsilon * self.global_counts.numel()
        )
        probability = global_probability.unsqueeze(0).expand(
            len(destination_ids),
            -1,
        )
        if self.time_counts.numel():
            time_id = torch.bucketize(
                timestamps.float(),
                self.time_boundaries,
            )
            probability = blend_dense_counts(
                probability,
                self.time_counts[time_id],
                self.time_smoothing,
                self.epsilon,
            )
        if self.frequency_counts.numel() and self.hash_to_frequency.numel():
            safe_destination = destination_ids.clamp(
                0,
                self.hash_to_frequency.numel() - 1,
            )
            frequency_id = self.hash_to_frequency[safe_destination]
            valid = frequency_id >= 0
            if valid.any():
                probability = probability.clone()
                probability[valid] = blend_dense_counts(
                    probability[valid],
                    self.frequency_counts[frequency_id[valid]],
                    self.frequency_smoothing,
                    self.epsilon,
                )
        if self.exact_hash_to_row.numel() and self.exact_totals.numel():
            safe_destination = destination_ids.clamp(
                0,
                self.exact_hash_to_row.numel() - 1,
            )
            exact_row = self.exact_hash_to_row[safe_destination]
            valid = exact_row >= 0
            if valid.any():
                probability = probability.clone()
                rows = exact_row[valid]
                totals = self.exact_totals[rows]
                denominator = (
                    totals + self.destination_smoothing
                ).unsqueeze(1)
                updated = (
                    probability[valid]
                    * self.destination_smoothing
                    / denominator
                )
                support_ids = self.exact_support_ids[rows]
                support_counts = self.exact_support_counts[rows]
                present = support_ids >= 0
                batch_rows = (
                    torch.arange(
                        len(rows),
                        device=rows.device,
                    )
                    .unsqueeze(1)
                    .expand_as(support_ids)
                )
                updated.index_put_(
                    (
                        batch_rows[present],
                        support_ids[present],
                    ),
                    support_counts[present]
                    / denominator.expand_as(support_counts)[present],
                    accumulate=True,
                )
                probability[valid] = updated
        return torch.log(
            probability.clamp_min(self.epsilon)
        )


class DiscreteSupportNumericalHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        metadata: dict[str, Any],
    ):
        super().__init__()
        self.linear = nn.Linear(
            int(hidden_dim),
            int(metadata["support_size"]),
        )
        self.label_smoothing = float(
            metadata.get("label_smoothing", 0.0)
        )
        self.ordinal_weight = float(
            metadata.get("ordinal_regularization_weight", 0.0)
        )
        self.global_calibration_weight = float(
            metadata.get("global_calibration_weight", 0.0)
        )
        self.register_buffer(
            "support_standardized",
            tensor(
                metadata["support_values_standardized"],
                torch.float32,
            ),
        )
        self.register_buffer(
            "support_original",
            tensor(
                metadata["support_values_original"],
                support_torch_dtype(metadata),
            ),
        )
        self.register_buffer(
            "class_weights",
            tensor(metadata["class_weights"], torch.float32),
        )
        counts = tensor(metadata["support_counts"], torch.float32)
        self.register_buffer(
            "global_probability",
            counts / counts.sum().clamp_min(1.0),
        )
        self.prior = SmoothedSupportPrior(metadata.get("prior") or {})

    def forward(
        self,
        hidden: torch.Tensor,
        destination_ids: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> dict[str, Any]:
        neural_logits = self.linear(hidden)
        prior_logits = None
        logits = neural_logits
        if self.prior.enabled:
            prior_logits = self.prior(destination_ids, timestamps)
            logits = (
                neural_logits
                + self.prior.lambda_prior * prior_logits
            )
        return {
            "mode": "discrete_support",
            "logits": logits,
            "neural_logits": neural_logits,
            "prior_logits": prior_logits,
            "support_standardized": self.support_standardized,
            "support_original": self.support_original,
            "class_weights": self.class_weights,
            "global_probability": self.global_probability,
            "label_smoothing": self.label_smoothing,
            "ordinal_weight": self.ordinal_weight,
            "global_calibration_weight": (
                self.global_calibration_weight
            ),
        }

    def sample(
        self,
        output: dict[str, Any],
        *,
        temperature: float,
    ) -> torch.Tensor:
        logits = output["logits"] / max(float(temperature), 1e-6)
        ids = torch.multinomial(
            torch.softmax(logits.float(), dim=-1),
            num_samples=1,
        ).squeeze(1)
        return self.support_original[ids]


class HierarchicalSupportNumericalHead(nn.Module):
    """Coarse-bin then within-bin support decoder."""

    def __init__(
        self,
        hidden_dim: int,
        metadata: dict[str, Any],
    ):
        super().__init__()
        offsets = np.asarray(
            metadata["hierarchical_bin_offsets"],
            dtype=np.int64,
        )
        self.coarse = nn.Linear(int(hidden_dim), len(offsets) - 1)
        self.fine = nn.ModuleList(
            [
                nn.Linear(int(hidden_dim), int(end - start))
                for start, end in zip(offsets[:-1], offsets[1:])
            ]
        )
        self.label_smoothing = float(
            metadata.get("label_smoothing", 0.0)
        )
        self.ordinal_weight = float(
            metadata.get("ordinal_regularization_weight", 0.0)
        )
        self.register_buffer(
            "offsets",
            tensor(offsets, torch.long),
        )
        self.register_buffer(
            "bin_ids",
            tensor(metadata["hierarchical_bin_ids"], torch.long),
        )
        self.register_buffer(
            "support_standardized",
            tensor(
                metadata["support_values_standardized"],
                torch.float32,
            ),
        )
        self.register_buffer(
            "support_original",
            tensor(
                metadata["support_values_original"],
                support_torch_dtype(metadata),
            ),
        )
        self.register_buffer(
            "class_weights",
            tensor(metadata["class_weights"], torch.float32),
        )
        counts = tensor(metadata["support_counts"], torch.float32)
        self.register_buffer(
            "global_probability",
            counts / counts.sum().clamp_min(1.0),
        )
        self.prior = SmoothedSupportPrior(metadata.get("prior") or {})

    def forward(
        self,
        hidden: torch.Tensor,
        destination_ids: torch.Tensor,
        timestamps: torch.Tensor,
        target_values: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        target_ids = (
            nearest_support_indices_torch(
                target_values,
                self.support_standardized,
            )
            if target_values is not None
            else None
        )
        coarse_logits = self.coarse(hidden)
        prior_logits = None
        if self.prior.enabled:
            prior_logits = self.prior(destination_ids, timestamps)
            coarse_prior = torch.stack(
                [
                    torch.logsumexp(
                        prior_logits[:, start:end],
                        dim=1,
                    )
                    for start, end in zip(
                        self.offsets[:-1].tolist(),
                        self.offsets[1:].tolist(),
                    )
                ],
                dim=1,
            )
            coarse_logits = (
                coarse_logits
                + self.prior.lambda_prior * coarse_prior
            )
        output: dict[str, Any] = {
            "mode": "hierarchical_support",
            "coarse_logits": coarse_logits,
            "hidden": hidden,
            "target_ids": target_ids,
            "support_standardized": self.support_standardized,
            "support_original": self.support_original,
            "class_weights": self.class_weights,
            "global_probability": self.global_probability,
            "label_smoothing": self.label_smoothing,
            "ordinal_weight": self.ordinal_weight,
            "prior_logits": prior_logits,
        }
        if target_ids is not None:
            target_bins = self.bin_ids[target_ids]
            max_width = max(
                int(layer.out_features)
                for layer in self.fine
            )
            fine_logits = hidden.new_full(
                (len(hidden), max_width),
                -float("inf"),
            )
            fine_targets = torch.empty_like(target_ids)
            fine_groups: list[dict[str, torch.Tensor]] = []
            for bin_id, layer in enumerate(self.fine):
                selected = target_bins == bin_id
                if not selected.any():
                    continue
                local = layer(hidden[selected])
                if prior_logits is not None:
                    start = int(self.offsets[bin_id].item())
                    end = int(self.offsets[bin_id + 1].item())
                    local = (
                        local
                        + self.prior.lambda_prior
                        * prior_logits[selected, start:end]
                    )
                fine_logits[selected, : local.shape[1]] = local
                fine_targets[selected] = (
                    target_ids[selected] - self.offsets[bin_id]
                )
                fine_groups.append(
                    {
                        "row_mask": selected,
                        "logits": local,
                        "targets": fine_targets[selected],
                    }
                )
            output.update(
                {
                    "target_bins": target_bins,
                    "fine_logits": fine_logits,
                    "fine_targets": fine_targets,
                    "fine_groups": fine_groups,
                }
            )
        return output

    def sample(
        self,
        output: dict[str, Any],
        *,
        temperature: float,
    ) -> torch.Tensor:
        coarse = torch.multinomial(
            torch.softmax(
                output["coarse_logits"].float()
                / max(float(temperature), 1e-6),
                dim=-1,
            ),
            1,
        ).squeeze(1)
        support_ids = torch.empty(
            len(coarse),
            dtype=torch.long,
            device=coarse.device,
        )
        hidden = output["hidden"]
        prior_logits = output.get("prior_logits")
        for bin_id, layer in enumerate(self.fine):
            selected = coarse == bin_id
            if not selected.any():
                continue
            local_logits = layer(hidden[selected])
            if prior_logits is not None:
                start = int(self.offsets[bin_id].item())
                end = int(self.offsets[bin_id + 1].item())
                local_logits = (
                    local_logits
                    + self.prior.lambda_prior
                    * prior_logits[selected, start:end]
                )
            local = torch.multinomial(
                torch.softmax(
                    local_logits.float()
                    / max(float(temperature), 1e-6),
                    dim=-1,
                ),
                1,
            ).squeeze(1)
            support_ids[selected] = local + self.offsets[bin_id]
        return self.support_original[support_ids]


def support_numerical_loss(
    output: dict[str, Any],
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mode = output["mode"]
    support = output["support_standardized"]
    target_ids = nearest_support_indices_torch(target, support)
    if mode == "discrete_support":
        nll = F.cross_entropy(
            output["logits"],
            target_ids,
            weight=output["class_weights"],
            label_smoothing=float(output["label_smoothing"]),
        )
        probabilities = torch.softmax(output["logits"].float(), dim=-1)
    elif mode == "hierarchical_support":
        target_weights = output["class_weights"][target_ids]
        coarse_values = F.cross_entropy(
            output["coarse_logits"],
            output["target_bins"],
            reduction="none",
            label_smoothing=float(output["label_smoothing"]),
        )
        fine_numerator = output["coarse_logits"].new_zeros(())
        fine_denominator = output["coarse_logits"].new_zeros(())
        for group in output["fine_groups"]:
            group_weights = target_weights[group["row_mask"]]
            group_values = F.cross_entropy(
                group["logits"],
                group["targets"],
                reduction="none",
                label_smoothing=float(output["label_smoothing"]),
            )
            fine_numerator = fine_numerator + (
                group_values * group_weights
            ).sum()
            fine_denominator = fine_denominator + group_weights.sum()
        coarse = (
            coarse_values * target_weights
        ).sum() / target_weights.sum().clamp_min(1e-8)
        fine = fine_numerator / fine_denominator.clamp_min(1e-8)
        nll = coarse + fine
        probabilities = None
    else:
        raise ValueError(f"Not a support numerical output: {mode!r}")
    components = {"nll": nll}
    ordinal_weight = float(output.get("ordinal_weight", 0.0))
    if ordinal_weight > 0.0 and probabilities is not None:
        scale = (
            support.max() - support.min()
        ).abs().clamp_min(1e-6)
        ordinal = (
            probabilities
            * (support.unsqueeze(0) - target.unsqueeze(1)).abs()
            / scale
        ).sum(dim=1).mean()
        components["ordinal"] = ordinal * ordinal_weight
    calibration_weight = float(
        output.get("global_calibration_weight", 0.0)
    )
    if calibration_weight > 0.0 and probabilities is not None:
        marginal = F.l1_loss(
            probabilities.mean(dim=0),
            output["global_probability"],
        )
        components["global_calibration"] = (
            marginal * calibration_weight
        )
    return sum(components.values()), components


def nearest_support_indices_torch(
    values: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    values = values.float()
    position = torch.searchsorted(support, values)
    right = position.clamp(max=len(support) - 1)
    left = (position - 1).clamp(min=0)
    choose_right = (
        (support[right] - values).abs()
        < (values - support[left]).abs()
    )
    return torch.where(choose_right, right, left)


def nearest_support_indices_numpy(
    values: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    position = np.searchsorted(support, values)
    right = np.clip(position, 0, len(support) - 1)
    left = np.clip(position - 1, 0, len(support) - 1)
    choose_right = (
        np.abs(support[right] - values)
        < np.abs(values - support[left])
    )
    return np.where(choose_right, right, left).astype(np.int64)


def inverse_numpy(
    values: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    output = (
        np.asarray(values, dtype=float)
        * float(metadata.get("std", 1.0))
        + float(metadata.get("mean", 0.0))
    )
    if str(metadata.get("preprocessing", "standardize")).startswith(
        "log1p"
    ):
        output = np.maximum(np.expm1(output), 0.0)
    if bool(metadata.get("clip_to_train_range", True)):
        output = np.clip(
            output,
            float(metadata.get("min_train", -np.inf)),
            float(metadata.get("max_train", np.inf)),
        )
    if metadata.get("semantic_type") == "count_numerical":
        output = np.maximum(np.round(output), 0.0)
    return output


def support_output_dtype(
    values: pd.Series,
    numerical_metadata: dict[str, Any],
) -> str:
    if (
        pd.api.types.is_integer_dtype(values.dtype)
        or numerical_metadata.get("semantic_type")
        == "count_numerical"
    ):
        return "int64"
    return "float64"


def support_torch_dtype(metadata: dict[str, Any]) -> torch.dtype:
    if str(metadata.get("support_output_dtype")) == "int64":
        return torch.int64
    return torch.float64


def original_support_for_standardized(
    standardized_values: np.ndarray,
    original_values: np.ndarray,
    standardized_support: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Retain exact observed values when raw training rows are available."""

    standardized_values = np.asarray(
        standardized_values,
        dtype=np.float32,
    )
    original_values = np.asarray(original_values)
    mapping: dict[float, Any] = {}
    for standardized, original in zip(
        standardized_values,
        original_values,
    ):
        mapping.setdefault(float(standardized), original)
    if all(float(value) in mapping for value in standardized_support):
        return np.asarray(
            [mapping[float(value)] for value in standardized_support]
        )
    return inverse_numpy(standardized_support, metadata)


def fit_hash_frequency_buckets(
    counts: pd.Series,
    num_buckets: int,
) -> dict[int, int]:
    if not len(counts):
        return {}
    try:
        labels = pd.qcut(
            counts.astype(float).rank(method="first"),
            q=min(max(int(num_buckets), 1), len(counts)),
            labels=False,
            duplicates="drop",
        )
        return {
            int(entity): int(bucket)
            for entity, bucket in labels.items()
        }
    except ValueError:
        return {int(entity): 0 for entity in counts.index}


def fit_timestamp_boundaries(
    timestamps: np.ndarray,
    num_buckets: int,
) -> np.ndarray:
    finite = np.asarray(timestamps, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.asarray([], dtype=np.float32)
    boundaries = np.quantile(
        finite,
        np.linspace(
            0.0,
            1.0,
            max(int(num_buckets), 1) + 1,
        )[1:-1],
    )
    return np.unique(boundaries).astype(np.float32)


def grouped_support_counts(
    group_ids: np.ndarray,
    support_ids: np.ndarray,
    num_groups: int,
    support_size: int,
) -> np.ndarray:
    output = np.zeros(
        (int(num_groups), int(support_size)),
        dtype=np.float32,
    )
    valid = (
        (group_ids >= 0)
        & (group_ids < int(num_groups))
    )
    np.add.at(
        output,
        (group_ids[valid], support_ids[valid]),
        1.0,
    )
    return output


def blend_dense_counts(
    base: torch.Tensor,
    counts: torch.Tensor,
    smoothing: float,
    epsilon: float,
) -> torch.Tensor:
    totals = counts.sum(dim=1, keepdim=True)
    empirical = (counts + epsilon) / (
        totals + epsilon * counts.shape[1]
    ).clamp_min(epsilon)
    weight = totals / (totals + float(smoothing))
    return (1.0 - weight) * base + weight * empirical


def tensor(values: Any, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(values, dtype=dtype)
