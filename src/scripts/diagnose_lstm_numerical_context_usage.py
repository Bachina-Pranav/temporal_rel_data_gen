#!/usr/bin/env python3
"""Paired fixed-latent diagnostics for LSTM numerical context usage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.graph_dataset import (  # noqa: E402
    build_temporal_history_index,
)
from attribute_generation.conditional_tabdlm.lstm_joint import (  # noqa: E402
    encode_conditions,
    load_lstm_checkpoint,
)
from attribute_generation.conditional_tabdlm.numerical import (  # noqa: E402
    inverse_transform_numerical,
)
from attribute_generation.conditional_tabdlm.numerical_head import (  # noqa: E402
    resolve_event_role_indices,
)
from attribute_generation.conditional_tabdlm.train import resolve_device  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--graph-history-prefix", default=None)
    parser.add_argument("--evaluation-real", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-rows", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    model, config, _, _, graph_encoder = load_lstm_checkpoint(
        args.checkpoint,
        device=device,
        include_graph=True,
    )
    spine_full = pd.read_csv(args.synthetic_spine, low_memory=False)
    frame = spine_full.head(int(args.num_rows)).reset_index(drop=True)
    graph_query_offset = 0
    history_frame = spine_full
    if args.graph_history_prefix:
        prefix = pd.read_csv(
            args.graph_history_prefix,
            low_memory=False,
        )
        missing = [
            column
            for column in config.schema.condition_columns
            if column not in prefix.columns
        ]
        if missing:
            raise ValueError(
                "Graph history prefix is missing condition columns: "
                f"{missing}"
            )
        graph_query_offset = int(len(prefix))
        history_frame = pd.concat(
            [
                prefix.loc[:, config.schema.condition_columns],
                spine_full.loc[:, config.schema.condition_columns],
            ],
            ignore_index=True,
        )
    real = (
        pd.read_csv(args.evaluation_real, low_memory=False)
        .head(len(frame))
        .reset_index(drop=True)
        if args.evaluation_real
        else None
    )
    report = context_usage_diagnostics(
        model,
        config,
        graph_encoder,
        history_frame,
        frame,
        real=real,
        device=device,
        seed=int(args.seed),
        graph_query_offset=graph_query_offset,
    )
    save_json(report, args.output)
    print(args.output)


@torch.no_grad()
def context_usage_diagnostics(
    model: Any,
    config: Any,
    graph_encoder: Any,
    history_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    *,
    real: pd.DataFrame | None,
    device: str,
    seed: int,
    graph_query_offset: int = 0,
) -> dict[str, Any]:
    roles = resolve_event_role_indices(
        config.raw,
        config.schema.foreign_key_columns,
        config.schema.datetime_columns,
    )
    num_hash_buckets = int(
        (config.raw.get("id_encoding") or {}).get(
            "num_buckets",
            262144,
        )
    )
    foreign_keys, datetimes = encode_conditions(
        query_frame,
        config.schema,
        num_hash_buckets,
        device,
    )
    graph_context = None
    if graph_encoder is not None:
        history = build_temporal_history_index(
            history_frame,
            config,
            seed=seed,
        )
        graph_context = graph_encoder(
            history.build_batch(
                list(
                    range(
                        int(graph_query_offset),
                        int(graph_query_offset)
                        + len(query_frame),
                    )
                ),
                device=device,
                deterministic=True,
            )
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    fixed_noise = torch.randn(
        len(query_frame),
        model.latent_noise_dim,
        device=device,
        generator=generator,
    )
    variants = condition_variants(
        foreign_keys,
        datetimes,
        roles,
        seed=seed,
    )
    outputs: dict[str, dict[str, Any]] = {}
    for label, (variant_fk, variant_dt) in variants.items():
        condition = model.encode_condition(
            variant_fk,
            variant_dt,
            graph_context=graph_context,
        )
        row = model.row_latent(condition, noise=fixed_noise)
        numerical = model.numerical_params(
            row,
            variant_fk,
            variant_dt,
            graph_context,
        )
        outputs[label] = {
            column: numerical_output_summary(
                model,
                column,
                output,
                config,
            )
            for column, output in numerical.items()
        }
    baseline = outputs["correct_context"]
    comparisons: dict[str, Any] = {}
    for label, columns in outputs.items():
        comparisons[label] = {
            column: compare_output_summaries(
                baseline[column],
                summary,
                query_frame,
                real,
                destination_column=roles["destination_fk"],
                numerical_column=column,
            )
            for column, summary in columns.items()
        }
    support_head_calibration = {}
    for column, summary in baseline.items():
        calibration = dict(summary.get("calibration_diagnostics") or {})
        if (
            real is not None
            and summary.get("probabilities") is not None
            and column in real
        ):
            calibration["support_probability_ece"] = (
                support_probability_ece(
                    summary["probabilities"],
                    summary["support_original"],
                    real[column],
                )
            )
        if calibration:
            support_head_calibration[column] = calibration
    return {
        "checkpoint": "loaded",
        "num_rows": int(len(query_frame)),
        "fixed_latent": True,
        "graph_context_held_fixed_across_pair": True,
        "graph_history_prefix_rows": int(graph_query_offset),
        "event_roles": roles,
        "conditions": list(variants),
        "comparisons": comparisons,
        "support_head_calibration": support_head_calibration,
        "continuous_variance_diagnostics": {
            column: baseline[column].get("variance_diagnostics")
            for column in config.schema.numerical_targets
        },
    }


def condition_variants(
    foreign_keys: torch.Tensor,
    datetimes: torch.Tensor,
    roles: dict[str, Any],
    *,
    seed: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device=foreign_keys.device)
    generator.manual_seed(int(seed) + 1009)
    permutation = torch.randperm(
        len(foreign_keys),
        generator=generator,
        device=foreign_keys.device,
    )
    source_index = int(roles["source_fk_index"])
    destination_index = int(roles["destination_fk_index"])
    timestamp_index = int(roles["timestamp_index"])

    shuffled_destination = foreign_keys.clone()
    shuffled_destination[:, destination_index] = foreign_keys[
        permutation,
        destination_index,
    ]
    zero_destination = foreign_keys.clone()
    zero_destination[:, destination_index] = 0
    shuffled_source = foreign_keys.clone()
    shuffled_source[:, source_index] = foreign_keys[
        permutation,
        source_index,
    ]
    shuffled_time = datetimes.clone()
    shuffled_time[:, timestamp_index] = datetimes[
        permutation,
        timestamp_index,
    ]
    return {
        "correct_context": (foreign_keys, datetimes),
        "shuffled_destination": (
            shuffled_destination,
            datetimes,
        ),
        "zero_destination": (zero_destination, datetimes),
        "shuffled_source": (shuffled_source, datetimes),
        "shuffled_timestamp": (foreign_keys, shuffled_time),
    }


def numerical_output_summary(
    model: Any,
    column: str,
    output: Any,
    config: Any,
) -> dict[str, Any]:
    if not isinstance(output, dict):
        mean = output[:, 0].float()
        log_std = output[:, 1].float().clamp(-7.0, 5.0)
        original_mean = inverse_transform_numerical(
            mean,
            (
                config.raw.get("_numerical_metadata") or {}
            ).get(column, {}),
        )
        return {
            "mode": "continuous_baseline",
            "mean": mean,
            "log_std": log_std,
            "expected_original": original_mean,
            "top_ids": None,
            "probabilities": None,
            "variance_diagnostics": {
                "predicted_std_mean": float(
                    torch.exp(log_std).mean().cpu()
                ),
                "predicted_std_std_across_rows": float(
                    torch.exp(log_std).std().cpu()
                ),
                "predicted_log_std_min": float(log_std.min().cpu()),
                "predicted_log_std_max": float(log_std.max().cpu()),
            },
        }
    if output["mode"] == "discrete_support":
        probabilities = torch.softmax(
            output["logits"].float(),
            dim=-1,
        )
    else:
        head = model.support_numerical_heads[column]
        probabilities = hierarchical_dense_probabilities(
            head,
            output,
        )
    support = output["support_original"].float()
    expected = probabilities @ support
    calibration = support_logit_diagnostics(
        model.support_numerical_heads[column],
        output,
        probabilities,
    )
    return {
        "mode": output["mode"],
        "probabilities": probabilities,
        "top_ids": probabilities.argmax(dim=1),
        "expected_original": expected,
        "support_original": support,
        "global_probability": output.get("global_probability"),
        "calibration_diagnostics": calibration,
        "variance_diagnostics": None,
    }


def support_logit_diagnostics(
    head: Any,
    output: dict[str, Any],
    probabilities: torch.Tensor,
) -> dict[str, Any]:
    eps = 1e-12
    entropy_by_row = -(
        probabilities.clamp_min(eps)
        * probabilities.clamp_min(eps).log()
    ).sum(dim=1)
    global_probability = output["global_probability"].float()
    target_entropy = -(
        global_probability.clamp_min(eps)
        * global_probability.clamp_min(eps).log()
    ).sum()
    if output["mode"] == "discrete_support":
        total = output["logits"].float()
        residual = output.get("residual_logits")
        prior = output.get("global_prior_logits")
    else:
        total = probabilities.clamp_min(eps).log()
        residual = hierarchical_dense_residual_logits(head, output)
        prior = output.get("global_prior_logits")
    residual = residual.float() if residual is not None else None
    prior = prior.float() if prior is not None else None
    ratio = None
    if residual is not None and prior is not None:
        prior_norm = prior.norm().clamp_min(eps)
        ratio = float(
            (residual.norm(dim=1) / prior_norm).mean().cpu()
        )
    return {
        "mean_max_softmax_probability": float(
            probabilities.max(dim=1).values.mean().cpu()
        ),
        "mean_predictive_entropy_nats": float(
            entropy_by_row.mean().cpu()
        ),
        "target_entropy_nats": float(target_entropy.cpu()),
        "logit_mean": float(total.mean().cpu()),
        "logit_std": float(total.std().cpu()),
        "residual_logit_mean": tensor_stat(residual, "mean"),
        "residual_logit_std": tensor_stat(residual, "std"),
        "prior_logit_mean": tensor_stat(prior, "mean"),
        "prior_logit_std": tensor_stat(prior, "std"),
        "gamma_scaled_residual_to_prior_norm_ratio": ratio,
    }


def hierarchical_dense_residual_logits(
    head: Any,
    output: dict[str, Any],
) -> torch.Tensor:
    rows = len(output["hidden"])
    dense = output["hidden"].new_zeros(
        (rows, len(head.support_standardized)),
        dtype=torch.float32,
    )
    coarse = output["coarse_residual_logits"].float()
    if head.global_prior.enabled:
        coarse = head.global_prior.residual_logits(coarse)
    for bin_id, layer in enumerate(head.fine):
        start = int(head.offsets[bin_id].item())
        end = int(head.offsets[bin_id + 1].item())
        local = layer(output["hidden"]).float()
        if head.global_prior.enabled:
            local = head.global_prior.residual_logits(local)
        dense[:, start:end] = coarse[:, bin_id : bin_id + 1] + local
    return dense


def tensor_stat(
    value: torch.Tensor | None,
    statistic: str,
) -> float | None:
    if value is None or not value.numel():
        return None
    return float(getattr(value, statistic)().cpu())


def hierarchical_dense_probabilities(
    head: Any,
    output: dict[str, Any],
) -> torch.Tensor:
    coarse = torch.softmax(
        output["coarse_logits"].float(),
        dim=-1,
    )
    dense = output["hidden"].new_zeros(
        (
            len(output["hidden"]),
            len(head.support_standardized),
        ),
        dtype=torch.float32,
    )
    prior = output.get("prior_logits")
    global_prior = output.get("global_prior_logits")
    for bin_id, layer in enumerate(head.fine):
        start = int(head.offsets[bin_id].item())
        end = int(head.offsets[bin_id + 1].item())
        local_logits = layer(output["hidden"]).float()
        if global_prior is not None:
            local_logits = (
                head.global_prior.residual_logits(local_logits)
                if head.global_prior.enabled
                else local_logits
            ) + global_prior[start:end].unsqueeze(0)
        if prior is not None:
            local_logits = (
                local_logits
                + head.prior.lambda_prior * prior[:, start:end]
            )
        dense[:, start:end] = (
            coarse[:, bin_id : bin_id + 1]
            * torch.softmax(local_logits, dim=-1)
        )
    return dense


def compare_output_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    spine: pd.DataFrame,
    real: pd.DataFrame | None,
    *,
    destination_column: str,
    numerical_column: str,
) -> dict[str, Any]:
    expected_change = (
        baseline["expected_original"]
        - candidate["expected_original"]
    ).abs()
    result: dict[str, Any] = {
        "mode": candidate["mode"],
        "mean_expected_value_change": float(
            expected_change.mean().cpu()
        ),
        "p95_expected_value_change": float(
            torch.quantile(expected_change.float(), 0.95).cpu()
        ),
    }
    if baseline["probabilities"] is not None:
        first = baseline["probabilities"].clamp_min(1e-12)
        second = candidate["probabilities"].clamp_min(1e-12)
        result.update(
            {
                "mean_absolute_probability_change": float(
                    (first - second).abs().mean().cpu()
                ),
                "mean_kl_divergence": float(
                    (
                        first * (first.log() - second.log())
                    )
                    .sum(dim=1)
                    .mean()
                    .cpu()
                ),
                "top1_support_agreement": float(
                    (
                        baseline["top_ids"]
                        == candidate["top_ids"]
                    )
                    .float()
                    .mean()
                    .cpu()
                ),
                "generated_value_agreement": float(
                    (
                        baseline["top_ids"]
                        == candidate["top_ids"]
                    )
                    .float()
                    .mean()
                    .cpu()
                ),
            }
        )
        global_probability = baseline.get("global_probability")
        if global_probability is not None:
            result["global_marginal_total_variation"] = float(
                0.5
                * (
                    candidate["probabilities"].mean(dim=0)
                    - global_probability
                )
                .abs()
                .sum()
                .cpu()
            )
        if real is not None and numerical_column in real:
            result["support_probability_ece"] = support_probability_ece(
                candidate["probabilities"],
                candidate["support_original"],
                real[numerical_column],
            )
    else:
        first_mean = baseline["mean"]
        second_mean = candidate["mean"]
        first_log_std = baseline["log_std"]
        second_log_std = candidate["log_std"]
        first_var = torch.exp(2.0 * first_log_std)
        second_var = torch.exp(2.0 * second_log_std)
        normal_kl = (
            second_log_std
            - first_log_std
            + (
                first_var
                + (first_mean - second_mean).pow(2)
            )
            / (2.0 * second_var)
            - 0.5
        )
        result["mean_gaussian_kl_divergence"] = float(
            normal_kl.mean().cpu()
        )
    if real is not None and numerical_column in real:
        expected = (
            candidate["expected_original"].detach().cpu().numpy()
        )
        predicted = pd.DataFrame(
            {
                destination_column: spine[destination_column]
                .astype(str)
                .to_numpy(),
                numerical_column: expected,
            }
        )
        real_values = pd.to_numeric(
            real[numerical_column],
            errors="coerce",
        )
        real_means = real.assign(
            **{destination_column: real[destination_column].astype(str)}
        ).groupby(destination_column)[numerical_column].mean()
        predicted_means = predicted.groupby(destination_column)[
            numerical_column
        ].mean()
        common = real_means.index.intersection(predicted_means.index)
        scale = max(float(real_values.std()), 1e-12)
        result[
            "destination_conditioned_standardized_mae"
        ] = (
            float(
                np.mean(
                    np.abs(
                        real_means.loc[common].to_numpy(float)
                        - predicted_means.loc[common].to_numpy(float)
                    )
                )
                / scale
            )
            if len(common)
            else None
        )
    return result


def support_probability_ece(
    probabilities: torch.Tensor,
    support: torch.Tensor,
    target: pd.Series,
    *,
    num_bins: int = 10,
) -> float | None:
    numeric = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(numeric)
    if not valid.any():
        return None
    rows = np.flatnonzero(valid)
    rows = rows[rows < len(probabilities)]
    if not len(rows):
        return None
    support_np = support.detach().cpu().numpy().astype(float)
    target_np = numeric[rows]
    distances = np.abs(target_np[:, None] - support_np[None, :])
    target_ids = distances.argmin(axis=1)
    exact = np.isclose(
        target_np,
        support_np[target_ids],
        rtol=1e-7,
        atol=1e-10,
    )
    rows = rows[exact]
    target_ids = target_ids[exact]
    if not len(rows):
        return None
    row_ids = torch.as_tensor(
        rows,
        dtype=torch.long,
        device=probabilities.device,
    )
    selected = probabilities[row_ids]
    confidence, prediction = selected.max(dim=1)
    correct = prediction.detach().cpu().numpy() == target_ids
    confidence_np = confidence.detach().cpu().numpy()
    edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        member = (confidence_np > lower) & (confidence_np <= upper)
        if member.any():
            ece += float(member.mean()) * abs(
                float(correct[member].mean())
                - float(confidence_np[member].mean())
            )
    return float(ece)


if __name__ == "__main__":
    main()
