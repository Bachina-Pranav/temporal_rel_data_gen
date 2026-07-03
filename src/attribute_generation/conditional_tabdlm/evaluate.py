"""Evaluation for Conditional TABDLM attribute generation."""

from __future__ import annotations

import math
import warnings
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graph_schema import graph_conditioning_enabled, graph_metadata
from .schema import ConditionalTABDLMConfig
from .tokenization import normalize_text, summary_length_bucket_name
from .utils import (
    distribution_l1,
    ensure_dir,
    js_divergence,
    ks_statistic,
    safe_corr,
    save_json,
)


def evaluate_from_config(
    config: ConditionalTABDLMConfig,
    synthetic_reviews_path: str | Path | None = None,
    real_reviews_path: str | Path | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    synthetic_reviews_path = Path(synthetic_reviews_path or config.output_dir / "synthetic_review_attrs.csv")
    real_reviews_path = Path(real_reviews_path or config.train_data_path)
    if output_path is not None:
        output_path = Path(output_path)
        output_dir = ensure_dir(output_path.parent)
    else:
        output_dir = ensure_dir(output_dir or config.output_dir / "evaluation")
        output_path = output_dir / "eval_metrics.json"
    real = pd.read_csv(real_reviews_path)
    synthetic = pd.read_csv(synthetic_reviews_path)
    debug_examples_path = synthetic_reviews_path.parent / "debug" / "generated_examples.jsonl"
    metrics = evaluate_frames(real, synthetic, config, debug_examples_path=debug_examples_path)
    sample_metadata_path = synthetic_reviews_path.parent / "sample_metadata.json"
    if sample_metadata_path.exists():
        with sample_metadata_path.open() as handle:
            sample_metadata = json.load(handle)
        if "graph_conditioning" not in metrics:
            metrics["graph_conditioning"] = {}
        for key in [
            "uses_graph_context",
            "graph_conditioning_mode",
            "temporal_filter_enabled",
            "temporal_filter_mode",
            "graph_uses_future_events",
            "graph_uses_target_attributes",
            "real_graph_used_at_sampling",
            "synthetic_graph_history_source",
        ]:
            if key in sample_metadata:
                metrics["graph_conditioning"][key] = sample_metadata[key]
    save_json(metrics, output_path)
    write_report(metrics, output_dir / "eval_report.md")
    print(f"Wrote {output_path}")
    print(f"Wrote {output_dir / 'eval_report.md'}")
    return metrics


def evaluate_frames(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    config: ConditionalTABDLMConfig,
    debug_examples_path: str | Path | None = None,
) -> dict[str, Any]:
    schema = config.schema
    real = normalize_for_eval(real, config)
    synthetic = normalize_for_eval(synthetic, config)
    rating_col = schema.categorical_targets[0] if schema.categorical_targets else None
    verified_col = "verified" if "verified" in schema.categorical_targets else (
        schema.categorical_targets[1] if len(schema.categorical_targets) > 1 else None
    )
    summary_col = schema.text_targets[0] if schema.text_targets else None
    timestamp_col = schema.datetime_columns[0]
    customer_col = schema.foreign_key_columns[0]
    product_col = schema.foreign_key_columns[1] if len(schema.foreign_key_columns) > 1 else schema.foreign_key_columns[0]

    metrics: dict[str, Any] = {
        "validity": validity_metrics(real, synthetic, schema),
        "marginal_categorical": {},
        "temporal": {},
        "joint": {},
        "text": {},
        "length_diagnostics": {},
        "text_privacy": {},
        "text_consistency": {},
        "conditional_fidelity": {},
    }
    if graph_conditioning_enabled(config.raw):
        metrics["graph_conditioning"] = graph_metadata(config.raw, real_graph_used_at_sampling=False)
    if rating_col:
        metrics["marginal_categorical"].update(categorical_distribution_metrics(real, synthetic, rating_col, numeric=True))
    if verified_col:
        metrics["marginal_categorical"].update(categorical_distribution_metrics(real, synthetic, verified_col, numeric=False))
    if rating_col:
        metrics["temporal"].update(monthly_numeric_metrics(real, synthetic, timestamp_col, rating_col, "monthly_rating_mean"))
    if verified_col:
        metrics["temporal"].update(monthly_numeric_metrics(real, synthetic, timestamp_col, verified_col, "monthly_verified_rate"))
    if summary_col:
        real_len = summary_lengths(real[summary_col])
        syn_len = summary_lengths(synthetic[summary_col])
        metrics["temporal"].update(
            monthly_series_metrics(
                real.assign(_summary_length=real_len),
                synthetic.assign(_summary_length=syn_len),
                timestamp_col,
                "_summary_length",
                "monthly_summary_length",
            )
        )
        metrics["text"].update(text_metrics(real[summary_col], synthetic[summary_col]))
        metrics["length_diagnostics"].update(
            summary_length_diagnostics(
                real[summary_col],
                synthetic[summary_col],
                config,
                debug_examples_path=debug_examples_path,
            )
        )
        metrics["text_privacy"].update(text_privacy_metrics(real[summary_col], synthetic[summary_col]))
    if rating_col and verified_col:
        metrics["joint"].update(joint_rating_verified_metrics(real, synthetic, rating_col, verified_col))
    if summary_col and rating_col:
        metrics["text_consistency"].update(text_consistency_metrics(real, synthetic, summary_col, rating_col, verified_col))
    if rating_col:
        metrics["conditional_fidelity"].update(top_entity_mae(real, synthetic, product_col, rating_col, "product_rating", numeric=True))
        metrics["conditional_fidelity"].update(top_entity_mae(real, synthetic, customer_col, rating_col, "customer_rating", numeric=True))
    if verified_col:
        metrics["conditional_fidelity"].update(top_entity_mae(real, synthetic, product_col, verified_col, "product_verified", numeric=True))
        metrics["conditional_fidelity"].update(top_entity_mae(real, synthetic, customer_col, verified_col, "customer_verified", numeric=True))
    metrics["conditional_fidelity"]["condition_columns_used"] = list(schema.condition_columns)
    return metrics


def normalize_for_eval(frame: pd.DataFrame, config: ConditionalTABDLMConfig) -> pd.DataFrame:
    frame = frame.copy()
    for column in config.schema.datetime_columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in config.schema.text_targets:
        if column in frame.columns:
            frame[column] = frame[column].map(normalize_text)
    for column in config.schema.categorical_targets:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    return frame.dropna(subset=list(config.schema.datetime_columns)).reset_index(drop=True)


def validity_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, schema) -> dict[str, Any]:
    out: dict[str, Any] = {
        "num_real_rows": int(len(real)),
        "num_synthetic_rows": int(len(synthetic)),
    }
    for column in schema.categorical_targets:
        valid = set(real[column].astype(str).dropna().unique()) if column in real else set()
        values = synthetic[column].astype(str) if column in synthetic else pd.Series([], dtype=str)
        out[f"invalid_{column}_rate"] = float((~values.isin(valid)).mean()) if len(values) else None
    for column in schema.text_targets:
        if column in synthetic:
            syn_len = summary_lengths(synthetic[column])
            real_len = summary_lengths(real[column])
            out[f"empty_{column}_rate"] = float((syn_len == 0).mean()) if len(syn_len) else None
            out[f"{column}_length_mean_real"] = float(real_len.mean()) if len(real_len) else None
            out[f"{column}_length_mean_synthetic"] = float(syn_len.mean()) if len(syn_len) else None
            out[f"{column}_length_ks"] = ks_statistic(real_len, syn_len)
    return out


def summary_length_diagnostics(
    real_text: pd.Series,
    synthetic_text: pd.Series,
    config: ConditionalTABDLMConfig,
    debug_examples_path: str | Path | None = None,
) -> dict[str, Any]:
    real_len = summary_lengths(real_text)
    syn_len = summary_lengths(synthetic_text)
    buckets = config.schema.summary_length_buckets or {
        "len_0": (0, 0),
        "len_1_2": (1, 2),
        "len_3_5": (3, 5),
        "len_6_10": (6, 10),
        "len_11_16": (11, 16),
        "len_17_32": (17, 32),
    }
    real_buckets = real_len.map(lambda length: summary_length_bucket_name(int(length), buckets))
    syn_buckets = syn_len.map(lambda length: summary_length_bucket_name(int(length), buckets))
    max_tokens = int(config.schema.text_max_lengths.get(config.schema.text_targets[0], 32)) if config.schema.text_targets else 32
    max_content = max(0, max_tokens - 2)
    diagnostics = {
        "summary_length_mean_real": float(real_len.mean()) if len(real_len) else None,
        "summary_length_mean_synthetic": float(syn_len.mean()) if len(syn_len) else None,
        "summary_length_median_real": float(real_len.median()) if len(real_len) else None,
        "summary_length_median_synthetic": float(syn_len.median()) if len(syn_len) else None,
        "summary_length_p90_real": float(real_len.quantile(0.9)) if len(real_len) else None,
        "summary_length_p90_synthetic": float(syn_len.quantile(0.9)) if len(syn_len) else None,
        "summary_length_ks": ks_statistic(real_len, syn_len),
        "summary_length_bucket_distribution_real": normalized_value_counts(real_buckets),
        "summary_length_bucket_distribution_synthetic": normalized_value_counts(syn_buckets),
        "summary_length_bucket_l1": distribution_l1(real_buckets, syn_buckets),
        "summary_length_bucket_js": js_divergence(real_buckets, syn_buckets),
        "generated_to_max_length_rate": float((syn_len >= max_content).mean()) if len(syn_len) else None,
    }
    diagnostics.update(debug_eos_diagnostics(debug_examples_path))
    if debug_examples_path is not None:
        summary_metrics_path = Path(debug_examples_path).parent / "summary_length_decoding_metrics.json"
        if summary_metrics_path.exists():
            with summary_metrics_path.open() as handle:
                diagnostics.update(json.load(handle))
    return diagnostics


def normalized_value_counts(series: pd.Series) -> dict[str, float]:
    return {str(key): float(value) for key, value in series.value_counts(normalize=True).sort_index().items()}


def debug_eos_diagnostics(debug_examples_path: str | Path | None) -> dict[str, Any]:
    if debug_examples_path is None or not Path(debug_examples_path).exists():
        return {
            "eos_missing_rate": None,
            "pad_after_eos_violation_rate": None,
            "mean_eos_position": None,
            "eos_position_distribution": {},
        }
    examples = load_debug_examples(debug_examples_path)
    if not examples:
        return {
            "eos_missing_rate": None,
            "pad_after_eos_violation_rate": None,
            "mean_eos_position": None,
            "eos_position_distribution": {},
        }
    eos_positions = [row.get("eos_position") for row in examples]
    missing = [pos is None for pos in eos_positions]
    violations = []
    for row in examples:
        tokens = row.get("raw_summary_tokens") or []
        if "[EOS]" not in tokens:
            violations.append(False)
            continue
        eos_idx = tokens.index("[EOS]")
        violations.append(any(token != "[PAD]" for token in tokens[eos_idx + 1 :]))
    observed = [int(pos) for pos in eos_positions if pos is not None]
    return {
        "eos_missing_rate": float(np.mean(missing)),
        "pad_after_eos_violation_rate": float(np.mean(violations)),
        "mean_eos_position": float(np.mean(observed)) if observed else None,
        "eos_position_distribution": {str(key): float(value) for key, value in pd.Series(observed).value_counts(normalize=True).sort_index().items()},
    }


def load_debug_examples(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def categorical_distribution_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, column: str, numeric: bool) -> dict[str, Any]:
    out = {
        f"{column}_distribution_l1": distribution_l1(real[column], synthetic[column]),
        f"{column}_distribution_js": js_divergence(real[column], synthetic[column]),
        f"{column}_total_variation": 0.5 * distribution_l1(real[column], synthetic[column]),
    }
    if numeric:
        out[f"{column}_ks"] = ks_statistic(pd.to_numeric(real[column], errors="coerce"), pd.to_numeric(synthetic[column], errors="coerce"))
    if column == "verified":
        real_rate = boolish_numeric(real[column]).mean()
        syn_rate = boolish_numeric(synthetic[column]).mean()
        out["verified_rate_real"] = float(real_rate)
        out["verified_rate_synthetic"] = float(syn_rate)
        out["verified_rate_diff"] = float(syn_rate - real_rate)
    return out


def monthly_numeric_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, timestamp_col: str, column: str, prefix: str) -> dict[str, Any]:
    return monthly_series_metrics(
        real.assign(_value=boolish_numeric(real[column]) if column == "verified" else pd.to_numeric(real[column], errors="coerce")),
        synthetic.assign(_value=boolish_numeric(synthetic[column]) if column == "verified" else pd.to_numeric(synthetic[column], errors="coerce")),
        timestamp_col,
        "_value",
        prefix,
    )


def monthly_series_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    prefix: str,
) -> dict[str, Any]:
    real_series = monthly_mean_series(real, timestamp_col, value_col)
    syn_series = monthly_mean_series(synthetic, timestamp_col, value_col)
    index = real_series.index.union(syn_series.index)
    r = real_series.reindex(index)
    s = syn_series.reindex(index)
    return {
        f"{prefix}_corr": safe_corr(r.to_numpy(dtype=float), s.to_numpy(dtype=float)),
        f"{prefix}_mae": float(np.nanmean(np.abs(r.to_numpy(dtype=float) - s.to_numpy(dtype=float)))) if len(index) else None,
    }


def monthly_mean_series(frame: pd.DataFrame, timestamp_col: str, value_col: str) -> pd.Series:
    monthly = frame[[timestamp_col, value_col]].dropna().copy()
    if monthly.empty:
        return pd.Series(dtype=float)
    monthly["_month"] = pd.to_datetime(monthly[timestamp_col], errors="coerce").dt.to_period("M")
    monthly = monthly.dropna(subset=["_month"])
    return monthly.groupby("_month")[value_col].mean()


def joint_rating_verified_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, rating_col: str, verified_col: str) -> dict[str, Any]:
    real_joint = real.groupby([rating_col, verified_col]).size() / max(len(real), 1)
    syn_joint = synthetic.groupby([rating_col, verified_col]).size() / max(len(synthetic), 1)
    index = real_joint.index.union(syn_joint.index)
    real_given_verified = real.groupby(verified_col)[rating_col].value_counts(normalize=True)
    syn_given_verified = synthetic.groupby(verified_col)[rating_col].value_counts(normalize=True)
    gv_index = real_given_verified.index.union(syn_given_verified.index)
    return {
        "rating_verified_joint_l1": float(np.abs(real_joint.reindex(index, fill_value=0.0) - syn_joint.reindex(index, fill_value=0.0)).sum()),
        "verified_rate_by_rating_mae": grouped_rate_mae(real, synthetic, rating_col, verified_col),
        "rating_distribution_given_verified_l1": float(
            np.abs(real_given_verified.reindex(gv_index, fill_value=0.0) - syn_given_verified.reindex(gv_index, fill_value=0.0)).sum()
        ),
    }


def grouped_rate_mae(real: pd.DataFrame, synthetic: pd.DataFrame, group_col: str, value_col: str) -> float | None:
    r = real.assign(_v=boolish_numeric(real[value_col])).groupby(group_col)["_v"].mean()
    s = synthetic.assign(_v=boolish_numeric(synthetic[value_col])).groupby(group_col)["_v"].mean()
    index = r.index.intersection(s.index)
    if len(index) == 0:
        return None
    return float(np.mean(np.abs(r.reindex(index).to_numpy(dtype=float) - s.reindex(index).to_numpy(dtype=float))))


def text_metrics(real_text: pd.Series, synthetic_text: pd.Series) -> dict[str, Any]:
    syn_tokens = [tokenize(text) for text in synthetic_text]
    flat = [token for row in syn_tokens for token in row]
    bigrams = [tuple(row[idx : idx + 2]) for row in syn_tokens for idx in range(max(0, len(row) - 1))]
    top_real = set(real_text.map(normalize_text).value_counts().head(100).index)
    top_syn = set(synthetic_text.map(normalize_text).value_counts().head(100).index)
    return {
        "distinct_1": float(len(set(flat)) / max(len(flat), 1)),
        "distinct_2": float(len(set(bigrams)) / max(len(bigrams), 1)),
        "summary_length_ks": ks_statistic(summary_lengths(real_text), summary_lengths(synthetic_text)),
        "unique_summary_rate": float(synthetic_text.map(normalize_text).nunique() / max(len(synthetic_text), 1)),
        "top_100_summary_overlap_rate": float(len(top_real.intersection(top_syn)) / max(len(top_real), 1)),
    }


def text_privacy_metrics(real_text: pd.Series, synthetic_text: pd.Series, sample_size: int = 5000) -> dict[str, Any]:
    real_norm = real_text.map(normalize_text)
    syn_norm = synthetic_text.map(normalize_text)
    metrics = {
        "exact_summary_train_overlap_rate": float(syn_norm.isin(set(real_norm)).mean()) if len(syn_norm) else None,
        "nearest_neighbor_rougeL_mean": None,
        "nearest_neighbor_token_jaccard_mean": None,
        "nearest_neighbor_sample_size": int(min(sample_size, len(syn_norm), len(real_norm))),
    }
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return metrics
    n = metrics["nearest_neighbor_sample_size"]
    if n <= 0:
        return metrics
    real_sample = real_norm.sample(n=n, random_state=17).tolist()
    syn_sample = syn_norm.sample(n=n, random_state=23).tolist()
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    real_matrix = vectorizer.fit_transform(real_sample)
    syn_matrix = vectorizer.transform(syn_sample)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(real_matrix)
    _, indices = nn.kneighbors(syn_matrix)
    rouge = []
    jaccard = []
    for syn, idx in zip(syn_sample, indices[:, 0]):
        real = real_sample[int(idx)]
        rouge.append(rouge_l_f1(tokenize(syn), tokenize(real)))
        jaccard.append(token_jaccard(tokenize(syn), tokenize(real)))
    metrics["nearest_neighbor_rougeL_mean"] = float(np.mean(rouge))
    metrics["nearest_neighbor_token_jaccard_mean"] = float(np.mean(jaccard))
    return metrics


def text_consistency_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    summary_col: str,
    rating_col: str,
    verified_col: str | None,
    max_train_rows: int = 100000,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.exceptions import ConvergenceWarning
    except Exception:
        return metrics
    train = real.dropna(subset=[summary_col, rating_col])
    if len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=29)
    if train[rating_col].nunique() > 1:
        clf = make_pipeline(
            TfidfVectorizer(max_features=50000, ngram_range=(1, 2)),
            LogisticRegression(max_iter=1000, n_jobs=1),
        )
        metrics["rating_text_predictor_converged"] = fit_text_classifier(
            clf,
            train[summary_col].map(normalize_text),
            train[rating_col].astype(str),
            ConvergenceWarning,
        )
        pred = clf.predict(synthetic[summary_col].map(normalize_text))
        pred_dist = pd.Series(pred).value_counts(normalize=True).to_dict()
        metrics["predicted_rating_distribution"] = {str(k): float(v) for k, v in pred_dist.items()}
        metrics["rating_text_consistency_accuracy"] = float(accuracy_score(synthetic[rating_col].astype(str), pred))
    if verified_col and verified_col in real and verified_col in synthetic:
        y = boolish_numeric(train[verified_col])
        if y.nunique() > 1:
            clf = make_pipeline(
                TfidfVectorizer(max_features=50000, ngram_range=(1, 2)),
                LogisticRegression(max_iter=1000, n_jobs=1),
            )
            metrics["verified_text_predictor_converged"] = fit_text_classifier(
                clf,
                train[summary_col].map(normalize_text),
                y,
                ConvergenceWarning,
            )
            if hasattr(clf[-1], "predict_proba"):
                prob = clf.predict_proba(synthetic[summary_col].map(normalize_text))[:, 1]
                syn_y = boolish_numeric(synthetic[verified_col])
                if syn_y.nunique() > 1:
                    metrics["verified_text_predictor_auc"] = float(roc_auc_score(syn_y, prob))
    return metrics


def fit_text_classifier(clf, texts: pd.Series, labels: pd.Series, convergence_warning_type) -> bool:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", convergence_warning_type)
        clf.fit(texts, labels)
    return not any(issubclass(item.category, convergence_warning_type) for item in caught)


def top_entity_mae(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    entity_col: str,
    value_col: str,
    prefix: str,
    numeric: bool,
    top_k: int = 1000,
) -> dict[str, Any]:
    top = real[entity_col].value_counts().head(top_k).index
    if value_col == "verified":
        real_values = real.assign(_value=boolish_numeric(real[value_col]))
        syn_values = synthetic.assign(_value=boolish_numeric(synthetic[value_col]))
    else:
        real_values = real.assign(_value=pd.to_numeric(real[value_col], errors="coerce"))
        syn_values = synthetic.assign(_value=pd.to_numeric(synthetic[value_col], errors="coerce"))
    r = real_values[real_values[entity_col].isin(top)].groupby(entity_col)["_value"].mean()
    s = syn_values[syn_values[entity_col].isin(top)].groupby(entity_col)["_value"].mean()
    index = r.index.intersection(s.index)
    return {
        f"{prefix}_top_{top_k}_mae": float(np.mean(np.abs(r.reindex(index).to_numpy(dtype=float) - s.reindex(index).to_numpy(dtype=float)))) if len(index) else None,
        f"{prefix}_top_{top_k}_coverage": float(len(index) / max(len(top), 1)),
    }


def boolish_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().all():
        return numeric.fillna(0.0).astype(float)
    lowered = series.astype(str).str.lower()
    return lowered.isin({"true", "1", "yes", "y", "verified"}).astype(float)


def summary_lengths(series: pd.Series) -> pd.Series:
    return series.map(lambda text: len(tokenize(text))).astype(float)


def tokenize(text: Any) -> list[str]:
    return normalize_text(text).lower().split()


def token_jaccard(left: list[str], right: list[str]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    return float(len(a.intersection(b)) / max(len(a.union(b)), 1))


def rouge_l_f1(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    lcs = longest_common_subsequence(left, right)
    precision = lcs / len(left)
    recall = lcs / len(right)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def longest_common_subsequence(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for j, other in enumerate(right, start=1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def write_report(metrics: dict[str, Any], path: str | Path) -> None:
    lines = ["# Conditional TABDLM Evaluation", ""]
    warnings = evaluation_warnings(metrics)
    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    for section, values in metrics.items():
        lines.append(f"## {section}")
        if isinstance(values, dict):
            for key, value in sorted(values.items()):
                if isinstance(value, float):
                    if math.isfinite(value):
                        value_repr = f"{value:.6g}"
                    else:
                        value_repr = str(value)
                else:
                    value_repr = str(value)
                lines.append(f"- {key}: {value_repr}")
        else:
            lines.append(str(values))
        lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def evaluation_warnings(metrics: dict[str, Any]) -> list[str]:
    warnings_out: list[str] = []
    privacy = metrics.get("text_privacy", {})
    text = metrics.get("text", {})
    length = metrics.get("length_diagnostics", {})
    exact_overlap = privacy.get("exact_summary_train_overlap_rate")
    unique_rate = text.get("unique_summary_rate")
    mean_length = length.get("summary_length_mean_synthetic")
    length_ks = length.get("summary_length_ks")
    if exact_overlap is not None and exact_overlap > 0.2:
        warnings_out.append(f"exact_summary_train_overlap_rate is high: {exact_overlap:.4g} > 0.2")
    if unique_rate is not None and unique_rate < 0.5:
        warnings_out.append(f"unique_summary_rate is low: {unique_rate:.4g} < 0.5")
    if mean_length is not None and not (3.0 <= mean_length <= 6.0):
        warnings_out.append(f"summary_length_mean_synthetic is outside [3.0, 6.0]: {mean_length:.4g}")
    if length_ks is not None and length_ks > 0.35:
        warnings_out.append(f"summary_length_ks is high: {length_ks:.4g} > 0.35")
    return warnings_out
