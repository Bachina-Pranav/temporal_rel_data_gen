"""Evaluation helpers for Text V1 summaries."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .masked_summary_dataset import TOKEN_RE, normalize_summary_text
from .text_privacy_metrics import compute_text_privacy_metrics


POSITIVE_WORDS = {
    "great",
    "good",
    "excellent",
    "perfect",
    "love",
    "loved",
    "nice",
    "best",
    "amazing",
    "awesome",
    "wonderful",
}
NEGATIVE_WORDS = {
    "bad",
    "poor",
    "terrible",
    "awful",
    "worst",
    "hate",
    "hated",
    "broken",
    "disappointed",
    "disappointing",
    "not",
}


def evaluate_summary_text_v1(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    text_col: str = "summary",
    rating_col: str = "rating",
    verified_col: str = "verified",
    timestamp_col: str = "review_time",
    customer_id_col: str = "customer_id",
    product_id_col: str = "product_id",
    privacy_sample_size: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    real_text = [normalize_summary_text(text) for text in real[text_col].fillna("")]
    synthetic_text = [normalize_summary_text(text) for text in synthetic[text_col].fillna("")]
    split = chronological_train_test_split(real, timestamp_col)
    metrics: Dict[str, Any] = {
        "method": "temporal_summary_text_v1",
        "text_column": text_col,
        "basic": basic_text_metrics(real_text, synthetic_text),
        "conditional": conditional_text_metrics(real, synthetic, text_col, rating_col),
        "temporal": temporal_text_metrics(real, synthetic, text_col, rating_col, timestamp_col),
        "relational": relational_text_metrics(real, synthetic, text_col, customer_id_col, product_id_col),
        "privacy": compute_text_privacy_metrics(
            split["train"][text_col].fillna(""),
            synthetic[text_col].fillna(""),
            test_summaries=split["test"][text_col].fillna(""),
            sample_size=privacy_sample_size,
            random_state=seed,
        ),
    }
    if verified_col in real.columns and verified_col in synthetic.columns:
        metrics["conditional"]["verified_summary_length_correlation"] = safe_corr(
            normalize_binary(synthetic[verified_col]), token_lengths(synthetic_text)
        )
    return metrics


def basic_text_metrics(real_text: List[str], synthetic_text: List[str]) -> Dict[str, Any]:
    real_tokens = [tokens(text) for text in real_text]
    syn_tokens = [tokens(text) for text in synthetic_text]
    real_lengths = np.asarray([len(item) for item in real_tokens], dtype=float)
    syn_lengths = np.asarray([len(item) for item in syn_tokens], dtype=float)
    return {
        "real_average_token_length": float(np.mean(real_lengths)) if len(real_lengths) else 0.0,
        "synthetic_average_token_length": float(np.mean(syn_lengths)) if len(syn_lengths) else 0.0,
        "real_median_token_length": float(np.median(real_lengths)) if len(real_lengths) else 0.0,
        "synthetic_median_token_length": float(np.median(syn_lengths)) if len(syn_lengths) else 0.0,
        "length_ks": ks_stat(real_lengths, syn_lengths),
        "real_vocabulary_size": len(vocab(real_tokens)),
        "synthetic_vocabulary_size": len(vocab(syn_tokens)),
        "real_distinct_1": distinct_n(real_tokens, 1),
        "synthetic_distinct_1": distinct_n(syn_tokens, 1),
        "real_distinct_2": distinct_n(real_tokens, 2),
        "synthetic_distinct_2": distinct_n(syn_tokens, 2),
        "real_distinct_3": distinct_n(real_tokens, 3),
        "synthetic_distinct_3": distinct_n(syn_tokens, 3),
        "real_type_token_ratio": type_token_ratio(real_tokens),
        "synthetic_type_token_ratio": type_token_ratio(syn_tokens),
        "non_empty_rate": float(sum(bool(text.strip()) for text in synthetic_text) / max(len(synthetic_text), 1)),
        "contains_special_token_rate": special_token_rate(synthetic_text),
    }


def conditional_text_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, text_col: str, rating_col: str) -> Dict[str, Any]:
    synthetic_sentiment = np.asarray([sentiment_score(text) for text in synthetic[text_col].fillna("")], dtype=float)
    real_sentiment = np.asarray([sentiment_score(text) for text in real[text_col].fillna("")], dtype=float)
    result = {
        "real_sentiment_by_rating": grouped_mean(real, real_sentiment, rating_col),
        "synthetic_sentiment_by_rating": grouped_mean(synthetic, synthetic_sentiment, rating_col),
        "rating_text_consistency_correlation": safe_corr(
            pd.to_numeric(synthetic[rating_col], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            synthetic_sentiment,
        ),
    }
    result.update(rating_classifier_metrics(real, synthetic, text_col, rating_col))
    return result


def rating_classifier_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, text_col: str, rating_col: str) -> Dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        train_text = real[text_col].fillna("").astype(str).tolist()
        train_y = pd.to_numeric(real[rating_col], errors="coerce").fillna(0).round().astype(int)
        model = make_pipeline(
            CountVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1),
            LogisticRegression(max_iter=200, multi_class="auto"),
        )
        model.fit(train_text, train_y)
        real_pred = model.predict(train_text)
        synthetic_text = synthetic[text_col].fillna("").astype(str).tolist()
        syn_pred = model.predict(synthetic_text)
        syn_rating = pd.to_numeric(synthetic[rating_col], errors="coerce").fillna(0).round().astype(int)
        return {
            "real_summary_to_rating_accuracy": float(np.mean(real_pred == train_y.to_numpy())),
            "synthetic_summary_to_rating_pred_distribution": distribution_dict(syn_pred),
            "synthetic_predicted_rating_mean_by_generated_rating": {
                str(key): float(np.mean(syn_pred[syn_rating.to_numpy() == key]))
                for key in sorted(set(syn_rating))
                if np.any(syn_rating.to_numpy() == key)
            },
            "rating_classifier_pred_actual_correlation": safe_corr(syn_pred, syn_rating.to_numpy()),
        }
    except Exception:
        return {
            "real_summary_to_rating_accuracy": None,
            "synthetic_summary_to_rating_pred_distribution": {},
            "rating_classifier_pred_actual_correlation": None,
        }


def temporal_text_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, text_col: str, rating_col: str, timestamp_col: str) -> Dict[str, Any]:
    real_month = pd.to_datetime(real[timestamp_col], errors="coerce").dt.to_period("M").astype(str)
    syn_month = pd.to_datetime(synthetic[timestamp_col], errors="coerce").dt.to_period("M").astype(str)
    real_len = pd.Series(token_lengths(real[text_col].fillna(""))).groupby(real_month).mean()
    syn_len = pd.Series(token_lengths(synthetic[text_col].fillna(""))).groupby(syn_month).mean()
    months = sorted(set(real_len.index).intersection(set(syn_len.index)))
    return {
        "summary_length_by_month_correlation": safe_corr(real_len.reindex(months).to_numpy(), syn_len.reindex(months).to_numpy()),
        "synthetic_summary_sentiment_by_month": grouped_mean(synthetic.assign(_sent=[sentiment_score(x) for x in synthetic[text_col].fillna("")]), "_sent", timestamp_col),
    }


def relational_text_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, text_col: str, customer_id_col: str, product_id_col: str) -> Dict[str, Any]:
    syn_sentiment = pd.Series([sentiment_score(text) for text in synthetic[text_col].fillna("")], index=synthetic.index)
    return {
        "product_average_summary_sentiment_std": float(syn_sentiment.groupby(synthetic[product_id_col]).mean().std(ddof=0)),
        "customer_average_summary_sentiment_std": float(syn_sentiment.groupby(synthetic[customer_id_col]).mean().std(ddof=0)),
        "synthetic_product_summary_length_std": float(pd.Series(token_lengths(synthetic[text_col].fillna(""))).groupby(synthetic[product_id_col]).mean().std(ddof=0)),
        "synthetic_customer_summary_length_std": float(pd.Series(token_lengths(synthetic[text_col].fillna(""))).groupby(synthetic[customer_id_col]).mean().std(ddof=0)),
    }


def chronological_train_test_split(real: pd.DataFrame, timestamp_col: str) -> Dict[str, pd.DataFrame]:
    sorted_real = real.assign(_ts=pd.to_datetime(real[timestamp_col], errors="coerce")).sort_values("_ts")
    cut = int(0.8 * len(sorted_real))
    return {
        "train": sorted_real.iloc[:cut].drop(columns=["_ts"]),
        "test": sorted_real.iloc[cut:].drop(columns=["_ts"]),
    }


def tokens(text: Any) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(normalize_summary_text(text))]


def token_lengths(texts: Iterable[Any]) -> List[int]:
    return [len(tokens(text)) for text in texts]


def vocab(token_lists: List[List[str]]) -> set[str]:
    return {token for item in token_lists for token in item}


def distinct_n(token_lists: List[List[str]], n: int) -> float:
    grams = []
    for item in token_lists:
        grams.extend(tuple(item[i : i + n]) for i in range(max(len(item) - n + 1, 0)))
    return float(len(set(grams)) / max(len(grams), 1))


def type_token_ratio(token_lists: List[List[str]]) -> float:
    all_tokens = [token for item in token_lists for token in item]
    return float(len(set(all_tokens)) / max(len(all_tokens), 1))


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def sentiment_score(text: Any) -> float:
    item = tokens(text)
    if not item:
        return 0.0
    pos = sum(1 for token in item if token in POSITIVE_WORDS)
    neg = sum(1 for token in item if token in NEGATIVE_WORDS)
    return float((pos - neg) / max(len(item), 1))


def grouped_mean(frame: pd.DataFrame, values: Any, group_col: str) -> Dict[str, float]:
    if isinstance(values, str):
        series = pd.to_numeric(frame[values], errors="coerce").fillna(0.0)
    else:
        series = pd.Series(values, index=frame.index)
    grouped = series.groupby(frame[group_col]).mean()
    return {str(key): float(value) for key, value in grouped.items()}


def safe_corr(a: Any, b: Any) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def distribution_dict(values: Iterable[Any]) -> Dict[str, float]:
    values = list(values)
    counts = Counter(values)
    total = max(len(values), 1)
    return {str(key): float(count / total) for key, count in sorted(counts.items())}


def special_token_rate(texts: Iterable[str]) -> float:
    markers = ("[MASK]", "[PAD]", "[CLS]", "[SEP]")
    texts = list(texts)
    return float(sum(any(marker in text for marker in markers) for text in texts) / max(len(texts), 1))


def normalize_binary(values: pd.Series) -> np.ndarray:
    return values.map(lambda value: 1.0 if str(value).lower() in {"true", "1", "yes", "y"} else 0.0).to_numpy(dtype=float)
