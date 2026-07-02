"""Privacy and memorization diagnostics for generated summaries."""

from __future__ import annotations

import re
import string
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .masked_summary_dataset import TOKEN_RE, normalize_summary_text


PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def compute_text_privacy_metrics(
    real_summaries: Iterable[Any],
    synthetic_summaries: Iterable[Any],
    test_summaries: Iterable[Any] | None = None,
    sample_size: int = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    real = [normalize_summary_text(text) for text in real_summaries]
    synthetic = [normalize_summary_text(text) for text in synthetic_summaries]
    test = [normalize_summary_text(text) for text in test_summaries] if test_summaries is not None else []
    real_set = set(real)
    real_norm_set = {normalize_for_copy(text) for text in real}
    exact_flags = [text in real_set and bool(text) for text in synthetic]
    norm_flags = [normalize_for_copy(text) in real_norm_set and bool(normalize_for_copy(text)) for text in synthetic]
    nearest = nearest_similarity_summary(real, synthetic, test, sample_size=sample_size, random_state=random_state)
    duplicate_counter = Counter(synthetic)
    duplicate_rows = sum(count for text, count in duplicate_counter.items() if text and count > 1)
    top_duplicates = [
        {"summary": text, "count": int(count)}
        for text, count in duplicate_counter.most_common(20)
        if text and count > 1
    ]
    metrics = {
        "exact_copy_rate": safe_rate(exact_flags),
        "normalized_exact_copy_rate": safe_rate(norm_flags),
        "normalized_exact_copy_rate_by_length_bucket": copy_rate_by_length_bucket(synthetic, norm_flags),
        "nearest_train_levenshtein_similarity_mean": nearest["train_mean"],
        "nearest_train_levenshtein_similarity_p95": nearest["train_p95"],
        "nearest_train_levenshtein_similarity_max": nearest["train_max"],
        "train_vs_test_nearest_gap": nearest["train_vs_test_gap"],
        "ngram_overlap_with_train_3": ngram_overlap(real, synthetic, 3),
        "ngram_overlap_with_train_5": ngram_overlap(real, synthetic, 5),
        "duplicate_synthetic_rate": float(duplicate_rows / max(len(synthetic), 1)),
        "top_duplicate_summaries": top_duplicates,
    }
    return metrics


def nearest_similarity_summary(
    real: List[str],
    synthetic: List[str],
    test: List[str],
    sample_size: int,
    random_state: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(random_state))
    if len(synthetic) > int(sample_size):
        indices = rng.choice(len(synthetic), size=int(sample_size), replace=False)
        synthetic_sample = [synthetic[int(idx)] for idx in indices]
    else:
        synthetic_sample = list(synthetic)
    train_sims = nearest_similarities(real, synthetic_sample)
    test_sims = nearest_similarities(test, synthetic_sample) if test else []
    train_mean = float(np.mean(train_sims)) if train_sims else 0.0
    test_mean = float(np.mean(test_sims)) if test_sims else 0.0
    return {
        "train_mean": train_mean,
        "train_p95": float(np.percentile(train_sims, 95)) if train_sims else 0.0,
        "train_max": float(np.max(train_sims)) if train_sims else 0.0,
        "train_vs_test_gap": float(train_mean - test_mean) if test_sims else None,
    }


def nearest_similarities(real: List[str], synthetic: List[str]) -> List[float]:
    real = [text for text in real if text]
    synthetic = [text for text in synthetic if text]
    if not real or not synthetic:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000)
        train_matrix = vectorizer.fit_transform(real)
        syn_matrix = vectorizer.transform(synthetic)
        nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn.fit(train_matrix)
        distances, _ = nn.kneighbors(syn_matrix)
        return [float(max(0.0, 1.0 - distance)) for distance in distances[:, 0]]
    except Exception:
        capped_real = real[:5000]
        return [max(sequence_similarity(text, candidate) for candidate in capped_real) for text in synthetic]


def sequence_similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(None, a, b).ratio())


def normalize_for_copy(text: Any) -> str:
    text = normalize_summary_text(text).lower().translate(PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def copy_rate_by_length_bucket(summaries: List[str], flags: List[bool]) -> Dict[str, float]:
    buckets = {
        "length_le_2": [],
        "length_3_to_5": [],
        "length_6_to_10": [],
        "length_gt_10": [],
    }
    for text, flag in zip(summaries, flags):
        length = len(TOKEN_RE.findall(text))
        if length <= 2:
            key = "length_le_2"
        elif length <= 5:
            key = "length_3_to_5"
        elif length <= 10:
            key = "length_6_to_10"
        else:
            key = "length_gt_10"
        buckets[key].append(bool(flag))
    return {key: safe_rate(values) for key, values in buckets.items()}


def ngram_overlap(real: Iterable[str], synthetic: Iterable[str], n: int) -> float:
    real_ngrams = set()
    for text in real:
        real_ngrams.update(text_ngrams(text, n))
    syn_ngrams = []
    for text in synthetic:
        syn_ngrams.extend(text_ngrams(text, n))
    if not syn_ngrams:
        return 0.0
    return float(sum(1 for gram in syn_ngrams if gram in real_ngrams) / len(syn_ngrams))


def text_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    tokens = [token.lower() for token in TOKEN_RE.findall(normalize_summary_text(text))]
    if len(tokens) < int(n):
        return []
    return [tuple(tokens[i : i + int(n)]) for i in range(len(tokens) - int(n) + 1)]


def safe_rate(values: Iterable[bool]) -> float:
    values = list(values)
    return float(sum(bool(value) for value in values) / max(len(values), 1))


def privacy_neighbors_table(real_summaries: Iterable[Any], synthetic_frame: pd.DataFrame, text_col: str) -> pd.DataFrame:
    real = [normalize_summary_text(text) for text in real_summaries]
    syn = [normalize_summary_text(text) for text in synthetic_frame[text_col]]
    similarities = nearest_similarities(real, syn)
    real_norm = [normalize_for_copy(text) for text in real]
    rows = []
    for idx, text in enumerate(syn):
        best_summary = ""
        best_similarity = 0.0
        if real:
            if len(real) <= 5000:
                scores = [sequence_similarity(text, candidate) for candidate in real]
                best_index = int(np.argmax(scores))
                best_summary = real[best_index]
                best_similarity = float(scores[best_index])
            elif idx < len(similarities):
                best_similarity = float(similarities[idx])
        rows.append(
            {
                "row_id": idx,
                "generated_summary": text,
                "nearest_train_summary": best_summary,
                "nearest_train_similarity": best_similarity,
                "is_exact_copy": text in real,
                "is_normalized_exact_copy": normalize_for_copy(text) in set(real_norm),
                "summary_length": len(TOKEN_RE.findall(text)),
            }
        )
    return pd.DataFrame(rows)
