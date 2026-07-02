"""Simple summary generation baselines for Text V1 comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

import numpy as np

from .masked_summary_dataset import normalize_summary_text


class MarginalSummarySampler:
    """Non-private copying baseline that samples real training summaries."""

    non_private_copying_baseline = True

    def __init__(self, summaries: Iterable[Any], seed: int = 42):
        self.summaries = [normalize_summary_text(text) for text in summaries if normalize_summary_text(text)]
        self.seed = int(seed)

    def sample(self, n: int) -> List[str]:
        rng = np.random.default_rng(self.seed)
        if not self.summaries:
            return [""] * int(n)
        indices = rng.integers(0, len(self.summaries), size=int(n))
        return [self.summaries[int(idx)] for idx in indices]

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "marginal_summary_sampler",
            "non_private_copying_baseline": True,
            "nearest_neighbor_decoder": False,
            "retrieval_augmented_generation": False,
            "contains_training_text_bank": True,
        }


class TemplateSummaryGenerator:
    """Low-quality non-retrieval template baseline conditioned on rating."""

    templates = {
        1: ["not good", "poor product", "very disappointing"],
        2: ["could be better", "not great", "below expectations"],
        3: ["okay product", "works fine", "average item"],
        4: ["good product", "nice item", "works well"],
        5: ["great product", "excellent item", "love it"],
    }

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def sample(self, ratings: Iterable[Any]) -> List[str]:
        rng = np.random.default_rng(self.seed)
        outputs = []
        for rating in ratings:
            try:
                key = int(round(float(rating)))
            except Exception:
                key = 3
            choices = self.templates.get(min(max(key, 1), 5), self.templates[3])
            outputs.append(choices[int(rng.integers(0, len(choices)))])
        return outputs

    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "template_summary_generator",
            "non_private_copying_baseline": False,
            "nearest_neighbor_decoder": False,
            "retrieval_augmented_generation": False,
            "contains_training_text_bank": False,
        }


class ConditionalUnigramSummaryGenerator:
    """Tiny rating-conditioned unigram baseline, not used by the main model."""

    def __init__(self, seed: int = 42, max_tokens: int = 8):
        self.seed = int(seed)
        self.max_tokens = int(max_tokens)
        self.by_rating: Dict[int, List[str]] = defaultdict(list)

    def fit(self, summaries: Iterable[Any], ratings: Iterable[Any]) -> "ConditionalUnigramSummaryGenerator":
        for summary, rating in zip(summaries, ratings):
            try:
                key = int(round(float(rating)))
            except Exception:
                key = 3
            self.by_rating[min(max(key, 1), 5)].extend(normalize_summary_text(summary).lower().split())
        return self

    def sample(self, ratings: Iterable[Any]) -> List[str]:
        rng = np.random.default_rng(self.seed)
        outputs = []
        fallback = [token for tokens in self.by_rating.values() for token in tokens] or ["good", "product"]
        for rating in ratings:
            try:
                key = int(round(float(rating)))
            except Exception:
                key = 3
            vocab = self.by_rating.get(min(max(key, 1), 5), fallback) or fallback
            length = int(rng.integers(2, self.max_tokens + 1))
            outputs.append(" ".join(vocab[int(rng.integers(0, len(vocab)))] for _ in range(length)))
        return outputs
