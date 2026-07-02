"""Tokenization and masked-summary datasets for Text V1."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class SimpleSummaryTokenizer:
    """Small deterministic tokenizer used when no pretrained tokenizer is available."""

    pad_token = "[PAD]"
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    mask_token = "[MASK]"
    unk_token = "[UNK]"
    empty_summary_token = "<empty_summary>"

    def __init__(self, vocab: Optional[Dict[str, int]] = None, lowercase: bool = True):
        self.lowercase = bool(lowercase)
        if vocab is None:
            tokens = [
                self.pad_token,
                self.cls_token,
                self.sep_token,
                self.mask_token,
                self.unk_token,
                self.empty_summary_token,
            ]
            vocab = {token: idx for idx, token in enumerate(tokens)}
        self.vocab = dict(vocab)
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def cls_token_id(self) -> int:
        return self.vocab[self.cls_token]

    @property
    def sep_token_id(self) -> int:
        return self.vocab[self.sep_token]

    @property
    def mask_token_id(self) -> int:
        return self.vocab[self.mask_token]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def special_token_ids(self) -> Dict[str, int]:
        return {
            "pad": self.pad_token_id,
            "cls": self.cls_token_id,
            "sep": self.sep_token_id,
            "mask": self.mask_token_id,
            "unk": self.unk_token_id,
        }

    def tokenize(self, text: Any) -> List[str]:
        text = normalize_summary_text(text)
        if not text:
            return [self.empty_summary_token]
        if self.lowercase:
            text = text.lower()
        return TOKEN_RE.findall(text) or [self.empty_summary_token]

    def fit(self, texts: Iterable[Any], max_vocab_size: int = 30000, min_freq: int = 1) -> "SimpleSummaryTokenizer":
        counts: Dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        protected = set(self.vocab)
        sorted_tokens = sorted(
            ((token, count) for token, count in counts.items() if count >= int(min_freq) and token not in protected),
            key=lambda item: (-item[1], item[0]),
        )
        for token, _ in sorted_tokens:
            if len(self.vocab) >= int(max_vocab_size):
                break
            self.vocab[token] = len(self.vocab)
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        return self

    def encode_summary(self, text: Any, max_summary_tokens: int) -> List[int]:
        tokens = self.tokenize(text)[: int(max_summary_tokens)]
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def pad_content_ids(self, ids: List[int], max_summary_tokens: int) -> tuple[List[int], List[int]]:
        ids = list(ids)[: int(max_summary_tokens)]
        mask = [1] * len(ids)
        while len(ids) < int(max_summary_tokens):
            ids.append(self.pad_token_id)
            mask.append(0)
        return ids, mask

    def decode_summary(self, ids: Iterable[int]) -> str:
        tokens: List[str] = []
        for idx in ids:
            token = self.inv_vocab.get(int(idx), self.unk_token)
            if token == self.sep_token:
                break
            if token in {
                self.pad_token,
                self.cls_token,
                self.mask_token,
                self.unk_token,
            }:
                continue
            if token == self.empty_summary_token:
                continue
            tokens.append(token)
        return clean_detokenized(tokens)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "simple_summary_tokenizer",
            "lowercase": self.lowercase,
            "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleSummaryTokenizer":
        return cls(vocab={str(k): int(v) for k, v in data["vocab"].items()}, lowercase=bool(data.get("lowercase", True)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleSummaryTokenizer":
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))


class MaskedSummaryDataset(Dataset):
    """Returns masked content tokens plus continuous conditioning features."""

    def __init__(
        self,
        frame,
        tokenizer: SimpleSummaryTokenizer,
        condition_features: np.ndarray,
        text_col: str = "summary",
        max_summary_tokens: int = 32,
        min_mask_prob: float = 0.15,
        max_mask_prob: float = 0.85,
        mask_schedule: str = "linear",
        seed: int = 42,
    ):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.condition_features = np.asarray(condition_features, dtype=np.float32)
        self.text_col = text_col
        self.max_summary_tokens = int(max_summary_tokens)
        self.min_mask_prob = float(min_mask_prob)
        self.max_mask_prob = float(max_mask_prob)
        self.mask_schedule = mask_schedule
        self.seed = int(seed)
        if len(self.frame) != len(self.condition_features):
            raise ValueError("frame and condition_features must have the same length")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[int(index)]
        ids = self.tokenizer.encode_summary(row.get(self.text_col, ""), self.max_summary_tokens)
        input_ids, attention_mask = self.tokenizer.pad_content_ids(ids, self.max_summary_tokens)
        original = np.asarray(input_ids, dtype=np.int64)
        attention = np.asarray(attention_mask, dtype=np.int64)
        rng = np.random.default_rng(self.seed + int(index) * 1009)
        t = float(rng.uniform(0.0, 1.0))
        if self.mask_schedule == "cosine":
            rate = self.min_mask_prob + (1.0 - np.cos(t * np.pi / 2.0)) * (self.max_mask_prob - self.min_mask_prob)
        else:
            rate = self.min_mask_prob + t * (self.max_mask_prob - self.min_mask_prob)
        candidate_positions = np.where(attention == 1)[0]
        mask_flags = np.zeros_like(original, dtype=bool)
        if len(candidate_positions):
            draws = rng.uniform(size=len(candidate_positions)) < rate
            if not bool(draws.any()):
                draws[int(rng.integers(0, len(candidate_positions)))] = True
            mask_flags[candidate_positions] = draws
        noisy = original.copy()
        noisy[mask_flags] = self.tokenizer.mask_token_id
        labels = np.full_like(original, fill_value=-100)
        labels[mask_flags] = original[mask_flags]
        return {
            "input_ids": torch.tensor(noisy, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "condition_features": torch.tensor(self.condition_features[int(index)], dtype=torch.float32),
            "row_id": torch.tensor(int(index), dtype=torch.long),
        }


def normalize_summary_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    text = str(text).strip()
    return re.sub(r"\s+", " ", text)


def clean_detokenized(tokens: List[str]) -> str:
    text = " ".join(tokens).strip()
    text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
