"""Tokenizers and deterministic encoders for conditional TABDLM."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip())


def normalize_category(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, float) and np.isnan(value):
        return "<missing>"
    return str(value)


def stable_hash_bucket(column_name: str, raw_id: Any, num_buckets: int) -> int:
    payload = f"{column_name}\x1f{raw_id}".encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % int(num_buckets)


@dataclass
class CategoryVocab:
    column: str
    token_to_id: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    @property
    def id_to_token(self) -> dict[int, str]:
        return {idx: token for token, idx in self.token_to_id.items()}

    @property
    def mask_id(self) -> int:
        return self.size

    def encode(self, value: Any) -> int:
        token = normalize_category(value)
        if token in self.token_to_id:
            return int(self.token_to_id[token])
        return 0

    def decode(self, idx: int) -> str:
        return self.id_to_token.get(int(idx), self.id_to_token.get(0, "<missing>"))

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "token_to_id": dict(self.token_to_id)}

    @classmethod
    def from_values(cls, column: str, values: Iterable[Any]) -> "CategoryVocab":
        counts: dict[str, int] = {}
        for value in values:
            token = normalize_category(value)
            counts[token] = counts.get(token, 0) + 1
        tokens = sorted(counts, key=lambda token: (-counts[token], token))
        if "<missing>" not in tokens:
            tokens.insert(0, "<missing>")
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(column=column, token_to_id=token_to_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CategoryVocab":
        return cls(
            column=str(data["column"]),
            token_to_id={str(k): int(v) for k, v in data["token_to_id"].items()},
        )


class SimpleTextTokenizer:
    """Small deterministic text tokenizer with token-level truncation."""

    pad_token = "[PAD]"
    bos_token = "[BOS]"
    mask_token = "[MASK]"
    unk_token = "[UNK]"
    eos_token = "[EOS]"
    empty_token = "<empty>"

    def __init__(self, vocab: dict[str, int] | None = None, lowercase: bool = True):
        self.lowercase = bool(lowercase)
        if vocab is None:
            protected = [
                self.pad_token,
                self.bos_token,
                self.mask_token,
                self.unk_token,
                self.eos_token,
                self.empty_token,
            ]
            vocab = {token: idx for idx, token in enumerate(protected)}
        self.vocab = dict(vocab)
        for token in [
            self.pad_token,
            self.bos_token,
            self.mask_token,
            self.unk_token,
            self.eos_token,
            self.empty_token,
        ]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def mask_id(self) -> int:
        return self.vocab[self.mask_token]

    @property
    def bos_id(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def unk_id(self) -> int:
        return self.vocab[self.unk_token]

    @property
    def eos_id(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: Any) -> list[str]:
        text = normalize_text(text)
        if self.lowercase:
            text = text.lower()
        tokens = TOKEN_RE.findall(text)
        return tokens if tokens else [self.empty_token]

    def fit(
        self,
        texts: Iterable[Any],
        max_vocab_size: int = 30000,
        min_frequency: int = 1,
    ) -> "SimpleTextTokenizer":
        counts: dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        protected = set(self.vocab)
        candidates = sorted(
            (
                (token, count)
                for token, count in counts.items()
                if count >= int(min_frequency) and token not in protected
            ),
            key=lambda item: (-item[1], item[0]),
        )
        for token, _ in candidates:
            if len(self.vocab) >= int(max_vocab_size):
                break
            self.vocab[token] = len(self.vocab)
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        return self

    def encode(self, text: Any, max_length: int) -> tuple[list[int], list[int]]:
        max_length = int(max_length)
        if max_length < 2:
            raise ValueError("max_length must be at least 2 for BOS/EOS")
        tokens = self.tokenize(text)[: self.max_content_tokens(max_length)]
        ids = [self.bos_id]
        ids.extend(self.vocab.get(token, self.unk_id) for token in tokens)
        ids.append(self.eos_id)
        ids = ids[:max_length]
        attention = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.pad_id)
            attention.append(1)
        return ids, attention

    def decode(self, ids: Iterable[int]) -> str:
        tokens: list[str] = []
        for idx in ids:
            token = self.inv_vocab.get(int(idx), self.unk_token)
            if token == self.eos_token:
                break
            if token == self.pad_token:
                break
            if token in {self.bos_token, self.mask_token, self.unk_token, self.empty_token}:
                continue
            tokens.append(token)
        return clean_detokenized(tokens)

    def content_length(self, ids: Iterable[int]) -> int:
        length = 0
        for idx in ids:
            idx = int(idx)
            if idx in {self.bos_id, self.pad_id, self.mask_id}:
                continue
            if idx == self.eos_id:
                break
            length += 1
        return int(length)

    def max_content_tokens(self, max_length: int) -> int:
        return max(0, int(max_length) - 2)

    @property
    def special_ids(self) -> set[int]:
        return {self.pad_id, self.bos_id, self.mask_id, self.unk_id, self.eos_id}

    @property
    def content_token_ids(self) -> list[int]:
        ids = [idx for token, idx in self.vocab.items() if idx not in self.special_ids and token != self.empty_token]
        if ids:
            return sorted(ids)
        return [self.unk_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "simple_text_tokenizer",
            "lowercase": self.lowercase,
            "vocab": dict(self.vocab),
            "special_ids": {
                "pad": self.pad_id,
                "bos": self.bos_id,
                "mask": self.mask_id,
                "unk": self.unk_id,
                "eos": self.eos_id,
            },
            "summary_vocab_size": self.vocab_size,
            "summary_pad_token_id": self.pad_id,
            "summary_bos_token_id": self.bos_id,
            "summary_eos_token_id": self.eos_id,
            "summary_unk_token_id": self.unk_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimpleTextTokenizer":
        return cls(
            vocab={str(k): int(v) for k, v in data["vocab"].items()},
            lowercase=bool(data.get("lowercase", True)),
        )


def clean_detokenized(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    return re.sub(r"\s+", " ", text).strip()


def default_summary_length_buckets() -> dict[str, tuple[int, int]]:
    return {
        "len_0": (0, 0),
        "len_1_2": (1, 2),
        "len_3_5": (3, 5),
        "len_6_10": (6, 10),
        "len_11_16": (11, 16),
        "len_17_32": (17, 32),
    }


def summary_length_bucket_name(content_length: int, buckets: dict[str, tuple[int, int]] | None = None) -> str:
    buckets = buckets or default_summary_length_buckets()
    content_length = int(content_length)
    for name, (low, high) in buckets.items():
        if int(low) <= content_length <= int(high):
            return str(name)
    return list(buckets)[-1]


def sample_length_from_bucket(
    bucket_name: str,
    buckets: dict[str, tuple[int, int]],
    max_content_tokens: int,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random
    low, high = buckets[str(bucket_name)]
    low = max(0, int(low))
    high = min(int(high), int(max_content_tokens))
    if high < low:
        high = low
    return int(rng.randint(low, high))
