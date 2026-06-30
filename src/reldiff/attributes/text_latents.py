"""Text latent extraction and nearest-neighbor latent decoding."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def safe_column_name(column: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", column)


class TextLatentEncoder:
    """Extract latent vectors for text columns with an offline-safe fallback.

    ``backend="auto"`` first tries locally available sentence-transformer or
    HuggingFace transformer weights. If neither is available locally, it uses a
    deterministic hashing encoder so tests and offline runs still work.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        backend: str = "auto",
        latent_dim: int = 384,
        batch_size: int = 64,
        device: str = "cpu",
        local_files_only: bool = True,
    ):
        if backend not in {"auto", "sentence_transformers", "transformers", "hashing"}:
            raise ValueError(
                "backend must be one of: auto, sentence_transformers, transformers, hashing."
            )
        self.model_name = model_name
        self.requested_backend = backend
        self.backend = "hashing"
        self.latent_dim = int(latent_dim)
        self.batch_size = int(batch_size)
        self.device = device
        self.local_files_only = local_files_only
        self._sentence_model = None
        self._tokenizer = None
        self._transformer_model = None
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        if self.requested_backend in {"auto", "sentence_transformers"}:
            try:
                from sentence_transformers import SentenceTransformer

                self._sentence_model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=None,
                    local_files_only=self.local_files_only,
                )
                dim = self._sentence_model.get_sentence_embedding_dimension()
                self.latent_dim = int(dim) if dim is not None else self.latent_dim
                self.backend = "sentence_transformers"
                return
            except Exception:
                if self.requested_backend == "sentence_transformers":
                    raise

        if self.requested_backend in {"auto", "transformers"}:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=self.local_files_only
                )
                self._transformer_model = AutoModel.from_pretrained(
                    self.model_name, local_files_only=self.local_files_only
                ).to(self.device)
                self._transformer_model.eval()
                hidden_size = getattr(self._transformer_model.config, "hidden_size", None)
                if hidden_size is not None:
                    self.latent_dim = int(hidden_size)
                self.backend = "transformers"
                return
            except Exception:
                if self.requested_backend == "transformers":
                    raise

        self.backend = "hashing"

    def encode(self, texts: Iterable[Any], max_length: Optional[int] = None) -> np.ndarray:
        values = ["" if pd.isna(text) else str(text) for text in texts]
        if self.backend == "sentence_transformers":
            return self._encode_sentence_transformers(values)
        if self.backend == "transformers":
            return self._encode_transformers(values, max_length=max_length)
        return self._encode_hashing(values, max_length=max_length)

    def fit_transform_columns(
        self,
        reviews: pd.DataFrame,
        text_cols: List[str],
        cache_dir: str | Path,
        max_lengths: Optional[Dict[str, int]] = None,
        force_recompute: bool = False,
    ) -> Dict[str, np.ndarray]:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        max_lengths = max_lengths or {}
        latents: Dict[str, np.ndarray] = {}
        metadata = {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "model_name": self.model_name,
            "latent_dim": self.latent_dim,
            "num_rows": int(len(reviews)),
            "columns": {},
        }

        for column in text_cols:
            path = cache_dir / f"{safe_column_name(column)}_latents.npy"
            if path.exists() and not force_recompute:
                encoded = np.load(path)
                if len(encoded) == len(reviews):
                    latents[column] = encoded.astype(np.float32)
                    metadata["columns"][column] = {
                        "cache_file": path.name,
                        "max_length": max_lengths.get(column),
                        "loaded_from_cache": True,
                    }
                    continue

            encoded = self.encode(
                reviews[column] if column in reviews.columns else [""] * len(reviews),
                max_length=max_lengths.get(column),
            ).astype(np.float32)
            np.save(path, encoded)
            latents[column] = encoded
            metadata["columns"][column] = {
                "cache_file": path.name,
                "max_length": max_lengths.get(column),
                "loaded_from_cache": False,
            }

        with (cache_dir / "text_latent_metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        return latents

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "model_name": self.model_name,
            "latent_dim": self.latent_dim,
            "local_files_only": self.local_files_only,
        }

    def _encode_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        encoded = self._sentence_model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(encoded, dtype=np.float32)

    def _encode_transformers(
        self, texts: List[str], max_length: Optional[int] = None
    ) -> np.ndarray:
        import torch

        outputs = []
        max_length = max_length or 256
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.no_grad():
                model_out = self._transformer_model(**tokens)
                hidden = model_out.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            outputs.append(pooled.cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def _encode_hashing(
        self, texts: List[str], max_length: Optional[int] = None
    ) -> np.ndarray:
        max_length = max_length or 256
        encoded = np.zeros((len(texts), self.latent_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[A-Za-z0-9_']+", text.lower())[:max_length]
            if not tokens:
                continue
            vectors = []
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
                rng = np.random.default_rng(seed)
                vector = rng.normal(0.0, 1.0, size=self.latent_dim).astype(np.float32)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                vectors.append(vector)
            encoded[row] = np.mean(vectors, axis=0)
        return encoded


class TextLatentNormalizer:
    """Column-wise latent standardization."""

    def __init__(self):
        self.stats: Dict[str, Dict[str, np.ndarray]] = {}

    def fit_transform(self, latents: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        normalized = {}
        for column, values in latents.items():
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            self.stats[column] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
            normalized[column] = ((values - mean) / std).astype(np.float32)
        return normalized

    def transform(self, column: str, values: np.ndarray) -> np.ndarray:
        stats = self.stats[column]
        return ((values - stats["mean"]) / stats["std"]).astype(np.float32)

    def inverse_transform(self, column: str, values: np.ndarray) -> np.ndarray:
        stats = self.stats[column]
        return (values * stats["std"] + stats["mean"]).astype(np.float32)

    def to_checkpoint(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self.stats

    @classmethod
    def from_checkpoint(
        cls, stats: Dict[str, Dict[str, np.ndarray]]
    ) -> "TextLatentNormalizer":
        normalizer = cls()
        normalizer.stats = {
            column: {
                "mean": np.asarray(values["mean"], dtype=np.float32),
                "std": np.asarray(values["std"], dtype=np.float32),
            }
            for column, values in stats.items()
        }
        return normalizer


class NearestNeighborTextDecoder:
    """Decode generated text latents by retrieving nearby training text."""

    def __init__(
        self,
        train_latents: Dict[str, np.ndarray],
        train_texts: Dict[str, List[str]],
        train_categories: Optional[Dict[str, List[Any]]] = None,
        metric: str = "cosine",
        min_conditioned_candidates: int = 3,
    ):
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'.")
        self.train_latents = {
            column: np.asarray(values, dtype=np.float32)
            for column, values in train_latents.items()
        }
        self.train_texts = train_texts
        self.train_categories = train_categories or {}
        self.metric = metric
        self.min_conditioned_candidates = int(min_conditioned_candidates)
        self._normalized_train = {
            column: self._normalize(values)
            for column, values in self.train_latents.items()
        }

    def decode(
        self,
        column: str,
        generated_latents: np.ndarray,
        generated_categories: Optional[Dict[str, List[Any]]] = None,
    ) -> List[str]:
        generated_latents = np.asarray(generated_latents, dtype=np.float32)
        decoded = []
        train_texts = self.train_texts[column]
        train_latents = self.train_latents[column]
        normalized_train = self._normalized_train[column]
        normalized_generated = self._normalize(generated_latents)

        for row in range(len(generated_latents)):
            candidates = self._candidate_indices(row, generated_categories)
            if self.metric == "cosine":
                scores = normalized_train[candidates] @ normalized_generated[row]
                choice = candidates[int(np.argmax(scores))]
            else:
                distances = np.linalg.norm(
                    train_latents[candidates] - generated_latents[row][None, :], axis=1
                )
                choice = candidates[int(np.argmin(distances))]
            decoded.append("" if pd.isna(train_texts[choice]) else str(train_texts[choice]))
        return decoded

    def _candidate_indices(
        self, row: int, generated_categories: Optional[Dict[str, List[Any]]]
    ) -> np.ndarray:
        total = len(next(iter(self.train_texts.values())))
        candidates = np.arange(total)
        if not generated_categories:
            return candidates

        mask = np.ones(total, dtype=bool)
        for column, generated_values in generated_categories.items():
            if column not in self.train_categories:
                continue
            train_values = np.asarray(self.train_categories[column], dtype=object)
            mask &= train_values == generated_values[row]
        conditioned = np.where(mask)[0]
        if len(conditioned) >= self.min_conditioned_candidates:
            return conditioned
        return candidates

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.clip(norms, 1e-8, None)
