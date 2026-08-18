"""Text embedding C2ST metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .c2st import run_binary_classifiers
from .utils import text_hash_embedding, write_json


def text_embedding_c2st_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    text_cfg = ((config.get("evaluation") or {}).get("text") or {})
    columns = text_cfg.get("text_columns") or [
        column for column, cfg in ((config.get("table") or {}).get("columns") or {}).items() if str((cfg or {}).get("type")) == "text"
    ]
    if not columns:
        return {"status": "skipped", "reason": "no_text_columns", "macro_auc": None, "macro_error": None, "per_text_column": {}}
    seed = int((config.get("evaluation") or {}).get("random_seed", 42))
    max_rows = int(text_cfg.get("max_text_rows", 50000))
    model_name = str(text_cfg.get("embedding_model", "deterministic_hash"))
    require_model = bool(text_cfg.get("require_embedding_model", False))
    cache = bool(text_cfg.get("cache_embeddings", True))
    classifiers = ((config.get("evaluation") or {}).get("c2st") or {}).get("classifiers") or ["logistic_regression"]
    per_column: dict[str, Any] = {}
    combined_real: list[np.ndarray] = []
    combined_synthetic: list[np.ndarray] = []
    for column in columns:
        if column not in real or column not in synthetic:
            continue
        n = min(len(real), len(synthetic), max_rows)
        real_text = real[column].sample(n=n, random_state=seed) if len(real) > n else real[column].head(n)
        syn_text = synthetic[column].sample(n=n, random_state=seed + 1) if len(synthetic) > n else synthetic[column].head(n)
        real_emb = embed_texts(
            real_text.tolist(),
            model_name,
            output_dir,
            f"{column}_real",
            cache,
            require_model=require_model,
        )
        syn_emb = embed_texts(
            syn_text.tolist(),
            model_name,
            output_dir,
            f"{column}_synthetic",
            cache,
            require_model=require_model,
        )
        combined_real.append(real_emb)
        combined_synthetic.append(syn_emb)
        x = np.vstack([real_emb, syn_emb])
        y = np.array([1] * len(real_emb) + [0] * len(syn_emb), dtype=int)
        results = run_binary_classifiers(x, y, classifiers, seed=seed)
        best_name = max(results, key=lambda name: results[name].get("auc", 0.5)) if results else None
        best = results.get(best_name, {}) if best_name else {}
        per_column[column] = {
            "auc": best.get("auc"),
            "accuracy": best.get("accuracy"),
            "error": best.get("error"),
            "classifier": best_name,
            "num_real": int(len(real_emb)),
            "num_synthetic": int(len(syn_emb)),
            "balanced_eval_n_real": int(len(real_emb)),
            "balanced_eval_n_synthetic": int(len(syn_emb)),
            "embedding_model": model_name,
            "feature_names": [f"embedding_dim_{idx}" for idx in range(real_emb.shape[1])] if real_emb.ndim == 2 else [],
            "per_classifier": results,
        }
    errors = [item["error"] for item in per_column.values() if item.get("error") is not None]
    aucs = [item["auc"] for item in per_column.values() if item.get("auc") is not None]
    combined = combined_text_c2st(
        combined_real,
        combined_synthetic,
        classifiers=classifiers,
        seed=seed,
        columns=list(per_column),
        model_name=model_name,
    )
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "macro_error": float(np.mean(errors)) if errors else None,
        "per_text_column": per_column,
        "combined_text_fields": combined,
    }


def combined_text_c2st(
    real_embeddings: list[np.ndarray],
    synthetic_embeddings: list[np.ndarray],
    *,
    classifiers: list[str],
    seed: int,
    columns: list[str],
    model_name: str,
) -> dict[str, Any]:
    if len(real_embeddings) < 2:
        return {
            "status": "not_applicable",
            "reason": "fewer_than_two_text_fields",
            "columns": columns,
            "error": None,
        }
    n = min(
        *(len(values) for values in real_embeddings),
        *(len(values) for values in synthetic_embeddings),
    )
    real = np.concatenate([values[:n] for values in real_embeddings], axis=1)
    synthetic = np.concatenate(
        [values[:n] for values in synthetic_embeddings], axis=1
    )
    x = np.vstack([real, synthetic])
    y = np.array([1] * n + [0] * n, dtype=int)
    results = run_binary_classifiers(x, y, classifiers, seed=seed)
    best_name = (
        max(results, key=lambda name: results[name].get("auc", 0.5))
        if results
        else None
    )
    best = results.get(best_name, {}) if best_name else {}
    return {
        "status": "computed",
        "columns": columns,
        "embedding_model": model_name,
        "fusion": "concatenated per-field embeddings",
        "auc": best.get("auc"),
        "accuracy": best.get("accuracy"),
        "error": best.get("error"),
        "classifier": best_name,
        "num_real": int(n),
        "num_synthetic": int(n),
        "per_classifier": results,
    }


def embed_texts(
    texts: list[Any],
    model_name: str,
    output_dir: str | Path,
    cache_key: str,
    cache: bool,
    *,
    require_model: bool = False,
) -> np.ndarray:
    fingerprint = text_embedding_fingerprint(texts, model_name)
    cache_path = (
        Path(output_dir)
        / "embedding_cache"
        / f"{cache_key}_{fingerprint[:16]}.npy"
    )
    metadata_path = cache_path.with_suffix(".json")
    if cache and cache_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("embedding_model") == model_name
            and int(metadata.get("num_texts", -1)) == len(texts)
            and (
                not require_model
                or metadata.get("embedding_backend") == model_name
            )
        ):
            return np.load(cache_path)
    embeddings: np.ndarray
    embedding_backend = model_name
    if model_name not in {"dummy", "deterministic_hash", "hash"}:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            embeddings = np.asarray(model.encode([str(text) for text in texts], show_progress_bar=False), dtype=float)
        except Exception as exc:
            if require_model:
                raise RuntimeError(
                    f"Required text embedding model {model_name!r} could not "
                    "be loaded; refusing to replace the paper metric with a "
                    "hash embedding."
                ) from exc
            embeddings = hash_embeddings(texts)
            embedding_backend = "deterministic_hash_fallback"
    else:
        embeddings = hash_embeddings(texts)
        embedding_backend = "deterministic_hash"
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        write_json(
            {
                "embedding_model": model_name,
                "embedding_backend": embedding_backend,
                "cache_key": cache_key,
                "shape": list(embeddings.shape),
                "num_texts": int(len(texts)),
                "fingerprint": fingerprint,
            },
            metadata_path,
        )
    return embeddings


def hash_embeddings(texts: list[Any], dim: int = 64) -> np.ndarray:
    return np.vstack([text_hash_embedding(text, dim=dim) for text in texts])


def text_embedding_fingerprint(texts: list[Any], model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(model_name).encode("utf-8"))
    digest.update(str(len(texts)).encode("ascii"))
    for text in texts:
        encoded = str(text).encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()
