"""Text embedding C2ST metrics."""

from __future__ import annotations

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
    cache = bool(text_cfg.get("cache_embeddings", True))
    classifiers = ((config.get("evaluation") or {}).get("c2st") or {}).get("classifiers") or ["logistic_regression"]
    per_column: dict[str, Any] = {}
    for column in columns:
        n = min(len(real), len(synthetic), max_rows)
        real_text = real[column].sample(n=n, random_state=seed) if len(real) > n else real[column].head(n)
        syn_text = synthetic[column].sample(n=n, random_state=seed + 1) if len(synthetic) > n else synthetic[column].head(n)
        real_emb = embed_texts(real_text.tolist(), model_name, output_dir, f"{column}_real", cache)
        syn_emb = embed_texts(syn_text.tolist(), model_name, output_dir, f"{column}_synthetic", cache)
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
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "macro_error": float(np.mean(errors)) if errors else None,
        "per_text_column": per_column,
    }


def embed_texts(texts: list[Any], model_name: str, output_dir: str | Path, cache_key: str, cache: bool) -> np.ndarray:
    cache_path = Path(output_dir) / "embedding_cache" / f"{cache_key}.npy"
    if cache and cache_path.exists():
        return np.load(cache_path)
    embeddings: np.ndarray
    if model_name not in {"dummy", "deterministic_hash", "hash"}:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            embeddings = np.asarray(model.encode([str(text) for text in texts], show_progress_bar=False), dtype=float)
        except Exception:
            embeddings = hash_embeddings(texts)
    else:
        embeddings = hash_embeddings(texts)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        write_json({"embedding_model": model_name, "cache_key": cache_key, "shape": list(embeddings.shape)}, cache_path.with_suffix(".json"))
    return embeddings


def hash_embeddings(texts: list[Any], dim: int = 64) -> np.ndarray:
    return np.vstack([text_hash_embedding(text, dim=dim) for text in texts])
