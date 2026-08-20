"""Deterministic building blocks for text-C2ST reproducibility audits."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from evaluation.paper_metrics.utils import text_hash_embedding


TEXT_FIELDS = ("summary", "review_text")


def c2st_error(auc: float) -> float:
    return float(2.0 * abs(float(auc) - 0.5))


def implied_auc(error: float) -> float:
    return float(0.5 + float(error) / 2.0)


def canonical_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    normalized = unicodedata.normalize("NFC", str(value))
    return " ".join(normalized.strip().split())


def historical_minilm_text(value: Any) -> str:
    """The historical MiniLM branch called ``str`` without normalization."""

    return str(value)


def preprocess_texts(values: Iterable[Any], policy: str) -> list[str]:
    if policy == "raw_str":
        return [historical_minilm_text(value) for value in values]
    if policy == "canonical":
        return [canonical_text(value) for value in values]
    if policy == "historical_hash":
        return list(values)
    raise ValueError(f"Unknown text preprocessing policy: {policy}")


def length_prefixed_sha256(values: Iterable[Any], *, canonical: bool) -> str:
    digest = hashlib.sha256()
    for value in values:
        text = canonical_text(value) if canonical else historical_minilm_text(value)
        encoded = text.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_csv(path: str | Path, text_fields: Iterable[str] = TEXT_FIELDS) -> tuple[dict[str, Any], pd.DataFrame]:
    resolved = Path(path).resolve()
    frame = pd.read_csv(resolved, low_memory=False)
    record: dict[str, Any] = {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "row_count": int(len(frame)),
        "file_size_bytes": int(resolved.stat().st_size),
        "columns": json.dumps(list(frame.columns)),
    }
    for field in text_fields:
        if field not in frame:
            record[f"{field}_canonical_sha256"] = None
            record[f"{field}_raw_sha256"] = None
            continue
        record[f"{field}_canonical_sha256"] = length_prefixed_sha256(
            frame[field], canonical=True
        )
        record[f"{field}_raw_sha256"] = length_prefixed_sha256(
            frame[field], canonical=False
        )
    return record, frame


def compare_text_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    fields: Iterable[str] = TEXT_FIELDS,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "same_row_count": bool(len(left) == len(right)),
        "fields": {},
    }
    n = min(len(left), len(right))
    for field in fields:
        if field not in left or field not in right:
            result["fields"][field] = {"status": "missing"}
            continue
        left_raw = left[field].head(n).map(historical_minilm_text)
        right_raw = right[field].head(n).map(historical_minilm_text)
        left_canonical = left[field].head(n).map(canonical_text)
        right_canonical = right[field].head(n).map(canonical_text)
        raw_equal = left_raw.eq(right_raw)
        canonical_equal = left_canonical.eq(right_canonical)
        left_lengths = left_canonical.str.split().map(len).astype(float)
        right_lengths = right_canonical.str.split().map(len).astype(float)
        left_multiset = sorted(left_canonical.tolist())
        right_multiset = sorted(right_canonical.tolist())
        result["fields"][field] = {
            "rows_compared": int(n),
            "identical_raw_fraction": float(raw_equal.mean()) if n else None,
            "identical_canonical_fraction": (
                float(canonical_equal.mean()) if n else None
            ),
            "changed_rows_raw": int((~raw_equal).sum()),
            "changed_rows_canonical": int((~canonical_equal).sum()),
            "same_canonical_multiset": bool(left_multiset == right_multiset),
            "row_order_only_difference": bool(
                left_multiset == right_multiset and not canonical_equal.all()
            ),
            "format_or_null_only_rows": int(
                ((~raw_equal) & canonical_equal).sum()
            ),
            "left_token_length_mean": float(left_lengths.mean()) if n else None,
            "right_token_length_mean": float(right_lengths.mean()) if n else None,
            "left_token_length_std": float(left_lengths.std(ddof=0)) if n else None,
            "right_token_length_std": float(right_lengths.std(ddof=0)) if n else None,
            "left_token_length_median": float(left_lengths.median()) if n else None,
            "right_token_length_median": float(right_lengths.median()) if n else None,
            "left_token_length_p95": float(left_lengths.quantile(0.95)) if n else None,
            "right_token_length_p95": float(right_lengths.quantile(0.95)) if n else None,
            "left_token_length_min": float(left_lengths.min()) if n else None,
            "right_token_length_min": float(right_lengths.min()) if n else None,
            "left_token_length_max": float(left_lengths.max()) if n else None,
            "right_token_length_max": float(right_lengths.max()) if n else None,
        }
    return result


@dataclass(frozen=True)
class TextC2STProtocol:
    name: str
    embedding_backend: str
    embedding_model: str
    preprocessing: str
    classifiers: tuple[str, ...]
    max_rows: int
    seed: int = 42
    n_splits: int = 5
    aggregation: str = "mean_per_field_normalized_error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "preprocessing": self.preprocessing,
            "classifiers": list(self.classifiers),
            "max_rows": int(self.max_rows),
            "seed": int(self.seed),
            "n_splits": int(self.n_splits),
            "aggregation": self.aggregation,
            "error_formula": "2 * abs(AUC - 0.5)",
        }


class EmbeddingStore:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        device: str = "auto",
        existing_cache_roots: Iterable[str | Path] = (),
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.existing_cache_roots = [Path(path) for path in existing_cache_roots]
        self._model: Any = None
        self._model_name: str | None = None
        self.model_metadata: dict[str, Any] = {}
        self.cache_events: list[dict[str, Any]] = []

    def embed(
        self,
        values: Iterable[Any],
        *,
        backend: str,
        model_name: str,
        preprocessing: str,
        label: str,
    ) -> np.ndarray:
        original = list(values)
        prepared = preprocess_texts(original, preprocessing)
        fingerprint = embedding_fingerprint(prepared, backend, model_name)
        path = self.cache_dir / f"{safe_name(label)}_{fingerprint[:20]}.npy"
        metadata_path = path.with_suffix(".json")
        cached = load_valid_cache(path, metadata_path, fingerprint, len(prepared))
        if cached is not None:
            self.cache_events.append({"label": label, "source": str(path)})
            return cached
        for candidate_metadata in self.cache_dir.glob(
            f"*_{fingerprint[:20]}.json"
        ):
            candidate = load_valid_cache(
                candidate_metadata.with_suffix(".npy"),
                candidate_metadata,
                fingerprint,
                len(prepared),
            )
            if candidate is not None:
                self.cache_events.append(
                    {"label": label, "source": str(candidate_metadata.with_suffix('.npy'))}
                )
                return candidate
        external = self._find_external_cache(
            original,
            model_name=model_name,
            preprocessing=preprocessing,
        )
        if external is not None and backend == "minilm":
            embeddings, source = external
            self.cache_events.append({"label": label, "source": str(source)})
        elif backend == "deterministic_hash":
            hash_inputs = original if preprocessing == "historical_hash" else prepared
            embeddings = np.vstack(
                [text_hash_embedding(value, dim=64) for value in hash_inputs]
            ).astype(np.float64)
            self.cache_events.append({"label": label, "source": "computed_hash"})
        elif backend == "minilm":
            model = self._get_model(model_name)
            embeddings = np.asarray(
                model.encode(
                    prepared,
                    batch_size=32,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                ),
                dtype=np.float64,
            )
            self.cache_events.append({"label": label, "source": "computed_minilm"})
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")
        np.save(path, embeddings)
        metadata_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "num_texts": len(prepared),
                    "shape": list(embeddings.shape),
                    "backend": backend,
                    "model_name": model_name,
                    "preprocessing": preprocessing,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return embeddings

    def _get_model(self, model_name: str) -> Any:
        if self._model is not None and self._model_name == model_name:
            return self._model
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if self.device != "auto":
            kwargs["device"] = self.device
        self._model = SentenceTransformer(model_name, **kwargs)
        self._model_name = model_name
        first = self._model[0] if len(self._model) else None
        auto_model = getattr(first, "auto_model", None)
        config = getattr(auto_model, "config", None)
        tokenizer = getattr(first, "tokenizer", None)
        try:
            model_dtype = str(next(auto_model.parameters()).dtype)
        except (AttributeError, StopIteration, TypeError):
            model_dtype = None
        self.model_metadata = {
            "library": "sentence-transformers",
            "library_version": module_version("sentence_transformers"),
            "transformers_version": module_version("transformers"),
            "torch_version": module_version("torch"),
            "model_name": model_name,
            "revision": getattr(config, "_commit_hash", None),
            "embedding_dimension": self._model.get_sentence_embedding_dimension(),
            "max_sequence_length": getattr(self._model, "max_seq_length", None),
            "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
            "truncation_behavior": (
                "SentenceTransformer tokenization truncates to max_sequence_length"
            ),
            "device": str(getattr(self._model, "device", self.device)),
            "model_dtype": model_dtype,
            "output_dtype_after_evaluator_cast": "float64",
            "encode_batch_size": 32,
            "normalize_embeddings": False,
            "pooling": describe_pooling(self._model),
        }
        return self._model

    def _find_external_cache(
        self,
        original: list[Any],
        *,
        model_name: str,
        preprocessing: str,
    ) -> tuple[np.ndarray, Path] | None:
        if preprocessing != "raw_str":
            return None
        old_fingerprint = historical_embedding_fingerprint(original, model_name)
        for root in self.existing_cache_roots:
            if not root.exists():
                continue
            for metadata_path in root.rglob(f"*_{old_fingerprint[:16]}.json"):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                array_path = metadata_path.with_suffix(".npy")
                if (
                    metadata.get("fingerprint") == old_fingerprint
                    and metadata.get("embedding_backend") == model_name
                    and int(metadata.get("num_texts", -1)) == len(original)
                    and array_path.is_file()
                ):
                    values = np.load(array_path, mmap_mode="r")
                    if values.ndim == 2 and values.shape[1] == 384:
                        return np.asarray(values), array_path
        return None


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def evaluate_protocol(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    protocol: TextC2STProtocol,
    store: EmbeddingStore,
    *,
    fields: Iterable[str] = TEXT_FIELDS,
    label: str,
) -> dict[str, Any]:
    prepared = prepare_protocol_embeddings(
        real,
        synthetic,
        protocol,
        store,
        fields=fields,
        label=label,
    )
    return evaluate_prepared_embeddings(
        prepared,
        protocol.classifiers,
        seed=protocol.seed,
        n_splits=protocol.n_splits,
        protocol=protocol,
    )


def prepare_protocol_embeddings(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    protocol: TextC2STProtocol,
    store: EmbeddingStore,
    *,
    fields: Iterable[str] = TEXT_FIELDS,
    label: str,
) -> dict[str, Any]:
    n = min(len(real), len(synthetic), int(protocol.max_rows))
    real_sample = (
        real.sample(n=n, random_state=protocol.seed) if len(real) > n else real.head(n)
    )
    synthetic_sample = (
        synthetic.sample(n=n, random_state=protocol.seed + 1)
        if len(synthetic) > n
        else synthetic.head(n)
    )
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for field in fields:
        if field not in real_sample or field not in synthetic_sample:
            continue
        real_embeddings = store.embed(
            real_sample[field].tolist(),
            backend=protocol.embedding_backend,
            model_name=protocol.embedding_model,
            preprocessing=protocol.preprocessing,
            label=f"{label}_{field}_real",
        )
        synthetic_embeddings = store.embed(
            synthetic_sample[field].tolist(),
            backend=protocol.embedding_backend,
            model_name=protocol.embedding_model,
            preprocessing=protocol.preprocessing,
            label=f"{label}_{field}_synthetic",
        )
        arrays[field] = {
            "real": real_embeddings,
            "synthetic": synthetic_embeddings,
        }
    return {"num_real": int(n), "num_synthetic": int(n), "arrays": arrays}


def evaluate_prepared_embeddings(
    prepared: dict[str, Any],
    classifiers: Iterable[str],
    *,
    seed: int,
    n_splits: int,
    protocol: TextC2STProtocol | None = None,
) -> dict[str, Any]:
    per_field: dict[str, Any] = {}
    real_arrays: list[np.ndarray] = []
    synthetic_arrays: list[np.ndarray] = []
    used_fields: list[str] = []
    for field, arrays in (prepared.get("arrays") or {}).items():
        result = evaluate_embedding_pair(
            arrays["real"],
            arrays["synthetic"],
            classifiers,
            seed=seed,
            n_splits=n_splits,
        )
        per_field[field] = result
        real_arrays.append(arrays["real"])
        synthetic_arrays.append(arrays["synthetic"])
        used_fields.append(field)
    errors = [value["error"] for value in per_field.values()]
    aucs = [value["auc"] for value in per_field.values()]
    combined = None
    if len(real_arrays) >= 2:
        combined = evaluate_embedding_pair(
            np.concatenate(real_arrays, axis=1),
            np.concatenate(synthetic_arrays, axis=1),
            classifiers,
            seed=seed,
            n_splits=n_splits,
        )
    macro_auc = float(np.mean(aucs)) if aucs else None
    macro_error = float(np.mean(errors)) if errors else None
    transformed_macro_auc = c2st_error(macro_auc) if macro_auc is not None else None
    return {
        "protocol": protocol.as_dict() if protocol is not None else None,
        "num_real": int(prepared.get("num_real", 0)),
        "num_synthetic": int(prepared.get("num_synthetic", 0)),
        "fields": used_fields,
        "per_field": per_field,
        "macro_auc": macro_auc,
        "macro_error": macro_error,
        "macro_error_from_macro_auc": transformed_macro_auc,
        "aggregation_identity_gap": (
            macro_error - transformed_macro_auc
            if macro_error is not None and transformed_macro_auc is not None
            else None
        ),
        "combined": combined,
    }


def evaluate_embedding_pair(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    classifiers: Iterable[str],
    *,
    seed: int,
    n_splits: int,
) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([real_embeddings, synthetic_embeddings])
    y = np.array(
        [1] * len(real_embeddings) + [0] * len(synthetic_embeddings), dtype=int
    )
    splits = min(int(n_splits), int(np.bincount(y).min()))
    cv = list(
        StratifiedKFold(
            n_splits=splits, shuffle=True, random_state=seed
        ).split(x, y)
    )
    model_factories = {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500)
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=1
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            random_state=seed
        ),
    }
    results: dict[str, Any] = {}
    for name in classifiers:
        if name not in model_factories:
            continue
        model = model_factories[name]()
        started = time.perf_counter()
        print(
            f"[text-c2st] classifier={name} rows={len(y):,} "
            f"features={x.shape[1]:,} folds={splits}",
            flush=True,
        )
        try:
            scores = cross_val_predict(
                model, x, y, cv=cv, method="predict_proba"
            )[:, 1]
            auc = float(roc_auc_score(y, scores))
            accuracy = float(accuracy_score(y, (scores >= 0.5).astype(int)))
            results[name] = {
                "auc": auc,
                "error": c2st_error(auc),
                "accuracy": accuracy,
                "status": "ok",
                "wall_clock_seconds": float(time.perf_counter() - started),
            }
        except Exception as exc:
            results[name] = {
                "status": "failed",
                "reason": str(exc),
                "wall_clock_seconds": float(time.perf_counter() - started),
            }
    successful = {
        name: value
        for name, value in results.items()
        if value.get("status") == "ok"
    }
    if not successful:
        raise RuntimeError("All requested text-C2ST classifiers failed")
    best_name = max(successful, key=lambda name: successful[name]["auc"])
    best = successful[best_name]
    return {
        "auc": best["auc"],
        "error": best["error"],
        "accuracy": best["accuracy"],
        "classifier": best_name,
        "per_classifier": results,
        "num_real": int(len(real_embeddings)),
        "num_synthetic": int(len(synthetic_embeddings)),
        "n_splits": int(splits),
        "splitter": "StratifiedKFold(shuffle=True, random_state=seed)",
    }


def flatten_protocol_result(
    result: dict[str, Any], *, data_label: str, evaluator_label: str
) -> list[dict[str, Any]]:
    rows = []
    for field, values in (result.get("per_field") or {}).items():
        rows.append(
            {
                "data": data_label,
                "evaluator": evaluator_label,
                "field": field,
                "auc": values.get("auc"),
                "error": values.get("error"),
                "classifier": values.get("classifier"),
                "num_real": result.get("num_real"),
                "num_synthetic": result.get("num_synthetic"),
            }
        )
    rows.append(
        {
            "data": data_label,
            "evaluator": evaluator_label,
            "field": "macro",
            "auc": result.get("macro_auc"),
            "error": result.get("macro_error"),
            "classifier": "per-field best",
            "num_real": result.get("num_real"),
            "num_synthetic": result.get("num_synthetic"),
        }
    )
    combined = result.get("combined") or {}
    if combined:
        rows.append(
            {
                "data": data_label,
                "evaluator": evaluator_label,
                "field": "combined",
                "auc": combined.get("auc"),
                "error": combined.get("error"),
                "classifier": combined.get("classifier"),
                "num_real": result.get("num_real"),
                "num_synthetic": result.get("num_synthetic"),
            }
        )
    return rows


def embedding_fingerprint(
    texts: Iterable[Any], backend: str, model_name: str
) -> str:
    digest = hashlib.sha256()
    digest.update(backend.encode("utf-8"))
    digest.update(model_name.encode("utf-8"))
    values = list(texts)
    digest.update(str(len(values)).encode("ascii"))
    for value in values:
        encoded = historical_minilm_text(value).encode(
            "utf-8", errors="surrogatepass"
        )
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def historical_embedding_fingerprint(
    texts: Iterable[Any], model_name: str
) -> str:
    digest = hashlib.sha256()
    values = list(texts)
    digest.update(model_name.encode("utf-8"))
    digest.update(str(len(values)).encode("ascii"))
    for value in values:
        encoded = historical_minilm_text(value).encode(
            "utf-8", errors="surrogatepass"
        )
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def load_valid_cache(
    path: Path, metadata_path: Path, fingerprint: str, num_texts: int
) -> np.ndarray | None:
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        metadata.get("fingerprint") != fingerprint
        or int(metadata.get("num_texts", -1)) != int(num_texts)
    ):
        return None
    return np.asarray(np.load(path, mmap_mode="r"))


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def describe_pooling(model: Any) -> dict[str, Any] | None:
    for module in model:
        if module.__class__.__name__.lower() == "pooling":
            config = getattr(module, "get_config_dict", None)
            return config() if callable(config) else {"class": module.__class__.__name__}
    return None


def finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
