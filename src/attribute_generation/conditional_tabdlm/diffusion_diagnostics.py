"""Reusable diagnostics and experiment metadata for hierarchical diffusion."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .tokenization import SimpleTextTokenizer


@dataclass(frozen=True)
class ProgressiveConditionSpec:
    """One controlled text-generation conditioning intervention."""

    name: str
    oracle_structured: bool
    oracle_lengths: bool
    graph_mode: str
    graph_history_source: str
    valid_generative_baseline: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROGRESSIVE_CONDITION_SPECS: dict[str, ProgressiveConditionSpec] = {
    "O1": ProgressiveConditionSpec(
        name="O1",
        oracle_structured=True,
        oracle_lengths=True,
        graph_mode="both_with_coverage",
        graph_history_source="real_prefix",
        valid_generative_baseline=False,
        description="Real structured attributes, exact real lengths, and real strict-past graph history.",
    ),
    "O2": ProgressiveConditionSpec(
        name="O2",
        oracle_structured=False,
        oracle_lengths=True,
        graph_mode="both_with_coverage",
        graph_history_source="real_prefix",
        valid_generative_baseline=False,
        description="Generated structured attributes, exact real lengths, and real strict-past graph history.",
    ),
    "O3": ProgressiveConditionSpec(
        name="O3",
        oracle_structured=False,
        oracle_lengths=False,
        graph_mode="both_with_coverage",
        graph_history_source="real_prefix",
        valid_generative_baseline=False,
        description="Generated structured attributes and lengths with real strict-past graph history.",
    ),
    "O4": ProgressiveConditionSpec(
        name="O4",
        oracle_structured=False,
        oracle_lengths=False,
        graph_mode="both_with_coverage",
        graph_history_source="evaluation_spine",
        valid_generative_baseline=True,
        description="Standard generated pipeline with graph history built from the evaluation spine.",
    ),
    "O5": ProgressiveConditionSpec(
        name="O5",
        oracle_structured=False,
        oracle_lengths=False,
        graph_mode="no_graph",
        graph_history_source="none",
        valid_generative_baseline=True,
        description="Generated structured attributes and lengths with graph history removed.",
    ),
}


def progressive_condition_spec(value: str | None) -> ProgressiveConditionSpec | None:
    if value is None or not str(value).strip():
        return None
    name = str(value).strip().upper()
    if name not in PROGRESSIVE_CONDITION_SPECS:
        raise ValueError(
            f"Unknown progressive conditioning mode {value!r}; "
            f"expected one of {sorted(PROGRESSIVE_CONDITION_SPECS)}"
        )
    return PROGRESSIVE_CONDITION_SPECS[name]


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_fingerprint(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame.loc[:, list(columns)] if columns is not None else frame
    digest = hashlib.sha256()
    digest.update("|".join(map(str, selected.columns)).encode("utf-8"))
    digest.update("|".join(map(str, selected.dtypes)).encode("utf-8"))
    hashes = pd.util.hash_pandas_object(selected, index=True).to_numpy(dtype=np.uint64)
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def current_git_commit(repo_root: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def unique_run_root(output_root: str | Path, experiment_name: str) -> Path:
    """Create a UTC-timestamped root without overwriting an earlier run."""

    root = Path(output_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    candidate = root / f"{safe_name(experiment_name)}_{timestamp}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def safe_name(value: Any) -> str:
    text = "".join(char if str(char).isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return text.strip("_") or "run"


def text_generation_diagnostics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    schema: Any,
    tokenizer: SimpleTextTokenizer | None = None,
    repeated_ngram_size: int = 3,
) -> dict[str, Any]:
    """Compute dataset-agnostic text quality and decoding-integrity metrics."""

    per_column: dict[str, Any] = {}
    for column in schema.text_targets:
        if column not in real or column not in synthetic:
            per_column[column] = {
                "status": "missing",
                "real_present": column in real,
                "synthetic_present": column in synthetic,
            }
            continue
        real_text = real[column].fillna("").astype(str)
        synthetic_text = synthetic[column].fillna("").astype(str)
        real_tokens = real_text.map(tokenize_for_diagnostics)
        synthetic_tokens = synthetic_text.map(tokenize_for_diagnostics)
        real_token_counts = real_tokens.map(len).to_numpy(dtype=float)
        synthetic_token_counts = synthetic_tokens.map(len).to_numpy(dtype=float)
        real_char_counts = real_text.map(len).to_numpy(dtype=float)
        synthetic_char_counts = synthetic_text.map(len).to_numpy(dtype=float)
        special_markers = tokenizer_special_markers(tokenizer)
        leaked = synthetic_text.map(
            lambda text: any(marker in text for marker in special_markers)
        )
        invalid_utf8 = synthetic_text.map(lambda text: not valid_utf8(text))
        distinct_1, distinct_2 = distinct_n_metrics(synthetic_tokens)
        bucket_column = length_bucket_for_text(schema, column)
        per_column[column] = {
            "status": "ok",
            "num_real": int(len(real_text)),
            "num_synthetic": int(len(synthetic_text)),
            "empty_rate": float((synthetic_text.str.strip() == "").mean()),
            "real_empty_rate": float((real_text.str.strip() == "").mean()),
            "special_token_leakage_rate": float(leaked.mean()),
            "padding_token_leakage_rate": float(
                synthetic_text.str.contains(r"\[PAD\]", regex=True).mean()
            ),
            "invalid_utf8_rate": float(invalid_utf8.mean()),
            "vocabulary_size": int(len({token for row in synthetic_tokens for token in row})),
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "repeated_ngram_rate": repeated_ngram_rate(
                synthetic_tokens, n=int(repeated_ngram_size)
            ),
            "token_count_ks": ks_distance(real_token_counts, synthetic_token_counts),
            "character_count_ks": ks_distance(real_char_counts, synthetic_char_counts),
            "text_length_ks": ks_distance(real_token_counts, synthetic_token_counts),
            "exact_or_valid_length_satisfaction_rate": length_satisfaction_rate(
                synthetic,
                synthetic_token_counts,
                schema,
                bucket_column,
            ),
            "exact_training_row_duplication_rate": exact_duplication_rate(
                real_text, synthetic_text
            ),
        }

    cross_field: dict[str, Any] = {}
    text_columns = [
        column
        for column in schema.text_targets
        if column in real and column in synthetic
    ]
    if len(text_columns) >= 2:
        left, right = text_columns[:2]
        cross_field["first_second_text_hash_cosine"] = {
            "columns": [left, right],
            "real_mean": mean_hash_cosine(real[left], real[right]),
            "synthetic_mean": mean_hash_cosine(
                synthetic[left], synthetic[right]
            ),
        }
    return {
        "text_columns": list(schema.text_targets),
        "per_column": per_column,
        "cross_field": cross_field,
    }


def tokenizer_special_markers(tokenizer: SimpleTextTokenizer | None) -> set[str]:
    if tokenizer is None:
        return {"[PAD]", "[BOS]", "[MASK]", "[UNK]", "[EOS]", "<empty>"}
    return {
        tokenizer.pad_token,
        tokenizer.bos_token,
        tokenizer.mask_token,
        tokenizer.unk_token,
        tokenizer.eos_token,
        tokenizer.empty_token,
    }


def tokenize_for_diagnostics(text: Any) -> list[str]:
    return [token for token in str(text).strip().split() if token]


def distinct_n_metrics(rows: pd.Series) -> tuple[float, float]:
    unigrams: list[tuple[str, ...]] = []
    bigrams: list[tuple[str, ...]] = []
    for tokens in rows:
        unigrams.extend((token,) for token in tokens)
        bigrams.extend(tuple(tokens[idx : idx + 2]) for idx in range(max(0, len(tokens) - 1)))
    distinct_1 = float(len(set(unigrams)) / len(unigrams)) if unigrams else 0.0
    distinct_2 = float(len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    return distinct_1, distinct_2


def repeated_ngram_rate(rows: pd.Series, n: int = 3) -> float:
    repeated = 0
    total = 0
    for tokens in rows:
        ngrams = [
            tuple(tokens[idx : idx + n])
            for idx in range(max(0, len(tokens) - n + 1))
        ]
        total += len(ngrams)
        repeated += max(0, len(ngrams) - len(set(ngrams)))
    return float(repeated / total) if total else 0.0


def ks_distance(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if not len(left) or not len(right):
        return None
    try:
        from scipy.stats import ks_2samp

        return float(ks_2samp(left, right).statistic)
    except Exception:
        support = np.sort(np.unique(np.concatenate([left, right])))
        left_cdf = np.searchsorted(np.sort(left), support, side="right") / len(left)
        right_cdf = np.searchsorted(np.sort(right), support, side="right") / len(right)
        return float(np.max(np.abs(left_cdf - right_cdf)))


def valid_utf8(text: str) -> bool:
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeError:
        return False
    return True


def exact_duplication_rate(real: pd.Series, synthetic: pd.Series) -> float:
    real_values = set(real.fillna("").astype(str))
    if not len(synthetic):
        return 0.0
    return float(synthetic.fillna("").astype(str).isin(real_values).mean())


def length_bucket_for_text(schema: Any, text_column: str) -> str | None:
    for column in schema.length_bucket_targets:
        try:
            if schema.text_column_for_length_bucket(column) == text_column:
                return str(column)
        except KeyError:
            continue
    return None


def length_satisfaction_rate(
    synthetic: pd.DataFrame,
    token_counts: np.ndarray,
    schema: Any,
    bucket_column: str | None,
) -> float | None:
    if bucket_column is None or bucket_column not in synthetic:
        return None
    buckets = schema.buckets_for_length_bucket(bucket_column)
    valid = []
    for count, bucket_name in zip(token_counts, synthetic[bucket_column].astype(str)):
        bounds = buckets.get(str(bucket_name))
        if bounds is None:
            valid.append(False)
            continue
        low, high = bounds
        valid.append(int(low) <= int(count) <= int(high))
    return float(np.mean(valid)) if valid else None


def mean_hash_cosine(left: pd.Series, right: pd.Series, dim: int = 64) -> float | None:
    n = min(len(left), len(right))
    if n == 0:
        return None
    left_emb = np.vstack([hash_text(value, dim) for value in left.head(n)])
    right_emb = np.vstack([hash_text(value, dim) for value in right.head(n)])
    denom = np.linalg.norm(left_emb, axis=1) * np.linalg.norm(right_emb, axis=1)
    values = np.divide(
        np.sum(left_emb * right_emb, axis=1),
        denom,
        out=np.zeros(n, dtype=float),
        where=denom > 0,
    )
    return float(np.mean(values))


def hash_text(text: Any, dim: int) -> np.ndarray:
    vector = np.zeros(int(dim), dtype=float)
    for token in tokenize_for_diagnostics(text):
        digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        vector[value % int(dim)] += 1.0 if ((value >> 8) & 1) else -1.0
    return vector


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
