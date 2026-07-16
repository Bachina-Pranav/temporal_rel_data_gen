"""Constrained categorical decoding and validation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from .tokenization import CategoryVocab, normalize_category


MISSING_CATEGORY_TOKEN = "<missing>"


def normalize_rating_value(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if not np.isfinite(numeric):
        return None
    doubled = round(numeric * 2.0)
    if abs(numeric * 2.0 - doubled) > 1e-8:
        return None
    normalized = float(doubled) / 2.0
    if normalized <= 0.0:
        return None
    rounded = int(round(normalized))
    if abs(normalized - rounded) <= 1e-8:
        return rounded
    return normalized


def normalize_categorical_value(column: str, value: Any) -> str | None:
    if str(column) == "rating":
        rating = normalize_rating_value(value)
        return None if rating is None else str(rating)
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return normalize_category(text)


def normalized_valid_values(column: str, values: Any) -> list[str]:
    normalized = []
    for value in values:
        item = normalize_categorical_value(column, value)
        if item is not None and item != MISSING_CATEGORY_TOKEN:
            normalized.append(item)
    return sorted(set(normalized), key=category_sort_key)


def category_sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def valid_category_ids(column: str, vocab: CategoryVocab) -> list[int]:
    ids = []
    for token, idx in vocab.token_to_id.items():
        normalized = normalize_categorical_value(column, token)
        if normalized is None or normalized == MISSING_CATEGORY_TOKEN:
            continue
        if str(column) == "rating" and normalize_rating_value(token) is None:
            continue
        ids.append(int(idx))
    if not ids:
        raise ValueError(f"No valid categorical IDs for {column!r}")
    return sorted(set(ids))


def valid_category_values(column: str, vocab: CategoryVocab) -> list[Any]:
    values: list[Any] = []
    for idx in valid_category_ids(column, vocab):
        token = vocab.decode(idx)
        if str(column) == "rating":
            rating = normalize_rating_value(token)
            if rating is not None:
                values.append(rating)
        else:
            values.append(token)
    if str(column) == "rating":
        return sorted(set(values))
    return sorted(set(str(value) for value in values), key=category_sort_key)


def mask_invalid_category_logits(logits: torch.Tensor, column: str, vocab: CategoryVocab) -> torch.Tensor:
    valid_ids = valid_category_ids(column, vocab)
    mask = torch.full_like(logits, -float("inf"))
    index = torch.tensor(valid_ids, dtype=torch.long, device=logits.device)
    mask.index_copy_(1, index, logits.index_select(1, index))
    return mask


def decode_category_id(column: str, vocab: CategoryVocab, idx: int) -> Any:
    token = vocab.decode(int(idx))
    if str(column) == "rating":
        rating = normalize_rating_value(token)
        if rating is None:
            raise ValueError(f"Decoded invalid rating token {token!r} from id={idx}")
        return rating
    normalized = normalize_categorical_value(column, token)
    if normalized is None or normalized == MISSING_CATEGORY_TOKEN:
        raise ValueError(f"Decoded invalid categorical token {token!r} for {column!r}")
    return token


def categorical_validity_mask(series: pd.Series, column: str, valid_values: Any) -> pd.Series:
    valid = set(normalized_valid_values(column, valid_values))
    normalized = series.map(lambda value: normalize_categorical_value(column, value))
    return normalized.map(lambda value: value in valid if value is not None else False)


def validate_output_categoricals(
    frame: pd.DataFrame,
    categorical_vocabs: dict[str, CategoryVocab],
    *,
    repair_invalid: bool = False,
) -> pd.DataFrame:
    output = frame.copy()
    errors: list[str] = []
    for column, vocab in categorical_vocabs.items():
        if column not in output.columns:
            continue
        valid_values = valid_category_values(column, vocab)
        mask = categorical_validity_mask(output[column], column, valid_values)
        if bool(mask.all()):
            if column == "rating":
                output[column] = output[column].map(normalize_rating_value)
            continue
        bad_rows = output.loc[~mask, [column]].head(10)
        if repair_invalid:
            replacement = valid_values[0]
            output.loc[~mask, column] = replacement
            continue
        errors.append(f"{column}: invalid rows={bad_rows.to_dict(orient='records')}")
    if errors:
        raise ValueError("Invalid categorical outputs detected before CSV write: " + "; ".join(errors))
    return output
