from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.pretokenized import (  # noqa: E402
    PretokenizedLSTMDataset,
    load_pretokenized_bundle,
)
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import save_json  # noqa: E402


def test_pretokenized_dataset_loads_memmap_without_raw_text_tokenization(tmp_path):
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating",),
        text_targets=("summary",),
        text_max_lengths={"summary": 5},
    )
    root = tmp_path / "pretokenized"
    root.mkdir()
    save_json({"num_rows": 3, "text_fields": {"summary": {"shape": [3, 5], "dtype": "int32"}}}, root / "metadata.json")
    tokenizer = SimpleTextTokenizer().fit(["alpha beta", "gamma"])
    save_json(tokenizer.to_dict(), root / "tokenizer_metadata.json")
    save_json(CategoryVocab.from_values("rating", ["5", "1"]).to_dict(), root / "vocab_rating.json")
    np.save(root / "train_indices.npy", np.asarray([0, 2], dtype=np.int64))
    np.save(root / "valid_indices.npy", np.asarray([1], dtype=np.int64))
    np.save(root / "foreign_key_ids.npy", np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int64))
    np.save(root / "datetime_values.npy", np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32))
    np.save(root / "categorical_ids.npy", np.asarray([[0], [1], [0]], dtype=np.int64))
    np.save(root / "review_time_ns.npy", np.asarray([10, 20, 30], dtype=np.int64))
    token_ids = np.memmap(root / "summary_token_ids.memmap", dtype=np.int32, mode="w+", shape=(3, 5))
    token_ids[:] = np.asarray([[1, 6, 7, 4, 0], [1, 8, 4, 0, 0], [1, 6, 4, 0, 0]], dtype=np.int32)
    token_ids.flush()
    np.save(root / "summary_lengths.npy", np.asarray([2, 1, 1], dtype=np.int32))

    bundle = load_pretokenized_bundle(root, schema)
    dataset = PretokenizedLSTMDataset(bundle, "train")
    sample = dataset[1]

    assert len(dataset) == 2
    assert dataset.timestamps_ns.tolist() == [10, 30]
    assert sample["row_id"].item() == 2
    assert sample["text_ids"]["summary"].shape[0] == 5
    assert sample["foreign_key_ids"].tolist() == [5, 6]
