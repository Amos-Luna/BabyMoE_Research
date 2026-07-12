from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from babymoe.config import LANGUAGES, load_config
from babymoe.data import (
    ExposureLimitReached,
    PreparedBlocks,
    TrilingualBatchStream,
    _select_indices,
    adjusted_units,
    language_quotas,
    prepare_all,
)

from conftest import REPO_ROOT


def test_byte_premium_arithmetic_and_quotas():
    assert adjusted_units(10_516, 1.0516) == 10_000
    assert adjusted_units(9_894, 0.9894) == 10_000
    quotas = language_quotas(
        100_000_000, {"eng": 0.333334, "nld": 0.333333, "zho": 0.333333}
    )
    assert sum(quotas.values()) == 100_000_000
    assert quotas["eng"] >= quotas["nld"]


def test_synthetic_preparation_is_idempotent(tmp_path):
    config = load_config(
        REPO_ROOT / "configs/smoke.yaml", artifacts_root=tmp_path
    )
    config = deepcopy(config)
    config["data"]["unique_exposure_units"] = 30_000
    config["data"]["validation_units_per_language"] = 2_000
    config["data"]["minimum_quota_fill_ratio"] = 0.90
    config["data"]["max_epochs"] = 1
    config["tokenizer"]["vocab_size"] = 256
    config["tokenizer"]["train_chars_per_language"] = 5_000
    config["tokenizer"]["model_max_length"] = 32
    config["training"]["target_exposure_units"] = 30_000
    config["training"]["sequence_length"] = 32
    config["model"]["max_position_embeddings"] = 32

    first = prepare_all(config, synthetic=True)
    second = prepare_all(config, synthetic=True)
    assert first == second
    tokenizer_config = json.loads(
        (Path(first["tokenizer_manifest"]).parent / "tokenizer_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert tokenizer_config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert "backend" not in tokenizer_config
    manifest = json.loads(
        open(first["blocks_manifest"], encoding="utf-8").read()
    )
    assert 0 < manifest["actual_unique_exposure_units"] <= 30_000
    assert manifest["max_compliant_exposure_units"] == pytest.approx(
        manifest["actual_unique_exposure_units"]
    )
    prepared = PreparedBlocks(first["blocks_manifest"])
    assert all(len(prepared.store(lang, "train")) > 0 for lang in LANGUAGES)


def test_validation_excludes_every_row_with_a_training_document_id():
    config = load_config(REPO_ROOT / "configs/smoke.yaml")
    dataset = Dataset.from_list(
        [
            {
                "text": f"document {index}",
                "doc-id": document_id,
                "category": "test",
                "num-tokens": 100,
            }
            for index, document_id in enumerate(("dup", "dup", "b", "c", "d", "e"))
        ]
    )
    train_indices, _, _ = _select_indices(
        dataset, config=config, lang="eng", quota=200
    )
    train_ids = {dataset[index]["doc-id"] for index in train_indices}
    validation_indices, _, _ = _select_indices(
        dataset,
        config=config,
        lang="eng",
        quota=400,
        excluded_document_ids=train_ids,
    )
    validation_ids = {dataset[index]["doc-id"] for index in validation_indices}
    assert train_ids.isdisjoint(validation_ids)


class _Store:
    def __init__(self, offset: int):
        self.rows = [
            {
                "input_ids": np.asarray([offset + index, 1], dtype=np.int64),
                "exposure": 10.0,
                "length": 2,
                "group_id": index,
                "domain_id": 0,
            }
            for index in range(4)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class _Prepared:
    def __init__(self):
        self.manifest = {
            "max_compliant_exposure_units": 240,
            "splits": {
                lang: {"train": {"adjusted_exposure_units": 40.0}}
                for lang in LANGUAGES
            },
        }
        self.stores = {
            lang: _Store(offset)
            for lang, offset in zip(LANGUAGES, (10, 20, 30))
        }

    def store(self, lang, split):
        assert split == "train"
        return self.stores[lang]


def _stream_config():
    return {
        "data": {
            "mixture": {"eng": 1 / 3, "nld": 1 / 3, "zho": 1 / 3},
            "max_epochs": 2,
        },
        "training": {"target_exposure_units": 240},
    }


def test_deterministic_stream_and_extended_resume():
    prepared = _Prepared()
    first = TrilingualBatchStream(prepared, _stream_config(), seed=42, stop_at_exposure=60)
    second = TrilingualBatchStream(prepared, _stream_config(), seed=42, stop_at_exposure=60)
    assert np.array_equal(first.next_batch(3)["input_ids"], second.next_batch(3)["input_ids"])
    state = first.state_dict()

    resumed = TrilingualBatchStream(
        prepared, _stream_config(), seed=42, stop_at_exposure=120
    )
    resumed.load_state_dict(state)
    while True:
        try:
            resumed.next_batch(2)
        except ExposureLimitReached:
            break
    assert resumed.total_consumed == 120
    assert all(value <= 80 for value in resumed.consumed.values())
