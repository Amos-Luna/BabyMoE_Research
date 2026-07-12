"""Auditable BabyBabelLM preparation and memory-mapped block access."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
from collections import Counter, deque
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from .config import LANGUAGES, artifact_path, config_fingerprint
from .utils import (
    atomic_directory,
    atomic_write_json,
    file_record,
    manifest_is_valid,
    sha256_file,
)

SELECTION_MANIFEST = "manifest.json"
BLOCKS_MANIFEST = "manifest.json"


def adjusted_units(raw_count: int | float, byte_premium: float) -> float:
    """Convert a language-specific count to English-reference exposure units."""

    if raw_count < 0 or byte_premium <= 0:
        raise ValueError("raw_count must be non-negative and byte_premium positive")
    if raw_count == 0:
        return 0.0
    value = Decimal(str(raw_count)) / Decimal(str(byte_premium))
    return float(value)


def language_quotas(total: int, mixture: Mapping[str, float]) -> dict[str, int]:
    """Allocate an integer budget exactly, using largest remainders."""

    exact = {lang: total * float(mixture[lang]) for lang in LANGUAGES}
    quotas = {lang: math.floor(exact[lang]) for lang in LANGUAGES}
    remainder = total - sum(quotas.values())
    order = sorted(LANGUAGES, key=lambda lang: exact[lang] - quotas[lang], reverse=True)
    for lang in order[:remainder]:
        quotas[lang] += 1
    if sum(quotas.values()) != total:
        raise AssertionError("integer quota allocation failed")
    return quotas


def _stable_order_key(seed: int, lang: str, document_id: str) -> bytes:
    return hashlib.sha256(f"{seed}|{lang}|{document_id}".encode("utf-8")).digest()


def _record_count(record: Mapping[str, Any], config: Mapping[str, Any], lang: str) -> int:
    field = config["data"]["token_count_field"]
    value = record.get(field)
    if value is not None and int(value) > 0:
        return int(value)
    raise ValueError(
        f"{lang} record is missing required official count field '{field}'"
    )


def _select_indices(
    dataset: Dataset,
    *,
    config: Mapping[str, Any],
    lang: str,
    quota: int,
    excluded: set[int] | None = None,
    excluded_document_ids: set[str] | None = None,
) -> tuple[list[int], list[float], float]:
    data_cfg = config["data"]
    doc_field = data_cfg["document_id_field"]
    seed = int(data_cfg["selection_seed"])
    premium = float(data_cfg["byte_premium"][lang])
    excluded = excluded or set()
    excluded_document_ids = excluded_document_ids or set()

    order: list[tuple[bytes, int]] = []
    seen_rows: set[tuple[str, bytes]] = set()
    for index in range(len(dataset)):
        if index in excluded:
            continue
        record = dataset[index]
        document_id = str(record.get(doc_field) or index)
        if document_id in excluded_document_ids:
            continue
        text_digest = hashlib.sha256(
            str(record.get(data_cfg["text_field"], "")).encode("utf-8")
        ).digest()
        row_identity = (document_id, text_digest)
        if row_identity in seen_rows:
            continue
        seen_rows.add(row_identity)
        key = (
            _stable_order_key(seed, lang, document_id)
            if bool(data_cfg.get("shuffle_documents", True))
            else index.to_bytes(16, "big")
        )
        order.append((key, index))
    order.sort(key=lambda item: item[0])

    selected: list[int] = []
    units: list[float] = []
    consumed = Decimal(0)
    decimal_quota = Decimal(quota)
    for _, index in order:
        count = Decimal(str(_record_count(dataset[index], config, lang))) / Decimal(
            str(premium)
        )
        if count <= 0 or consumed + count > decimal_quota:
            continue
        selected.append(index)
        units.append(float(count))
        consumed += count
        if consumed == decimal_quota:
            break
    return selected, units, float(consumed)


def _decorate_dataset(
    dataset: Dataset,
    indices: list[int],
    units: list[float],
    *,
    config: Mapping[str, Any],
    lang: str,
    split: str,
) -> Dataset:
    selected = dataset.select(indices)
    raw = [_record_count(dataset[index], config, lang) for index in indices]
    selected = selected.add_column("_babymoe_language", [lang] * len(selected))
    selected = selected.add_column("_babymoe_split", [split] * len(selected))
    selected = selected.add_column("_babymoe_raw_units", raw)
    selected = selected.add_column("_babymoe_adjusted_units", units)
    return selected


def _synthetic_dataset(lang: str, config: Mapping[str, Any]) -> Dataset:
    snippets = {
        "eng": "the child reads a small book and talks with a friend",
        "nld": "het kind leest een klein boek en praat met een vriend",
        "zho": "孩子读一本小书并和朋友说话",
    }
    target = int(config["data"]["unique_exposure_units"] / 3) + int(
        config["data"]["validation_units_per_language"]
    )
    per_document = 1_000
    records = []
    for index in range(max(6, math.ceil(target / per_document) + 3)):
        text = " ".join([snippets[lang]] * 100)
        records.append(
            {
                "text": text,
                "doc-id": f"synthetic-{lang}-{index}",
                "category": "synthetic",
                "data-source": "generated-smoke-data",
                "script": {"eng": "Latn", "nld": "Latn", "zho": "Hani"}[lang],
                "language": lang,
                "num-tokens": per_document,
            }
        )
    return Dataset.from_list(records)


def selection_fingerprint(config: Mapping[str, Any], synthetic: bool) -> str:
    payload = {
        "data": config["data"],
        "synthetic": synthetic,
        "schema": 1,
    }
    return config_fingerprint(payload)


def prepare_selection(
    config: Mapping[str, Any],
    *,
    token: str | None = None,
    synthetic: bool = False,
    force: bool = False,
) -> Path:
    data_root = artifact_path(config, "data_dir")
    selected_root = data_root / "selected"
    fingerprint = selection_fingerprint(config, synthetic)
    manifest_path = selected_root / SELECTION_MANIFEST
    if not force and manifest_is_valid(
        manifest_path, expected_fingerprint=fingerprint, root=selected_root
    ):
        print(f"[prepare] selection valid; skipping: {selected_root}")
        return manifest_path

    quotas = language_quotas(
        int(config["data"]["unique_exposure_units"]), config["data"]["mixture"]
    )
    cache_dir = artifact_path(config, "cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    language_summaries: dict[str, Any] = {}

    with atomic_directory(selected_root) as temporary:
        outputs: list[Path] = []
        for lang in LANGUAGES:
            print(f"[prepare] loading {lang} from {config['data']['repos'][lang]}")
            dataset = (
                _synthetic_dataset(lang, config)
                if synthetic
                else load_dataset(
                    config["data"]["repos"][lang],
                    split="train",
                    revision=config["data"]["revisions"][lang],
                    cache_dir=str(cache_dir),
                    token=token,
                    trust_remote_code=bool(config["data"].get("trust_remote_code", False)),
                )
            )
            train_indices, train_units, train_total = _select_indices(
                dataset, config=config, lang=lang, quota=quotas[lang]
            )
            doc_field = config["data"]["document_id_field"]
            train_document_ids = {
                str(dataset[index].get(doc_field) or index)
                for index in train_indices
            }
            validation_indices, validation_units, validation_total = _select_indices(
                dataset,
                config=config,
                lang=lang,
                quota=int(config["data"]["validation_units_per_language"]),
                excluded=set(train_indices),
                excluded_document_ids=train_document_ids,
            )
            if not train_indices:
                raise RuntimeError(f"no training documents selected for {lang}")
            fill_ratio = train_total / quotas[lang]
            minimum_fill = float(config["data"]["minimum_quota_fill_ratio"])
            if fill_ratio < minimum_fill:
                raise RuntimeError(
                    f"{lang} filled only {fill_ratio:.6%} of its quota; "
                    f"minimum is {minimum_fill:.6%}"
                )
            train_dataset = _decorate_dataset(
                dataset,
                train_indices,
                train_units,
                config=config,
                lang=lang,
                split="train",
            )
            validation_dataset = _decorate_dataset(
                dataset,
                validation_indices,
                validation_units,
                config=config,
                lang=lang,
                split="validation",
            )
            train_path = temporary / f"{lang}_train.parquet"
            validation_path = temporary / f"{lang}_validation.parquet"
            train_dataset.to_parquet(str(train_path))
            validation_dataset.to_parquet(str(validation_path))
            outputs.extend([train_path, validation_path])

            domain_field = config["data"]["domain_field"]
            domains = Counter(str(value) for value in train_dataset[domain_field])
            language_summaries[lang] = {
                "source_repo": config["data"]["repos"][lang],
                "source_revision": config["data"]["revisions"][lang],
                "source_fingerprint": getattr(dataset, "_fingerprint", "synthetic"),
                "source_documents": len(dataset),
                "train_documents": len(train_dataset),
                "train_adjusted_units": train_total,
                "train_quota": quotas[lang],
                "train_quota_fill_ratio": fill_ratio,
                "validation_documents": len(validation_dataset),
                "validation_adjusted_units": validation_total,
                "byte_premium": float(config["data"]["byte_premium"][lang]),
                "domains": dict(sorted(domains.items())),
            }
            print(
                f"[prepare] {lang}: {train_total:,}/{quotas[lang]:,} train units, "
                f"{validation_total:,} validation units"
            )

        actual_unique = sum(
            value["train_adjusted_units"] for value in language_summaries.values()
        )
        configured_unique = int(config["data"]["unique_exposure_units"])
        if actual_unique > configured_unique + 1e-6:
            raise AssertionError(
                f"selected exposure {actual_unique} exceeds budget {configured_unique}"
            )
        manifest = {
            "complete": True,
            "fingerprint": fingerprint,
            "synthetic": synthetic,
            "accounting_formula": config["data"]["accounting_formula"],
            "byte_premium_source": config["data"]["byte_premium_source"],
            "configured_unique_exposure_units": configured_unique,
            "actual_unique_exposure_units": actual_unique,
            "languages": language_summaries,
            "files": [file_record(path, relative_to=temporary) for path in outputs],
        }
        atomic_write_json(temporary / SELECTION_MANIFEST, manifest)
    return manifest_path


def _load_parquet(path: Path) -> Dataset:
    # Reading the Arrow table directly avoids an extra Hugging Face cache copy
    # of the already-persistent selected corpus.
    return Dataset(pq.read_table(path, memory_map=True))


def tokenizer_fingerprint(
    config: Mapping[str, Any], selection_manifest: Path
) -> str:
    payload = {
        "tokenizer": config["tokenizer"],
        "selection_manifest_sha256": sha256_file(selection_manifest),
        "schema": 1,
    }
    return config_fingerprint(payload)


def prepare_tokenizer(
    config: Mapping[str, Any],
    selection_manifest: Path,
    *,
    force: bool = False,
) -> Path:
    tokenizer_root = artifact_path(config, "tokenizer_dir")
    fingerprint = tokenizer_fingerprint(config, selection_manifest)
    manifest_path = tokenizer_root / "manifest.json"
    if not force and manifest_is_valid(
        manifest_path, expected_fingerprint=fingerprint, root=tokenizer_root
    ):
        print(f"[prepare] tokenizer valid; skipping: {tokenizer_root}")
        return manifest_path

    selected_root = selection_manifest.parent
    chars_per_language = int(config["tokenizer"]["train_chars_per_language"])
    special = config["tokenizer"]["special_tokens"]

    def text_iterator() -> Iterator[str]:
        for lang in LANGUAGES:
            dataset = _load_parquet(selected_root / f"{lang}_train.parquet")
            consumed = 0
            for text in dataset[config["data"]["text_field"]]:
                text = str(text)
                if not text:
                    continue
                remaining = chars_per_language - consumed
                if remaining <= 0:
                    break
                sample = text[:remaining]
                consumed += len(sample)
                yield sample

    with atomic_directory(tokenizer_root) as temporary:
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            text_iterator(),
            vocab_size=int(config["tokenizer"]["vocab_size"]),
            min_frequency=int(config["tokenizer"]["min_frequency"]),
            special_tokens=[
                special["unk_token"],
                special["bos_token"],
                special["eos_token"],
                special["pad_token"],
            ],
        )
        tokenizer.save_model(str(temporary))
        tokenizer_json = temporary / "tokenizer.json"
        tokenizer.save(str(tokenizer_json))
        fast = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json),
            unk_token=special["unk_token"],
            bos_token=special["bos_token"],
            eos_token=special["eos_token"],
            pad_token=special["pad_token"],
            model_max_length=int(config["tokenizer"]["model_max_length"]),
        )
        fast.save_pretrained(str(temporary))
        output_files = [
            path
            for path in temporary.iterdir()
            if path.is_file() and path.name != "manifest.json"
        ]
        manifest = {
            "complete": True,
            "fingerprint": fingerprint,
            "selection_manifest_sha256": sha256_file(selection_manifest),
            "vocab_size": len(fast),
            "special_token_ids": {
                "unk": fast.unk_token_id,
                "bos": fast.bos_token_id,
                "eos": fast.eos_token_id,
                "pad": fast.pad_token_id,
            },
            "files": [
                file_record(path, relative_to=temporary) for path in output_files
            ],
        }
        atomic_write_json(temporary / "manifest.json", manifest)
    return manifest_path


@dataclass
class _ShardWriter:
    root: Path
    prefix: str
    sequence_length: int
    dtype: np.dtype
    shard_blocks: int = 4_096

    def __post_init__(self) -> None:
        self.ids: list[np.ndarray] = []
        self.exposure: list[float] = []
        self.lengths: list[int] = []
        self.group_ids: list[int] = []
        self.domain_ids: list[int] = []
        self.records: list[dict[str, Any]] = []
        self.total_blocks = 0

    def add(
        self,
        ids: Iterable[int],
        exposure: float,
        *,
        length: int | None = None,
        group_id: int = 0,
        domain_id: int = 0,
    ) -> None:
        array = np.asarray(list(ids), dtype=self.dtype)
        if array.shape != (self.sequence_length,):
            raise ValueError(f"block has shape {array.shape}, expected {self.sequence_length}")
        self.ids.append(array)
        self.exposure.append(float(exposure))
        self.lengths.append(int(length or self.sequence_length))
        self.group_ids.append(int(group_id))
        self.domain_ids.append(int(domain_id))
        if len(self.ids) >= self.shard_blocks:
            self.flush()

    def flush(self) -> None:
        if not self.ids:
            return
        index = len(self.records)
        stem = f"{self.prefix}_{index:04d}"
        paths = {
            "ids": self.root / f"{stem}_ids.npy",
            "exposure": self.root / f"{stem}_exposure.npy",
            "lengths": self.root / f"{stem}_lengths.npy",
            "group_ids": self.root / f"{stem}_group_ids.npy",
            "domain_ids": self.root / f"{stem}_domain_ids.npy",
        }
        np.save(paths["ids"], np.stack(self.ids))
        np.save(paths["exposure"], np.asarray(self.exposure, dtype=np.float64))
        np.save(paths["lengths"], np.asarray(self.lengths, dtype=np.int32))
        np.save(paths["group_ids"], np.asarray(self.group_ids, dtype=np.uint64))
        np.save(paths["domain_ids"], np.asarray(self.domain_ids, dtype=np.int32))
        count = len(self.ids)
        self.records.append(
            {
                "count": count,
                **{name: path.name for name, path in paths.items()},
            }
        )
        self.total_blocks += count
        self.ids.clear()
        self.exposure.clear()
        self.lengths.clear()
        self.group_ids.clear()
        self.domain_ids.clear()


def _consume_segments(segments: deque[list[float]], count: int) -> float:
    remaining = count
    units = 0.0
    while remaining:
        segment_count, unit_per_token = segments[0]
        take = min(remaining, int(segment_count))
        units += take * unit_per_token
        segment_count -= take
        remaining -= take
        if segment_count <= 0:
            segments.popleft()
        else:
            segments[0][0] = segment_count
    return units


def _group_hash(document_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(document_id.encode("utf-8"), digest_size=8).digest(), "little"
    )


def blocks_fingerprint(
    config: Mapping[str, Any], selection_manifest: Path, tokenizer_manifest: Path
) -> str:
    payload = {
        "sequence_length": int(config["training"]["sequence_length"]),
        "selection_manifest_sha256": sha256_file(selection_manifest),
        "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest),
        "schema": 1,
    }
    return config_fingerprint(payload)


def prepare_blocks(
    config: Mapping[str, Any],
    selection_manifest: Path,
    tokenizer_manifest: Path,
    *,
    force: bool = False,
) -> Path:
    data_root = artifact_path(config, "data_dir")
    blocks_root = data_root / "blocks"
    fingerprint = blocks_fingerprint(config, selection_manifest, tokenizer_manifest)
    manifest_path = blocks_root / BLOCKS_MANIFEST
    if not force and manifest_is_valid(
        manifest_path, expected_fingerprint=fingerprint, root=blocks_root
    ):
        print(f"[prepare] packed blocks valid; skipping: {blocks_root}")
        return manifest_path

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(artifact_path(config, "tokenizer_dir"))
    )
    sequence_length = int(config["training"]["sequence_length"])
    dtype = np.uint16 if len(tokenizer) <= np.iinfo(np.uint16).max else np.int32
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id)
    selected_root = selection_manifest.parent

    with atomic_directory(blocks_root) as temporary:
        split_manifest: dict[str, Any] = {}
        token_frequency = np.zeros(len(tokenizer), dtype=np.int64)
        all_outputs: list[Path] = []
        for lang in LANGUAGES:
            split_manifest[lang] = {}
            for split in ("train", "validation"):
                dataset = _load_parquet(selected_root / f"{lang}_{split}.parquet")
                writer = _ShardWriter(
                    temporary,
                    f"{lang}_{split}",
                    sequence_length,
                    np.dtype(dtype),
                )
                domain_values = sorted(
                    {str(value) for value in dataset[config["data"]["domain_field"]]}
                )
                domain_to_id = {value: index for index, value in enumerate(domain_values)}
                total_exposure = 0.0
                total_model_tokens = 0

                if split == "train":
                    token_buffer: list[int] = []
                    frequency_buffer: list[int] = []
                    offset = 0
                    segments: deque[list[float]] = deque()
                    for record in dataset:
                        ids = tokenizer.encode(
                            str(record[config["data"]["text_field"]]),
                            add_special_tokens=False,
                            verbose=False,
                        )
                        ids.append(eos_id)
                        if not ids:
                            continue
                        units = float(record["_babymoe_adjusted_units"])
                        token_buffer.extend(ids)
                        frequency_buffer.extend(ids)
                        segments.append([float(len(ids)), units / len(ids)])
                        if len(frequency_buffer) >= 1_000_000:
                            token_frequency += np.bincount(
                                frequency_buffer, minlength=len(tokenizer)
                            )
                            frequency_buffer.clear()
                        while len(token_buffer) - offset >= sequence_length:
                            block = token_buffer[offset : offset + sequence_length]
                            exposure = _consume_segments(segments, sequence_length)
                            writer.add(block, exposure)
                            offset += sequence_length
                            total_exposure += exposure
                            total_model_tokens += sequence_length
                        if offset >= sequence_length * 16:
                            token_buffer = token_buffer[offset:]
                            offset = 0
                    if frequency_buffer:
                        token_frequency += np.bincount(
                            frequency_buffer, minlength=len(tokenizer)
                        )
                else:
                    doc_field = config["data"]["document_id_field"]
                    domain_field = config["data"]["domain_field"]
                    for row_index, record in enumerate(dataset):
                        ids = tokenizer.encode(
                            str(record[config["data"]["text_field"]]),
                            add_special_tokens=False,
                            verbose=False,
                        )
                        ids.append(eos_id)
                        units = float(record["_babymoe_adjusted_units"])
                        unit_per_token = units / max(1, len(ids))
                        document_id = str(record.get(doc_field) or row_index)
                        domain = str(record.get(domain_field) or "unknown")
                        for start in range(0, len(ids), sequence_length):
                            chunk = ids[start : start + sequence_length]
                            length = len(chunk)
                            padded = chunk + [pad_id] * (sequence_length - length)
                            exposure = length * unit_per_token
                            writer.add(
                                padded,
                                exposure,
                                length=length,
                                group_id=_group_hash(document_id),
                                domain_id=domain_to_id[domain],
                            )
                            total_exposure += exposure
                            total_model_tokens += length

                writer.flush()
                for record in writer.records:
                    for key in ("ids", "exposure", "lengths", "group_ids", "domain_ids"):
                        all_outputs.append(temporary / record[key])
                split_manifest[lang][split] = {
                    "blocks": writer.total_blocks,
                    "model_tokens": total_model_tokens,
                    "adjusted_exposure_units": total_exposure,
                    "domains": domain_values,
                    "shards": writer.records,
                }
                print(
                    f"[prepare] {lang}/{split}: {writer.total_blocks:,} blocks, "
                    f"{total_exposure:,.0f} adjusted units"
                )

        frequency_path = temporary / "token_frequency.npy"
        np.save(frequency_path, token_frequency)
        all_outputs.append(frequency_path)
        actual_unique = sum(
            split_manifest[lang]["train"]["adjusted_exposure_units"]
            for lang in LANGUAGES
        )
        manifest = {
            "complete": True,
            "fingerprint": fingerprint,
            "selection_manifest_sha256": sha256_file(selection_manifest),
            "tokenizer_manifest_sha256": sha256_file(tokenizer_manifest),
            "sequence_length": sequence_length,
            "dtype": np.dtype(dtype).name,
            "vocab_size": len(tokenizer),
            "actual_unique_exposure_units": actual_unique,
            "max_compliant_exposure_units": actual_unique
            * int(config["data"]["max_epochs"]),
            "splits": split_manifest,
            "token_frequency": frequency_path.name,
            "files": [file_record(path, relative_to=temporary) for path in all_outputs],
        }
        atomic_write_json(temporary / BLOCKS_MANIFEST, manifest)
    return manifest_path


def prepare_all(
    config: Mapping[str, Any],
    *,
    token: str | None = None,
    synthetic: bool = False,
    force: bool = False,
) -> dict[str, str]:
    selection = prepare_selection(
        config, token=token, synthetic=synthetic, force=force
    )
    tokenizer = prepare_tokenizer(config, selection, force=force)
    blocks = prepare_blocks(config, selection, tokenizer, force=force)
    return {
        "selection_manifest": str(selection),
        "tokenizer_manifest": str(tokenizer),
        "blocks_manifest": str(blocks),
    }


class BlockStore:
    """Random access over a split's memory-mapped shards."""

    def __init__(self, root: Path, split_payload: Mapping[str, Any]) -> None:
        self.root = root
        self.shards = list(split_payload["shards"])
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def _load_shard(self, index: int) -> dict[str, np.ndarray]:
        if index not in self._cache:
            record = self.shards[index]
            self._cache[index] = {
                key: np.load(self.root / record[key], mmap_mode="r")
                for key in ("ids", "exposure", "lengths", "group_ids", "domain_ids")
            }
        return self._cache[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[shard_index - 1] if shard_index else 0
        local = index - previous
        arrays = self._load_shard(shard_index)
        return {
            "input_ids": np.asarray(arrays["ids"][local], dtype=np.int64),
            "exposure": float(arrays["exposure"][local]),
            "length": int(arrays["lengths"][local]),
            "group_id": int(arrays["group_ids"][local]),
            "domain_id": int(arrays["domain_ids"][local]),
        }


class PreparedBlocks:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("complete"):
            raise ValueError(f"incomplete block manifest: {manifest_path}")
        self._stores: dict[tuple[str, str], BlockStore] = {}

    def store(self, lang: str, split: str) -> BlockStore:
        key = (lang, split)
        if key not in self._stores:
            self._stores[key] = BlockStore(
                self.root, self.manifest["splits"][lang][split]
            )
        return self._stores[key]

    @property
    def token_frequency(self) -> np.ndarray:
        return np.load(self.root / self.manifest["token_frequency"], mmap_mode="r")


class ExposureLimitReached(StopIteration):
    """Raised when no additional full training block is compliant."""


class TrilingualBatchStream:
    """Deterministic deficit-balanced block stream with resumable cursors."""

    def __init__(
        self,
        prepared: PreparedBlocks,
        config: Mapping[str, Any],
        *,
        seed: int,
        stop_at_exposure: int | None = None,
    ) -> None:
        self.prepared = prepared
        self.config = config
        self.seed = int(seed)
        self.mixture = {
            lang: float(config["data"]["mixture"][lang]) for lang in LANGUAGES
        }
        self.max_epochs = int(config["data"]["max_epochs"])
        configured_target = int(config["training"]["target_exposure_units"])
        compliant_target = math.floor(
            float(prepared.manifest["max_compliant_exposure_units"])
        )
        self.target_exposure = min(configured_target, compliant_target)
        if stop_at_exposure is not None:
            self.target_exposure = min(self.target_exposure, int(stop_at_exposure))
        self.stores = {
            lang: prepared.store(lang, "train") for lang in LANGUAGES
        }
        if any(len(store) == 0 for store in self.stores.values()):
            raise ValueError("every language must contain at least one training block")
        self.unique_exposure = {
            lang: float(
                prepared.manifest["splits"][lang]["train"][
                    "adjusted_exposure_units"
                ]
            )
            for lang in LANGUAGES
        }
        self.epochs = {lang: 0 for lang in LANGUAGES}
        self.cursors = {lang: 0 for lang in LANGUAGES}
        self.consumed = {lang: 0.0 for lang in LANGUAGES}
        self.total_consumed = 0.0
        self._permutations: dict[str, np.ndarray] = {}
        self._rng = random.Random(self.seed)

    def _permutation(self, lang: str) -> np.ndarray:
        key = f"{lang}:{self.epochs[lang]}"
        cached = self._permutations.get(key)
        if cached is None:
            lang_offset = {"eng": 11, "nld": 23, "zho": 37}[lang]
            generator = np.random.default_rng(
                self.seed + lang_offset + 1_000_003 * self.epochs[lang]
            )
            cached = generator.permutation(len(self.stores[lang]))
            self._permutations[key] = cached
        return cached

    def _next_for_language(self, lang: str) -> dict[str, Any]:
        if self.cursors[lang] >= len(self.stores[lang]):
            self.epochs[lang] += 1
            self.cursors[lang] = 0
        if self.epochs[lang] >= self.max_epochs:
            raise ExposureLimitReached(f"{lang} reached {self.max_epochs} epochs")
        permutation = self._permutation(lang)
        index = int(permutation[self.cursors[lang]])
        self.cursors[lang] += 1
        return self.stores[lang][index]

    def _pick_language(self) -> str:
        if self.total_consumed <= 0:
            return self._rng.choices(
                list(LANGUAGES),
                weights=[self.mixture[lang] for lang in LANGUAGES],
                k=1,
            )[0]
        deficits = {
            lang: self.mixture[lang] - self.consumed[lang] / self.total_consumed
            for lang in LANGUAGES
        }
        maximum = max(deficits.values())
        contenders = [
            lang for lang in LANGUAGES if deficits[lang] >= maximum - 1e-12
        ]
        return self._rng.choice(contenders)

    def next_batch(self, batch_size: int) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        languages: list[str] = []
        for _ in range(batch_size):
            if self.total_consumed >= self.target_exposure:
                break
            lang = self._pick_language()
            row = self._next_for_language(lang)
            exposure = float(row["exposure"])
            rows.append(row)
            languages.append(lang)
            self.consumed[lang] += exposure
            self.total_consumed += exposure
            # A checkpoint is genuine once the threshold has been crossed.
            # Stop on that block rather than under-counting by refusing it.
            if self.total_consumed >= self.target_exposure:
                break
        if not rows:
            raise ExposureLimitReached(
                f"target adjusted exposure reached ({self.total_consumed:,.0f})"
            )
        return {
            "input_ids": np.stack([row["input_ids"] for row in rows]),
            "exposure": np.asarray([row["exposure"] for row in rows], dtype=np.float64),
            "languages": languages,
        }

    def epochs_seen(self) -> dict[str, float]:
        return {
            lang: self.consumed[lang] / max(1.0, self.unique_exposure[lang])
            for lang in LANGUAGES
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "target_exposure": self.target_exposure,
            "epochs": dict(self.epochs),
            "cursors": dict(self.cursors),
            "consumed": dict(self.consumed),
            "total_consumed": self.total_consumed,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("cannot resume a data stream with a different seed")
        previous_target = int(state["target_exposure"])
        if self.target_exposure < previous_target:
            raise ValueError("resume target may be extended but not reduced")
        if self.target_exposure < float(state["total_consumed"]):
            raise ValueError("resume target is below already consumed exposure")
        self.epochs = {lang: int(state["epochs"][lang]) for lang in LANGUAGES}
        self.cursors = {lang: int(state["cursors"][lang]) for lang in LANGUAGES}
        self.consumed = {
            lang: float(state["consumed"][lang]) for lang in LANGUAGES
        }
        self.total_consumed = float(state["total_consumed"])
        self._rng.setstate(state["rng_state"])
        self._permutations.clear()


def validation_batches(
    prepared: PreparedBlocks,
    *,
    lang: str,
    batch_size: int,
    max_batches: int,
    pad_token_id: int,
) -> Iterator[dict[str, np.ndarray]]:
    store = prepared.store(lang, "validation")
    cursor = 0
    for _ in range(max_batches):
        rows = [store[index] for index in range(cursor, min(cursor + batch_size, len(store)))]
        if not rows:
            return
        ids = np.stack([row["input_ids"] for row in rows])
        lengths = np.asarray([row["length"] for row in rows], dtype=np.int32)
        positions = np.arange(ids.shape[1])[None, :]
        attention = positions < lengths[:, None]
        labels = ids.copy()
        labels[~attention] = -100
        yield {
            "input_ids": ids,
            "attention_mask": attention.astype(np.int64),
            "labels": labels,
            "group_ids": np.asarray([row["group_id"] for row in rows], dtype=np.uint64),
            "domain_ids": np.asarray([row["domain_id"] for row in rows], dtype=np.int32),
            "lengths": lengths,
        }
        cursor += len(rows)
