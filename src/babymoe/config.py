"""YAML configuration loading, validation, and reproducibility helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

LANGUAGES = ("eng", "nld", "zho")
MAX_UNIQUE_EXPOSURE = 100_000_000
MAX_TOTAL_EXPOSURE = 1_000_000_000
MAX_EPOCHS = 10


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_tree(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"cyclic config include detected at {path}")
    seen.add(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    include = raw.pop("include", None)
    if include is None:
        return raw
    parent = _load_yaml_tree(path.parent / include, seen)
    return _deep_merge(parent, raw)


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(
    path: str | Path,
    *,
    seed: int | None = None,
    artifacts_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load an included YAML config and apply operational overrides."""

    config_path = Path(path).resolve()
    config = _load_yaml_tree(config_path)
    config.setdefault("experiment", {})["config_path"] = str(config_path)
    if seed is not None:
        config["experiment"]["seed"] = int(seed)
    else:
        config["experiment"].setdefault(
            "seed", int(config["experiment"].get("primary_seed", 42))
        )
    if artifacts_root is not None:
        config.setdefault("paths", {})["artifacts_root"] = str(
            Path(artifacts_root).expanduser()
        )
    validate_config(config)
    config["experiment"]["config_fingerprint"] = config_fingerprint(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"experiment", "paths", "data", "tokenizer", "model", "training"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing config sections: {missing}")

    data = config["data"]
    repos = data.get("repos", {})
    revisions = data.get("revisions", {})
    premiums = data.get("byte_premium", {})
    mixture = data.get("mixture", {})
    if tuple(repos) != LANGUAGES and set(repos) != set(LANGUAGES):
        raise ValueError(f"data.repos must contain exactly {LANGUAGES}")
    if set(revisions) != set(LANGUAGES) or any(
        not str(value) for value in revisions.values()
    ):
        raise ValueError("data.revisions must pin eng/nld/zho source commits")
    if set(premiums) != set(LANGUAGES) or any(float(v) <= 0 for v in premiums.values()):
        raise ValueError("data.byte_premium must contain positive eng/nld/zho values")
    if set(mixture) != set(LANGUAGES) or any(float(v) < 0 for v in mixture.values()):
        raise ValueError("data.mixture must contain non-negative eng/nld/zho values")
    if abs(sum(float(v) for v in mixture.values()) - 1.0) > 1e-6:
        raise ValueError("data.mixture must sum to 1")

    unique_budget = int(data.get("unique_exposure_units", 0))
    minimum_fill = float(data.get("minimum_quota_fill_ratio", 0))
    total_budget = int(config["training"].get("target_exposure_units", 0))
    max_epochs = int(data.get("max_epochs", 0))
    if not 0 < unique_budget <= MAX_UNIQUE_EXPOSURE:
        raise ValueError(f"unique exposure must be in (0, {MAX_UNIQUE_EXPOSURE}]")
    if not 0 < minimum_fill <= 1:
        raise ValueError("minimum_quota_fill_ratio must be in (0, 1]")
    if not 0 < total_budget <= MAX_TOTAL_EXPOSURE:
        raise ValueError(f"total exposure must be in (0, {MAX_TOTAL_EXPOSURE}]")
    if not 0 < max_epochs <= MAX_EPOCHS:
        raise ValueError(f"max epochs must be in (0, {MAX_EPOCHS}]")
    if total_budget > unique_budget * max_epochs:
        raise ValueError("target exposure exceeds unique corpus size × max_epochs")

    model = config["model"]
    if model.get("family") not in {"mistral", "mixtral"}:
        raise ValueError("model.family must be 'mistral' or 'mixtral'")
    hidden = int(model.get("hidden_size", 0))
    heads = int(model.get("num_attention_heads", 0))
    if hidden <= 0 or heads <= 0 or hidden % heads:
        raise ValueError("hidden_size must be positive and divisible by attention heads")
    if int(model.get("max_position_embeddings", 0)) < int(
        config["training"].get("sequence_length", 0)
    ):
        raise ValueError("model context is shorter than training.sequence_length")
    if model["family"] == "mixtral":
        experts = int(model.get("num_local_experts", 0))
        top_k = int(model.get("num_experts_per_tok", 0))
        if not 0 < top_k <= experts:
            raise ValueError("Mixtral top-k must be between 1 and num_local_experts")


def artifact_path(config: Mapping[str, Any], key: str) -> Path:
    paths = config["paths"]
    root = Path(paths["artifacts_root"]).expanduser()
    value = Path(paths[key]).expanduser()
    return value if value.is_absolute() else root / value


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)
