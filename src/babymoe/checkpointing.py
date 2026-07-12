"""BabyLM exposure milestones and interruption-safe checkpoint utilities."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .utils import atomic_directory, atomic_write_json


def checkpoint_milestones(
    checkpoint_cfg: Mapping[str, Any], max_exposure: int
) -> list[int]:
    phases = (
        (
            int(checkpoint_cfg["phase1_every_units"]),
            int(checkpoint_cfg["phase1_max_units"]),
        ),
        (
            int(checkpoint_cfg["phase2_every_units"]),
            int(checkpoint_cfg["phase2_max_units"]),
        ),
        (
            int(checkpoint_cfg["phase3_every_units"]),
            int(checkpoint_cfg["phase3_max_units"]),
        ),
    )
    milestones: set[int] = set()
    previous_max = 0
    for every, phase_max in phases:
        if every <= 0 or phase_max <= 0:
            continue
        start = every if previous_max == 0 else previous_max + every
        first = ((start + every - 1) // every) * every
        milestones.update(range(first, min(phase_max, max_exposure) + 1, every))
        previous_max = max(previous_max, phase_max)
    return sorted(value for value in milestones if 0 < value <= max_exposure)


def revision_name(exposure_units: int, prefix: str = "chck_") -> str:
    if exposure_units % 1_000_000:
        raise ValueError("official revision exposure must be a whole million")
    return f"{prefix}{exposure_units // 1_000_000}M"


def save_inference_checkpoint(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    checkpoint_root: str | Path,
    milestone_units: int,
    actual_exposure_units: float,
    metadata: Mapping[str, Any],
    prefix: str = "chck_",
) -> Path:
    root = Path(checkpoint_root)
    destination = root / revision_name(milestone_units, prefix)
    with atomic_directory(destination) as temporary:
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(temporary)
        atomic_write_json(
            temporary / "checkpoint_meta.json",
            {
                **dict(metadata),
                "milestone_exposure_units": milestone_units,
                "actual_exposure_units": actual_exposure_units,
                "revision": destination.name,
            },
        )
    return destination


def save_final_checkpoint(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    destination: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    path = Path(destination)
    with atomic_directory(path) as temporary:
        model.save_pretrained(
            temporary,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(temporary)
        atomic_write_json(temporary / "checkpoint_meta.json", dict(metadata))
    return path


def save_resume_state(path: str | Path, state: Mapping[str, Any]) -> Path:
    """Atomically rotate latest/previous trainer-state files."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = destination.with_name("previous.pt")
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(state), handle)
        handle.flush()
        os.fsync(handle.fileno())
    if previous.exists():
        previous.unlink()
    if destination.exists():
        os.replace(destination, previous)
    os.replace(temporary, destination)
    return destination


def load_resume_state(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def find_resume_state(resume_dir: str | Path) -> Path | None:
    root = Path(resume_dir)
    for name in ("latest.pt", "previous.pt"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def clean_temporary_resume(path: str | Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        Path(path).with_suffix(".tmp").unlink()
