"""Shared filesystem, hashing, logging, and environment utilities."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


@contextlib.contextmanager
def atomic_directory(path: str | Path) -> Iterator[Path]:
    """Build a directory beside its destination and swap it in on success."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.previous")
    if not destination.exists() and backup.exists():
        os.replace(backup, destination)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        yield temporary
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            os.replace(destination, backup)
        os.replace(temporary, destination)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if not destination.exists() and backup.exists():
            os.replace(backup, destination)
        raise


def manifest_is_valid(
    manifest_path: str | Path,
    *,
    expected_fingerprint: str,
    root: str | Path | None = None,
) -> bool:
    path = Path(manifest_path)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not payload.get("complete") or payload.get("fingerprint") != expected_fingerprint:
        return False
    base = Path(root) if root is not None else path.parent
    for item in payload.get("files", []):
        candidate = base / item["path"]
        if not candidate.is_file() or candidate.stat().st_size != int(item["size"]):
            return False
        if sha256_file(candidate) != item["sha256"]:
            return False
    return True


def file_record(path: str | Path, *, relative_to: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    return {
        "path": str(candidate.relative_to(relative_to)),
        "size": candidate.stat().st_size,
        "sha256": sha256_file(candidate),
    }


def get_git_sha(repo_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        summary.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except ImportError:
        summary["torch"] = None
    for package in (
        "transformers",
        "datasets",
        "tokenizers",
        "accelerate",
        "numpy",
        "huggingface-hub",
    ):
        try:
            summary[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            summary[package] = None
    return summary


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()
