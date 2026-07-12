"""Compact multilingual router telemetry and artifact serialization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .config import LANGUAGES
from .utils import atomic_write_json


def jensen_shannon(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * np.log(p / midpoint))
        + 0.5 * np.sum(q * np.log(q / midpoint))
    )


def _reshape_router(
    logits: torch.Tensor, batch_size: int, sequence_length: int
) -> torch.Tensor:
    if logits.ndim == 3:
        if logits.shape[:2] != (batch_size, sequence_length):
            raise ValueError(f"unexpected router shape: {tuple(logits.shape)}")
        return logits
    if logits.ndim == 2 and logits.shape[0] == batch_size * sequence_length:
        return logits.reshape(batch_size, sequence_length, logits.shape[-1])
    raise ValueError(f"unexpected router shape: {tuple(logits.shape)}")


@dataclass
class RouterAccumulator:
    language: str
    num_layers: int
    num_experts: int
    top_k: int
    snapshot_limit: int

    def __post_init__(self) -> None:
        self.sum_probabilities = np.zeros(
            (self.num_layers, self.num_experts), dtype=np.float64
        )
        self.selected_counts = np.zeros_like(self.sum_probabilities)
        self.entropy_sum = np.zeros(self.num_layers, dtype=np.float64)
        self.token_count = np.zeros(self.num_layers, dtype=np.int64)
        self.snapshots: dict[str, list[np.ndarray]] = {
            "router_probabilities": [],
            "token_ids": [],
            "positions": [],
            "group_ids": [],
            "domain_ids": [],
        }
        self.snapshot_count = 0

    def update(
        self,
        router_logits: Iterable[torch.Tensor],
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_ids: np.ndarray,
        domain_ids: np.ndarray,
    ) -> None:
        layers = tuple(router_logits)
        if len(layers) != self.num_layers:
            raise ValueError(f"received {len(layers)} router layers, expected {self.num_layers}")
        batch_size, sequence_length = input_ids.shape
        valid = attention_mask.bool()
        layer_probabilities: list[torch.Tensor] = []
        for layer_index, logits in enumerate(layers):
            shaped = _reshape_router(logits, batch_size, sequence_length)
            probabilities = torch.softmax(shaped.float(), dim=-1)
            selected = torch.topk(probabilities, self.top_k, dim=-1).indices
            valid_probabilities = probabilities[valid]
            valid_selected = selected[valid]
            if valid_probabilities.numel() == 0:
                continue
            self.sum_probabilities[layer_index] += (
                valid_probabilities.sum(dim=0).detach().cpu().numpy()
            )
            self.selected_counts[layer_index] += (
                torch.nn.functional.one_hot(
                    valid_selected, num_classes=self.num_experts
                )
                .sum(dim=(0, 1))
                .detach()
                .cpu()
                .numpy()
            )
            entropy = -(
                valid_probabilities
                * torch.log(valid_probabilities.clamp_min(1e-12))
            ).sum(dim=-1)
            self.entropy_sum[layer_index] += float(entropy.sum().item())
            self.token_count[layer_index] += int(valid_probabilities.shape[0])
            layer_probabilities.append(probabilities)

        remaining = self.snapshot_limit - self.snapshot_count
        if remaining <= 0 or len(layer_probabilities) != self.num_layers:
            return
        flat_valid = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        chosen = flat_valid[:remaining]
        if chosen.numel() == 0:
            return
        stacked = torch.stack(layer_probabilities, dim=2)
        chosen_probabilities = stacked.reshape(
            -1, self.num_layers, self.num_experts
        )[chosen]
        flat_ids = input_ids.reshape(-1)[chosen]
        flat_positions = (
            torch.arange(sequence_length, device=input_ids.device)
            .repeat(batch_size)[chosen]
        )
        sequence_indices = torch.div(chosen, sequence_length, rounding_mode="floor")
        sequence_indices_cpu = sequence_indices.detach().cpu().numpy()
        self.snapshots["router_probabilities"].append(
            chosen_probabilities.detach().cpu().to(torch.float16).numpy()
        )
        self.snapshots["token_ids"].append(flat_ids.detach().cpu().numpy())
        self.snapshots["positions"].append(flat_positions.detach().cpu().numpy())
        self.snapshots["group_ids"].append(group_ids[sequence_indices_cpu])
        self.snapshots["domain_ids"].append(domain_ids[sequence_indices_cpu])
        self.snapshot_count += int(chosen.numel())

    def finalize(self) -> dict[str, Any]:
        layers = []
        for layer_index in range(self.num_layers):
            count = max(1, int(self.token_count[layer_index]))
            mean_probabilities = self.sum_probabilities[layer_index] / count
            load = self.selected_counts[layer_index] / max(1, count * self.top_k)
            load_mean = float(load.mean())
            load_cv = float(load.std() / load_mean) if load_mean else float("inf")
            normalized_entropy = float(
                self.entropy_sum[layer_index]
                / count
                / max(1e-12, math.log(self.num_experts))
            )
            layers.append(
                {
                    "layer": layer_index,
                    "tokens": count,
                    "mean_probabilities": mean_probabilities.tolist(),
                    "topk_load": load.tolist(),
                    "normalized_entropy": normalized_entropy,
                    "load_cv": load_cv,
                    "minimum_expert_share": float(load.min()),
                }
            )
        return {"language": self.language, "layers": layers}

    def snapshot_arrays(self) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for key, parts in self.snapshots.items():
            if parts:
                output[key] = np.concatenate(parts, axis=0)
            elif key == "router_probabilities":
                output[key] = np.empty(
                    (0, self.num_layers, self.num_experts), dtype=np.float16
                )
            else:
                output[key] = np.empty((0,), dtype=np.int64)
        output["language"] = np.asarray(
            [self.language] * len(output["token_ids"]), dtype="U3"
        )
        return output


def summarize_multilingual_routing(
    accumulators: Mapping[str, RouterAccumulator],
    *,
    previous_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    languages = {lang: accumulators[lang].finalize() for lang in LANGUAGES}
    num_layers = accumulators[LANGUAGES[0]].num_layers
    pairwise: dict[str, list[float]] = {}
    for left, right in (("eng", "nld"), ("eng", "zho"), ("nld", "zho")):
        values = []
        for layer in range(num_layers):
            p = np.asarray(
                languages[left]["layers"][layer]["mean_probabilities"]
            )
            q = np.asarray(
                languages[right]["layers"][layer]["mean_probabilities"]
            )
            values.append(jensen_shannon(p, q))
        pairwise[f"{left}-{right}"] = values

    edge_width = max(1, num_layers // 6)
    middle_width = max(1, num_layers // 3)
    edge_indices = list(range(edge_width)) + list(
        range(num_layers - edge_width, num_layers)
    )
    middle_start = (num_layers - middle_width) // 2
    middle_indices = list(range(middle_start, middle_start + middle_width))
    en_zho = np.asarray(pairwise["eng-zho"])
    u_index = float(en_zho[edge_indices].mean() - en_zho[middle_indices].mean())

    stability: dict[str, Any] | None = None
    if previous_summary is not None:
        stability = {}
        for lang in LANGUAGES:
            current = np.asarray(
                [
                    row["mean_probabilities"]
                    for row in languages[lang]["layers"]
                ],
                dtype=np.float64,
            )
            previous = np.asarray(
                [
                    row["mean_probabilities"]
                    for row in previous_summary["languages"][lang]["layers"]
                ],
                dtype=np.float64,
            )
            stability[lang] = {
                "mean_js_from_previous": float(
                    np.mean(
                        [
                            jensen_shannon(current[layer], previous[layer])
                            for layer in range(num_layers)
                        ]
                    )
                )
            }

    return {
        "languages": languages,
        "pairwise_js_by_layer": pairwise,
        "edge_layers": edge_indices,
        "middle_layers": middle_indices,
        "edge_minus_middle_en_zho_js": u_index,
        "stability": stability,
    }


def save_routing_artifacts(
    output_dir: str | Path,
    *,
    milestone_units: int,
    summary: Mapping[str, Any],
    accumulators: Mapping[str, RouterAccumulator],
    save_snapshot: bool = True,
) -> tuple[Path, Path | None]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"routing_{milestone_units // 1_000_000}M"
    summary_path = root / f"{stem}.json"
    snapshot_path = root / f"{stem}.npz"
    atomic_write_json(summary_path, dict(summary))
    if not save_snapshot:
        return summary_path, None

    arrays: dict[str, list[np.ndarray]] = {}
    for lang in LANGUAGES:
        for key, value in accumulators[lang].snapshot_arrays().items():
            arrays.setdefault(key, []).append(value)
    combined = {
        key: np.concatenate(values, axis=0) if values else np.empty((0,))
        for key, values in arrays.items()
    }
    temporary = snapshot_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **combined)
    temporary.replace(snapshot_path)
    return summary_path, snapshot_path


def load_latest_routing_summary(
    output_dir: str | Path, before_milestone: int
) -> dict[str, Any] | None:
    candidates: list[tuple[int, Path]] = []
    for path in Path(output_dir).glob("routing_*M.json"):
        try:
            milestone = int(path.stem.removeprefix("routing_").removesuffix("M"))
        except ValueError:
            continue
        if milestone * 1_000_000 < before_milestone:
            candidates.append((milestone, path))
    if not candidates:
        return None
    return json.loads(max(candidates, key=lambda item: item[0])[1].read_text())
