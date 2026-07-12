from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from babymoe.routing import (
    RouterAccumulator,
    jensen_shannon,
    save_routing_artifacts,
    summarize_multilingual_routing,
)


def test_jensen_shannon_known_cases():
    assert jensen_shannon(np.asarray([0.5, 0.5]), np.asarray([0.5, 0.5])) == pytest.approx(0)
    assert jensen_shannon(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == pytest.approx(
        np.log(2), rel=1e-6
    )


def _accumulator(lang: str, edge_expert: int) -> RouterAccumulator:
    accumulator = RouterAccumulator(
        language=lang,
        num_layers=6,
        num_experts=2,
        top_k=1,
        snapshot_limit=8,
    )
    layers = []
    for layer in range(6):
        if layer in (0, 5):
            logits = [6.0, -6.0] if edge_expert == 0 else [-6.0, 6.0]
        else:
            logits = [0.0, 0.0]
        layers.append(torch.tensor([logits, logits], dtype=torch.float32))
    accumulator.update(
        layers,
        input_ids=torch.tensor([[4, 5]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        group_ids=np.asarray([11], dtype=np.uint64),
        domain_ids=np.asarray([0], dtype=np.int32),
    )
    return accumulator


def test_router_aggregation_depth_statistic_and_save(tmp_path):
    accumulators = {
        "eng": _accumulator("eng", 0),
        "nld": _accumulator("nld", 0),
        "zho": _accumulator("zho", 1),
    }
    summary = summarize_multilingual_routing(accumulators)
    assert summary["edge_minus_middle_en_zho_js"] > 0
    assert summary["pairwise_js_by_layer"]["eng-nld"] == pytest.approx([0] * 6)
    first_layer = summary["languages"]["eng"]["layers"][0]
    assert first_layer["tokens"] == 2
    assert first_layer["minimum_expert_share"] == 0

    summary_path, snapshot_path = save_routing_artifacts(
        tmp_path,
        milestone_units=1_000_000,
        summary=summary,
        accumulators=accumulators,
    )
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    snapshot = np.load(snapshot_path)
    assert saved["edge_minus_middle_en_zho_js"] > 0
    assert snapshot["router_probabilities"].shape == (6, 6, 2)
