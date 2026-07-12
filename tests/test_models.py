from __future__ import annotations

from copy import deepcopy

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from babymoe.config import load_config
from babymoe.models import (
    assert_active_compute_match,
    build_model,
    model_accounting,
)

from conftest import REPO_ROOT


def _tiny_moe_config():
    config = load_config(REPO_ROOT / "configs/moe_learned.yaml")
    config = deepcopy(config)
    config["model"].update(
        {
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 128,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    config["training"]["sequence_length"] = 128
    return config


def test_dense_and_moe_active_ffn_match():
    dense = load_config(REPO_ROOT / "configs/dense_match.yaml")
    moe = load_config(REPO_ROOT / "configs/moe_learned.yaml")
    assert_active_compute_match(dense["model"], moe["model"])


def test_native_moe_build_freeze_and_auto_loading(tmp_path, tiny_tokenizer):
    config = _tiny_moe_config()
    model = build_model(config, tiny_tokenizer)
    accounting = model_accounting(model, config)
    assert accounting.total_parameters > accounting.active_ffn_parameters_per_token
    batch = torch.tensor([[1, 4, 5, 2]])
    outputs = model(
        input_ids=batch,
        labels=batch,
        output_router_logits=True,
        use_cache=False,
    )
    assert torch.isfinite(outputs.loss)
    assert len(outputs.router_logits) == 2

    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    tiny_tokenizer.save_pretrained(checkpoint)
    causal = AutoModelForCausalLM.from_pretrained(checkpoint)
    sequence = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3, ignore_mismatched_sizes=True
    )
    token = AutoModelForTokenClassification.from_pretrained(
        checkpoint, num_labels=3, ignore_mismatched_sizes=True
    )
    assert causal.config.model_type == "mixtral"
    assert sequence.config.num_labels == 3
    assert token.config.num_labels == 3

    frozen = deepcopy(config)
    frozen["model"]["router_mode"] = "frozen_random_gate"
    frozen_model = build_model(frozen, tiny_tokenizer)
    gate_parameters = [
        parameter
        for name, parameter in frozen_model.named_parameters()
        if name.endswith("block_sparse_moe.gate.weight")
    ]
    assert len(gate_parameters) == 2
    assert all(not parameter.requires_grad for parameter in gate_parameters)
