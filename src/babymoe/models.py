"""Native Transformers model construction and dense/MoE fairness accounting."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import torch
from transformers import (
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass(frozen=True)
class ModelAccounting:
    family: str
    total_parameters: int
    trainable_parameters: int
    active_ffn_parameters_per_token: int
    total_ffn_parameters: int
    active_ffn_flops_per_token: int
    estimated_forward_flops_per_token: int
    estimated_training_flops_per_token: int
    num_experts: int
    top_k: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _common_config_kwargs(
    model_cfg: Mapping[str, Any], tokenizer: PreTrainedTokenizerBase
) -> dict[str, Any]:
    return {
        "vocab_size": len(tokenizer),
        "hidden_size": int(model_cfg["hidden_size"]),
        "intermediate_size": int(model_cfg["intermediate_size"]),
        "num_hidden_layers": int(model_cfg["num_hidden_layers"]),
        "num_attention_heads": int(model_cfg["num_attention_heads"]),
        "num_key_value_heads": int(model_cfg["num_key_value_heads"]),
        "hidden_act": str(model_cfg["hidden_act"]),
        "max_position_embeddings": int(model_cfg["max_position_embeddings"]),
        "initializer_range": float(model_cfg["initializer_range"]),
        "rms_norm_eps": float(model_cfg["rms_norm_eps"]),
        "use_cache": bool(model_cfg.get("use_cache", False)),
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "tie_word_embeddings": bool(model_cfg.get("tie_word_embeddings", True)),
        "attention_dropout": float(model_cfg.get("attention_dropout", 0.0)),
    }


def build_model(
    config: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedModel:
    model_cfg = config["model"]
    common = _common_config_kwargs(model_cfg, tokenizer)
    family = model_cfg["family"]
    if family == "mistral":
        hf_config = MistralConfig(**common)
        model: PreTrainedModel = MistralForCausalLM(hf_config)
    elif family == "mixtral":
        hf_config = MixtralConfig(
            **common,
            num_local_experts=int(model_cfg["num_local_experts"]),
            num_experts_per_tok=int(model_cfg["num_experts_per_tok"]),
            router_aux_loss_coef=float(model_cfg["router_aux_loss_coef"]),
            router_jitter_noise=float(model_cfg.get("router_jitter_noise", 0.0)),
            # The trainer requests logits explicitly. Keeping this false in the
            # saved config avoids unnecessary router outputs in official eval.
            output_router_logits=False,
        )
        model = MixtralForCausalLM(hf_config)
        if model_cfg.get("router_mode") == "frozen_random_gate":
            frozen = freeze_router_gates(model)
            expected = int(model_cfg["num_hidden_layers"])
            if frozen != expected:
                raise RuntimeError(
                    f"expected to freeze {expected} router gates, froze {frozen}"
                )
    else:
        raise ValueError(f"unsupported model family: {family}")

    if bool(config["training"].get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
    return model


def freeze_router_gates(model: PreTrainedModel) -> int:
    """Freeze only token-to-expert linear gates, never SwiGLU gate projections."""

    frozen = 0
    for name, module in model.named_modules():
        if name.endswith("block_sparse_moe.gate") and isinstance(
            module, torch.nn.Linear
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False
            frozen += 1
    return frozen


def model_accounting(
    model: PreTrainedModel, config: Mapping[str, Any]
) -> ModelAccounting:
    cfg = config["model"]
    family = str(cfg["family"])
    layers = int(cfg["num_hidden_layers"])
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    experts = int(cfg["num_local_experts"]) if family == "mixtral" else 1
    top_k = int(cfg["num_experts_per_tok"]) if family == "mixtral" else 1
    # SwiGLU has gate, up, and down matrices.
    per_expert = 3 * hidden * intermediate
    active_ffn = layers * per_expert * top_k
    total_ffn = layers * per_expert * experts
    attention_heads = int(cfg["num_attention_heads"])
    key_value_heads = int(cfg["num_key_value_heads"])
    head_dim = hidden // attention_heads
    key_value_width = key_value_heads * head_dim
    attention_projection_parameters = layers * (
        2 * hidden * hidden + 2 * hidden * key_value_width
    )
    attention_interaction_flops = (
        4 * layers * int(config["training"]["sequence_length"]) * hidden
    )
    vocabulary_projection_flops = 2 * hidden * int(model.config.vocab_size)
    forward_flops = (
        2 * (active_ffn + attention_projection_parameters)
        + attention_interaction_flops
        + vocabulary_projection_flops
    )
    return ModelAccounting(
        family=family,
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        active_ffn_parameters_per_token=active_ffn,
        total_ffn_parameters=total_ffn,
        active_ffn_flops_per_token=2 * active_ffn,
        estimated_forward_flops_per_token=forward_flops,
        estimated_training_flops_per_token=3 * forward_flops,
        num_experts=experts,
        top_k=top_k,
    )


def active_ffn_parameters(model_cfg: Mapping[str, Any]) -> int:
    family = str(model_cfg["family"])
    top_k = int(model_cfg["num_experts_per_tok"]) if family == "mixtral" else 1
    return (
        int(model_cfg["num_hidden_layers"])
        * 3
        * int(model_cfg["hidden_size"])
        * int(model_cfg["intermediate_size"])
        * top_k
    )


def assert_active_compute_match(
    dense_model_cfg: Mapping[str, Any],
    moe_model_cfg: Mapping[str, Any],
) -> None:
    structural = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
    )
    mismatches = [
        key for key in structural if dense_model_cfg[key] != moe_model_cfg[key]
    ]
    if mismatches:
        raise ValueError(f"dense/MoE structural mismatch: {mismatches}")
    dense_active = active_ffn_parameters(dense_model_cfg)
    moe_active = active_ffn_parameters(moe_model_cfg)
    if dense_active != moe_active:
        raise ValueError(
            f"active FFN mismatch: dense={dense_active:,}, MoE={moe_active:,}"
        )
