"""Single-GPU, exposure-aware BabyMoE training loop."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .checkpointing import (
    checkpoint_milestones,
    find_resume_state,
    load_resume_state,
    revision_name,
    save_final_checkpoint,
    save_inference_checkpoint,
    save_resume_state,
)
from .config import LANGUAGES, artifact_path, save_resolved_config
from .data import (
    ExposureLimitReached,
    PreparedBlocks,
    TrilingualBatchStream,
    validation_batches,
)
from .models import build_model, model_accounting
from .routing import (
    RouterAccumulator,
    load_latest_routing_summary,
    save_routing_artifacts,
    summarize_multilingual_routing,
)
from .utils import (
    JsonlWriter,
    atomic_write_json,
    environment_summary,
    get_git_sha,
    sha256_file,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def learning_rate_at_exposure(config: Mapping[str, Any], exposure: float) -> float:
    training = config["training"]
    peak = float(training["learning_rate"])
    floor = peak * float(training["min_learning_rate_ratio"])
    warmup = float(training["warmup_exposure_units"])
    target = float(training["target_exposure_units"])
    if exposure < warmup:
        return peak * exposure / max(1.0, warmup)
    decay_fraction = float(training["wsd_decay_fraction"])
    decay_start = target * (1.0 - decay_fraction)
    if exposure < decay_start:
        return peak
    fraction = min(1.0, max(0.0, (exposure - decay_start) / max(1.0, target - decay_start)))
    return peak + (floor - peak) * fraction


def _autocast_context(device: torch.device, dtype_name: str):
    if device.type == "cuda":
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _unwrap_model(model: PreTrainedModel) -> PreTrainedModel:
    return getattr(model, "_orig_mod", model)


def _optimizer_to_device(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def evaluate(
    *,
    model: PreTrainedModel,
    prepared: PreparedBlocks,
    config: Mapping[str, Any],
    device: torch.device,
    collect_routing: bool,
    snapshot_routing: bool = True,
) -> tuple[dict[str, Any], dict[str, RouterAccumulator] | None]:
    model.eval()
    micro_batch = int(config["training"]["micro_batch_size"])
    max_batches = int(config["training"]["validation_batches_per_language"])
    if collect_routing:
        max_batches = max(
            max_batches, int(config["routing"]["aggregate_batches_per_language"])
        )
    is_moe = config["model"]["family"] == "mixtral"
    num_layers = int(config["model"]["num_hidden_layers"])
    num_experts = int(config["model"]["num_local_experts"])
    top_k = int(config["model"]["num_experts_per_tok"])
    snapshot_limit = (
        int(config["routing"]["snapshot_tokens_per_language"])
        if snapshot_routing
        else 0
    )
    accumulators = (
        {
            lang: RouterAccumulator(
                language=lang,
                num_layers=num_layers,
                num_experts=num_experts,
                top_k=top_k,
                snapshot_limit=snapshot_limit,
            )
            for lang in LANGUAGES
        }
        if collect_routing and is_moe
        else None
    )
    metrics: dict[str, Any] = {}
    with torch.no_grad():
        for lang in LANGUAGES:
            weighted_loss = 0.0
            weighted_aux = 0.0
            tokens = 0
            batches = 0
            for batch in validation_batches(
                prepared,
                lang=lang,
                batch_size=micro_batch,
                max_batches=max_batches,
                pad_token_id=int(model.config.pad_token_id),
            ):
                input_ids = torch.from_numpy(batch["input_ids"]).to(device)
                attention_mask = torch.from_numpy(batch["attention_mask"]).to(device)
                labels = torch.from_numpy(batch["labels"]).to(device)
                with _autocast_context(device, config["training"]["dtype"]):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                        output_router_logits=bool(accumulators),
                    )
                valid_tokens = int((labels[:, 1:] != -100).sum().item())
                aux = (
                    float(outputs.aux_loss.detach().float().item())
                    if getattr(outputs, "aux_loss", None) is not None
                    else 0.0
                )
                total_loss = float(outputs.loss.detach().float().item())
                ce_loss = total_loss - float(config["model"]["router_aux_loss_coef"]) * aux
                weighted_loss += ce_loss * valid_tokens
                weighted_aux += aux * valid_tokens
                tokens += valid_tokens
                batches += 1
                if accumulators is not None:
                    accumulators[lang].update(
                        outputs.router_logits,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        group_ids=batch["group_ids"],
                        domain_ids=batch["domain_ids"],
                    )
            mean_loss = weighted_loss / max(1, tokens)
            metrics[lang] = {
                "ce_loss": mean_loss,
                "perplexity": math.exp(min(20.0, mean_loss)),
                "aux_loss": weighted_aux / max(1, tokens),
                "tokens": tokens,
                "batches": batches,
            }
    model.train()
    return metrics, accumulators


def benchmark_training(
    config: Mapping[str, Any],
    *,
    steps: int = 3,
) -> dict[str, Any]:
    """Measure representative forward/backward throughput without writing a run."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_root = artifact_path(config, "tokenizer_dir")
    blocks_manifest = artifact_path(config, "data_dir") / "blocks" / "manifest.json"
    if not tokenizer_root.exists() or not blocks_manifest.exists():
        raise FileNotFoundError("prepare data before running the training benchmark")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_root))
    prepared = PreparedBlocks(blocks_manifest)
    set_seed(int(config["experiment"]["seed"]))
    model = build_model(config, tokenizer).to(device).train()
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
    )
    batch_size = int(config["training"]["micro_batch_size"])
    sequence_length = int(config["training"]["sequence_length"])
    input_ids = torch.randint(
        low=0,
        high=len(tokenizer),
        size=(batch_size, sequence_length),
        device=device,
    )
    durations = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with _autocast_context(device, config["training"]["dtype"]):
            outputs = model(
                input_ids=input_ids,
                labels=input_ids,
                use_cache=False,
                output_router_logits=config["model"]["family"] == "mixtral",
            )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["max_grad_norm"])
        )
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - started
        if step:
            durations.append(duration)
    tokens_per_second = batch_size * sequence_length / float(np.mean(durations))
    train_splits = [
        prepared.manifest["splits"][lang]["train"] for lang in LANGUAGES
    ]
    exposure_per_model_token = sum(
        float(split["adjusted_exposure_units"]) for split in train_splits
    ) / sum(int(split["model_tokens"]) for split in train_splits)
    exposure_per_second = tokens_per_second * exposure_per_model_token
    target = int(config["training"]["target_exposure_units"])
    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "micro_batch_size": batch_size,
        "sequence_length": sequence_length,
        "measured_steps": steps,
        "model_tokens_per_second": tokens_per_second,
        "estimated_exposure_units_per_second": exposure_per_second,
        "estimated_hours_to_configured_target": target
        / max(1e-9, exposure_per_second)
        / 3600,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available()
            else 0.0
        ),
        "model_accounting": model_accounting(model, config).to_dict(),
    }
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _trainer_state(
    *,
    model: PreTrainedModel,
    optimizer: AdamW,
    stream: TrilingualBatchStream,
    global_step: int,
    config: Mapping[str, Any],
    blocks_hash: str,
    tokenizer_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_fingerprint": config["experiment"]["config_fingerprint"],
        "blocks_manifest_sha256": blocks_hash,
        "tokenizer_manifest_sha256": tokenizer_hash,
        "global_step": global_step,
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {
            "name": "exposure_wsd",
            "last_exposure_units": stream.total_consumed,
            "last_learning_rate": optimizer.param_groups[0]["lr"],
        },
        "stream": stream.state_dict(),
        "rng": capture_rng_state(),
    }


def _load_trainer_state(
    *,
    state: Mapping[str, Any],
    model: PreTrainedModel,
    optimizer: AdamW,
    stream: TrilingualBatchStream,
    config: Mapping[str, Any],
    blocks_hash: str,
    tokenizer_hash: str,
    device: torch.device,
) -> int:
    expected = {
        "config_fingerprint": config["experiment"]["config_fingerprint"],
        "blocks_manifest_sha256": blocks_hash,
        "tokenizer_manifest_sha256": tokenizer_hash,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"resume refused: {key} differs")
    _unwrap_model(model).load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    _optimizer_to_device(optimizer, device)
    stream.load_state_dict(state["stream"])
    scheduler = state.get("scheduler", {})
    if scheduler and abs(
        float(scheduler["last_exposure_units"]) - stream.total_consumed
    ) > 1e-6:
        raise ValueError("resume refused: scheduler and exposure ledger disagree")
    restore_rng_state(state["rng"])
    return int(state["global_step"])


def run_training(
    config: Mapping[str, Any],
    *,
    run_dir: str | Path,
    repo_root: str | Path,
    resume: bool = True,
    stop_at_exposure: int | None = None,
) -> dict[str, Any]:
    torch.set_float32_matmul_precision("high")
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoints_root = run_path / "checkpoints"
    routing_root = run_path / "routing"
    resume_dir = run_path / "resume"
    metrics_writer = JsonlWriter(run_path / "metrics.jsonl")

    blocks_manifest = artifact_path(config, "data_dir") / "blocks" / "manifest.json"
    tokenizer_root = artifact_path(config, "tokenizer_dir")
    tokenizer_manifest = tokenizer_root / "manifest.json"
    if not blocks_manifest.exists() or not tokenizer_manifest.exists():
        raise FileNotFoundError(
            "prepared blocks/tokenizer missing; run scripts/prepare_data.py first"
        )
    blocks_hash = sha256_file(blocks_manifest)
    tokenizer_hash = sha256_file(tokenizer_manifest)
    prepared = PreparedBlocks(blocks_manifest)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_root))
    current_git_sha = get_git_sha(repo_root)

    existing_meta = run_path / "run_meta.json"
    if existing_meta.exists():
        old = json.loads(existing_meta.read_text(encoding="utf-8"))
        if not resume:
            raise ValueError(
                "run directory already exists; --no-resume requires a new --run-dir"
            )
        if (
            old.get("config_fingerprint")
            != config["experiment"]["config_fingerprint"]
        ):
            raise ValueError("run directory belongs to a different resolved config")
        if resume and old.get("git_sha") != current_git_sha:
            raise ValueError(
                "resume refused: Git SHA changed; use a new run directory for changed code"
            )

    save_resolved_config(config, run_path / "resolved_config.yaml")
    model = build_model(config, tokenizer).to(device)
    accounting = model_accounting(model, config)
    if bool(config["training"].get("compile", False)):
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        betas=(
            float(config["training"]["adam_beta1"]),
            float(config["training"]["adam_beta2"]),
        ),
        eps=float(config["training"]["adam_epsilon"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    stream = TrilingualBatchStream(
        prepared,
        config,
        seed=seed,
        stop_at_exposure=stop_at_exposure,
    )

    run_meta = {
        "condition_id": config["experiment"]["condition_id"],
        "seed": seed,
        "config_fingerprint": config["experiment"]["config_fingerprint"],
        "blocks_manifest_sha256": blocks_hash,
        "tokenizer_manifest_sha256": tokenizer_hash,
        "git_sha": current_git_sha,
        "environment": environment_summary(),
        "model_accounting": accounting.to_dict(),
        "data_accounting": {
            "source_repositories": dict(config["data"]["repos"]),
            "source_revisions": dict(config["data"]["revisions"]),
            "accounting_formula": config["data"]["accounting_formula"],
            "byte_premium": dict(config["data"]["byte_premium"]),
            "byte_premium_source": config["data"]["byte_premium_source"],
            "packed_unique_exposure_units": prepared.manifest[
                "actual_unique_exposure_units"
            ],
            "max_compliant_exposure_units": prepared.manifest[
                "max_compliant_exposure_units"
            ],
        },
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "manifest_sha256": tokenizer_hash,
        },
        "requested_target_exposure": int(config["training"]["target_exposure_units"]),
        "operational_target_exposure": stream.target_exposure,
    }
    atomic_write_json(run_path / "run_meta.json", run_meta)
    print(
        f"[train] {config['experiment']['condition_id']} seed={seed} device={device} "
        f"params={accounting.total_parameters:,} "
        f"active_ffn={accounting.active_ffn_parameters_per_token:,} "
        f"target={stream.target_exposure:,.0f}"
    )

    global_step = 0
    resume_state_path = find_resume_state(resume_dir) if resume else None
    if resume and resume_state_path is None and (
        any(checkpoints_root.glob("chck_*")) or (run_path / "final").exists()
    ):
        raise RuntimeError(
            "inference checkpoints exist but no trainer state is recoverable"
        )
    if resume_state_path is not None:
        try:
            state = load_resume_state(resume_state_path, device="cpu")
        except Exception as error:
            previous = resume_dir / "previous.pt"
            if resume_state_path.name != "latest.pt" or not previous.exists():
                raise
            print(
                f"[resume-warning] latest state is unreadable ({error}); "
                "falling back to previous.pt"
            )
            resume_state_path = previous
            state = load_resume_state(resume_state_path, device="cpu")
        global_step = _load_trainer_state(
            state=state,
            model=model,
            optimizer=optimizer,
            stream=stream,
            config=config,
            blocks_hash=blocks_hash,
            tokenizer_hash=tokenizer_hash,
            device=device,
        )
        print(
            f"[resume] step={global_step:,} exposure={stream.total_consumed:,.0f} "
            f"from={resume_state_path}"
        )

    max_exposure = math.floor(stream.target_exposure)
    milestones = checkpoint_milestones(config["checkpointing"], max_exposure)
    saved: set[int] = set()
    for path in checkpoints_root.glob("chck_*M"):
        metadata_path = path / "checkpoint_meta.json"
        completion_path = path / "evaluation_complete.json"
        if not metadata_path.exists() or not completion_path.exists():
            continue
        milestone = int(
            json.loads(metadata_path.read_text())["milestone_exposure_units"]
        )
        if milestone <= stream.total_consumed:
            saved.add(milestone)

    wb = None
    if config["logging"].get("wandb_mode") != "disabled":
        try:
            import wandb

            wb = wandb.init(
                project=config["logging"]["wandb_project"],
                entity=config["logging"].get("wandb_entity"),
                name=f"{config['experiment']['condition_id']}-seed{seed}",
                id=f"{config['experiment']['condition_id']}-seed{seed}",
                mode=config["logging"]["wandb_mode"],
                config=dict(config),
                resume="allow",
            )
        except Exception as error:  # pragma: no cover - external service
            print(f"[wandb-warning] disabled after initialization error: {error}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    micro_batch = int(config["training"]["micro_batch_size"])
    log_every = int(config["training"]["log_every_steps"])
    resume_every = int(config["training"]["resume_every_steps"])
    resume_minutes = float(config["training"]["resume_every_minutes"])
    started = time.time()
    last_log = started
    last_resume = started
    running_total = 0.0
    running_ce = 0.0
    running_aux = 0.0
    running_microbatches = 0
    running_model_tokens = 0
    should_stop = False
    session_start_exposure = stream.total_consumed

    while not should_stop:
        optimizer.zero_grad(set_to_none=True)
        step_model_tokens = 0
        completed_microbatches = 0
        for _ in range(accumulation_steps):
            try:
                batch = stream.next_batch(micro_batch)
            except ExposureLimitReached:
                should_stop = True
                break
            input_ids = torch.from_numpy(batch["input_ids"]).to(device)
            labels = input_ids
            with _autocast_context(device, config["training"]["dtype"]):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    use_cache=False,
                    output_router_logits=config["model"]["family"] == "mixtral",
                )
                total_loss = outputs.loss
                aux_loss = getattr(outputs, "aux_loss", None)
                aux_value = (
                    float(aux_loss.detach().float().item())
                    if aux_loss is not None
                    else 0.0
                )
                ce_loss = total_loss - float(
                    config["model"]["router_aux_loss_coef"]
                ) * (aux_loss if aux_loss is not None else 0.0)
                scaled_loss = total_loss / accumulation_steps
            scaled_loss.backward()
            completed_microbatches += 1
            tokens = int(input_ids.numel())
            step_model_tokens += tokens
            running_model_tokens += tokens
            running_total += float(total_loss.detach().float().item())
            running_ce += float(ce_loss.detach().float().item())
            running_aux += aux_value
            running_microbatches += 1
            if batch["input_ids"].shape[0] < micro_batch:
                should_stop = True
                break

        if completed_microbatches == 0:
            break
        if completed_microbatches < accumulation_steps:
            scale = accumulation_steps / completed_microbatches
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["max_grad_norm"])
        )
        learning_rate = learning_rate_at_exposure(config, stream.total_consumed)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        global_step += 1

        now = time.time()
        if global_step % log_every == 0 or should_stop:
            elapsed = max(1e-6, now - last_log)
            exposure_rate = (
                stream.total_consumed - session_start_exposure
            ) / max(1e-6, now - started)
            eta_seconds = (
                max(0.0, stream.target_exposure - stream.total_consumed)
                / max(1e-6, exposure_rate)
            )
            payload = {
                "phase": "train",
                "step": global_step,
                "exposure_units": stream.total_consumed,
                "exposure_by_language": dict(stream.consumed),
                "epochs_by_language": stream.epochs_seen(),
                "total_loss": running_total / max(1, running_microbatches),
                "ce_loss": running_ce / max(1, running_microbatches),
                "aux_loss": running_aux / max(1, running_microbatches),
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
                "model_tokens_per_second": running_model_tokens / elapsed,
                "eta_seconds": eta_seconds,
                "max_cuda_memory_bytes": (
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                ),
                "wallclock_seconds": now - started,
            }
            metrics_writer.write(payload)
            if wb is not None:
                wb.log(payload, step=global_step)
            print(
                f"[step {global_step:>6}] exposure={stream.total_consumed:>12,.0f} "
                f"ce={payload['ce_loss']:.4f} aux={payload['aux_loss']:.4f} "
                f"lr={learning_rate:.2e} tok/s={payload['model_tokens_per_second']:,.0f} "
                f"eta={eta_seconds / 3600:.1f}h"
            )
            running_total = running_ce = running_aux = 0.0
            running_microbatches = running_model_tokens = 0
            last_log = now

        reached = [
            milestone
            for milestone in milestones
            if milestone not in saved and stream.total_consumed >= milestone
        ]
        for milestone in reached:
            # Persist the exact optimizer/data/RNG boundary before the slower
            # inference save and evaluation so a Colab disconnect never leaves
            # a checkpoint without a recoverable trainer state.
            state = _trainer_state(
                model=model,
                optimizer=optimizer,
                stream=stream,
                global_step=global_step,
                config=config,
                blocks_hash=blocks_hash,
                tokenizer_hash=tokenizer_hash,
            )
            save_resume_state(resume_dir / "latest.pt", state)
            last_resume = time.time()
            checkpoint_meta = {
                **run_meta,
                "global_step": global_step,
                "exposure_by_language": dict(stream.consumed),
                "epochs_by_language": stream.epochs_seen(),
            }
            unwrapped = _unwrap_model(model)
            destination = save_inference_checkpoint(
                model=unwrapped,
                tokenizer=tokenizer,
                checkpoint_root=checkpoints_root,
                milestone_units=milestone,
                actual_exposure_units=stream.total_consumed,
                metadata=checkpoint_meta,
                prefix=config["checkpointing"]["branch_prefix"],
            )
            print(f"[checkpoint] {destination.name} -> {destination}")
            snapshot_routing = milestone in {
                int(value) for value in config["routing"]["analysis_milestones"]
            }
            validation_metrics, accumulators = evaluate(
                model=model,
                prepared=prepared,
                config=config,
                device=device,
                collect_routing=bool(config["routing"]["enabled"]),
                snapshot_routing=snapshot_routing,
            )
            validation_payload = {
                "phase": "validation",
                "step": global_step,
                "milestone_exposure_units": milestone,
                "actual_exposure_units": stream.total_consumed,
                "languages": validation_metrics,
            }
            metrics_writer.write(validation_payload)
            if wb is not None:
                flat = {
                    f"validation/{lang}/{key}": value
                    for lang, values in validation_metrics.items()
                    for key, value in values.items()
                    if isinstance(value, (int, float))
                }
                wb.log(flat, step=global_step)
            if accumulators is not None:
                previous = load_latest_routing_summary(routing_root, milestone)
                routing_summary = summarize_multilingual_routing(
                    accumulators, previous_summary=previous
                )
                routing_summary.update(
                    {
                        "milestone_exposure_units": milestone,
                        "actual_exposure_units": stream.total_consumed,
                        "step": global_step,
                    }
                )
                save_routing_artifacts(
                    routing_root,
                    milestone_units=milestone,
                    summary=routing_summary,
                    accumulators=accumulators,
                    save_snapshot=snapshot_routing,
                )
                metrics_writer.write(
                    {
                        "phase": "routing",
                        "step": global_step,
                        "milestone_exposure_units": milestone,
                        "edge_minus_middle_en_zho_js": routing_summary[
                            "edge_minus_middle_en_zho_js"
                        ],
                    }
                )
            atomic_write_json(
                destination / "evaluation_complete.json",
                {
                    "milestone_exposure_units": milestone,
                    "global_step": global_step,
                    "validation": validation_metrics,
                    "routing_collected": accumulators is not None,
                    "routing_snapshot_saved": bool(
                        accumulators is not None and snapshot_routing
                    ),
                },
            )
            saved.add(milestone)

        if (
            global_step % resume_every == 0
            or now - last_resume >= resume_minutes * 60
        ):
            state = _trainer_state(
                model=model,
                optimizer=optimizer,
                stream=stream,
                global_step=global_step,
                config=config,
                blocks_hash=blocks_hash,
                tokenizer_hash=tokenizer_hash,
            )
            save_resume_state(resume_dir / "latest.pt", state)
            last_resume = time.time()
            print(f"[resume] saved at step={global_step:,}")

    final_meta = {
        **run_meta,
        "global_step": global_step,
        "actual_exposure_units": stream.total_consumed,
        "exposure_by_language": dict(stream.consumed),
        "epochs_by_language": stream.epochs_seen(),
        "source_revision": (
            revision_name(max(saved), config["checkpointing"]["branch_prefix"])
            if saved
            else None
        ),
    }
    save_final_checkpoint(
        model=_unwrap_model(model),
        tokenizer=tokenizer,
        destination=run_path / "final",
        metadata=final_meta,
    )
    state = _trainer_state(
        model=model,
        optimizer=optimizer,
        stream=stream,
        global_step=global_step,
        config=config,
        blocks_hash=blocks_hash,
        tokenizer_hash=tokenizer_hash,
    )
    save_resume_state(resume_dir / "latest.pt", state)
    summary = {
        **final_meta,
        "wallclock_seconds_this_session": time.time() - started,
        "saved_milestones": sorted(saved),
        "completed_operational_target": stream.total_consumed
        >= stream.target_exposure - 1.0,
    }
    atomic_write_json(run_path / "summary.json", summary)
    if wb is not None:
        wb.finish()
    print(
        f"[done] step={global_step:,} exposure={stream.total_consumed:,.0f} "
        f"final={run_path / 'final'}"
    )
    return summary
