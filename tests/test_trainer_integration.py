from __future__ import annotations

from copy import deepcopy

from transformers import AutoModelForCausalLM

from babymoe.config import config_fingerprint, load_config
from babymoe.data import prepare_all
from babymoe.trainer import run_training

from conftest import REPO_ROOT


def test_tiny_training_resume_and_final_hf_reload(tmp_path):
    config = deepcopy(
        load_config(REPO_ROOT / "configs/smoke.yaml", artifacts_root=tmp_path)
    )
    config["data"].update(
        {
            "unique_exposure_units": 30_000,
            "validation_units_per_language": 2_000,
            "minimum_quota_fill_ratio": 0.90,
            "max_epochs": 1,
        }
    )
    config["tokenizer"].update(
        {
            "vocab_size": 256,
            "train_chars_per_language": 5_000,
            "model_max_length": 32,
        }
    )
    config["model"].update(
        {
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
        }
    )
    config["training"].update(
        {
            "target_exposure_units": 30_000,
            "sequence_length": 32,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "log_every_steps": 100,
            "resume_every_steps": 2,
            "resume_every_minutes": 60,
            "validation_batches_per_language": 1,
        }
    )
    config["routing"]["enabled"] = False
    config["experiment"].pop("config_fingerprint")
    config["experiment"]["config_fingerprint"] = config_fingerprint(config)
    prepare_all(config, synthetic=True)

    run_dir = tmp_path / "runs" / "integration"
    first = run_training(
        config,
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        resume=True,
        stop_at_exposure=500,
    )
    second = run_training(
        config,
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        resume=True,
        stop_at_exposure=1_000,
    )
    assert second["global_step"] > first["global_step"]
    assert second["actual_exposure_units"] > first["actual_exposure_units"]
    model = AutoModelForCausalLM.from_pretrained(run_dir / "final")
    assert model.config.model_type == "mixtral"
