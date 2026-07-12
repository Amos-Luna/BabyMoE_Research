from __future__ import annotations

from copy import deepcopy

import pytest

from babymoe.config import artifact_path, config_fingerprint, load_config, validate_config

from conftest import REPO_ROOT


def test_layered_config_merge_and_override(tmp_path):
    config = load_config(
        REPO_ROOT / "configs/moe_frozen_gate.yaml",
        seed=71,
        artifacts_root=tmp_path,
    )
    assert config["model"]["family"] == "mixtral"
    assert config["model"]["hidden_size"] == 768
    assert config["model"]["router_mode"] == "frozen_random_gate"
    assert config["training"]["target_exposure_units"] == 10_000_000
    assert config["experiment"]["seed"] == 71
    assert artifact_path(config, "data_dir") == tmp_path / "data"


def test_config_fingerprint_is_order_independent():
    assert config_fingerprint({"a": 1, "b": {"c": 2}}) == config_fingerprint(
        {"b": {"c": 2}, "a": 1}
    )


def test_validation_rejects_budget_above_ten_epochs():
    config = load_config(REPO_ROOT / "configs/dense_match.yaml")
    invalid = deepcopy(config)
    invalid["data"]["max_epochs"] = 1
    with pytest.raises(ValueError, match="max_epochs"):
        validate_config(invalid)
