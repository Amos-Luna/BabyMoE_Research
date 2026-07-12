from __future__ import annotations

import torch

from babymoe.checkpointing import (
    checkpoint_milestones,
    find_resume_state,
    load_resume_state,
    revision_name,
    save_resume_state,
)
from babymoe.config import load_config

from conftest import REPO_ROOT


def test_official_checkpoint_schedule():
    config = load_config(REPO_ROOT / "configs/moe_learned.yaml")
    milestones = checkpoint_milestones(config["checkpointing"], 1_000_000_000)
    assert milestones[:10] == [value * 1_000_000 for value in range(1, 11)]
    assert milestones[10] == 20_000_000
    assert 100_000_000 in milestones
    assert milestones[-1] == 1_000_000_000
    assert len(milestones) == 28
    assert revision_name(100_000_000) == "chck_100M"


def test_resume_state_rotation_and_round_trip(tmp_path):
    destination = tmp_path / "resume" / "latest.pt"
    save_resume_state(destination, {"step": 1, "tensor": torch.arange(4)})
    save_resume_state(destination, {"step": 2, "tensor": torch.arange(5)})
    assert find_resume_state(destination.parent) == destination
    latest = load_resume_state(destination)
    previous = load_resume_state(destination.parent / "previous.pt")
    assert latest["step"] == 2
    assert previous["step"] == 1
    assert torch.equal(latest["tensor"], torch.arange(5))
