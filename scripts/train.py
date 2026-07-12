#!/usr/bin/env python3
"""Train or resume one configured BabyMoE experimental condition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from babymoe.config import artifact_path, load_config
from babymoe.trainer import benchmark_training, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--artifacts-root")
    parser.add_argument("--run-dir")
    parser.add_argument("--stop-at-exposure", type=int)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(
        args.config,
        seed=args.seed,
        artifacts_root=args.artifacts_root,
    )
    cache_dir = artifact_path(config, "cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.dry_run:
        print(
            json.dumps(
                benchmark_training(config, steps=args.benchmark_steps),
                indent=2,
                default=str,
            )
        )
        return

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else artifact_path(config, "runs_dir")
        / f"{config['experiment']['condition_id']}_seed{config['experiment']['seed']}"
    )
    summary = run_training(
        config,
        run_dir=run_dir,
        repo_root=repo_root,
        resume=args.resume,
        stop_at_exposure=args.stop_at_exposure,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
