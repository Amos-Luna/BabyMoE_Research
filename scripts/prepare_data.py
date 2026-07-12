#!/usr/bin/env python3
"""Prepare the auditable multilingual corpus, tokenizer, and packed blocks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from babymoe.config import artifact_path, load_config, save_resolved_config
from babymoe.data import prepare_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts-root")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, artifacts_root=args.artifacts_root)
    cache_dir = artifact_path(config, "cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfp_zho = float(config["data"]["byte_premium"]["zho"])
    card_zho = float(config["data"]["dataset_card_byte_premium"]["zho"])
    if cfp_zho != card_zho:
        print(
            "[compliance-warning] Chinese Byte Premium differs between sources: "
            f"CfP={cfp_zho}, dataset card={card_zho}. Using the configured CfP value "
            "and recording it in the manifest."
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not args.synthetic and not hf_token:
        raise RuntimeError(
            "HF_TOKEN is required: accept the three gated BabyLM dataset terms "
            "and add HF_TOKEN to Colab Secrets before preparation"
        )

    data_root = artifact_path(config, "data_dir")
    data_root.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, data_root / "preparation_config.yaml")
    outputs = prepare_all(
        config,
        token=hf_token,
        synthetic=args.synthetic,
        force=args.force,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
