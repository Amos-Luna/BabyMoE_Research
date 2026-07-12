#!/usr/bin/env python3
"""Audit and idempotently publish BabyLM checkpoints as Hugging Face revisions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from babymoe.utils import atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--only-main", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-local-model-audit", action="store_true")
    remote_group = parser.add_mutually_exclusive_group()
    remote_group.add_argument(
        "--verify-remote-models", dest="verify_remote_models", action="store_true"
    )
    remote_group.add_argument(
        "--skip-remote-model-verify",
        dest="verify_remote_models",
        action="store_false",
    )
    parser.set_defaults(verify_remote_models=None)
    return parser.parse_args()


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(candidate.relative_to(path)).encode("utf-8"))
        digest.update(str(candidate.stat().st_size).encode("ascii"))
        digest.update(sha256_file(candidate).encode("ascii"))
    return digest.hexdigest()


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "checkpoint_meta.json"
    if not metadata_path.exists():
        raise ValueError(f"checkpoint lacks checkpoint_meta.json: {path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _discover(run_dir: Path) -> tuple[Path, list[tuple[int, str, Path]]]:
    final = run_dir / "final"
    if not final.is_dir():
        raise FileNotFoundError(final)
    checkpoints = []
    for path in run_dir.joinpath("checkpoints").glob("chck_*M"):
        metadata = _checkpoint_metadata(path)
        if not (path / "evaluation_complete.json").exists():
            raise ValueError(f"checkpoint evaluation is incomplete: {path}")
        milestone = int(metadata["milestone_exposure_units"])
        expected = f"chck_{milestone // 1_000_000}M"
        if milestone % 1_000_000 or path.name != expected:
            raise ValueError(f"non-canonical checkpoint revision: {path}")
        if float(metadata["actual_exposure_units"]) < milestone:
            raise ValueError(f"checkpoint claims exposure it did not reach: {path}")
        checkpoints.append((milestone, path.name, path))
    checkpoints.sort()
    return final, checkpoints


def _audit_local_checkpoint(path: Path, classification_heads: bool = False) -> None:
    print(f"[audit] AutoModelForCausalLM <- {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError(
            f"vocab mismatch at {path}: model={model.config.vocab_size}, "
            f"tokenizer={len(tokenizer)}"
        )
    del model, tokenizer
    gc.collect()
    if classification_heads:
        for auto_class in (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        ):
            print(f"[audit] {auto_class.__name__} <- {path}")
            model = auto_class.from_pretrained(
                path,
                local_files_only=True,
                low_cpu_mem_usage=True,
                num_labels=3,
                ignore_mismatched_sizes=True,
            )
            del model
            gc.collect()


def _model_card(repo_id: str, run_dir: Path, summary: dict[str, Any]) -> str:
    condition = summary["condition_id"]
    exposure = float(summary["actual_exposure_units"])
    complete_curve = (
        len(summary.get("saved_milestones", [])) == 28
        and max(summary.get("saved_milestones", [0])) == 1_000_000_000
    )
    learning_curve_command = (
        f"bash scripts/eval_model_fast.sh --model_name {repo_id} --bos_fix 0"
        if complete_curve
        else (
            "bash scripts/zeroshot_model.sh --model_name "
            f"{repo_id} --revision chck_NM --bos_fix 0  # each listed revision"
        )
    )
    return f"""---
library_name: transformers
pipeline_tag: text-generation
language:
- en
- nl
- zh
tags:
- babylm
- mixture-of-experts
- multilingual
---

# {repo_id}

BabyLM 2026 Multilingual model for the **{condition}** condition. The model and
joint tokenizer were initialized and trained from scratch on the permitted
English, Dutch, and Chinese BabyBabelLM mixture.

- Adjusted exposure reached: `{exposure:,.0f}`
- Seed: `{summary['seed']}`
- Git commit: `{summary['git_sha']}`
- Model family: `{summary['model_accounting']['family']}`
- Total parameters: `{summary['model_accounting']['total_parameters']:,}`
- Active FFN parameters per token: `{summary['model_accounting']['active_ffn_parameters_per_token']:,}`

Intermediate revisions use the required `chck_{{N}}M` naming. Exact accounting,
hashes, and per-language exposure are in `compliance_manifest.json`.

```bash
git clone https://github.com/babylm-org/babylm-eval
cd babylm-eval/multilingual
bash scripts/eval_model_full.sh --model_name {repo_id} --bos_fix 0
{learning_curve_command}
```

This model is a research artifact; routing specialization is not evidence of a
human-interpretable linguistic expert without the accompanying controls.
"""


def _remote_revisions(api: HfApi, repo_id: str) -> set[str]:
    try:
        refs = api.list_repo_refs(repo_id)
    except Exception:
        return set()
    return {branch.name for branch in refs.branches}


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    resolved_config = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    verify_remote_models = (
        bool(resolved_config["hub"]["verify_after_upload"])
        if args.verify_remote_models is None
        else bool(args.verify_remote_models)
    )
    final, checkpoints = _discover(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    final_meta = _checkpoint_metadata(final)
    if float(final_meta["actual_exposure_units"]) != float(
        summary["actual_exposure_units"]
    ):
        raise ValueError("final checkpoint and run summary exposure differ")

    items = [("main", final)]
    if not args.only_main:
        items.extend((revision, path) for _, revision, path in checkpoints)
    fingerprints = {revision: _directory_fingerprint(path) for revision, path in items}
    print("[upload] planned revisions:")
    for revision, path in items:
        print(f"  {revision:>10}  {fingerprints[revision][:12]}  {path}")

    if not args.skip_local_model_audit:
        for revision, path in items:
            _audit_local_checkpoint(path, classification_heads=revision == "main")
    if args.dry_run:
        print("[upload] dry run complete; no Hub changes made")
        return

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for upload")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    try:
        api.update_repo_settings(args.repo_id, private=args.private)
    except Exception as error:
        print(f"[upload-warning] could not update repository visibility: {error}")

    upload_state_path = run_dir / "upload_state.json"
    upload_state = (
        json.loads(upload_state_path.read_text(encoding="utf-8"))
        if upload_state_path.exists()
        else {"repo_id": args.repo_id, "revisions": {}}
    )
    if upload_state.get("repo_id") != args.repo_id:
        upload_state = {"repo_id": args.repo_id, "revisions": {}}

    remote = _remote_revisions(api, args.repo_id)
    # Main must exist before branches can be created.
    ordered_items = sorted(items, key=lambda item: item[0] != "main")
    for revision, path in ordered_items:
        fingerprint = fingerprints[revision]
        state_matches = (
            upload_state["revisions"].get(revision, {}).get("fingerprint")
            == fingerprint
        )
        remote_exists = revision in remote
        if state_matches and remote_exists:
            print(f"[upload] unchanged; skipping {revision}")
            continue
        if revision != "main" and revision not in remote:
            api.create_branch(args.repo_id, branch=revision, revision="main")
            remote.add(revision)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(path),
            revision=revision,
            commit_message=f"Upload BabyLM checkpoint {revision}",
        )
        upload_state["revisions"][revision] = {
            "fingerprint": fingerprint,
            "path": str(path),
        }
        remote.add(revision)
        atomic_write_json(upload_state_path, upload_state)
        print(f"[upload] completed {revision}")

    compliance = {
        "repo_id": args.repo_id,
        "run_summary": summary,
        "revisions": [
            {
                "revision": revision,
                "fingerprint": fingerprints[revision],
                "metadata": _checkpoint_metadata(path),
            }
            for revision, path in items
        ],
    }
    hub_dir = run_dir / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    readme_path = hub_dir / "README.md"
    readme_path.write_text(
        _model_card(args.repo_id, run_dir, summary), encoding="utf-8"
    )
    compliance_path = hub_dir / "compliance_manifest.json"
    atomic_write_json(compliance_path, compliance)
    for local_path, remote_path in (
        (readme_path, "README.md"),
        (compliance_path, "compliance_manifest.json"),
    ):
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=args.repo_id,
            repo_type="model",
            revision="main",
            commit_message=f"Update {remote_path}",
        )

    for revision, _ in items:
        AutoConfig.from_pretrained(args.repo_id, revision=revision, token=token)
        AutoTokenizer.from_pretrained(args.repo_id, revision=revision, token=token)
        if verify_remote_models:
            model = AutoModelForCausalLM.from_pretrained(
                args.repo_id,
                revision=revision,
                token=token,
                low_cpu_mem_usage=True,
            )
            del model
            gc.collect()
        print(f"[verify] remote revision {revision} is loadable")
    print(f"[upload] public model ready: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
