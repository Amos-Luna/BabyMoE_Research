#!/usr/bin/env python3
"""Run confound-controlled language probes over saved routing snapshots."""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from transformers import PreTrainedTokenizerFast

from babymoe.analysis import LANGUAGE_TO_ID, ProbeInput, run_layerwise_probes
from babymoe.config import LANGUAGES, artifact_path
from babymoe.data import PreparedBlocks
from babymoe.utils import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--milestone", type=int, help="Exposure in millions; default latest")
    parser.add_argument("--dense-run-dir")
    parser.add_argument("--max-tokens", type=int, default=60_000)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--permutation-samples", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _milestone_from_path(path: Path) -> int:
    return int(path.stem.removeprefix("routing_").removesuffix("M"))


def _select_snapshot(run_dir: Path, milestone: int | None) -> tuple[Path, Path, int]:
    snapshots = sorted(
        run_dir.joinpath("routing").glob("routing_*M.npz"),
        key=_milestone_from_path,
    )
    if not snapshots:
        raise FileNotFoundError(f"no routing snapshots found under {run_dir / 'routing'}")
    if milestone is None:
        snapshot = snapshots[-1]
    else:
        matches = [path for path in snapshots if _milestone_from_path(path) == milestone]
        if not matches:
            raise FileNotFoundError(f"routing snapshot {milestone}M not found")
        snapshot = matches[0]
    value = _milestone_from_path(snapshot)
    summary = snapshot.with_suffix(".json")
    if not summary.exists():
        raise FileNotFoundError(summary)
    return snapshot, summary, value


def _unicode_script(token: str) -> str:
    counts: dict[str, int] = {}
    for character in token:
        category = unicodedata.category(character)
        if category[0] in {"Z", "C"}:
            continue
        name = unicodedata.name(character, "")
        if "LATIN" in name:
            script = "Latn"
        elif any(value in name for value in ("CJK", "IDEOGRAPH", "HIRAGANA", "KATAKANA")):
            script = "Hani"
        elif "HANGUL" in name:
            script = "Hang"
        elif category[0] in {"P", "S"}:
            script = "PunctSymbol"
        elif category[0] == "N":
            script = "Number"
        else:
            script = "Other"
        counts[script] = counts.get(script, 0) + 1
    return max(counts, key=counts.get) if counts else "WhitespaceOrByte"


def _stratified_subsample(
    languages: np.ndarray, max_tokens: int, seed: int
) -> np.ndarray:
    if len(languages) <= max_tokens:
        return np.arange(len(languages))
    rng = np.random.default_rng(seed)
    selected = []
    per_language = max_tokens // len(LANGUAGES)
    for lang in LANGUAGES:
        indices = np.flatnonzero(languages == lang)
        selected.append(
            rng.choice(indices, size=min(per_language, len(indices)), replace=False)
        )
    return np.sort(np.concatenate(selected))


def _validation_at_milestone(run_dir: Path, milestone_units: int) -> dict[str, Any] | None:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    candidates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("phase") == "validation":
            candidates.append(payload)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda value: abs(
            int(value["milestone_exposure_units"]) - milestone_units
        ),
    )


def _mean_validation_ce(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    return float(
        np.mean(
            [payload["languages"][lang]["ce_loss"] for lang in LANGUAGES]
        )
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    if config["model"]["family"] != "mixtral":
        raise ValueError("routing analysis requires a Mixtral-family run")
    snapshot_path, summary_path, milestone_m = _select_snapshot(
        run_dir, args.milestone
    )
    milestone_units = milestone_m * 1_000_000
    snapshot = np.load(snapshot_path)
    languages = snapshot["language"].astype(str)
    selected = _stratified_subsample(languages, args.max_tokens, args.seed)
    languages = languages[selected]
    token_ids = snapshot["token_ids"][selected].astype(np.int64)
    positions = snapshot["positions"][selected].astype(np.int64)
    group_ids = snapshot["group_ids"][selected]
    domain_ids = snapshot["domain_ids"][selected].astype(np.int64)
    routing = snapshot["router_probabilities"][selected].astype(np.float32)

    prepared = PreparedBlocks(
        artifact_path(config, "data_dir") / "blocks" / "manifest.json"
    )
    frequencies = np.asarray(prepared.token_frequency[token_ids], dtype=np.int64)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(artifact_path(config, "tokenizer_dir"))
    )
    unique_token_ids = np.unique(token_ids)
    script_by_id = {
        int(token_id): _unicode_script(
            tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        for token_id in unique_token_ids
    }
    scripts = np.asarray([script_by_id[int(token_id)] for token_id in token_ids])
    frequency_bins = np.floor(np.log2(frequencies + 1)).astype(np.int64)
    position_bins = np.minimum(
        9,
        positions * 10 // int(config["training"]["sequence_length"]),
    )
    domains = []
    for lang, domain_id in zip(languages, domain_ids):
        domain_names = prepared.manifest["splits"][str(lang)]["validation"]["domains"]
        domains.append(
            domain_names[int(domain_id)]
            if 0 <= int(domain_id) < len(domain_names)
            else "unknown"
        )
    controls = np.column_stack(
        [token_ids, scripts, frequency_bins, position_bins, np.asarray(domains)]
    )
    labels = np.asarray([LANGUAGE_TO_ID[str(lang)] for lang in languages], dtype=np.int64)
    strata = np.asarray(
        [
            f"{script}|{frequency}|{position}"
            for script, frequency, position in zip(
                scripts, frequency_bins, position_bins
            )
        ]
    )

    bootstrap_samples = (
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else int(config["routing"]["bootstrap_samples"])
    )
    permutation_samples = (
        args.permutation_samples
        if args.permutation_samples is not None
        else int(config["routing"]["permutation_samples"])
    )
    probes = run_layerwise_probes(
        ProbeInput(
            controls=controls,
            routing=routing,
            labels=labels,
            groups=group_ids,
            strata=strata,
        ),
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=args.seed,
    )
    routing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    significant_layers = [
        layer["layer"]
        for layer in probes["layers"]
        if layer["delta_macro_auroc"] >= 0.03
        and layer["bootstrap_95_ci"][0] > 0
        and layer["within_stratum_permutation_pvalue"] < 0.05
    ]
    minimum_share = min(
        row["minimum_expert_share"]
        for lang in LANGUAGES
        for row in routing_summary["languages"][lang]["layers"]
    )
    maximum_cv = max(
        row["load_cv"]
        for lang in LANGUAGES
        for row in routing_summary["languages"][lang]["layers"]
    )

    moe_validation = _validation_at_milestone(run_dir, milestone_units)
    dense_validation = (
        _validation_at_milestone(Path(args.dense_run_dir), milestone_units)
        if args.dense_run_dir
        else None
    )
    moe_ce = _mean_validation_ce(moe_validation)
    dense_ce = _mean_validation_ce(dense_validation)
    relative_ce_gap = (
        (moe_ce - dense_ce) / dense_ce
        if moe_ce is not None and dense_ce is not None and dense_ce
        else None
    )
    result = {
        "run_dir": str(run_dir),
        "snapshot": str(snapshot_path),
        "milestone_exposure_units": milestone_units,
        "tokens_analyzed": int(len(labels)),
        "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
        "probe": probes,
        "routing_structure": {
            "pairwise_js_by_layer": routing_summary["pairwise_js_by_layer"],
            "edge_layers": routing_summary["edge_layers"],
            "middle_layers": routing_summary["middle_layers"],
            "edge_minus_middle_en_zho_js": routing_summary[
                "edge_minus_middle_en_zho_js"
            ],
            "minimum_expert_share": minimum_share,
            "maximum_load_cv": maximum_cv,
        },
        "dense_comparison": {
            "moe_mean_validation_ce": moe_ce,
            "dense_mean_validation_ce": dense_ce,
            "relative_ce_gap": relative_ce_gap,
        },
        "decision": {
            "residual_language_signal": bool(significant_layers),
            "significant_layers": significant_layers,
            "edge_greater_than_middle": routing_summary[
                "edge_minus_middle_en_zho_js"
            ]
            > 0,
            "router_healthy": minimum_share >= 0.05 and maximum_cv <= 1.0,
            "moe_within_5_percent_of_dense": (
                relative_ce_gap <= 0.05 if relative_ce_gap is not None else None
            ),
        },
    }
    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"analysis_{milestone_m}M.json"
    atomic_write_json(output_path, result)
    print(json.dumps(result["decision"], indent=2))
    print(f"[analysis] wrote {output_path}")


if __name__ == "__main__":
    main()
