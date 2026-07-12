#!/usr/bin/env python3
"""Combine three seed-level routing analyses into the preregistered pilot gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from babymoe.utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="append", required=True)
    parser.add_argument("--output", default="pilot_summary.json")
    args = parser.parse_args()
    if len(args.analysis) < 3:
        raise ValueError("provide at least three seed-level analysis JSON files")
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.analysis]
    residual = sum(bool(row["decision"]["residual_language_signal"]) for row in rows)
    u_shape = sum(bool(row["decision"]["edge_greater_than_middle"]) for row in rows)
    healthy = sum(bool(row["decision"]["router_healthy"]) for row in rows)
    within_dense = [
        row["decision"]["moe_within_5_percent_of_dense"]
        for row in rows
        if row["decision"]["moe_within_5_percent_of_dense"] is not None
    ]
    positive_mechanism = residual >= 2 and u_shape >= 2
    controlled_null = residual == 0 and healthy == len(rows)
    result = {
        "seed_analyses": args.analysis,
        "n_seeds": len(rows),
        "seeds_with_residual_language_signal": residual,
        "seeds_with_edge_greater_than_middle": u_shape,
        "seeds_with_healthy_router": healthy,
        "positive_mechanistic_gate": positive_mechanism,
        "controlled_null_gate": controlled_null,
        "all_available_seeds_within_5_percent_dense": (
            all(within_dense) if within_dense else None
        ),
        "proceed_to_100M": healthy >= 2 and (positive_mechanism or controlled_null),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
