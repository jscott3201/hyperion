#!/usr/bin/env python3
"""Create deterministic positive/negative variants of the model-free M1 trace fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in values)
    )


def mutate(values: list[dict[str, Any]], mutation: str) -> list[dict[str, Any]]:
    if mutation == "missing-controller-start":
        return values[1:]
    if mutation == "missing-pre-boundary":
        for index, value in enumerate(values):
            if (
                value.get("kind") == "os_sample"
                and value.get("phase") == "measured"
                and value.get("trial_index") == 0
                and value.get("boundary") == "pre_trial"
            ):
                del values[index]
                return values
    if mutation == "cross-trial-hash":
        measured = [
            value
            for value in values
            if value.get("kind") == "trial" and value.get("phase") == "measured"
        ]
        measured[-1]["output_token_sha256"] = "9" * 64
        return values
    if mutation == "oracle-tree-drift":
        next(value for value in values if value.get("kind") == "worker_start")[
            "mlx_lm_tree_sha256"
        ] = "9" * 64
        return values
    if mutation == "controller-envelope-drift":
        values[0]["oracle_lock_sha256"] = "9" * 64
        return values
    if mutation == "worker-source-drift":
        next(value for value in values if value.get("kind") == "worker_start")[
            "generate_source_sha256"
        ] = "9" * 64
        return values
    if mutation == "os-summary-drift":
        next(value for value in values if value.get("kind") == "os_trial_summary")[
            "max_lifetime_phys_footprint_bytes"
        ] += 1
        return values
    if mutation == "missing-warmup-boundary":
        for index, value in enumerate(values):
            if (
                value.get("kind") == "os_sample"
                and value.get("phase") == "warmup"
                and value.get("boundary") == "pre_trial"
            ):
                del values[index]
                values[-1]["os_samples"] -= 1
                return values
    if mutation == "controller-sample-count":
        values[-1]["os_samples"] += 1
        return values
    if mutation == "trial-event-reorder":
        pending = next(
            index
            for index, value in enumerate(values)
            if value.get("kind") == "trial_start_pending" and value.get("phase") == "measured"
        )
        values[pending], values[pending + 1] = values[pending + 1], values[pending]
        return values
    if mutation in ("controlled-failure", "uncontrolled-oom"):
        controller_start = values[0]
        worker_start = next(value for value in values if value.get("kind") == "worker_start")
        model_loaded = next(value for value in values if value.get("kind") == "model_loaded")
        controller_end = values[-1]
        controller_end.update(
            {
                "worker_exit": {"code": 1, "signal": None},
                "worker_success": False,
                "uncontrolled_oom": mutation == "uncontrolled-oom",
                "os_samples": 0,
                "os_sample_errors": 0,
                "validation_errors": ["synthetic controlled worker failure"],
                "valid": False,
            }
        )
        failure = {
            "schema": "hyperion.m1-worker-event.v1",
            "kind": "failure",
            "unix_ns": 120,
            "monotonic_ns": 20,
            "error_type": "RuntimeError",
            "message": "synthetic allocation refusal",
            "traceback": ["Traceback (most recent call last):", "RuntimeError: synthetic allocation refusal"],
        }
        return [controller_start, worker_start, model_loaded, failure, controller_end]
    raise RuntimeError(f"unknown mutation: {mutation}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mutation", required=True)
    args = parser.parse_args()
    write(args.output, mutate(load(args.input), args.mutation))


if __name__ == "__main__":
    main()
