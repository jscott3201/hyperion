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
    if mutation == "worker-environment-drift":
        next(value for value in values if value.get("kind") == "worker_start")[
            "environment"
        ]["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        return values
    if mutation == "worker-platform-drift":
        next(value for value in values if value.get("kind") == "worker_start")[
            "platform"
        ]["macos"] = "26.2"
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
    if mutation in (
        "controlled-failure",
        "uncontrolled-oom",
        "sigkill-relabel",
        "worker-exit-drift",
        "non-capacity-failure",
        "capacity-protocol-error",
        "alloc-substring-failure",
        "traceback-non-string",
        "failure-missing-field",
        "failure-extra-field",
    ):
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
            "message": "synthetic out of memory",
            "traceback": ["Traceback (most recent call last):", "RuntimeError: synthetic out of memory"],
        }
        if mutation == "sigkill-relabel":
            controller_end["worker_exit"] = {"code": None, "signal": 9}
            controller_end["uncontrolled_oom"] = False
        elif mutation == "worker-exit-drift":
            controller_end["worker_exit"] = {"code": 2, "signal": None}
        elif mutation == "non-capacity-failure":
            failure["message"] = "synthetic protocol bug"
            failure["traceback"][-1] = "RuntimeError: synthetic protocol bug"
        elif mutation == "capacity-protocol-error":
            failure["error_type"] = "ProtocolError"
            failure["traceback"][-1] = "ProtocolError: synthetic out of memory"
        elif mutation == "alloc-substring-failure":
            failure["message"] = "synthetic allocator protocol corruption"
            failure["traceback"][-1] = (
                "RuntimeError: synthetic allocator protocol corruption"
            )
        elif mutation == "traceback-non-string":
            failure["traceback"][0] = {"forged": True}
        elif mutation == "failure-missing-field":
            del failure["message"]
        elif mutation == "failure-extra-field":
            failure["capacity"] = True
        return [controller_start, worker_start, model_loaded, failure, controller_end]
    if mutation == "stderr-hash-drift":
        values[-1]["stderr_sha256"] = "0" * 64
        return values
    if mutation == "missing-stderr":
        return values
    raise RuntimeError(f"unknown mutation: {mutation}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mutation", required=True)
    args = parser.parse_args()
    values = mutate(load(args.input), args.mutation)
    values[-1]["stderr_file"] = args.output.with_suffix(".stderr.log").name
    write(args.output, values)
    output_stderr = args.output.with_suffix(".stderr.log")
    if args.mutation != "missing-stderr":
        output_stderr.write_bytes(args.input.with_suffix(".stderr.log").read_bytes())


if __name__ == "__main__":
    main()
