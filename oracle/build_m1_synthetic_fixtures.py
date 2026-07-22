#!/usr/bin/env python3
"""Build model-free full-envelope fixtures for M1 validator CI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTROLLER = "hyperion.m1-controller-event.v1"
WORKER = "hyperion.m1-worker-event.v1"
TRIAL = "hyperion.m1-trial.v1"
OS_SAMPLE = "hyperion.m1-os-sample.v1"
OUTPUT_SHA256 = "a" * 64
RUN_SHA256 = "b" * 64
SOURCE_COMMIT = "c" * 40
MODEL_LABEL = "gemma-4-12B-QAT-Q4-g64-affine"
MODEL_MANIFEST_SHA256 = "9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144"
ORACLE_LOCK_SHA256 = "b3603b4ebbc7f5883afe3d8cc10fc1767239837f5256985bd591b777c993dbaf"
ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "TZ": "UTC",
}
STARTUP_FLAGS = {
    "bytes_warning": 0,
    "debug": 0,
    "dev_mode": False,
    "dont_write_bytecode": 1,
    "hash_randomization": 0,
    "ignore_environment": 0,
    "inspect": 0,
    "int_max_str_digits": 4300,
    "interactive": 0,
    "isolated": 0,
    "no_site": 1,
    "no_user_site": 1,
    "optimize": 0,
    "quiet": 0,
    "safe_path": True,
    "utf8_mode": 1,
    "verbose": 0,
    "warn_default_encoding": 0,
}
STDERR_BYTES = b"synthetic worker stderr\n"


def emit(output: Any, value: dict[str, Any]) -> None:
    output.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def memory(seed: int) -> dict[str, int]:
    return {
        "pageins": seed,
        "wired_size_bytes": 1_000 + seed,
        "resident_size_bytes": 2_000 + seed,
        "phys_footprint_bytes": 3_000 + seed,
        "lifetime_max_phys_footprint_bytes": 4_000 + seed,
        "interval_max_phys_footprint_bytes": 5_000 + seed,
    }


def trial(phase: str, trial_index: int) -> dict[str, Any]:
    offsets = [100 + index * 10 for index in range(1025)]
    return {
        "schema": TRIAL,
        "kind": "trial",
        "phase": phase,
        "trial_index": trial_index,
        "arm": "core-default",
        "model_key": "12b",
        "model_label": MODEL_LABEL,
        "input_tokens": 1024,
        "generated_tokens": 1025,
        "decode_intervals": 1024,
        "prompt_token_sha256": "d" * 64,
        "output_token_sha256": OUTPUT_SHA256,
        "token_offsets_ns": offsets,
        "ttft_ns": 100,
        "prefill_tok_s": 10_240_000_000.0,
        "decode_duration_ns": 10_240,
        "decode_tok_s": 100_000_000.0,
        "itl_n": 1024,
        "itl_p50_ns": 10,
        "itl_p95_ns": 10,
        "itl_p99_ns": 10,
        "mlx_memory_start": {"active_bytes": 10, "cache_bytes": 20, "peak_bytes": 30},
        "mlx_memory_end": {"active_bytes": 11, "cache_bytes": 21, "peak_bytes": 31},
        "mlx_peak_bytes": 31,
        "requested_wired_limit_bytes": 12_713_115_648,
        "unix_ns": 110,
        "started_monotonic_ns": 1_000,
        "finished_monotonic_ns": 20_000,
    }


def build(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".stderr.log").write_bytes(STDERR_BYTES)
    with path.open("w", encoding="utf-8") as output:
        emit(
            output,
            {
                "schema": CONTROLLER,
                "kind": "controller_start",
                "started_unix_ns": 100,
                "source_commit": SOURCE_COMMIT,
                "run_id": "synthetic",
                "session_id": "synthetic-1",
                "run_manifest": "benchmarks/raw/m1/synthetic/run-manifest.json",
                "run_manifest_sha256": RUN_SHA256,
                "schedule_sha256": "e" * 64,
                "model_verification_pre_sha256": "f" * 64,
                "model_key": "12b",
                "model_label": MODEL_LABEL,
                "context_tokens": 1024,
                "generated_tokens": 1025,
                "warmups": 1,
                "trials": 5,
                "arm": "core-default",
                "wired_limit": "default",
                "corpus_manifest_sha256": "1" * 64,
                "fixture_rendered_sha256": "2" * 64,
                "fixture_token_sha256": "d" * 64,
                "model_manifest_sha256": MODEL_MANIFEST_SHA256,
                "oracle_lock_sha256": ORACLE_LOCK_SHA256,
                "worker_sha256": "5" * 64,
                "oracle_identity_source_sha256": "8" * 64,
                "oracle_launcher_sha256": "7" * 64,
                "model_identity_source_sha256": "4" * 64,
                "executable_sha256": "6" * 64,
                "command": {
                    "program": "hyperion-bench",
                    "subcommand": "m1 run-cell",
                    "output": "benchmarks/raw/m1/synthetic/core/12b-1024.jsonl",
                },
                "os_sample_period_ms": 25,
                "worktree_clean": True,
                "worker_environment": ENVIRONMENT,
                "native_canary": {
                    "macos": "26.2.0",
                    "gpu_name": "Apple M5",
                    "mlx_runtime": "0.32.0",
                    "recommended_working_set_bytes": 12_713_115_648,
                },
            },
        )
        emit(
            output,
            {
                "schema": WORKER,
                "kind": "worker_start",
                "pid": 42,
                "unix_ns": 101,
                "monotonic_ns": 1,
                "source_commit": SOURCE_COMMIT,
                "run_id": "synthetic",
                "session_id": "synthetic-1",
                "run_manifest_sha256": RUN_SHA256,
                "schedule_sha256": "e" * 64,
                "model_verification_receipt_sha256": "f" * 64,
                "model_key": "12b",
                "model_label": MODEL_LABEL,
                "model_manifest_sha256": MODEL_MANIFEST_SHA256,
                "model_payload_tree_sha256": "60386542c026e72aa7b8b4a3ffb3e2356fd3e80c3d54939ad75d932d59bff2d7",
                "model_payload_file_count": 9,
                "model_exact_inventory": True,
                "token_file": "synthetic.tokens.u32le",
                "token_sha256": "d" * 64,
                "input_tokens": 1024,
                "generated_tokens": 1025,
                "warmups": 1,
                "trials": 5,
                "arm": "core-default",
                "python": "3.12.13",
                "python_executable_sha256": "01564940172b2811e1f39a4dc90e84c7a26a19cf071bbc5de67e456d82627bec",
                "python_runtime_tree_sha256": "01a580d385a91f4b8bc195c8b2f56c4c2d156f6c1e1ad8768fc4501987c4e12f",
                "python_runtime_file_count": 1897,
                "site_packages_tree_sha256": "db258e22404a3937d46d72ff44083400aafcf34636b8444a91a29c858b297006",
                "site_packages_file_count": 5470,
                "startup_flags": STARTUP_FLAGS,
                "pycache_prefix": "/dev/null",
                "hash_seed_probe": 1_244_036_990_071_903_237,
                "platform": {"macos": "26.2.0", "machine": "arm64"},
                "mlx_version": "0.32.0",
                "mlx_metal_version": "0.32.0",
                "mlx_lm_version": "0.31.3",
                "mlx_lm_commit": "8239c72de5a0e42c539e30489021db73c7fe258c",
                "mlx_tree_sha256": "bacebd4f46680155a129301ffefc516402142183584f2b47673bc91b561f0cd9",
                "mlx_tree_file_count": 40,
                "mlx_metal_tree_sha256": "628a99548b65855148fb03f71cac83ce46eae42140f119fa8d1b51285c2abefd",
                "mlx_metal_tree_file_count": 406,
                "mlx_lm_tree_sha256": "40dc49399a07cdf22e3516070cfe222e89ec2f0ff29cd6e257e1b069edc3472f",
                "mlx_lm_tree_file_count": 176,
                "mlx_lm_package_sha256": "f9ffa88772d26e537a98aa39ab16488a7a0d13cc1fac5d665376132c94b49608",
                "generate_source_sha256": "270778ad53eaca55a8533d82e6752660fe5d2605c4aa0879b48a50a91f69345f",
                "worker_sha256": "5" * 64,
                "oracle_identity_source_sha256": "8" * 64,
                "oracle_launcher_sha256": "7" * 64,
                "model_identity_source_sha256": "4" * 64,
                "environment": ENVIRONMENT,
                "device_info": {"max_recommended_working_set_size": 12_713_115_648},
                "requested_wired_limit_bytes": 12_713_115_648,
                "recommended_working_set_bytes": 12_713_115_648,
                "previous_wired_limit_bytes": 12_713_115_648,
                "wired_limit_effective": True,
            },
        )
        emit(
            output,
            {
                "schema": WORKER,
                "kind": "model_loaded",
                "unix_ns": 102,
                "monotonic_ns": 2,
                "load_duration_ns": 1,
                "memory": {"active_bytes": 1, "cache_bytes": 2, "peak_bytes": 3},
            },
        )
        phases = [("warmup", 0)] + [("measured", index) for index in range(5)]
        for phase, trial_index in phases:
            key = {"phase": phase, "trial_index": trial_index}
            emit(
                output,
                {
                    "schema": WORKER,
                    "kind": "trial_start_pending",
                    "unix_ns": 103,
                    "monotonic_ns": 3,
                    **key,
                },
            )
            emit(
                output,
                {
                    "schema": WORKER,
                    "kind": "trial_start",
                    "unix_ns": 104,
                    "monotonic_ns": 4,
                    **key,
                },
            )
            emit(output, trial(phase, trial_index))
            emit(
                output,
                {
                    "schema": WORKER,
                    "kind": "trial_post_cleanup_pending",
                    "unix_ns": 112,
                    "monotonic_ns": 21_000,
                    **key,
                    "mlx_memory_post_cleanup": {
                        "active_bytes": 1,
                        "cache_bytes": 0,
                        "peak_bytes": 31,
                    },
                },
            )
            emit(
                output,
                {
                    "schema": WORKER,
                    "kind": "trial_end",
                    "unix_ns": 113,
                    "monotonic_ns": 21_001,
                    **key,
                },
            )
        emit(
            output,
            {
                "schema": WORKER,
                "kind": "worker_end",
                "unix_ns": 150,
                "monotonic_ns": 50,
                "measured_trials": 5,
                "warmups": 1,
                "measured_output_token_sha256": OUTPUT_SHA256,
                "memory": {"active_bytes": 1, "cache_bytes": 0, "peak_bytes": 31},
            },
        )
        for phase, trial_index in phases:
            base = 10_000 + (trial_index + (0 if phase == "warmup" else 1)) * 1_000
            for offset, boundary in ((0, "pre_trial"), (100, None), (200, "post_trial")):
                emit(
                    output,
                    {
                        "schema": OS_SAMPLE,
                        "kind": "os_sample",
                        "elapsed_ns": base + offset,
                        "phase": phase,
                        "trial_index": trial_index,
                        "boundary": boundary,
                        **memory(offset // 100),
                    },
                )
            emit(
                output,
                {
                    "schema": OS_SAMPLE,
                    "kind": "os_trial_summary",
                    "phase": phase,
                    "trial_index": trial_index,
                    "sample_count": 3,
                    "pre_trial": memory(0),
                    "post_trial": memory(2),
                    "first": memory(0),
                    "last": memory(2),
                    "max_wired_size_bytes": 1002,
                    "max_resident_size_bytes": 2002,
                    "max_phys_footprint_bytes": 3002,
                    "max_interval_phys_footprint_bytes": 5002,
                    "max_lifetime_phys_footprint_bytes": 4002,
                    "pageins_first": 0,
                    "pageins_last": 2,
                },
            )
        emit(
            output,
            {
                "schema": CONTROLLER,
                "kind": "controller_end",
                "started_unix_ns": 100,
                "finished_unix_ns": 200,
                "source_commit": SOURCE_COMMIT,
                "run_id": "synthetic",
                "session_id": "synthetic-1",
                "run_manifest_sha256": RUN_SHA256,
                "worker_exit": {"code": 0, "signal": None},
                "worker_success": True,
                "uncontrolled_oom": False,
                "stderr_file": path.with_suffix(".stderr.log").name,
                "stderr_sha256": hashlib.sha256(STDERR_BYTES).hexdigest(),
                "os_samples": 18,
                "os_sample_errors": 0,
                "validation_errors": [],
                "valid": True,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
