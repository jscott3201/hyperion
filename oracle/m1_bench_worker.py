#!/usr/bin/env python3
"""Pinned stock-mlx-lm worker for one Hyperion M1 benchmark cell."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any


TRIAL_SCHEMA = "hyperion.m1-trial.v1"
EVENT_SCHEMA = "hyperion.m1-worker-event.v1"


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_token_ids(path: Path, expected_count: int, expected_sha256: str) -> list[int]:
    payload = path.read_bytes()
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"token fixture SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    if len(payload) != expected_count * 4:
        raise RuntimeError(
            f"token fixture byte count mismatch: expected {expected_count * 4}, got {len(payload)}"
        )
    tokens = [item[0] for item in struct.iter_unpack("<I", payload)]
    if any(token >= 262_144 for token in tokens):
        raise RuntimeError("token fixture contains an out-of-vocabulary ID")
    return tokens


def nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        raise RuntimeError("nearest-rank percentile requires at least one value")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def memory_snapshot(mx: Any) -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-key", choices=("12b", "e4b"), required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--token-sha256", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--generated-tokens", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--wired-limit", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input_tokens <= 0 or args.generated_tokens < 2:
        raise RuntimeError("input tokens must be positive and generated tokens must be >= 2")
    if args.warmups != 1 or args.trials != 5:
        raise RuntimeError("M1 scored cells require exactly one warmup and five trials")

    # Import only after argument and fixture validation so malformed invocations stay model-free.
    token_ids = load_token_ids(args.token_file, args.input_tokens, args.token_sha256)
    model_manifest = args.model_path / "SHA256SUMS"
    actual_model_manifest_sha256 = sha256_file(model_manifest)
    if actual_model_manifest_sha256 != args.model_manifest_sha256:
        raise RuntimeError(
            "model manifest SHA-256 mismatch: "
            f"expected {args.model_manifest_sha256}, got {actual_model_manifest_sha256}"
        )

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    device_info = json_safe(mx.device_info())
    recommended = int(device_info["max_recommended_working_set_size"])
    requested_wired_limit = (
        recommended if args.wired_limit == "default" else int(args.wired_limit)
    )
    if requested_wired_limit <= 0 or requested_wired_limit > recommended:
        raise RuntimeError(
            f"wired limit {requested_wired_limit} is outside (0, {recommended}]"
        )

    previous_wired_limit = int(mx.set_wired_limit(requested_wired_limit))
    effective_probe_previous = int(mx.set_wired_limit(requested_wired_limit))
    if effective_probe_previous != requested_wired_limit:
        raise RuntimeError(
            "wired-limit probe failed: the second set did not observe the requested prior value"
        )

    worker_path = Path(__file__).resolve()
    generate_path = Path(sys.modules[generate_step.__module__].__file__).resolve()
    package_path = Path(mlx_lm.__file__).resolve()
    emit(
        {
            "schema": EVENT_SCHEMA,
            "kind": "worker_start",
            "pid": os.getpid(),
            "source_commit": args.source_commit,
            "model_key": args.model_key,
            "model_label": args.model_label,
            "model_manifest_sha256": actual_model_manifest_sha256,
            "token_file": args.token_file.name,
            "token_sha256": args.token_sha256,
            "input_tokens": args.input_tokens,
            "generated_tokens": args.generated_tokens,
            "warmups": args.warmups,
            "trials": args.trials,
            "arm": args.arm,
            "python": platform.python_version(),
            "platform": {"macos": platform.mac_ver()[0], "machine": platform.machine()},
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_lm_version": importlib.metadata.version("mlx-lm"),
            "mlx_lm_package_sha256": sha256_file(package_path),
            "generate_source_sha256": sha256_file(generate_path),
            "worker_sha256": sha256_file(worker_path),
            "device_info": device_info,
            "requested_wired_limit_bytes": requested_wired_limit,
            "recommended_working_set_bytes": recommended,
            "previous_wired_limit_bytes": previous_wired_limit,
            "wired_limit_effective": True,
        }
    )

    load_started_ns = time.perf_counter_ns()
    model, _tokenizer = load(str(args.model_path), lazy=False)
    mx.synchronize()
    load_finished_ns = time.perf_counter_ns()
    prompt = mx.array(token_ids, dtype=mx.int32)
    mx.eval(prompt)
    mx.synchronize()
    emit(
        {
            "schema": EVENT_SCHEMA,
            "kind": "model_loaded",
            "load_duration_ns": load_finished_ns - load_started_ns,
            "memory": memory_snapshot(mx),
        }
    )

    measured_output_sha256: str | None = None
    phases = [("warmup", index) for index in range(args.warmups)] + [
        ("measured", index) for index in range(args.trials)
    ]
    for phase, trial_index in phases:
        mx.synchronize()
        mx.clear_cache()
        mx.synchronize()
        mx.reset_peak_memory()
        mx.random.seed(0)
        prompt_cache = make_prompt_cache(model)
        mx.synchronize()
        memory_start = memory_snapshot(mx)

        emit(
            {
                "schema": EVENT_SCHEMA,
                "kind": "trial_start",
                "phase": phase,
                "trial_index": trial_index,
            }
        )
        started_ns = time.perf_counter_ns()
        iterator = generate_step(
            prompt,
            model,
            max_tokens=args.generated_tokens,
            prompt_cache=prompt_cache,
            prefill_step_size=2048,
        )
        outputs: list[int] = []
        offsets_ns: list[int] = []
        for token, logprobs in iterator:
            materialized_ns = time.perf_counter_ns()
            outputs.append(int(token))
            offsets_ns.append(materialized_ns - started_ns)
            del logprobs
        mx.synchronize()
        memory_end = memory_snapshot(mx)

        if len(outputs) != args.generated_tokens:
            raise RuntimeError(
                f"generate_step yielded {len(outputs)} IDs, expected {args.generated_tokens}"
            )
        if len(offsets_ns) < 2 or any(
            later <= earlier for earlier, later in zip(offsets_ns, offsets_ns[1:])
        ):
            raise RuntimeError("token materialization timestamps are not strictly increasing")
        itls_ns = [later - earlier for earlier, later in zip(offsets_ns, offsets_ns[1:])]
        ttft_ns = offsets_ns[0]
        decode_duration_ns = offsets_ns[-1] - offsets_ns[0]
        output_bytes = b"".join(struct.pack("<I", token) for token in outputs)
        output_sha256 = sha256_bytes(output_bytes)
        if phase == "measured":
            if measured_output_sha256 is None:
                measured_output_sha256 = output_sha256
            elif output_sha256 != measured_output_sha256:
                raise RuntimeError(
                    "greedy measured-trial output hash changed within one benchmark cell"
                )

        emit(
            {
                "schema": TRIAL_SCHEMA,
                "kind": "trial",
                "phase": phase,
                "trial_index": trial_index,
                "arm": args.arm,
                "model_key": args.model_key,
                "model_label": args.model_label,
                "input_tokens": args.input_tokens,
                "generated_tokens": len(outputs),
                "decode_intervals": len(itls_ns),
                "prompt_token_sha256": args.token_sha256,
                "output_token_sha256": output_sha256,
                "token_offsets_ns": offsets_ns,
                "ttft_ns": ttft_ns,
                "prefill_tok_s": args.input_tokens * 1_000_000_000.0 / ttft_ns,
                "decode_duration_ns": decode_duration_ns,
                "decode_tok_s": len(itls_ns) * 1_000_000_000.0 / decode_duration_ns,
                "itl_n": len(itls_ns),
                "itl_p50_ns": nearest_rank(itls_ns, 0.50),
                "itl_p95_ns": nearest_rank(itls_ns, 0.95),
                "itl_p99_ns": nearest_rank(itls_ns, 0.99),
                "mlx_memory_start": memory_start,
                "mlx_memory_end": memory_end,
                "mlx_peak_bytes": memory_end["peak_bytes"],
                "requested_wired_limit_bytes": requested_wired_limit,
            }
        )
        emit(
            {
                "schema": EVENT_SCHEMA,
                "kind": "trial_end",
                "phase": phase,
                "trial_index": trial_index,
            }
        )

        del iterator
        del prompt_cache
        del outputs
        del offsets_ns
        mx.synchronize()
        mx.clear_cache()

    emit(
        {
            "schema": EVENT_SCHEMA,
            "kind": "worker_end",
            "measured_trials": args.trials,
            "warmups": args.warmups,
            "measured_output_token_sha256": measured_output_sha256,
            "memory": memory_snapshot(mx),
        }
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        emit(
            {
                "schema": EVENT_SCHEMA,
                "kind": "failure",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc().splitlines(),
            }
        )
        raise SystemExit(1)
