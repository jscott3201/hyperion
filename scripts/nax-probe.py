#!/usr/bin/env python3
"""M0 empirical MLX route probe; bench-only and never imported by serving code."""

from __future__ import annotations

import json
import platform
import statistics
import time

import mlx.core as mx


K = 3840
N = 3840
ROWS = (1, 64, 512, 2048)
TRIALS = 5


def timed(operation):
    output = operation()
    mx.eval(output)
    warm_checksum = float(output[0, 0].item())
    elapsed = []
    for _ in range(TRIALS):
        started = time.perf_counter_ns()
        output = operation()
        mx.eval(output)
        elapsed.append(time.perf_counter_ns() - started)
    checksum = float(output[0, 0].item())
    if not checksum == warm_checksum:
        raise RuntimeError("probe checksum drifted across identical operations")
    return elapsed, checksum


def emit(mode: str, rows: int, elapsed: list[int], checksum: float) -> None:
    median_ns = int(statistics.median(elapsed))
    operations = 2 * rows * K * N
    payload = {
        "schema": "hyperion.nax-route-probe.v1",
        "mode": mode,
        "shape": [rows, K, N],
        "warmups": 1,
        "trials": TRIALS,
        "nanoseconds": elapsed,
        "median_nanoseconds": median_ns,
        "nominal_tflops": operations / median_ns / 1_000,
        "checksum": checksum,
    }
    print("MEASURED " + json.dumps(payload, separators=(",", ":"), sort_keys=True))


def main() -> None:
    mx.random.seed(0)
    info = mx.device_info()
    header = {
        "schema": "hyperion.nax-route-probe-host.v1",
        "mlx": mx.__version__,
        "macos": platform.mac_ver()[0],
        "architecture": info.get("architecture", "unknown"),
        "recommended_working_set_bytes": info.get(
            "max_recommended_working_set_size", 0
        ),
    }
    print("MEASURED " + json.dumps(header, separators=(",", ":"), sort_keys=True))

    base_weight = mx.random.normal((N, K))
    mx.eval(base_weight)
    weights = {
        "bfloat16": base_weight.astype(mx.bfloat16),
        "float16": base_weight.astype(mx.float16),
    }
    mx.eval(*weights.values())
    quantized, scales, biases = mx.quantize(
        weights["float16"], group_size=64, bits=4, mode="affine"
    )
    mx.eval(quantized, scales, biases)

    for rows in ROWS:
        for mode, dtype in (("bfloat16", mx.bfloat16), ("float16", mx.float16)):
            inputs = mx.random.normal((rows, K)).astype(dtype)
            mx.eval(inputs)
            elapsed, checksum = timed(lambda: inputs @ weights[mode].T)
            emit(mode, rows, elapsed, checksum)
            del inputs
            mx.clear_cache()

        inputs = mx.random.normal((rows, K)).astype(mx.float16)
        mx.eval(inputs)
        elapsed, checksum = timed(
            lambda: mx.quantized_matmul(
                inputs,
                quantized,
                scales,
                biases,
                transpose=True,
                group_size=64,
                bits=4,
                mode="affine",
            )
        )
        emit("q4-affine-g64", rows, elapsed, checksum)
        del inputs
        mx.clear_cache()


if __name__ == "__main__":
    main()
