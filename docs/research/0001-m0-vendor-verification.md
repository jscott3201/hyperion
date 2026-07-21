# M0 vendor verification — 2026-07-21

This record binds M0 implementation choices to the exact external revisions inspected. It is
not a benchmark ledger and makes no performance-promotion claim.

## Native MLX

- Pin: MLX tag `v0.32.0`, commit `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`.
- Local package: Homebrew MLX 0.32.0; `MLXConfig.cmake`, headers, dylib, and bundled metallib
  are from the same keg. Hyperion uses `find_package(MLX 0.32.0 EXACT CONFIG REQUIRED)` and
  separately checks header macros plus `mlx::core::version()` at runtime.
- The exact
  [`ScaledDotProductAttention::use_fallback`](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/scaled_dot_product_attention.cpp#L591-L640)
  source admits full attention only at equal Q/V head dimensions 64, 80, or 128. Its vector
  path admits equal dimensions 64, 96, 128, or 256 plus the 192/128 special case, requires
  `q_len <= 8`, and caps `q_len × GQA <= 32`. Gemma 4 local head dimension 256 therefore has
  a fused decode path but not fused full prefill; global dimension 512 falls back in both.
- The exact
  [`quantized.cpp`](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/quantized.cpp)
  source selects generation-aware vector limits, dispatches small-batch `qmv_wide` for M≥2,
  and uses split-K QMM after the vector limit. Hyperion does not hardcode those thresholds.
- Direct C++ MLX is the chosen native dependency. `mlx-c` is not used, so no claim about its
  `fast::metal_kernel` coverage enters M0.

## Oracle and conversion

- Oracle package metadata remains version 0.31.3, sourced from exact upstream commit
  [`8239c72de5a0e42c539e30489021db73c7fe258c`](https://github.com/ml-explore/mlx-lm/commit/8239c72de5a0e42c539e30489021db73c7fe258c),
  paired with `mlx==0.32.0`. `oracle/uv.lock` binds the full dependency graph.
- At that commit, `MODEL_REMAPPING` maps `gemma4_unified` to `gemma4`. Gemma 4 sanitation
  drops vision/audio/projector weights, range-stat tensors, rotary-embedding tensors, and
  redundant shared-KV projections. This is the reviewed text-only conversion behavior; the
  checkpoint config is not rewritten.
- The Gemma 4 quantization predicate forces router projections to affine group-64 8-bit and
  otherwise allows the requested recipe. The dense 12B conversion therefore uses the declared
  default recipe: affine group-64, 4-bit for eligible layers. M2 separately measures any QAT
  quality delta; M0 makes no quality claim.

## Checkpoint

- Primary source: `google/gemma-4-12B-it-qat-q4_0-unquantized`, immutable revision
  `b6ed86275a6a5735884e208bfed95b445a684ca2`.
- The reviewed local `config.json` declares `model_type: gemma4_unified`, 48 text layers,
  hidden size 3840, 16 query heads, local/global head dimensions 256/512, and a 262,144-token
  vocabulary. These are observations only; the validated Rust geometry arrives at M2/M8.
- The reviewed local model-card frontmatter declares `apache-2.0` and links the Gemma 4
  license. The source snapshot and converted output are bound by local SHA-256 manifests before
  M0 acceptance.

## Empirical M5 probe

`HYP-M0-NAX-PROBE-001` in `benchmarks/BENCHMARKS.md` records the required BF16/FP16/Q4
3840-wide sweep. It establishes working capability and shape-dependent throughput only. Since
MLX does not expose the selected internal matmul route as stable public telemetry, the row does
not assert that any individual cell used NAX.
