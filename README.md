# Hyperion

Hyperion is a clean-sheet local inference engine for the Google Gemma 4 family on Apple M5
Macs. The hard runtime floor is macOS 26.2 and MLX 0.32.0; there are no M1–M4 runtime paths and
no alternate/helper backend.

The repository is milestone-gated. M0 establishes the six-crate Rust 1.95 workspace, one C++
MLX backend behind a narrow C ABI, a compiled-and-loaded custom metallib, a fail-loud startup
canary, a pinned bench-only `mlx-lm` oracle, and append-only evidence. HTTP serving is not
advertised before M3.

## M0 commands

```sh
# Fast/model-free tier 1 (also used by development PR CI)
scripts/ci-pr.sh

# Real M5 native canary
cargo run --locked -p hyperion-bench -- canary

# Project-owned oracle environment and real 12B smoke
scripts/setup-oracle.sh
scripts/download-m0-model.sh
scripts/convert-m0-model.sh
scripts/oracle-smoke.sh
```

The goal contract and milestone gates live in [`docs/goal-package`](docs/goal-package/INDEX.md).
Benchmark claims are valid only when present as `MEASURED` rows in the append-only ledger.

The self-hosted `release-gate` keeps checkout cleanup enabled and recreates the locked oracle.
Its protected `hyperion-m5-release` environment must define `HYPERION_M0_SOURCE_MODEL` and
`HYPERION_M0_ORACLE_MODEL` as paths outside the checkout. Each path must contain the reviewed
`SHA256SUMS`; the release script verifies both manifests and every payload before execution.
