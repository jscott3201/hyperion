# Hyperion evidence ledger

This file is append-only after rows are accepted. Raw traces belong in `benchmarks/raw/` and
are intentionally ignored. Every claim is classified as `MEASURED`, `DECIDED`, or
`ASPIRATIONAL`; only `MEASURED` rows may support benchmark claims.

## Safe machine-state profile: `m5-16g-local-2026-07-21`

- Hardware class: MacBook Pro, base Apple M5, 16 GB unified memory, arm64
- OS: macOS 26.6
- Public Metal family: Apple10 (`1010`)
- MLX: 0.32.0 headers and dylib
- Rust: 1.95.0
- CMake: 4.4.0
- Privacy rule: never record serial numbers, platform UUIDs, account names, or credentials

## Row schema

Each row records an immutable ID, classification, source revision/state, exact command,
machine-state profile, inputs and hashes, complete metrics/output, trial count, and verdict.
Performance promotions additionally require A-C-C-A ordering, five measured trials after one
warmup, candidate-min greater than baseline-max, and G1–G4 evidence appropriate to the current
milestone.

## M0 rows

M0 accepted rows are appended after the final reviewed commit is rerun. The initial local
canary result is intentionally not frozen while the implementation worktree is still dirty.
