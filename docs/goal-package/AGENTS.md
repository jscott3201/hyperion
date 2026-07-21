# hyperion — agent operating contract (BUILD-LAW)

You are working on `hyperion`, a clean-sheet Rust + MLX inference engine exclusively for the
Google Gemma 4 family on Apple M5 (16 GB MacBook Pro first), for non-coding agentic
workloads. This file is law inside the repo; the goal package (`docs/goal-package/`) is the
spec of record.

## Hard constraints

- Platform floor M5 / macOS 26.2+ / MLX 0.32.0 pinned. No M1–M4 paths, no capability ladder.
- Single native backend behind the narrow C ABI in `native/hyperion_mlx/include/`. No Python
  on the request path. No helper/stub serving backends (test doubles live in tests only).
- Model scope: Gemma 4 family ONLY, config-driven geometry. 12B primary, E4B second tier;
  26B-A4B/31B geometry-validated, never runtime-gated on 16 GB.
- Execution model (immutable without an ADR): shape-bucketed compiled steps; in-place
  `slice_update` KV (local ring 1024 / global capacity-stepped K=V); no `concatenate` in hot
  loops; no per-step `reset_peak_memory`; explicit streams; engine thread owns MLX.
- 16 GB budget: device-derived ceiling ≈12.06 GB effective; governor fail-closed (529);
  zero uncontrolled OOM is a standing gate.
- Correctness before speed: G1 parity outranks every benchmark. Never move a floor, edit a
  frozen fixture, or promote with overlapping noise distributions.

## Implementation discipline

1. Read the current milestone in `docs/goal-package/10-milestones-and-gates.md` + its spec
   files before coding. Do not jump ahead.
2. Baseline before optimizing; A-C-C-A ordering; candidate-min > baseline-max.
3. Record exact commands + machine state for every MEASURED claim; ledgers append-only.
4. `unsafe` only in `hyperion-ffi`; every ABI handle has lifecycle tests.
5. Small reviewable commits; every milestone ends with a numbered decision record.
6. Real-tensor rule: no fixture-gated milestone where real weights are feasible.

## Subagent policy

Use subagents for isolated research/verification/profiling/review; they return evidence,
not opinions. Recommended lanes: codebase-mapper, external-researcher (MLX/Metal/HF
version drift), kernel-engineer (Metal/NAX), correctness-reviewer (parity/fixtures),
perf-analyst (ledger interpretation), adversarial-reviewer (pre-promotion refute pass —
mandatory for kernel/cache/MTP promotions), security-reliability-reviewer (FFI/unsafe/
server). Optimal team size is 3–5 focused agents; verification quality beats agent count
(gemma-challenge process lesson).

## Completion rule

A milestone is complete only when its gate criteria hold on concrete evidence: tests,
ledger rows, decision records, or documented blockers. If blocked: stop with attempted
paths, evidence, blocker, and the decision needed from the owner.
