# hyperion — agent operating contract (BUILD-LAW)

You are working on `hyperion`, a clean-sheet Rust + MLX inference engine for the Google Gemma 4
and Qwen `qwen3_5` hybrid families on Apple M5 (16 GB MacBook Pro first), for local agentic
workloads. This file is law inside the repo; the goal package (`docs/goal-package/`) and
[ADR 0006](../decisions/0006-dual-family-v1-scope-and-gates.md) are the spec of record.

## Hard constraints

- Platform floor M5 / macOS 26.2+ / MLX 0.32.0 pinned. No M1–M4 paths, no capability ladder.
- Single native backend behind the narrow C ABI in `native/hyperion_mlx/include/`. No Python
  on the request path. No helper/stub serving backends (test doubles live in tests only).
- Model scope: Gemma 4 plus Qwen's `qwen3_5` hybrid architecture, beginning with the pinned
  Qwen3.8-27B text path. Recognition, implementation, and accepted support are separate states.
- Execution model (immutable without an ADR): family-specific architecture adapters behind one
  native runtime; shape-stable compiled steps; transactional state mutation; no unbounded
  `concatenate` in hot loops; no per-step `reset_peak_memory`; explicit streams; one engine
  thread owns MLX. Gemma retains its local-ring/global-K=V layout. Qwen uses FP32 Gated DeltaNet
  recurrent state plus KV only for its full-attention layers until evidence changes that policy.
- 16 GB budget: device-derived ceiling ≈12.06 GB effective; governor fail-closed (529);
  zero uncontrolled OOM is a standing gate.
- Correctness before speed: G1 parity outranks every benchmark. Never move a floor, edit a
  frozen fixture, or promote with overlapping noise distributions.

## Implementation discipline

1. Read ADR 0006 and the current phase in `docs/goal-package/10-milestones-and-gates.md` plus its
   spec files before coding. Follow the dependency DAG; do not claim a later phase's support.
2. Baseline before optimizing; A-C-C-A ordering; candidate-min > baseline-max.
3. Record exact commands + machine state for every MEASURED claim; ledgers append-only.
4. `unsafe` only in `hyperion-ffi`; every ABI handle has lifecycle tests.
5. Small reviewable commits; every accepted phase ends with a numbered decision record.
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
