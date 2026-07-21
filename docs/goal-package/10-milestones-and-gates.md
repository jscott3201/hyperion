# 10 — Milestones & gates

Rules: strict order; a milestone is accepted only on concrete evidence (tests, ledger rows,
decision record); **real-tensor rule** — no milestone may gate on fixtures/stubs where real
weights/tensors are feasible (Helios R1: M06–M11 "passed" on fixtures and the work was done
twice). Every milestone ends with a numbered decision record.

## M0 — Repo bootstrap & canary
Workspace (6 crates + native), CI tier-1 (build/clippy/tests/no-orphan-crates), CMake +
metallib build, startup canary (M5 gen / macOS 26.2+ / MLX 0.32.0 pin /
recommendedMaxWorkingSetSize → budget line), oracle venv pinned (mlx-lm 0.31.3+), weights
downloaded, ledger + decisions scaffolding.
**Gate:** clean build on m5-16g; canary MEASURED line recorded; oracle generates with
gemma-4-12B on this machine.

## M1 — Baselines (the A-arm, forever) + the memory-cliff sweep
Stock mlx-lm rows on m5-16g: 12B-QAT-Q4 (convert with default recipe) and E4B-QAT-Q4 —
decode/prefill tok/s, TTFT, peak, ITL p50/p95/p99 at 1K/4K/8K/16K/32K; mlx-lm server
agent-eval smoke arm. Nobody has published these numbers; they anchor everything. **Also run a
resident-budget/working-set sweep (A4)** to locate the M5-16GB throughput cliff — NunSpark
MEASURED 6 GB optimal / 10 GB collapse on a 16 GB M4; the M5 optimum is unknown and the governor
(05) targets it, not the OOM ceiling.
**Gate:** reproducible (5-trial, A-C-C-A self-check) MEASURED rows; ledger seeded; low_n rows
labeled; throughput-vs-budget curve recorded with the optimum identified.

## M2 — Native forward, parity, memory spine
Geometry-driven graph (12B): quantized matmuls (stock), heterogeneous KV (local ring
1024+γ_max, global capacity-stepped K=V), shape-bucketed compiled decode step, chunked
prefill (2048),
softcap epilogue, greedy sampler, E1 quant ablation (g32-grid vs g64 vs mixed_4_6; needs the
lossy top-1/KL comparator built here), unfused-but-correct attention everywhere (**masks built
per layer-kind — A3**), and the **`near_tie_events` counter (A1)** wired into `--metrics`.
**Gate:** G1 parity (token-exact + two-sided logit thresholds, derived); **16K peak ≤
~10 GB** (vs Helios 21.874 GB — the headline; carried by chunked prefill + no-concat-grow
KV, i.e. the execution-model change — NOT the K2 banded kernel, which lands at M4); 32K
sentinel under budget; decode within 0.9× of M1 stock (correct-first, fast-enough); mask-by-kind
correctness test (no `broadcast_shapes` crash once a sliding layer rotates past its window).

## M3 — Serving core & agentic surface v0
axum dual dialect + real SSE + cancel; in-process tokenizer + template golden fixtures
(incl. thinking + tool renders vs transformers reference); tool-call parser + dedupe +
repair telemetry; sampler surface; error taxonomy contract tests; governor v1
(geometry-derived constants, calibration probe); /control surface.
**Gate:** contract tests green (all error codes); streamed greedy run byte-identical to
M2 CLI on fixtures; governor calibration report (predicted vs measured peaks across
sentinels, error band recorded); zero uncontrolled OOM across suite.

## M4 — Kernel lane (the moat)
K1 global-decode kernel (hd512, K=V single-read); K2 windowed-flash local prefill (hd256,
band 1024) + global flash (hd512); K3 NAX/TensorOps probe (stock-first discipline);
per-layer-type dispatch table; A/B audit rig ported (Gemma-parameterized).
**Gate:** each kernel promotes via full protocol (G1 = tokens-exact + logit within the
fault-boundary threshold — flash kernels reorder FP reductions, so bit-exact logits are not
achievable; G2 candidate-min > baseline-max, G3 quality hold, G4 sentinels); tail gate
**ITL p99 ≤ 2× p50** on
chat_short; prefill and peak strictly improve vs M2 rows; decision records per kernel
(including any "stock wins" outcome — that is a valid, recorded result).

## M5 — Agentic layer v1
Thinking-mode policy (strip/retain/budget) + SSE thinking lane; parallel-call bounded
support; constrained-JSON v1 (schema-mask automaton, O-4) vs post-hoc arm; **n-gram adaptive
prompt-lookup drafter (B1)** as a cheap second speculative lane (zero resident cost, tokenizer-
exact, disable-to-zero policy; fits structured tool-call output — a natural partner to the
constrained-JSON lane); agent-eval suite (09) live with sandbox network isolation; suite floor
MEASURED and frozen.
**Gate:** constrained arm strictly dominates post-hoc malformed-rate at ≤10% decode
overhead (else recorded + opt-in); n-gram lane is target-verified (every divergence near-tie-
explained) and **never a net loss** on the agent-eval suite (adaptive disable-to-zero proven,
disable threshold derived from hyperion's compute-bound cost model — not NunSpark's disk-bound
numbers); adv tasks green (injection, dedupe-bait); floor row frozen as the G3 reference.

## M6 — Conversation cache
Session-scoped snapshot/restore at 2048 boundaries (native snapshot handles + Rust
policy/LRU); bitwise restore-equality harness; multi-turn TTFT evidence on the r0x/x0x
transcripts.
**Gate:** restored-vs-fresh **byte-identical tokens AND logit frames** at every advertised
boundary (k=2); measured multi-turn TTFT reduction reported against the T0 anchor
(79% avoidable prefill; −81.6% TTFT live probe); agent-eval outcomes identical warm vs
cold; LRU under byte budget with zero governor violations.

## M7 — MTP lane (either outcome acceptable)
Drafter load (assistant checkpoint): 4-layer Q-only, **vocab/embedding tensor aliased** with the
target, **centroid logit head** (2048 clusters / top-32), **dtype-matched at the boundary** (B3);
Q-only cross-attn into the target's captured K/V; feed the previous token via the **target's**
embedding table; constant `position_ids`. γ autotune; one-pass bucketed verify (K1 global kernel
at q_len=γ; qmv_wide). **Append-only rollback (A2):** speculative writes live in the local ring's
γ_max slack and are discarded by *not committing* — never `trim` committed state; the trim-or-drop
module is the one from 05/`PrefixCache`, validated after a forced ring rotation. **Target-verified
exactness (A1):** every emitted token = a real verify-pass argmax; the `near_tie_events` counter
explains each divergence from single-token greedy; an *unexplained* divergence auto-disables +
records a fixture (do NOT gate on byte-identity). Scoped opt-in wiring.
**Gate:** promote default-on ONLY at ≥ +25% protected aggregate (full corpus, A-C-C-A); else park
as scoped opt-in with decision record. Hard stop: two full A/B rounds sub-gate → park (R2).
Target-verified exactness green (all greedy-divergences near-tie-explained) in both outcomes;
rollback correctness proven after a forced sliding-window rotation.

## M8 — Family breadth (16 GB tier complete)
E4B end-to-end: shared-KV wiring (config-derived), PLE path, K≠V global (2 KV heads),
window 512, 128K ctx handling; full gate matrix on m5-16g; 26B-A4B + 31B geometry/manifest
validation (incl. MoE fields + router-8-bit rule) headless; capability-ladder refusal
(31B on 16 GB → clean UNSUPPORTED, not OOM).
**Gate:** E4B G1–G4 + agent-eval on m5-16g; 12B-vs-E4B delta rows recorded; geometry tests
green for all five configs; clean-refusal test for over-budget models.

## M9 — 26B-A4B MoE tier via expert streaming [POST-V1, owner-gated — NOT on the v1 critical path]
Documented capability, promoted into the build only on explicit owner go (O-8/O-10). Runs
Gemma-4-26B-A4B (128 experts, top-8) on 16 GB by disk-streaming only the router-fired experts —
NunSpark MEASURED the same shape class (Qwen3-30B-A3B) at **2.8–3.2 tok/s greedy on a 16 GB M4**,
reading ~100 MB/token vs ~1 GB. New subsystem (not in the v1 spec): per-expert packing +
manifest; a byte-budgeted two-region piece cache (LRU experts / MRU core); **persistent scatter
buffers** — never rebuild zero-filled 128-expert buffers (NunSpark's biggest single MoE fix,
1.6→3.2 tok/s; safe because the expert-gather reads only router-selected rows); verify-pass-
granularity temporal expert prefetch into an eviction-safe staging buffer. **MTP/spec OFF for
MoE (A5)** — greedy + expert-caching is the sparse lever. Nothing here is Gemma-MoE-specific yet
(new territory vs 26B-A4B's actual `mlp.experts` layout; NunSpark's `qwen3_moe` path is a
structural template only). Full recipe + evidence: references/nunspark-adoptions.md §C.
**Gate (if built):** fired-union selective load bit-identical to loading all experts (atol 0);
MEASURED greedy tok/s on 26B-A4B at m5-16g; peak within the throughput-optimum budget (A4);
clean UNSUPPORTED (not OOM) if disk/RAM can't host it.

## Post-v1 lanes (recorded, not scheduled)
**M9 (26B-A4B MoE streaming — owner-gated, above)**; multimodal projector (O-5, nameplate OCR);
KV-quant (O-6, only if 128K hard requirement); continuous batching / queue depth; sampled-mode
speculative acceptance; **tree spec** (the batch-flatten approach works on MLX — 07/B5 — but
needs a compute-bound payoff hypothesis and hybrid-attention validation first); SSD tier (only
with a real-workload payoff hypothesis — R3); LoRA application; TUI; CUDA port of the kernel
designs (the schedule/tile decisions transfer; TileLang evaluation note in the dossier).
