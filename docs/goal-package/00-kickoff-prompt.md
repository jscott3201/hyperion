# Kickoff prompt (Claude Code / Codex goal)

Use after the repository is created and this package is placed at `_goals/hyperion-goal-package/`
(or repo root `docs/goal-package/`). Run milestones in order; never skip a gate.

```text
/goal Implement hyperion, a clean-sheet Rust 1.95+ + MLX 0.32.0 inference engine exclusively
for the Google Gemma 4 model family on Apple M5 Macs (macOS 26.2+, M5 generation is the hard
compatibility floor — no M1–M4 fallbacks), optimized first for the 16 GB MacBook Pro M5
profile and for non-coding agentic workloads (tool calling with strict JSON, thinking-mode
control, long multi-turn conversations). Follow the goal package at docs/goal-package/:
read 01-goal-contract.md and AGENTS.md first, then execute milestones M0–M8 from
10-milestones-and-gates.md strictly in order. Hard constraints: single native backend behind
a narrow C ABI (no Python on the request path, no helper subprocess backend); heterogeneous
KV from day one (O(1) ring for the 40 local sliding-window layers, capacity-stepped single
K=V tensor for the 8 global layers); shape-bucketed compiled decode step (no per-step graph
re-trace, no unbounded concatenate-grow KV, no per-step reset_peak_memory); device-derived
memory ceiling with a predictive pre-admission governor (fail-closed 529); real SSE streaming
on both OpenAI and Anthropic surfaces; in-process tokenizer and Gemma 4 chat template with
native tool-call wire format and thinking-mode policy; correctness before speed everywhere —
mlx-lm (>=0.31.3, pinned) is the parity oracle and the A/B verification protocol in
08-correctness-and-verification.md governs every promotion (G1 parity, G2 throughput,
G3 agent-eval floor, G4 memory/governor; candidate-min must beat baseline-max; append-only
ledger; never move a floor to pass). Speculative correctness is TARGET-VERIFIED, not
byte-identical — every emitted token must equal a real verify-pass argmax and each greedy-
divergence must be an explained sub-0.5 near-tie (ship the near_tie_events counter from M2);
do NOT gate on "MTP-on == MTP-off byte-identity" (unachievable at fp16/bf16 scale). Speculative
rollback is append-only (discard by not committing; never trim a rotated sliding-window ring).
Build attention masks per layer-kind from that kind's own cache. The governor targets the
MEASURED throughput optimum (below the OOM ceiling — run the M1 budget sweep), never live OS
availability. MTP speculative decoding is milestone M7 (default-off, promotes only at >= +25%
protected aggregate); a zero-cost n-gram adaptive prompt-lookup drafter is a second speculative
lane at M5. Do not build: SSD cache tiers, KV compression for active decode, LoRA application,
multimodal, TUI, tree speculative decoding, the M9 26B-A4B streaming tier (owner-gated), or any
M1–M4 support. Record MEASURED evidence rows for every benchmark claim with exact commands and
machine state. Between iterations, complete the next unaccepted milestone in order and stop if a
gate cannot be defensibly passed under the constraints.
```

## Session-zero checklist (before M0 coding)

1. Confirm hardware: `sysctl hw.model`, macOS ≥ 26.2, M5 (gen-17 GPU family canary — see
   02-architecture.md §Startup canary). Record `recommendedMaxWorkingSetSize`.
2. Pin toolchain: Rust 1.95+, MLX 0.32.0 (exact), mlx-lm 0.31.3+ (oracle venv, pinned),
   CMake + Metal toolchain. Record all versions in the ledger header.
3. Download weights (Hugging Face, Apache-2.0):
   - `google/gemma-4-12B-it-qat-q4_0-unquantized` (primary base for quantization)
   - `google/gemma-4-12B-it` (bf16 — parity/QAT-delta spot-checks ONLY; ≈24 GB, does NOT fit
     16 GB resident: run off-device or via streaming convert. The on-device oracle and native
     arm both run the SAME Q4 so parity isolates the engine, not the quant.)
   - `google/gemma-4-E4B-it-qat-q4_0-unquantized` (M8)
   - `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant` (MTP drafter, M7 — 4L, hidden 1024,
     centroid logit head; alias the vocab table with the target, match dtypes at the boundary)
   - `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant` (E4B MTP drafter, M8)
   Store under `artifacts/models/` (gitignored).
4. Verify the oracle: `mlx_lm.generate` runs gemma-4-12B (community MLX conversion or local
   `mlx_lm.convert` output) on this machine before any native code exists.
