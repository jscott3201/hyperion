# 01 — Goal contract

## North star

One sentence: **the best-in-class local inference engine for the Gemma 4 family on M5 Macs,
measured on agentic quality-per-watt-per-GB, not on generic benchmark theater.**

hyperion exists so that demanding local agentic workloads (facility diagnostics,
recommendations, point tagging, and operator-assistance workflows) run on a 16 GB MacBook Pro M5
with: fast time-to-first-token on long agent transcripts, decode throughput at or beyond the
memory-bandwidth ceiling via speculative decoding, strict tool-call reliability, and evidence
for every claim. It is also the dev/eval substrate for prompt & schema iteration ahead of
hosted deployment (A12 posture: engine+quant are identity-tuple fields; hyperion results are
iteration evidence, not production conformance).

## Why clean-sheet (ruled 2026-07-21)

- Helios/gemma4d proved the model math but its native execution model (define-by-run re-trace,
  deferred global-first KV eval, concatenate-grow global KV, per-step reset_peak) is the direct
  cause of decode-tail pathology (chat p99 510.9 ms vs p50 82.4 ms; code_review_8k p99
  2161.7 ms) and the 16K memory cliff (peak 21.874 GB vs a 14 GB gate). Retrofitting the
  execution model means rewriting the core anyway.
- mlx-bonsai's model lane (Ternary Qwen3.6) is dropped on measured evidence: agentic stretch
  1/3, expected-tool share 26.39%, well-formed 93.06% only after server repair, ~13–17 tok/s.
  Its serving/verification infra is the best-of-breed carry.
- M5 + MLX 0.32.0 + Gemma-4-only scope invalidates enough assumptions (NAX prefill economics,
  head-dim kernel gaps, QAT checkpoints, official MTP drafter) that a from-scratch spine with
  targeted organ transplants beats an in-place evolution.

## Scope

| Tier | Models | Status in v1 |
|---|---|---|
| Primary | `gemma-4-12B-it` from QAT-q4_0-unquantized, 4-bit | All gates run here |
| Secondary | `gemma-4-E4B-it` (QAT 4-bit) — edge/community tier | M8 gates on m5-16g |
| Architected | `gemma-4-26B-A4B` (MoE), `gemma-4-31B` | Geometry + config validation only; runtime gated on 32 GB+ hardware (O-8) |
| Supported geometry, ungated | `gemma-4-E2B` | O-3 |

Text-only v1. The 12B's encoder-free multimodal path (35M linear projector) is a designed-for
post-v1 lane (O-5, nameplate-OCR commissioning use case), not a gate.

## Hard constraints (violations = stop and report)

1. **Platform floor:** Apple M5 generation, macOS 26.2+, MLX 0.32.0. No M1–M4 code paths,
   no capability ladder. Startup canary asserts the floor and fails loudly (mlx-bonsai
   ADR 0003 pattern). A stock mlx-lm reference path must always run on the same machine as
   the correctness baseline.
2. **Single native backend.** Rust owns serving/policy; one C++ MLX graph behind a narrow
   C ABI. No Python anywhere on the request path; no helper-subprocess backend; no
   stub-vs-helper-vs-native backend forks (Helios R5).
3. **16 GB profile is the binding budget.** Device-derived ceiling (measured
   `recommendedMaxWorkingSetSize` ≈ 12,713,115,648 B on m5-16g → effective budget
   ≈ 12.06 GB, soft watermark 90%). Every milestone's G4 gate enforces it. The Helios 16K
   behavior (21.874 GB peak) is the canonical failure this design must never reproduce.
4. **Correctness before speed.** G1 parity outranks every throughput number. Greedy
   token-exactness vs the pinned oracle; two-sided fault-boundary thresholds for logits.
   Never move a floor or edit frozen fixtures to pass (A/B protocol, carried verbatim).
   Speculative/MTP correctness is **target-verified** — every emitted token equals a real
   verify-pass argmax, each greedy-divergence a logged sub-0.5 near-tie — **not** byte-identity
   to single-token greedy (unachievable at fp16/bf16 scale; A1, 08).
5. **Baseline-then-gate.** No invented absolute targets. M1 measures stock mlx-lm on this
   exact machine; all promotion gates are relative to those MEASURED rows. (Known physics
   for orientation only, never as a gate: ~153 GB/s ÷ ~7 GB Q4 working set ⇒ low-20s tok/s
   dense-decode ceiling; NAX moves prefill ~3.5–4× vs M4 class, decode ~1.2×.)
6. **Agentic contract:** ≥1 tool call per assistant turn supported (Gemma 4 emits parallel
   calls), server-minted IDs, schema-validated, degrade-never-drop; thinking blocks stripped
   across turns, retained within a turn's tool-call chain; greedy AND sampled modes (Gemma
   defaults t=1.0/top-p 0.95/top-k 64) — greedy-only is a bonsai limitation, not a feature.
7. **Evidence discipline:** MEASURED/DECIDED/ASPIRATIONAL ledger; append-only; exact
   commands + machine state per row; A-C-C-A run ordering; candidate-min > baseline-max
   noise bar; adversarial-refute review before any kernel promotion.

## Non-goals (v1)

Production internet-facing serving; multi-user tenancy beyond single-flight+small queue;
SSD cache tier; KV compression for active decode (measured dead end: 0.000% active
reduction, q4 breaks greedy); LoRA application (registry design carried, application
deferred); TUI; DiffusionGemma; non-Gemma models; CUDA (the design should not preclude a
later CUDA port, but zero CUDA code in v1).

## Definition of done (v1)

M0–M8 accepted with gates green; ledger holds MEASURED rows for 12B-QAT-Q4 and E4B-QAT-Q4 on
m5-16g covering decode/prefill/TTFT/peak at 1K/4K/8K/16K/32K, agent-eval floor met, governor
calibrated to the **throughput optimum** (below the OOM ceiling; the M1 budget sweep located
it) with zero uncontrolled OOM aborts across the suite, conversation-cache TTFT evidence
recorded, the `near_tie_events` counter live (target-verified exactness auditable), and MTP
either promoted at ≥ +25% protected aggregate or parked with evidence — both outcomes are
acceptable completions for M7. M9 (26B-A4B streaming) is explicitly **out of the v1 DoD**
(owner-gated, O-10).
