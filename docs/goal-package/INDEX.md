# hyperion — goal package v0.2 (2026-07-21)

Agent-executable spec for **hyperion** (working name, O-1): a clean-sheet Rust + MLX inference
engine exclusively for the **Google Gemma 4 family** on **Apple M5** Macs, 16 GB MacBook Pro
first, built for non-coding agentic workloads (smart buildings / energy systems).

> **v0.2 (2026-07-21):** incorporates findings from the NunSpark exploration — the target-verified
> exactness reframing (A1), append-only KV rollback (A2), mask-by-kind correctness (A3), the
> throughput-optimum governor (A4), the MoE-spec rule (A5), an n-gram adaptive drafter lane (B1),
> confirmed drafter details (B3), prefix rotation safety (B4), corrected tree-spec reasoning (B5),
> and a new owner-gated M9 (26B-A4B streaming tier). Full provenance:
> `references/nunspark-adoptions.md`.

Successor-of-record to two retired engines: **Helios/gemma4d** (Gemma 4 12B runtime; mined for
its model math, FFI shape, and XR evidence) and **mlx-bonsai** (Ternary-Qwen engine; mined for
its serving core, governor, eval harness, and A/B verification protocol). Bonsai model support
is dropped entirely — poor measured agentic quality (EVAL-RESULTS.md, 2026-07-15).

## Read order

1. `00-kickoff-prompt.md` — the goal prompt to hand a Claude Code / Codex agent.
2. `01-goal-contract.md` — north star, hard constraints, non-goals, completion rule.
3. `02-architecture.md` — workspace, crates, threading, native boundary.
4. `03-native-runtime-and-kernels.md` — graph strategy, the SDPA gap, kernel lanes K1–K3.
5. `04-model-family-and-weights.md` — family matrix, QAT quantization pipeline, drafter.
6. `05-kv-and-memory.md` — heterogeneous KV, memory budget, governor, conversation cache.
7. `06-serving-and-agentic-api.md` — dual API, streaming, tool calls, thinking mode, sampler.
8. `07-speculative-decoding.md` — the MTP lane: design, gate, sequencing discipline.
9. `08-correctness-and-verification.md` — parity oracle, A/B protocol, gates G1–G4.
10. `09-agent-eval.md` — the re-domained agentic eval for smart-buildings workloads.
11. `10-milestones-and-gates.md` — M0–M8 with promotion gates (+ M9 post-v1, owner-gated).
12. `11-risk-register.md` — risks R1–R14 with mitigations, from measured history.
13. `SOURCES.md` — evidence map (primary sources, confidence ledger).
14. `VERIFICATION.md` — adversarial-review record (v0.1 findings + v0.2 NunSpark incorporation).

## References (grounding — read before native/kernel work)

| File | Purpose |
|---|---|
| `references/gemma4-family-facts.md` | CONFIRMED per-size config table (from config.json ×5) + attach mechanics. |
| `references/mlx-m5-facts.md` | MLX 0.32.0 capability/gap sheet, M5/NAX numbers, thread-safety footguns. |
| `references/carry-drop-inventory.md` | Component-level carry/adapt/drop rulings vs Helios + mlx-bonsai, with evidence. |
| `references/nunspark-adoptions.md` | **v0.2 provenance:** NunSpark corrections A1–A5, mechanisms B1–B5, anti-lessons, and the M9 (26B streaming) recipe — with measured evidence and where each lands in the spec. |

## Standing conventions

- `AGENTS.md` is the operating contract for any agent working in the eventual repo.
- Baseline-then-gate: no invented performance targets; gates are relative to MEASURED baselines.
- MEASURED / DECIDED / ASPIRATIONAL ledger discipline (mirrors mlx-bonsai BENCHMARKS.md).
- Engine + quantization are identity-tuple fields (A12 posture): hyperion on MLX is local/dev/
  commissioning evidence, not a production substrate claim.

## Open owner decisions

| ID | Decision | Default until ruled |
|---|---|---|
| O-1 | Project name (hyperion; alternates: sequoialess naming, keep `gemma4d` lineage) | hyperion |
| O-2 | License / open-vs-private (target open-core posture applies) | Apache-2.0 OR MIT dual, private repo |
| O-3 | E2B tier inclusion in v1 gates | Out; geometry supported, ungated |
| O-4 | Constrained-JSON decoder: build token-mask engine vs port (xgrammar-style) vs post-hoc-only v1 | Build minimal schema-mask engine at M5, post-hoc fallback |
| O-5 | Multimodal (12B encoder-free 35M projector; nameplate OCR) timing | Post-v1 lane, not gated |
| O-6 | KV-quant lane (q8/q4 KV) | Parked; revisit only if 128K becomes a hard requirement |
| O-7 | Anthropic surface parity depth (count_tokens, system blocks) | Mirror mlx-bonsai's implemented subset |
| O-8 | 26B-A4B / 31B activation timing | **Upgraded (v0.2):** 26B-A4B is measured-feasible on 16 GB via expert streaming (M9, NunSpark recipe) — deferred, owner-gated, not v1. 31B still needs 32 GB+. Geometry-validated in v1 either way. |
| O-9 | n-gram adaptive drafter default (on / off / opt-in) at M5 | Ship on for structured-output lanes; disable threshold derived from compute-bound cost at M5 |
| O-10 | Promote M9 (26B-A4B streaming tier) into the active build? | Deferred — build only on explicit owner go; a new disk-streaming subsystem, orthogonal to the v1 resident engine |
