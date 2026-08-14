# hyperion — goal package v0.3 (2026-08-14)

Agent-executable spec for **Hyperion**: a clean-sheet Rust + MLX inference engine for
**Gemma 4 and Qwen's `qwen3_5` hybrid architecture** on **Apple M5** Macs, 16 GB MacBook Pro
first, built for local agentic workloads.

> **v0.3 (2026-08-14):** [ADR 0006](../decisions/0006-dual-family-v1-scope-and-gates.md)
> supersedes the Gemma-only forward scope and old active milestone order. It preserves accepted
> ADRs/evidence, makes Gemma a regression family, and adds the pinned Qwen3.8-27B text path behind
> separate architecture/conversation/deployment identities. Read ADR 0006 before older technical
> chapters; Gemma-specific facts remain valid only for Gemma.

> **v0.2 (2026-07-21):** incorporates findings from the NunSpark exploration — the target-verified
> exactness reframing (A1), append-only KV rollback (A2), mask-by-kind correctness (A3), the
> throughput-optimum governor (A4), the MoE-spec rule (A5), an n-gram adaptive drafter lane (B1),
> confirmed drafter details (B3), prefix rotation safety (B4), corrected tree-spec reasoning (B5),
> and a new owner-gated M9 (26B-A4B streaming tier). Full provenance:
> `references/nunspark-adoptions.md`.

Successor-of-record to two retired engines: **Helios/gemma4d** (Gemma 4 12B runtime; mined for
its model math, FFI shape, and XR evidence) and **mlx-bonsai** (Ternary-Qwen engine; mined for
its serving core, governor, eval harness, and A/B verification protocol). Bonsai model support
and its tested ternary artifact remain dropped on measured quality; that result does not apply
to the separately released Qwen3.8 dense hybrid checkpoint.

## Read order

1. `../decisions/0006-dual-family-v1-scope-and-gates.md` — current owner ruling and sequence.
2. `00-kickoff-prompt.md` — concise agent handoff.
3. `01-goal-contract.md` — north star, constraints, non-goals, completion rule.
4. `10-milestones-and-gates.md` — active P0–P7 dependency DAG and gate truth.
5. `02-architecture.md` — workspace, threading, family seam, native boundary.
6. `03-native-runtime-and-kernels.md` — current Gemma graph/kernel baseline.
7. `04-model-family-and-weights.md` — family-specific model and artifact facts.
8. `05-kv-and-memory.md` — family-specific state, budget, governor, and snapshots.
9. `06-serving-and-agentic-api.md` — shared API plus checkpoint conversation profiles.
10. `07-speculative-decoding.md` — deferred Gemma MTP research baseline.
11. `08-correctness-and-verification.md` — per-artifact oracle and promotion protocol.
12. `09-agent-eval.md` — shared tasks with per-artifact/profile evidence.
13. `11-risk-register.md` — inherited and dual-family risks.
14. `SOURCES.md` and `VERIFICATION.md` — source map and historical review record.

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

## Ruled owner decisions

| ID | Decision | Ruling |
|---|---|---|
| O-2 | License / open-vs-private | **Ruled 2026-07-31:** public source; Hyperion-authored source and documentation are `MIT OR Apache-2.0` at the recipient's option. Model artifacts and third-party material retain their own terms. See ADR 0005. |
| O-11 | Dual-family V1 direction | **Ruled 2026-08-14:** Gemma 4 remains first-class; add the pinned Qwen3.8 text path as a separate `qwen3_5` architecture. See ADR 0006. |

## Open owner decisions

O-3 through O-10 are retained as historical Gemma-subprogram rulings. ADR 0006 overrides their
old milestone scheduling; none expands the active P0–P7 scope implicitly.

| ID | Decision | Default until ruled |
|---|---|---|
| O-1 | Project name (hyperion; alternates: sequoialess naming, keep `gemma4d` lineage) | hyperion |
| O-3 | E2B tier inclusion in v1 gates | Out; geometry supported, ungated |
| O-4 | Constrained-JSON decoder: build token-mask engine vs port (xgrammar-style) vs post-hoc-only v1 | Build minimal schema-mask engine at M5, post-hoc fallback |
| O-5 | Multimodal (12B encoder-free 35M projector; nameplate OCR) timing | Post-v1 lane, not gated |
| O-6 | KV-quant lane (q8/q4 KV) | Parked; revisit only if 128K becomes a hard requirement |
| O-7 | Anthropic surface parity depth (count_tokens, system blocks) | Mirror mlx-bonsai's implemented subset |
| O-8 | 26B-A4B / 31B activation timing | **Upgraded (v0.2):** 26B-A4B is measured-feasible on 16 GB via expert streaming (M9, NunSpark recipe) — deferred, owner-gated, not v1. 31B still needs 32 GB+. Geometry-validated in v1 either way. |
| O-9 | n-gram adaptive drafter default (on / off / opt-in) at M5 | Ship on for structured-output lanes; disable threshold derived from compute-bound cost at M5 |
| O-10 | Promote M9 (26B-A4B streaming tier) into the active build? | Deferred — build only on explicit owner go; a new disk-streaming subsystem, orthogonal to the v1 resident engine |
