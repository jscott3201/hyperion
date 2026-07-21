# Carry / Adapt / Drop inventory (vs Helios @ c5f4b06 and mlx-bonsai @ e99e2ee+T0)

Ruling of record for what the greenfield lifts, redesigns, or abandons. Evidence in the
source repos; file:line cites available in the analysis of 2026-07-21.

| Component | Verdict | From | Rationale (evidence) |
|---|---|---|---|
| C ABI shape (opaque handles + status enum + thread-local last-error + synchronous pull-step + telemetry struct) | **CARRY** | Helios (richer of the two) | Proven on both engines; 30-fn surface incl. StepResult telemetry and borrowed hidden-state handle. Slim to ≤25 fns. |
| Native execution model (define-by-run re-trace, deferred end-of-step KV eval, concat-grow global KV, per-step reset_peak) | **DROP** | Helios | Direct cause: chat p99 510.9 ms / code_review_8k p99 2161.7 ms vs p50 ~82 ms; 16K peak 21.874 GB. Replaced by bucketed compiled steps + in-place slice_update (own XR71 measured slice-update at ~0.010 ms/token — cheap, never defaulted). |
| Native forward math (5:1 dispatch `(l+1)%6`, K=V global, dual RoPE incl. partial 128/512, softcap 30 epilogue) | **CARRY** | Helios | Hard-won correct Gemma 4 math, verified against config; port as reference implementation. |
| Server core (axum + dedicated engine thread + bounded mpsc(8) + cancel-on-drop + reset-around-generation; real SSE) | **CARRY** | bonsai | Helios: hand-rolled std::net, thread-per-conn, 3-frame post-hoc "SSE" → dropped. |
| Dual API surface (Anthropic /v1/messages + OpenAI /v1/chat/completions, one PreparedPrompt, dialect = framing only) | **CARRY** | bonsai | Implemented + tested; Anthropic surface is what agent frameworks want. |
| Error taxonomy (400/401/409/413×2/429/529-pre-stream + SSE-error-mid-stream/503/500, constant-time auth, fail-closed non-loopback) | **CARRY** | bonsai | Hard-won pattern with contract tests. |
| Memory governor (predict peak transient per chunk; 3-source working set; shrink-to-fit; typed fail-closed OOM; READY/SOFT_PAUSED/HARD_REJECT) | **ADAPT** | bonsai pattern | Structurally superior (settled-only check once passed while true peak was 1.96 GB over ceiling). MUST re-derive constants from Gemma geometry — the ×6 fudge and 16-attn/48-GDN split are Qwen-specific (their own gates doc says cross-model constants are invalid). |
| Transactional cache handles (snapshot/rollback/commit) | **CARRY** | bonsai | Built, unused there; exactly the M6 conversation-cache + M7 MTP-rollback substrate. |
| KV design (uniform-ish per-layer tensors, slice-after-concat local, unbounded global) | **DROP → redesign** | Helios | Replaced by O(1) local ring + capacity-stepped single-tensor K=V global (05). Keep Helios's layer-type knowledge. |
| Prefix cache | **ADAPT** | both | Helios P06 exact-restore warm TTFT <0.1 ms (works) but XR07 realistic reuse failed parity + OOM; bonsai T0: −81.6% TTFT live but REJECTED on one changed transcript; only 2048-boundary bitwise-exact. → session-scoped, 2048-boundary-only, bitwise-gated design (05). Gemma KV is sliceable (bonsai's GDN state was not) — tractable now. |
| SSD cold tier | **DROP v1** | both parked | Helios P07 disabled pending variance; premature vs RAM reuse. Post-v1 lane requires payoff hypothesis. |
| KV compression (active) | **DROP** | Helios | q8 = storage-only 7–17%, active reduction 0.000%; q4 breaks greedy. Architecture (8 KiB/token global) removes the pressure. |
| MTP harness | **ADAPT, resequenced** | Helios design + official drafter | Correct architecture (cross-attn drafter, one-pass verify direction) but marathon ended +19.969% < 25% gate with verifier=85.76% of cost on the slow substrate. M7 after kernels; same gate; 2-round cap. |
| Sampler | **DROP both → build** | — | Both greedy-only argmax (Helios sampler crate unused by server). Agentic target needs sampled defaults + seeds + penalties + constrained-JSON. |
| Tokenizer | **CARRY mechanism** | bonsai | In-process crate pattern; retarget to Gemma (HF tokenizers, 262144). Helios shelled to Python mlx_lm per request → banned. |
| Chat template | **ADAPT mechanism / REWRITE content** | bonsai render path | In-process minijinja + golden fixtures. Helios's two Gemma formatters were both wrong (roles/thinking) and the server ignored them — R6. |
| Tool-call layer | **ADAPT** | bonsai | Incremental fragment assembler + schema validation + degrade-never-drop + server-minted IDs carried; rewrite wire format for Gemma tokens; add parallel-call dedupe; add constrained-JSON lane (new). |
| Parity oracle (mlx-lm subprocess, two-phase reap-before-native, two-sided fault thresholds, opaque cache) | **CARRY** | bonsai | Retarget to gemma4 (mlx-lm 0.31.3 has it); drop GDN-state channel; re-derive thresholds. |
| Agent-eval harness (turn loop, sandbox path-safety, determinism/FNV fingerprints, honesty split raw-vs-repaired, multi-arm) | **ADAPT** | bonsai | Mechanism carried ~verbatim; task suite/tools/graders re-domained to smart-buildings (09); enforce network isolation (was convention-only — admitted gap). |
| Kernel A/B protocol (G1–G4, A-C-C-A, candidate-min>baseline-max, bit-exact tamper-proof audit binding SHAs, adversarial refute, append-only ledger) | **CARRY verbatim** | bonsai | Crown jewel; ~90% of capture/compare rig reusable (re-parameterize vocab/corpus). Build the specified-but-unbuilt lossy top-1/KL comparator (E1 needs it). |
| Real-workload corpus + protected-aggregate holdouts + low_n policy | **CARRY method / REPLACE contents** | Helios | The discipline that kept MTP honest. Contents become target W1–W5-shaped prompts, frozen with SHA manifests. |
| TUI (Ratatui, provider-boundary, snapshot-tested) | **DEFER** | Helios design | Clean but non-critical; post-v1. |
| LoRA adapters (registry/trust/manifest design) | **DEFER application, keep design** | Helios | Registry solid (12 tests); live application never truly worked (P09 no token change). Not a smart-buildings v1 need. |
| Router crate / pure-Rust engine crate / llama-baseline crate | **DROP** | Helios | Placeholder (18 lines) / unused island / obsolete baseline. |
| Ternary kernels (base3 codecs, scale-q2, BitLinear lineage) | **DROP** | bonsai | Dies with the Bonsai model lane. Keep only the A/B methodology those probes exercised. |
| M5-floor canary ADR (gen-17 assert, no ladder, stock-reference-required) | **CARRY** | bonsai ADR 0003 | Verbatim pattern, updated to macOS 26.2/MLX 0.32.0 pins. |
| Evidence ledger format (MEASURED/DECIDED/ASPIRATIONAL; append-only; machine-state headers) | **CARRY** | bonsai + Helios | Both engines' best shared habit. |
