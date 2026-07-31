# 08 — Correctness & verification (the carried crown jewels)

Two systems carried near-verbatim: mlx-bonsai's parity oracle + kernel A/B protocol, and
Helios's real-workload benchmark methodology. They are complementary; both are law here.

## Parity oracle (G1 substrate)

- **Reference:** pinned `mlx-lm` (≥0.31.3 — first release with complete Gemma 4 support:
  layer pattern, KV sharing, p-RoPE, tool parser). Persistent line-JSON subprocess, bench
  scope only, never on the serving path.
- **Two-phase memory discipline (16 GB law):** oracle runs first, dumps reference tokens +
  logit frames to ephemeral `.f32`, process is reaped BEFORE native MLX initializes — the
  machine cannot hold both stacks (bonsai ADR 0004). References re-derived each run; opaque
  `make_cache()` — never assume the oracle's cache internals.
- **Compared:** greedy token IDs exact (per-case 16 tokens, ≥1 case at 64); full-vocab
  (262144) last-position prefill logits max-abs; per-sentinel-context cases (1K/4K/8K/16K/32K)
  + template/tool/thinking-mode render fixtures.
- **Two-sided fault-boundary thresholds (no magic tolerances):** measure clean max-abs;
  inject a real single-layer fault (e.g., RoPE offset +1 on one layer); the gate sits
  between measured-clean and measured-fault. Re-derive when MLX/quant/kernels change.
  (bonsai measured clean 0.0176 vs fault 0.0017-scale separation on Qwen — Gemma numbers
  will differ; derive, don't port.)
- **Quant honesty:** oracle arm and native arm run the SAME quantization (convert once,
  load both sides) so parity isolates the engine, not the quant. The QAT-vs-bf16 delta is
  a separate MEASURED row (E1), never conflated with engine parity.

## Kernel A/B protocol (promotion law — carried verbatim)

A candidate (kernel, execution-model change, cache change, MTP, quant recipe) promotes ONLY
when all four gates pass on recorded evidence, A re-run in the same session/machine state:

- **G1 Parity (outranks all) — four tiers by change class:**
  - *Same-shape execution-model / cache changes* (no arithmetic reorder, no pass-shape change)
    = bit-exact greedy tokens AND logit frames vs the reference path.
  - *Kernel changes* (K1/K2 flash, fused epilogues — the floating-point reduction order is
    rewritten, so bit-exact logits are not physically achievable) = greedy tokens exact AND
    logit max-abs within the two-sided fault-boundary threshold.
  - *MTP / speculative / n-gram / any change that alters the forward-pass shape* =
    **target-verified lossless**, NOT byte-identical: every emitted token must equal some real
    verify-pass argmax, and every divergence from single-token greedy must be an explained
    sub-0.5 top-2 near-tie (the `near_tie_events` counter quantifies exposure per run); an
    **unexplained** divergence fails the gate as a real bug. (A1 — NunSpark MEASURED ~1
    near-tie flip per 110 tokens from fp16 pass-shape differences in the KV cache; a
    byte-identity gate on spec is unachievable at model scale and was retired there after it
    failed. Do NOT gate or advertise "MTP-on == MTP-off byte-identical.")
  - *Lossy quant recipes* = predeclared top-1 agreement + KL budget vs the declared
    reference — the comparator bonsai specified but never built; hyperion builds it at M2
    (E1 needs it).
- **`near_tie_events` instrumentation (ship from M2):** on every verify pass, count rows whose
  top-2 logit gap is <0.5; expose in `--metrics` and the ledger. This is what makes the
  target-verified tier auditable instead of asserted — a spec divergence is only acceptable if
  it coincides with a logged near-tie; a divergence with no near-tie is a bug, full stop.
- **G2 Throughput:** decode ≥ 1.10× baseline OR predeclared bpw/memory target; prefill
  ≥ 0.97×; TTFT ≤ 1.03× (no silent regressions elsewhere in the pipe).
- **G3 Quality floor:** agent-eval suite ≥ baseline floor (09); gemma-challenge eval-prompts
  quality hold (the FP8-logit-saturation lesson: a faster kernel that skews logits must be
  caught by CI, not users).
- **G4 Memory/governor:** peak ≤ budget at 8K AND 32K sentinels; zero uncontrolled OOM;
  governor prediction error within calibrated band.

Statistics: 512-token prompt / 1024-token decode standard case + real-workload corpus; 1
discarded warmup + 5 trials; first token excluded both sides; nearest-rank percentiles;
**A-C-C-A ordering; noise bar = candidate MIN > baseline MAX** (no overlapping-distribution
promotions); RSS sampled at 25 ms.

**Bit-exact tamper-proof audit** (bonsai `native_ab.rs`, ~90% reusable): per-arm capture of
full-vocab f32 logit frames + greedy tokens over frozen cases; compare requires byte-equal
metadata/prompts and REJECTS same-executable/same-commit arms; audit binds SHA-256 of
weights/config/tokenizer/executable/metallib; compiled-git-sha must equal HEAD, not dirty.
Re-parameterize vocab/corpus for Gemma; drop the GDN-state channel (no analog).

**Adversarial refute:** before promotion, a max-effort reviewer pass actively tries to break
the parity claim / benchmark validity / ABI safety; majority-refute kills the pass. Ledger is
append-only; floors and frozen fixtures are immutable (a failed gate is information, not an
obstacle).

## Real-workload benchmark methodology (Helios heritage)

- 8-family real-context corpus rebuilt for the actual use case: facility-diagnostic,
  recommendation, operator-assistance, and tagging tool chains; long-transcript
  multi-turn, RAG-ish document QA at 4K/8K/16K, short chat. Prompt files frozen with SHA-256
  manifests (Helios pattern; its corpus files are reusable formats, contents replaced —
  no Rust code-review prompts as "real workload" for a smart-buildings engine).
- **Protected-aggregate holdouts:** any scoped optimization (MTP lanes, cache) must report
  the aggregate across ALL lanes, not its favorite subset (the discipline that kept Helios
  honest through XR86).
- Tail latency is a first-class metric: ITL p50/p95/p99 per lane; the Helios pathology
  (p99/p50 > 6×) gets an explicit regression gate: **p99 ≤ 2× p50** on chat_short after M4
  (DECIDED target, justified by bonsai's measured ITL p99/p50 = 75.5/71.7 ≈ 1.05 on the
  non-pathological execution model — flat decode IS achievable on MLX).
- `low_n` policy: any row below trial minimums is labeled and cannot gate promotions.

## CI tiers

1. Per-commit: build, clippy, unit + ABI lifecycle tests, template golden fixtures, parser
   corpus, geometry validation (all 5 configs), no-orphan-crate check. No model needed.
2. Nightly (on the M5 box): G1 parity smoke (2 cases) + `near_tie_events` smoke on a spec
   case, 512/128 perf canary vs last accepted row (±5% alarm), governor calibration probe
   (predicted-vs-measured peak AND the throughput-vs-resident-budget cliff — the optimum is
   below the OOM ceiling, A4).
3. Per-milestone: full A/B protocol + agent-eval + sentinel matrix → ledger rows + decision
   record (`docs/decisions/NNNN-*.md`, numbered, append-only).
