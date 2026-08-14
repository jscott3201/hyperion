# 08 — Correctness & verification (the carried crown jewels)

Under ADR 0006, every family/checkpoint has a separately pinned oracle, conversation profile,
state trace, vocabulary, corpus, and baseline. Shared protocol does not mean shared tolerances.

Two systems carried near-verbatim: mlx-bonsai's parity oracle + kernel A/B protocol, and
Helios's real-workload benchmark methodology. They are complementary; both are law here.

## Parity oracle (G1 substrate)

- **Reference:** exact pinned runtime revisions per artifact. Gemma retains its reviewed mlx-lm
  reference. Qwen P2 pins independent Transformers and mlx-lm implementations plus execution
  mode and requires differential agreement. Persistent reference processes are bench scope only,
  never on the serving path.
- **Reference memory discipline:** for Gemma and any reference artifact that fits locally, the
  oracle runs first, writes reference tokens/logit/state frames, and is reaped before native MLX
  initializes (bonsai ADR 0004). The 55.56 GB Qwen BF16 teacher cannot run on the 16 GB target;
  P2 therefore produces an immutable trace bundle on a pinned remote/high-memory host, binding
  source, oracle/runtime, execution mode, hardware, commands, corpus, and payload hashes. Later
  transformed-artifact comparisons may run sequentially on the target. Oracle cache internals
  remain opaque in either case.
- **Compared:** input IDs, greedy token IDs, profile-sized full-vocab logits, selected layer/state
  slices, and free-running output at each artifact's declared context sentinels, plus exact
  template/stop/tool/thinking fixtures. Qwen includes recurrent/convolution state traces.
- **Two-sided fault-boundary thresholds (no magic tolerances):** measure clean max-abs;
  inject a real single-layer fault (e.g., RoPE offset +1 on one layer); the gate sits
  between measured-clean and measured-fault. Re-derive when MLX/quant/kernels change.
  A predecessor observed separable clean/fault bands on a different Qwen artifact; exact values
  are not portable and must be re-derived rather than copied.
- **Quant honesty:** oracle arm and native arm run the SAME quantization (convert once,
  load both sides) so parity isolates the engine, not the quant. Teacher/source-versus-quantized
  quality is a separate MEASURED row (the Gemma E1 row is historical), never conflated with
  engine parity.

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
    reference. The legacy Gemma M2/E1 work owns its comparator; Qwen P4 must bind its own
    teacher, corpus, and thresholds.
- **`near_tie_events` instrumentation:** on every speculative verify pass, count rows whose
  top-2 logit gap is <0.5; expose in `--metrics` and the ledger. This is what makes the
  target-verified tier auditable instead of asserted — a spec divergence is only acceptable if
  it coincides with a logged near-tie; a divergence with no near-tie is a bug, full stop.
- **G2 Throughput:** decode ≥ 1.10× baseline OR predeclared bpw/memory target; prefill
  ≥ 0.97×; TTFT ≤ 1.03× (no silent regressions elsewhere in the pipe).
- **G3 Quality floor:** each artifact/profile passes its separately frozen agent-eval floor and
  declared capability canaries. `gemma-challenge/eval-prompts` remains the Gemma canary; it does
  not substitute for Qwen P4's reasoning/code/instruction/structured-output/tool/long-context
  suite. A faster kernel that skews logits must be caught before promotion.
- **G4 Memory/governor:** each deployment passes its declared sentinel matrix with zero
  uncontrolled OOM and governor prediction error within its calibrated band. Legacy Gemma gates
  retain their matrix, including 8K/32K; Qwen V1 gates 8K plus the complete 16K envelope. Qwen
  32K is non-gating/post-V1.

Statistics: 512-token prompt / 1024-token decode standard case + real-workload corpus; 1
discarded warmup + 5 trials; first token excluded both sides; nearest-rank percentiles;
**A-C-C-A ordering; noise bar = candidate MIN > baseline MAX** (no overlapping-distribution
promotions); RSS sampled at 25 ms.

**Bit-exact tamper-proof audit** (bonsai `native_ab.rs`, ~90% reusable): per-arm capture of
full-vocab f32 logit frames + greedy tokens over frozen cases; compare requires byte-equal
metadata/prompts and REJECTS same-executable/same-commit arms. The audit binds the artifact root,
conversation profile, deployment profile, state-layout, weights/config/tokenizer/template,
runtime/executable/metallib, oracle, harness, corpus, and machine identities; compiled-git-sha
must equal HEAD, not dirty. Parameterize vocab/corpus/state channels by architecture. Gemma has no
GDN channel; Qwen requires GDN recurrent/convolution traces and commit/abort/replay checks.

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
  (p99/p50 > 6×) retains an explicit regression gate: **p99 ≤ 2× p50** on `chat_short` after
  the relevant fast-path/kernel promotion
  (DECIDED target, justified by bonsai's measured ITL p99/p50 = 75.5/71.7 ≈ 1.05 on the
  non-pathological execution model — flat decode IS achievable on MLX).
- `low_n` policy: any row below trial minimums is labeled and cannot gate promotions.

## CI tiers

1. Per-commit: build, clippy, unit + ABI lifecycle tests, family classifier, strict config tests,
   checkpoint-profile golden fixtures, parser corpus, all-five Gemma geometry validation and,
   once implemented, pinned Qwen geometry validation; no-orphan-crate check. No model needed.
2. Protected release/measurement (on the M5 box): G1 parity smoke (2 cases), plus a
   `near_tie_events` spec smoke only while a speculative lane is open; 512/128 perf canary vs
   last accepted row (±5% alarm), governor calibration probe
   (predicted-vs-measured peak AND the throughput-vs-resident-budget cliff — the optimum is
   below the OOM ceiling, A4).
3. Per-phase: the applicable full A/B protocol + artifact-specific agent eval and sentinel matrix
   → ledger rows + decision
   record (`docs/decisions/NNNN-*.md`, numbered, append-only).
