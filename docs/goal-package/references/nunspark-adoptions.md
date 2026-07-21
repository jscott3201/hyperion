# NunSpark adoptions — what v0.2 changed and why (2026-07-21)

Source: exploration of `github.com/sharma-open-source/NunSpark` (v0.12, MIT, ~9k LOC
Python/MLX) — an Apple-Silicon MLX engine that runs oversized LLMs by disk-streaming weights +
deep-K speculative decoding. Sibling problem (run what doesn't fit RAM), same substrate
(MLX/M5/Gemma-3/4/spec-decode/16GB/lossless discipline). This sheet is the provenance for every
v0.2 change; the changes themselves are folded into the numbered spec files + gates.

**One caveat that colors all of it:** NunSpark has **zero measured Gemma numbers** — Gemma is a
registered-but-never-load-tested arch there, its Gemma-MTP path carries a known latent bug, its
tree path is unvalidated on hybrid attention, its EAGLE path is untrained scaffolding. Its
correctness *lessons* (A1–A5) are worth more than its code. hyperion's Gemma-4 numbers remain
first-of-kind. Labels below: **MEASURED** (NunSpark gated A/B, M4 16GB unless noted),
**COMMUNITY** (external hardware), **CLAIM** (design thesis).

---

## A. Corrections folded into the spec (implementation-critical)

### A1 — Exactness: "MTP-on == MTP-off, byte-identical" is unachievable; use target-verified. → 07, 08, 10-M2/M7, 11-R11
NunSpark `plan5-m2-mismatch-investigation` (controlled 3-arm experiment): a K-token verify pass
builds the KV cache through a different forward-pass **shape** than single-token greedy, so fp16
rounding differs at every position (max-abs 1.29 by layer 29) → attention shifts logits ~0.1–0.15
→ a **near-tie argmax flips**. MEASURED: gaps 0.08–0.15 (always <0.5), **~1 flip per 110 tokens**,
self-healing (next token reconverges), emitted token is always *some* real verify pass's argmax
(never a hallucination). Their bit-identity gate FAILED on real workloads (fixed-K, adaptive-K,
greedy all pairwise disagreed) and only ever passed on fixtures too small to hit a near-tie.
**Second-order:** auto-disable-on-divergence itself changes verify shape → induces flips → it is
self-defeating as an invariant.
**hyperion change:** three-tier G1 (08): same-shape+state = bit-identical; **cross-shape (MTP,
any γ change) = target-verified lossless** (every emitted token = some real verify-pass argmax),
NOT byte-identical, each divergence explained as a sub-0.5 top-2 near-tie via a standing
`near_tie_events` counter shipped from M2; any *unexplained* divergence = real bug (auto-disable +
fixture). Drop "auto-disable on first divergence" from 07.

### A2 — KV rollback: clone-and-commit / append-only, never trim once the ring rotated. → 05, 07, 10-M7, 11-R12
`RotatingKVCache.is_trimmable()` is False once rotated. NunSpark's own `gemma4_mtp_speculative_
generate` rolls back with `kv.truncate()` — an **acknowledged latent bug** (their backlog #4) on
sliding-window (i.e. Gemma) targets; long agentic sessions (many spec rounds) are exactly what
rotates the window mid-session. They got it *right* in `PrefixCache`: probe the first
non-rotating layer for true length; **drop-if-rotated** rather than trim; `commit()` reconciles
against the actual cache offset instead of trusting its own token count.
**hyperion change:** MTP rollback (07/M7) is append-only — speculative writes land in the local
ring's γ_max slack and are **discarded by not committing**, never by trimming committed state; the
KV-safety module is modeled on PrefixCache's trim-or-drop + reconcile discipline. Validate
rollback specifically *after* ring rotation.

### A3 — Heterogeneous masks: build one mask per layer *kind*, from that kind's own cache. → 03, 05
NunSpark `MaskPlan.PerLayer` encodes a real crash they hit twice (incl. a community repro):
building a global-attention mask from a sliding layer's cache truncates it (the rotating cache
clamps offset to `window-1`), short by `offset-127` columns → `broadcast_shapes` crash once the
sequence exceeds the window.
**hyperion change:** the Geometry-driven graph builds masks **by layer kind, sourced from the
first layer of that kind** — never layer 0 unconditionally. Directly relevant to the 40-local/
8-global layout.

### A4 — Governor objective: target the throughput-optimal working set, not the OOM ceiling. → 05, 10-M1, 11-R13
MEASURED (16GB M4, Qwen3-30B budget sweep): **4/5/6/8/10 GB → 1.79 / 2.52 / 3.23 / 2.84 / 1.33
tok/s** — throughput is **non-monotonic**; past ~6GB the resident set fights the macOS memory
compressor (per-miss service latency 3.0ms → 18.7ms even as miss *count* drops). A live-
availability-based budget clamp was **tried and reverted** (macOS free-% too volatile mid-session;
starved a fine machine to the floor). Auto-budget = deterministic `0.75×(RAM−8GB)`.
**hyperion change:** the governor's objective is the *measured* throughput optimum (which is below
the OOM-safe ceiling), NOT "as much resident as safely fits"; add a budget/working-set sweep to M1
to locate the M5-16GB cliff; never size off live OS availability. Bites even the 12B-resident case
as weights+KV+workspace climb toward 12GB.

### A5 — Speculative decoding *hurts* MoE targets; spec is the dense lever. → 07, 10-M9, 11-R14
MEASURED across four MoE scales (30B/120B/235B/480B): spec decode ran **slower than greedy** —
each verify-pass position fires its own expert union, so union-I/O grows ~linearly with K while
acceptance doesn't. Rule: **spec = dense lever; expert-caching = sparse lever.**
**hyperion change:** Gemma-4-12B is dense → MTP correctly placed there (M7). If the 26B-A4B MoE
tier (M9) is ever built, it runs **greedy + expert-caching, MTP OFF** by default.

---

## B. Mechanisms folded in

### B1 — N-gram / prompt-lookup adaptive drafter (a second, cheap speculative lane). → 06, 07, 10-M5
Model-free spec: draft from the most recent prior occurrence of the current suffix. **Zero**
resident weight/KV cost, tokenizer-exact, lossless-by-construction. Adaptive policy: multiplicative
grow/shrink, **disable to exactly zero** when acceptance <25% (returns `[]` without even scanning),
cheap periodic re-probe (~every 50 steps). Fits agentic tool-calling: repeated JSON keys /
`{"name":…,"arguments":{…}}` scaffolding recur verbatim → likely *higher* acceptance than prose,
no shared-tokenizer constraint. **Complementary** to MTP (n-gram for structured repetition, MTP
for general). Re-tune the disable threshold for compute-bound (a wasted verify costs compute, not
a disk sweep → closer to always-on); the algorithm structure ports unchanged.

### B3 — Gemma-4 MTP drafter details (confirmed via config.json + NunSpark impl). → 04, 07, 10-M7
The two assistant `config.json`s + NunSpark `gemma4_assistant.py` confirm and enrich M7:
- **4-layer Q-only** drafter, no K/V projections — cross-attends the **target's captured K/V**
  (last non-KV-shared layer per attention type). 12B assistant: hidden 1024, 16Q/8KV, head_dim
  256. E4B assistant: hidden 256, 4Q/2KV, `backbone_hidden_size: 2560` = target hidden (**confirms
  it consumes target hidden states**).
- **Feed the drafter the previous token via the TARGET's embedding table**, not the drafter's
  ("omitting this wrecks acceptance"). `position_ids` **constant** across draft steps (attends one
  static target-KV snapshot).
- **Centroid logit head** (config: `num_centroids: 2048`, `centroid_intermediate_top_k: 32`,
  `use_ordered_embeddings: true`): score 2048 clusters → top-32 → gather+matmul only those rows,
  avoiding a full 262144-vocab matmul every draft step. Adopt for the drafter's per-step head.
- **dtype-match at the drafter boundary** (drafter bf16 vs target fp32 hidden → MLX promotes to
  fp32 → recasts the full embedding table every step → ~1000× slowdown).
- **Alias the vocab/embedding tensor** between target and drafter (hyperion controls both ends;
  NunSpark loads two 262144-vocab tables redundantly — a real 16GB cost to avoid).

### B4 — Single-slot prefix-cache rotation safety. → 05, 10-M6
NunSpark's prefix cache (trim to LCP, prefill only the suffix) is hyperion's M6 idea; adopt its
**rotation-safety discipline**: read true length from the first non-rotating layer's offset;
**drop-if-rotated** (don't trim past window); reconcile against actual cache offset rather than
trusting bookkeeping. Fold into M6's restore path.

### B5 — Tree spec is not "structurally impossible"; correct the stated reason. → 07
NunSpark makes tree verify work on MLX by **flattening the tree into the batch dimension**
(equal-length root-to-leaf paths from one shared prefix, tile the prefix KV across B rows, one
stock causal mask + single scalar offset — no custom tree mask/RoPE). So hyperion's "MLX cache
offset makes tree verification structurally hostile" reasoning is **wrong**. BUT the v1 conclusion
(skip tree) still holds for the *right* reasons: (i) compute scales with `num_paths × path_len`
(redundant ancestor recompute) which is free in NunSpark's disk-bound regime but **not** in
hyperion's compute-bound one; (ii) their tree path is a POC, unvalidated on hybrid sliding/global
attention (Gemma's shape), with the same unguarded Rotating-cache gap. **hyperion change:** keep
tree OUT of v1, fix the reason in 07.

### B2 — ArchSpec / MaskPlan / LayerRunner seam validates hyperion's Geometry dispatch. → 02
NunSpark's per-layer-kind strategy pattern (mask-plan / cache-kind / layer-runner resolved once
per kind, "only stash cross-layer KV when something downstream consumes it") is exactly hyperion's
`Geometry`-driven per-layer-kind dispatch. External corroboration; adopt the "stash KV only when
consumed" memory discipline explicitly.

---

## D. Anti-lessons (NunSpark measured dead — do not build)

- **Decode-time I/O / page-cache warming** for single-token decode: net-neutral-to-negative
  (a decode miss is ~0.36 pieces/layer — nothing to parallelize). Reinforces hyperion R3.
- **Availability-based budget clamp:** reverted (A4).
- **EAGLE drafter as a reference:** technique legit, NunSpark's impl is untrained/untested — not a
  port target.
- **Dead parallel adaptive-K code** (two disagreeing impls, no tests) — study the n-gram policy
  (B1), not those.

---

## C. Capability option — Gemma-4-26B-A4B on 16GB via streaming (M9, post-v1, owner-gated)

NunSpark runs Qwen3-30B-A3B — the **same 128-expert/top-8 shape class as Gemma-4-26B-A4B** — at
**2.8–3.2 tok/s greedy on a 16GB M4** (MEASURED), reading ~100MB/token vs ~1GB, by disk-streaming
only the fired experts. The measured recipe (the parts to carry):
1. **Persistent scatter buffers** (their biggest fix): never rebuild a zero-filled 128-expert
   buffer per layer (MEASURED 59–70% of every token, ~15GB transient writes/token); keep
   shape-stable buffers, scatter only fired rows, never re-zero (safe: the expert-gather reads only
   router-selected rows). **1.6 → 3.2 tok/s.** Worth adopting even resident (allocator/bandwidth).
2. **Two-region cache**: LRU for sparse experts, MRU for the cyclically-scanned core — expert hit
   rate 57–64% → **89–92%** at the same budget.
3. **Temporal prefetch** at verify-pass granularity (consecutive passes fire 74–80% overlapping
   experts) into an eviction-safe staging buffer.
Rules: MTP OFF for MoE (A5); nothing here is Gemma-MoE-specific yet (NunSpark never wired
`selective_moe` for Gemma's `enable_moe_block`) — new territory against 26B-A4B's actual expert
layout, using their qwen3_moe path as a structural template only.
**Status:** captured as milestone **M9 (post-v1, not on the v1 critical path)** — promote into v1
only on explicit owner go. O-8 upgraded from "needs 32GB+" to "measured-feasible on 16GB via
streaming, deferred."

---

## E. Methodology corroboration
NunSpark independently arrived at: Sonnet-mechanical / Opus-design subagents with gate criteria
written before spawning; a `backlog.md` ledger of refuted ideas *with the killing measurement*;
bit-identity-vs-mlx-lm as the correctness backbone; flushed interleaved A/B; a single MLX-owning
thread with raw `os.read` on worker pools and tensor materialization on one thread (validates
hyperion's engine-thread + explicit-streams design); "trust live numbers over adversarial flushed
probes." Strong external validation of hyperion's A/B protocol and AGENTS.md.
