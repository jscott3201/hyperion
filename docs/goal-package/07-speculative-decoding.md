# 07 — Speculative decoding (MTP lane) [M7 — sequenced late ON PURPOSE]

## Why decode needs it (physics)

M5 decode is memory-bandwidth-bound: 153 GB/s ÷ ~7 GB Q4 working set ⇒ ~low-20s tok/s
dense ceiling regardless of kernel quality (Apple's own M5 data: decode only +19–27% vs M4
while prefill jumped ~4×). Speculative decoding is the ONLY lever that beats the ceiling —
it converts k sequential bandwidth-bound steps into one wider, more compute-bound verify
step that can use idle NAX capacity. Everything else in this package optimizes toward the
ceiling; this lane goes through it.

## Why it is M7 and not M2 (evidence)

Helios ran an ~XR14→XR86 marathon on MTP and never cleared its own +25% protected-aggregate
default-on gate: final XR86 = **+19.969%**, weighted acceptance **0.706**, verifier forward =
**85.76%** of selected-phase cost; early real acceptance was 0.000–0.438 until the harness
matured. The lesson is NOT "MTP doesn't work" — acceptance 0.706 with a 400M drafter is
healthy, and Google reports up to 3× (H100-class) and ~2.2× on Apple silicon (26B, batch
4–8); llama.cpp measured 1.9–3.1× (DGX Spark, acceptance 0.588). The lesson is that
**verify-step cost decides everything**, and Helios's verifier ran on the slow re-trace
execution model with a serial rollback default. hyperion's sequencing: make the dense path
fast first (M2–M4: compiled step, K1/K2 kernels), then attach MTP where the verify forward
is cheap. Same gate (+25%), different substrate.

## Design (official drafter, one-pass verify)

- **Drafter:** `gemma-4-12B-it-…-assistant` (~400M, Apache-2.0). Architecture CONFIRMED from
  `config.json` + the NunSpark `gemma4_assistant.py` reference (see references/nunspark-
  adoptions.md B3): **4-layer, Q-only** (`num_hidden_layers: 4`, hidden 1024, 16 Q / 8 KV,
  head_dim 256) — **no k/v projections of its own**, it **cross-attends into the target's
  captured K/V** (the last non-KV-shared layer per attention type; E4B assistant's config
  literally carries `backbone_hidden_size: 2560` = target hidden, confirming it consumes
  target hidden states). Implementation details to honor:
  - Feed the drafter each previous drafted token via the **target's** embedding table, not
    the drafter's ("omitting this wrecks acceptance"); `position_ids` stays **constant**
    across the γ draft steps (it attends one static target-KV snapshot).
  - **Centroid logit head** (`num_centroids: 2048`, `centroid_intermediate_top_k: 32`,
    `use_ordered_embeddings: true`): score 2048 clusters → top-32 → gather+matmul only those
    rows, avoiding a full 262144-vocab matmul on every draft step. This is the drafter's
    per-step cost centre — build it, don't skip it.
  - **Match dtypes at the drafter boundary** (drafter bf16 vs target fp32 hidden promotes the
    whole matmul to fp32 and recasts the 262144-vocab table every step → ~1000× slowdown).
  - **Alias the vocab/embedding tensor** between target and drafter — hyperion owns both ends
    (NunSpark loads two 262144 tables redundantly; a real 16 GB cost to avoid).
- **Draft:** greedy, depth γ ∈ [2, 8] (autotuned per acceptance EWMA; Helios XR14 heritage).
  Single-position pinned drafting (linear chain). **Tree drafting is OUT of v1** — not because
  it's impossible on MLX (NunSpark shows it works by flattening the tree into the batch
  dimension: equal-length root-to-leaf paths from one shared prefix, tile the prefix KV across
  B rows, one stock causal mask + scalar offset — no custom tree mask/RoPE), but because its
  compute scales `num_paths × path_len` (redundant ancestor recompute), which is free in a
  disk-bound engine but **not** in hyperion's compute-bound regime, and NunSpark's own tree
  path is an unvalidated POC on hybrid sliding/global attention with an unguarded Rotating-
  cache gap. Revisit post-v1 only with a compute-bound payoff hypothesis (references/nunspark-
  adoptions.md B5).
- **Verify:** ONE batched forward over the γ candidate tokens (shape-bucketed like decode;
  qmv_wide is MLX 0.32.0's kernel for exactly this small-batch quantized matvec). Greedy
  acceptance: longest matching prefix + 1 corrected token (target distribution preserved
  under greedy by construction). Sampled-mode speculative acceptance is post-v1.
- **Rollback: append-only, never trim committed state (A2 — a real bug in NunSpark's own
  Gemma-MTP path).** `RotatingKVCache.is_trimmable()` is False once the sliding-window ring
  has rotated, and a long agentic session (many spec rounds) is exactly what rotates it.
  Speculative writes land in the local ring's **γ_max slack** and are **discarded by NOT
  committing them** — the write cursor only advances by the accepted count; rejected positions
  are never `trim`-ed out of committed state. The KV-safety module is modeled on the
  trim-or-drop + reconcile-against-actual-offset discipline (05 conversation cache / NunSpark
  `PrefixCache`), never on `kv.truncate`. Native `snapshot/rollback/commit` handles (bonsai
  transactional-cache heritage) back this; **validate rollback specifically after ring
  rotation**, not just before.
- **Exactness — target-verified, three tiers (A1 — the "byte-identical MTP" invariant is
  unachievable at fp16/bf16 scale).** NunSpark MEASURED that a multi-token verify pass builds
  KV through a different pass *shape* than single-token greedy, flipping a **near-tie argmax
  ~1 token in 110** (logit gaps always <0.5, self-healing, and the emitted token is always
  *some* real verify pass's own argmax — never a hallucination). So MTP-on is **NOT**
  byte-identical to MTP-off; and "auto-disable on first divergence" is self-defeating (changing
  γ itself changes the verify shape and induces flips). The invariant hyperion gates instead:
  (1) same-shape+same-state = bit-identical (cheap, real); (2) **cross-shape MTP =
  target-verified lossless** — every emitted token equals some real verify-pass argmax, and
  each divergence from single-token greedy must be an explained sub-0.5 top-2 near-tie via a
  standing **`near_tie_events` counter** (shipped from M2); (3) any divergence *not* explained
  by a near-tie gap = a real bug → auto-disable + record fixture. Parity is a per-session
  runtime check plus the counter, not a byte-identity assertion.

## Second lane — n-gram / prompt-lookup adaptive drafter [M5, complementary to MTP] (B1)

A cheap second speculative lane that pairs well with the agentic tool-calling target, adopted
from NunSpark's best-designed drafter. Model-free: draft the next tokens from the most recent
prior occurrence of the current context suffix. **Zero resident weight/KV cost**, tokenizer-
exact, lossless-by-construction (only ever accepts the target's own verify-pass argmax — same
target-verified tier as MTP). Why it fits: agentic output repeats verbatim structure — JSON
keys, `{"name": …, "arguments": {…}}` scaffolding, argument names recurring across calls in one
trace — which is precisely the "recent suffix recurs" pattern, plausibly *higher* acceptance
than prose and with no shared-tokenizer constraint. It is **complementary** to MTP (n-gram wins
on structured repetition, MTP on general generation); a request/lane may use either.

Adaptive self-disabling policy (port the structure, re-tune the threshold): multiplicative
grow/shrink of the proposal cap, **disable to exactly zero** when acceptance <25% (returns an
empty proposal without even scanning — genuinely zero marginal cost, not "small"), cheap
periodic re-probe (~every 50 steps) that re-enables if it starts paying. Re-tune the disable
threshold for hyperion's **compute-bound** cost model (a wasted verify pass costs extra compute,
not a disk-read sweep — so the break-even sits closer to always-on than NunSpark's disk-bound
tuning); derive it from the verify-pass-FLOPs-vs-tokens-saved ratio at M5, don't copy the
numbers. Same three-tier exactness (target-verified + `near_tie_events`) applies.

## MoE targets: speculative decoding is OFF by default (A5)

If the 26B-A4B MoE tier is ever built (M9), **do not** run MTP or n-gram spec against it:
NunSpark MEASURED spec *slower than greedy* across four MoE scales (30B/120B/235B/480B) because
each verify-pass position fires its own expert union, so union-I/O grows with γ while acceptance
doesn't. Spec is the **dense** lever (Gemma-4-12B, this milestone); expert-caching is the sparse
lever (M9). This note exists so M9 doesn't rediscover it.

## Gate (unchanged from Helios — the number that was never beaten)

Promote to default-on ONLY at **≥ +25% protected aggregate** across the real-workload
corpus (chat/tool/qa/long-context lanes, first token excluded, A-C-C-A, candidate-min >
baseline-max), with G1 exactness green and G4 memory within budget (drafter adds ~0.25 GB
Q4 + activations). Below gate → ships as scoped opt-in (`"speculative": true` per request /
config allowlist per workload lane, exactly how Helios ended), and that is an ACCEPTABLE
M7 completion. The ledger records either outcome.

## Expected economics on M5 (to be MEASURED, orientation only)

Acceptance 0.6–0.75 (Helios 0.706 measured; llama.cpp 0.588) × γ=4 ⇒ ~2.3–3.2 tokens
committed per verify forward. If the compiled verify forward costs ≤1.6× a single decode
step (plausible: same weights-read amortized over γ tokens is precisely the
bandwidth-sharing win), net ≥ +40–80% is in range — comfortably over the +25% gate. If the
verify forward stays ≥2.5× a decode step, it will land sub-gate again like Helios. The M7
kill-switch criterion is explicit up front: two full A/B rounds sub-gate → park, write the
decision record, move on (R2 discipline — no second marathon).

## Batch-4-8 note (Google's Apple-silicon claim context)

Google's ~2.2× figure was batch 4–8 on the 26B; hyperion v1 is batch-1 single-flight. Do
not import their number as an expectation; the M1 baseline + M7 A/B rows are the only
numbers that count (baseline-then-gate law).
