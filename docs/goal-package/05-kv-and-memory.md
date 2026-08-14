# 05 — KV, memory budget, governor, conversation cache

State layout and memory prediction are architecture/deployment-specific under ADR 0006. The first
sections retain the implemented Gemma baseline; Qwen has a distinct recurrent+KV topology.

## Gemma heterogeneous KV (implemented baseline)

Two storage classes, allocated from `Geometry`:

1. **Local layers (12B: 40 × [8 heads × 256]) — true O(1) ring.** Capacity = window 1024 **+
   γ_max (8) speculative slack** so rejected MTP draft writes never overwrite still-in-window
   KV; attention reads exclude the speculative region. `slice_update` writes; ring index
   arithmetic in the step function. Exact allocated ceiling (12B, bf16, including slack):
   40 × 8 × 256 × 2 (K+V) × 2 B × 1032 = 338,165,760 B ≈ **0.338 GB** — flat forever,
   regardless of context.
2. **Global layers (12B: 8 × [1 head × 512], K=V) — capacity-stepped single tensor.**
   One tensor per layer (K IS V), preallocated in 256-token capacity steps, `slice_update`
   append, bucket-growth at step boundaries only. Cost: 8 × 1 × 512 × 1 tensor × 2 B =
   **8 KiB/token** → 0.5 GB @ 64K, 1.05 GB @ 128K, 2.1 GB @ 256K.

Total 12B KV bf16: **~0.6 GB @ 32K, ~0.84 GB @ 64K, ~1.4 GB @ 128K, ~2.4 GB @ 256K.** The architecture is
the compression: 40/48 layers never scale with context and the 8 that do carry one 512-dim
K=V head instead of 8×256×2. (Uniform-KV framing would be ~10× worse; this is why the
legacy Gemma KV-quant lane was parked.) E4B variant: shared-KV layers allocate nothing; window
512; K≠V global — all from `Geometry`.

## Gemma 16 GB budget (device-derived, enforced)

| Line item (12B QAT-Q4, bf16 KV) | Estimate |
|---|---|
| Weights (affine g64 ≈4.5 bpw) | ~6.7 GB (g32 grid-exact variant ~7.5 GB; the E1 comparison is legacy evidence) |
| KV @ 32K (0.338 local ring + 0.27 global) | ~0.61 GB |
| Workspace/transients (chunked prefill, windowed-flash kernels) | ~1.0–1.5 GB (governor-predicted, measured) |
| **Total @ 32K** | **~8.3–8.8 GB vs ~12.06 GB ceiling** ✅ |
| KV @ 128K adds | +0.8 GB → ~9.1–9.6 GB — feasible with K2 windowed prefill + governor chunk-shrink; treat as stretch sentinel, low_n allowed |

## Qwen hybrid state and 16K envelope

Qwen carries two different state classes:

- 48 Gated DeltaNet recurrent matrices in FP32:
  `48 × 48 × 128 × 128 × 4 = 150,994,944 B` (144 MiB), plus about 2.8 MiB of BF16 convolution
  state. Keep it FP32 and transactional initially; old and staged copies may coexist at a step
  boundary and must be counted in peak admission.
- KV only for 16 full-attention layers:
  `2 × 16 × 4 KV heads × 256 × 2 B = 65,536 B/token`. BF16 KV is 0.5 GiB at 8K and 1 GiB at
  16K. It is the correctness control; compression is not presumed mandatory.

The current mixed-Q2 planning range (roughly 7.6–8.3 GiB weights) plus recurrent state and BF16
KV leaves a narrow static margin before scratch, compiled graphs, allocator cache, staging,
server, and reserve. The 16K profile promotes only on measured live peak. Native 262K is not a
credible promise on 16 GB.

Ceiling: measured `recommendedMaxWorkingSetSize` = 12,713,115,648 B on m5-16g → effective
budget `min(12 GiB, 94.9%)` ≈ **12.06 GB**, soft watermark 90% ≈ 10.86 GB (numbers re-read
from the device at startup, never hardcoded — the constants here are the m5-16g calibration
sample). `iogpu.wired_limit_mb` stays stock in v1; document but don't require sysctl tuning.

Legacy Gemma sentinels remain 1K / 4K / 8K / 16K / **32K**, with 128K as a low-N memory/TTFT
probe. Qwen V1 gates 8K and the complete 16K deployment envelope; 32K is deferred. Helios's
canonical Gemma failure—16K peak 21.874 GB—remains regression evidence, not a Qwen target.

## Governor (family/deployment-derived constants — R7)

Pre-admission, per prefill chunk and per decode block:

```
predicted_peak = settled_working_set            # max(MLX active+cache, phys_footprint, counted weights)
              + state_append_bytes(chunk)       # from the active family/deployment state plan
              + operation_transient(chunk)      # DERIVED — PER-OP PEAK, not Σ over layers
              + workspace_floor + reserve(512 MiB)
```

- Constants are derived independently from the active architecture, deployment, and operation,
  then calibrated against measured misses. Gemma equations cannot be reused for Qwen, nor can a
  prior Qwen fudge factor be reused for this checkpoint. Keep each calibration identity in the
  ledger and re-derive it after a kernel/cache change.
- For the current Gemma attention plan, the unfused transient control is the per-layer maximum of
  `q_chunk × min(ctx, 1024) × heads_local` and `q_chunk × ctx × heads_global`, multiplied by dtype
  and safety. This is a Gemma example, not the shared equation. Qwen's plan must additionally
  model GDN/convolution temporaries, staged recurrent state, and its full-GQA operations.
- Admission: if predicted_peak > budget → halve chunk, down to 1 token; if still over →
  typed `OOM_GOVERNOR` **before submission**, fail-closed → HTTP 529 (pre-stream) or SSE
  `error: overloaded` (mid-stream). Zero uncontrolled OOM aborts is a standing G4 gate.
- States READY / SOFT_PAUSED / HARD_REJECT surfaced in `/control/stats` + step telemetry.
- Gemma decode admission checks its 256-token global-KV bucket growth
  (`8 KiB/token × 256 = 2 MiB`) at bucket transitions. Qwen's deployment plan declares its own
  KV growth (16 MiB per 256 BF16 tokens at the initial geometry), recurrent staging, compilation,
  and any other allocation events; the shared governor must not assume Gemma's sole event.
- **Objective is the throughput-optimum working set, NOT "as much resident as safely fits"
  (A4 — NunSpark MEASURED the macOS memory cliff).** Resident-set-vs-throughput is
  non-monotonic: a 16 GB M4 sweep gave 4/5/6/8/10 GB → 1.79 / 2.52 / **3.23** / 2.84 / 1.33
  tok/s — past ~6 GB the resident set fights the macOS compressor (per-miss service latency
  3.0 → 18.7 ms even as miss *count* drops). Two rules: (i) the OOM-safe 12.06 GB ceiling is an
  upper bound, not a target — the preregistered family A-arm runs a budget/working-set sweep to
  locate the M5-16GB throughput optimum and the governor holds the working set there; (ii) **never size off live
  OS availability** (a macOS free-% clamp was tried and reverted upstream — too volatile
  mid-session, starved a fine machine to the floor; size off *total* RAM deterministically with
  an explicit override). Bites even the 12B-resident path: as weights + KV + workspace climb
  toward 12 GB, the compressor cliff can arrive before OOM does.

## Conversation cache (session-scoped in-memory exact-prefix reuse) [P6 candidate]

The agentic workload is append-only transcripts: same system prompt + growing tool-call
history. Measured stakes (mlx-bonsai T0 probe, m5-16g): full-history re-prefill was
**79.077% avoidable tokens** across a real 87-turn eval; live exact-prefix reuse cut
post-first-turn TTFT **84.17 s → 15.52 s (−81.6%)** — and was still REJECTED because one
repaired transcript changed (G1). Both engines proved near-miss reuse silently corrupts;
bonsai also proved **only the 2048-token chunk boundary was bitwise-exact**. Design encodes
those verdicts:

1. **Session-scoped, not global:** cache keyed by conversation (API `session_id` /
   fingerprint of the message-prefix token chain). One live state-snapshot chain per session;
   LRU across sessions under a byte budget (default ~2 GB, config).
2. **Snapshot only at validated family-specific boundaries.** The current Gemma boundary is
   2048. Its snapshot = global-KV slice
   reference up to boundary + local ring image (bounded: ≤0.34 GB worst-case, typically the
   ring is the cheap part) + token-chain hash + engine identity tuple (artifact, conversation,
   deployment, state layout, runtime/kernel). A future Qwen snapshot also contains the exact
   GDN recurrent and convolution state; it cannot trim or synthesize that state from KV alone.
3. **Restore = bitwise-identical state contract.** G1 gate: restored-then-continued
   generation must produce byte-identical tokens AND logit frames vs fresh full prefill at
   the same boundary, on frozen fixtures, k=2 repeats. If a boundary can't prove bitwise
   equality it is not a snapshot point. No "close enough" tier exists (T0's rejection is law).
4. **Turn flow:** new request → longest snapshot whose token chain is an exact prefix →
   restore → prefill only the suffix. Miss → full prefill (and lay down snapshots as you go).
5. Explicitly OUT (v1): cross-session dedup, durable/SSD-backed persistence (P8), edited-prefix
   partial invalidation (exact-prefix only), and mid-decode restore (Helios MidDecode rejection
   carried).
6. **Rotation safety (A2/B4 — NunSpark `PrefixCache` discipline).** Read true prefix length
   from the **first non-rotating (global) layer's** cache offset, never a local layer's (a
   rotated ring clamps its offset to the window and under-reports length); if a needed prefix
   falls inside a rotated local window, **drop the snapshot rather than trim it**
   (`RotatingKVCache` is not trimmable past rotation); `commit` reconciles against the actual
   cache offset instead of trusting bookkeeping. This is the *same* trim-or-drop module the MTP
   rollback uses (07) — build it once, model it on `PrefixCache`, never on `kv.truncate`.

Expected shape of the win (to be measured at P6): multi-turn TTFT collapses from
O(transcript) to O(new tokens); the T0 numbers are the anchor that makes this a top-3
feature for the agentic use case, not a nice-to-have.

## Prefill chunking

The current Gemma default is 2048 (matching its validated snapshot boundary; bonsai T0 showed
2048-chunk bitwise stability and mlx-lm uses 2048). A Qwen deployment must select and validate its
own prefill chunk; no Gemma boundary is inherited. The governor may shrink per admission. Chunk
boundaries are also cancellation points and, post-V1, possible decode-interleave points.
