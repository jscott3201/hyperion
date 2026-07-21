# 05 — KV, memory budget, governor, conversation cache

## Heterogeneous KV (day-one design, not retrofit)

Two storage classes, allocated from `Geometry`:

1. **Local layers (12B: 40 × [8 heads × 256]) — true O(1) ring.** Capacity = window 1024 **+
   γ_max (8) speculative slack** so rejected MTP draft writes never overwrite still-in-window
   KV; attention reads exclude the speculative region. `slice_update` writes; ring index
   arithmetic in the step function. Cost ceiling (12B, bf16):
   40 × 8 × 256 × 2 (K+V) × 2 B × 1024 = 335,544,320 B ≈ **0.335 GB** — flat forever,
   regardless of context.
2. **Global layers (12B: 8 × [1 head × 512], K=V) — capacity-stepped single tensor.**
   One tensor per layer (K IS V), preallocated in 256-token capacity steps, `slice_update`
   append, bucket-growth at step boundaries only. Cost: 8 × 1 × 512 × 1 tensor × 2 B =
   **8 KiB/token** → 0.5 GB @ 64K, 1.05 GB @ 128K, 2.1 GB @ 256K.

Total 12B KV bf16: **~0.6 GB @ 32K, ~0.84 GB @ 64K, ~1.4 GB @ 128K, ~2.4 GB @ 256K.** The architecture is
the compression: 40/48 layers never scale with context and the 8 that do carry one 512-dim
K=V head instead of 8×256×2. (Uniform-KV framing would be ~10× worse; this is why KV-quant
is parked, O-6.) E4B variant: shared-KV layers allocate nothing; window 512; K≠V global —
all from `Geometry`.

## The 16 GB budget (device-derived, enforced)

| Line item (12B QAT-Q4, bf16 KV) | Estimate |
|---|---|
| Weights (affine g64 ≈4.5 bpw) | ~6.7 GB (g32 grid-exact variant ~7.5 GB — E1 decides) |
| KV @ 32K (0.335 local ring + 0.27 global) | ~0.6 GB |
| Workspace/transients (chunked prefill, windowed-flash kernels) | ~1.0–1.5 GB (governor-predicted, measured) |
| **Total @ 32K** | **~8.3–8.8 GB vs ~12.06 GB ceiling** ✅ |
| KV @ 128K adds | +0.8 GB → ~9.1–9.6 GB — feasible with K2 windowed prefill + governor chunk-shrink; treat as stretch sentinel, low_n allowed |

Ceiling: measured `recommendedMaxWorkingSetSize` = 12,713,115,648 B on m5-16g → effective
budget `min(12 GiB, 94.9%)` ≈ **12.06 GB**, soft watermark 90% ≈ 10.86 GB (numbers re-read
from the device at startup, never hardcoded — the constants here are the m5-16g calibration
sample). `iogpu.wired_limit_mb` stays stock in v1; document but don't require sysctl tuning.

Sentinel contexts for all gates: 1K / 4K / 8K / 16K / **32K** (full perf rows), 128K
(memory + TTFT sentinel, low_n acceptable). Helios's canonical failure — 16K peak 21.874 GB
— must read ≤ ~10 GB here; that single number is the M2 headline gate.

## Governor (bonsai pattern, Gemma-derived constants — R7)

Pre-admission, per prefill chunk and per decode block:

```
predicted_peak = settled_working_set            # max(MLX active+cache, phys_footprint, counted weights)
              + kv_append_bytes(chunk)          # from Geometry: 8 KiB/token global + ring writes
              + attention_transient(chunk)      # DERIVED — PER-LAYER PEAK, not Σ over layers
                                                # (layers eval sequentially; one score tensor live):
                                                #   max( q_chunk×min(ctx,1024)×heads_local,
                                                #        q_chunk×ctx×heads_global ) × dtype × safety
              + workspace_floor + reserve(512 MiB)
```

- Constants derived from Gemma geometry FIRST, then calibrated against measured misses; the
  bonsai ×6 fudge and Qwen 16+48 layer split must NOT be ported (their `measurement-gates.md`
  says exactly this). With K2 kernels the local transient term drops to the banded tile —
  re-derive after M4 and keep both calibrations in the ledger.
- Admission: if predicted_peak > budget → halve chunk, down to 1 token; if still over →
  typed `OOM_GOVERNOR` **before submission**, fail-closed → HTTP 529 (pre-stream) or SSE
  `error: overloaded` (mid-stream). Zero uncontrolled OOM aborts is a standing G4 gate.
- States READY / SOFT_PAUSED / HARD_REJECT surfaced in `/control/stats` + step telemetry.
- Decode admission: bucket-boundary growth (256-token steps of 8 KiB×256 = 2 MiB global) is
  the only allocation event — check at bucket transitions, not every token.
- **Objective is the throughput-optimum working set, NOT "as much resident as safely fits"
  (A4 — NunSpark MEASURED the macOS memory cliff).** Resident-set-vs-throughput is
  non-monotonic: a 16 GB M4 sweep gave 4/5/6/8/10 GB → 1.79 / 2.52 / **3.23** / 2.84 / 1.33
  tok/s — past ~6 GB the resident set fights the macOS compressor (per-miss service latency
  3.0 → 18.7 ms even as miss *count* drops). Two rules: (i) the OOM-safe 12.06 GB ceiling is an
  upper bound, not a target — **M1 runs a budget/working-set sweep** to locate the M5-16GB
  throughput optimum and the governor holds the working set there; (ii) **never size off live
  OS availability** (a macOS free-% clamp was tried and reverted upstream — too volatile
  mid-session, starved a fine machine to the floor; size off *total* RAM deterministically with
  an explicit override). Bites even the 12B-resident path: as weights + KV + workspace climb
  toward 12 GB, the compressor cliff can arrive before OOM does.

## Conversation cache (session-scoped exact-prefix reuse) [M6]

The agentic workload is append-only transcripts: same system prompt + growing tool-call
history. Measured stakes (mlx-bonsai T0 probe, m5-16g): full-history re-prefill was
**79.077% avoidable tokens** across a real 87-turn eval; live exact-prefix reuse cut
post-first-turn TTFT **84.17 s → 15.52 s (−81.6%)** — and was still REJECTED because one
repaired transcript changed (G1). Both engines proved near-miss reuse silently corrupts;
bonsai also proved **only the 2048-token chunk boundary was bitwise-exact**. Design encodes
those verdicts:

1. **Session-scoped, not global:** cache keyed by conversation (API `session_id` /
   fingerprint of the message-prefix token chain). One live KV snapshot chain per session;
   LRU across sessions under a byte budget (default ~2 GB, config).
2. **Snapshot only at prefill-chunk boundaries (2048).** Snapshot = global-KV slice
   reference up to boundary + local ring image (bounded: ≤0.34 GB worst-case, typically the
   ring is the cheap part) + token-chain hash + engine identity tuple (model, quant, template
   hash, kernel build SHA — Helios cache-key discipline).
3. **Restore = bitwise-identical state contract.** G1 gate: restored-then-continued
   generation must produce byte-identical tokens AND logit frames vs fresh full prefill at
   the same boundary, on frozen fixtures, k=2 repeats. If a boundary can't prove bitwise
   equality it is not a snapshot point. No "close enough" tier exists (T0's rejection is law).
4. **Turn flow:** new request → longest snapshot whose token chain is an exact prefix →
   restore → prefill only the suffix. Miss → full prefill (and lay down snapshots as you go).
5. Explicitly OUT (v1): cross-session dedup, SSD spill (P07 stayed parked), edited-prefix
   partial invalidation (exact-prefix only), mid-decode restore (Helios MidDecode rejection
   carried).
6. **Rotation safety (A2/B4 — NunSpark `PrefixCache` discipline).** Read true prefix length
   from the **first non-rotating (global) layer's** cache offset, never a local layer's (a
   rotated ring clamps its offset to the window and under-reports length); if a needed prefix
   falls inside a rotated local window, **drop the snapshot rather than trim it**
   (`RotatingKVCache` is not trimmable past rotation); `commit` reconciles against the actual
   cache offset instead of trusting bookkeeping. This is the *same* trim-or-drop module the MTP
   rollback uses (07) — build it once, model it on `PrefixCache`, never on `kv.truncate`.

Expected shape of the win (to be MEASURED at M6): multi-turn TTFT collapses from
O(transcript) to O(new tokens); the T0 numbers are the anchor that makes this a top-3
feature for the agentic use case, not a nice-to-have.

## Prefill chunking

Default 2048 (matches snapshot boundary; bonsai T0 showed 2048-chunk bitwise stability;
mlx-lm uses 2048). Governor may shrink per admission. Chunk boundaries are also the cancel
points and (post-v1) decode-interleave points.
