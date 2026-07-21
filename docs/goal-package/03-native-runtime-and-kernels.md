# 03 — Native runtime & kernel lanes

## The execution-model decision (made up front — Helios R4)

Helios re-traced the define-by-run graph every step, deferred KV evaluation to end-of-step
(global layers first), grew global KV by full `concatenate` reallocation, and called
`reset_peak_memory()` every decode step. Result: p50 ~70–82 ms but p99 511–2162 ms tails
attributed to full-attention deferred eval. hyperion rules the opposite model:

1. **Shape-bucketed step functions.** KV buffers are capacity-stepped (256-token steps).
   Within a bucket, every decode step has identical shapes → the traced graph is stable.
   Use `mx::compile` on the decode step (and per-bucket prefill chunk) so kernel dispatch
   is amortized; bucket transitions (rare, at capacity boundaries) pay one re-trace by
   design, at a known point, never mid-token.
2. **In-place KV writes.** `slice_update` into preallocated buffers for both layer types.
   Zero `concatenate` in the decode hot loop. Helios's own XR71 measured slice-update
   overhead at ~0.010 ms/token — the mechanism is cheap; it just was never made the default.
3. **Eager, incremental KV eval.** KV materializes as part of the step evaluation itself
   (no deferred end-of-step global-first pass — that ordering was the tail).
4. **No per-step `reset_peak_memory`.** Peak tracking is sampled by the bench harness, not
   the hot loop.
5. **Explicit streams everywhere;** the engine thread owns them (02-architecture).

## The SDPA gap (provable from MLX 0.32.0 dispatch source — this is the moat)

MLX fused attention support at 0.32.0:

| Path | Supported head_dim | Gemma 4 hit? |
|---|---|---|
| Fused **full** (prefill, q_len > 8) | {64, 80, 128} | **Neither** 256 (local) nor 512 (global) — every prefill layer falls back to unfused matmul+softmax+matmul |
| Fused **vector** (decode, q_len ≤ 8) | {64, 96, 128, 256} (+192/128 MLA case) | local 256 **yes**; global 512 **no** |

Consequences measured in Helios: unfused prefill materializes [T_q, T_kv] score tensors
(the 16K memory blowup), and global-layer decode runs unfused. The custom-kernel program
exists to close exactly this, nothing more speculative than that.

### Kernel lane K1 — global-layer attention kernel (head_dim 512, K=V-aware) [M4]

Fused Metal kernel for the 8 global layers: GQA 16 Q-heads : 1 KV-head, head_dim 512,
**K=V unified** — the value tensor IS the key tensor. Must cover **q_len 1..γ_max** (decode
q_len=1 AND MTP-verify q_len=γ≤8): global attention exits MLX's fused vector path whenever
`q_len × 16 > 32`, i.e. q_len ≥ 3, so hyperion owns this shape across decode and verify.
Exploit: stream the single K=V tensor through threadgroup memory ONCE for both the score
dot-products and the value accumulation, instead of a generic kernel's separate K-then-V
reads. The CONFIRMED architectural figure is a **37.5% global-KV storage reduction** (arXiv);
the decode-time *bandwidth* win from the single read is **ASPIRATIONAL and A/B-gated (G2)**,
not a spec guarantee. Inputs arrive pre-rotated (RoPE is a separate fused op; partial
rotation = 128 of 512 dims, global).
Authoring: `mx::fast::metal_kernel` (C++ API; JIT, template-specialized on head_dim/dtypes;
forward-only is fine for inference). Fall back to a C++ `Primitive` only if graph-fusion
profiling demands it (decide at M4 exit, not later — retrofit rewrites the authoring layer).

### Kernel lane K2 — windowed-flash prefill kernel (head_dim 256 local; 512 global) [M4]

Flash-style online-softmax tiled prefill kernel:
- **Local layers (40): banded attention.** Each query attends ≤1024-token window → tile the
  band only. Working set is O(window), independent of context. The memory-cliff fix is
  **staged**: at **M2** the execution-model change alone (chunked prefill bounds the score
  tensor to q_chunk×ctx ≤ 2048×ctx, plus no-concat-grow KV) is what carries the 16K peak
  gate; **K2 at M4** then removes the remaining ctx factor for the 40 local layers
  (→ q_chunk×window), so they stop scaling with context entirely.
- **Global layers (8): standard causal flash** at head_dim 512 with the K=V single-read
  trick from K1.
Correctness gate: bitwise-stable greedy tokens vs the unfused reference path, logit max-abs
within the two-sided fault-boundary threshold (08-correctness).

### Kernel lane K3 — NAX/TensorOps experiments [M4, experimental sublane]

Apple's published M5 economics: NAX gives ~3.5–4.06× prefill vs M4 on 4-bit and bf16 models
via Metal 4 TensorOps; MLX ≥0.30.1 already routes quantized matmuls to NAX ("NAX with JIT";
qmv_wide added in 0.32.0). Discipline (from the kernel dossier + gemma-challenge lesson):
**benchmark stock `quantized_matmul` FIRST** — it may already win; only write custom NAX
tile kernels for the fixed 12B shapes (3840×15360 etc.) if profiling shows a real gap.
Constraint to respect: naive MSL compute shaders that don't route through TensorOps get
~2–3 TFLOPS-class throughput, not NAX throughput — so any custom GEMM must use the Metal 4
TensorOps/cooperative-tensor path, or it will LOSE to stock. K3 promotes only through the
full A/B protocol; expected outcome is "stock wins prefill GEMM, custom wins attention" —
that's fine, the attention kernels (K1/K2) are the moat.

## Quantization pipeline (weights)

- **Base checkpoint: `google/gemma-4-12B-it-qat-q4_0-unquantized`** (bf16 weights out of the
  QAT pipeline; Google positions it exactly for custom downstream compilation). NOT the GGUF:
  MLX/mlx-lm have no GGUF-q4_0 ingestion (unsupported types silently upcast to fp16); the
  correct path is requantize-from-QAT-bf16.
- **Grid match matters:** q4_0 = symmetric, block 32, one fp16 scale (w = s·(q−8)). MLX
  affine g32/b4 with bias = −8·s represents that grid exactly (~5.0 effective bits/weight,
  ≈7.5 GB for 11.95B). MLX affine **g64**/b4 is the memory-lean default (~4.5 bits/weight,
  ≈6.7 GB) but is NOT the grid QAT trained for.
- **E1 quant ablation (M2, MEASURED):** g32-grid-exact vs g64 vs mixed_4_6 (extra bits on
  down_proj/first-last-⅛ layers; note lm_head is TIED to the 262144×3840 embedding — mixed
  recipes that upcast lm_head upcast ~1.0 GB of embedding, on a 12 GB budget that is a real
  trade). Gate the default on: greedy parity vs oracle, gemma-challenge eval-prompt quality
  hold, decode tok/s, resident GB. No assumption survives contact with the ledger.
- Runtime quantized ops: MLX affine `quantized_matmul` (2/3/4/5/6/8-bit kernels exist;
  qmv/qmm/qmv_quad/qmm_splitk/qmv_wide dispatch is generation-aware — never hardcode M
  thresholds).
- Activations bf16; softmax fp32 (MLX default); final logits fp32 with softcap
  `30·tanh(x/30)` fused Rust-side of the lm_head? No — native, single fused epilogue op.

## Sampling (native epilogue)

Greedy argmax native (token + logit returned per step). Sampled mode: temperature/top-k/
top-p/min-p computed native-side on the fp32 logit vector with a seeded per-request RNG
(vocab 262144 → do NOT ship logits across the ABI per token; ship the sampled token + the
top-k logprobs requested). Repetition/presence/frequency penalties applied Rust-side via a
small recent-token state passed into the step call (bounded window, documented).

## What is explicitly NOT in the native layer

Tool-call parsing, template rendering, constrained-JSON masking (Rust, 06), prefix-cache
policy (Rust owns policy; native exposes snapshot/restore mechanics, 05), scheduling,
auth. The native layer is a fast, dumb, deterministic step machine.
