# MLX 0.32.0 + M5 — capability/gap sheet (2026-07-21)

## The SDPA head-dim gap (CONFIRMED from Metal dispatch source — the kernel program's reason to exist)

`ScaledDotProductAttention::use_fallback` (mlx/backend/metal/scaled_dot_product_attention.cpp):

- Fused **full** (prefill, q_len>8): head_dim ∈ {64, 80, 128} only, q==v dims. → **Gemma 4
  local (256) and global (512) BOTH fall back** for prefill → unfused matmul+softmax+matmul,
  materialized [T_q,T_kv] scores (the Helios 16K memory story).
- Fused **vector** (decode, q_len≤8, (q_len×gqa)≤32): head_dim ∈ {64, 96, 128, 256} +
  (192q/128v MLA case). → local 256 decode IS fused; **global 512 decode falls back**.
- 12B decode GQA factors: local 16Q/8KV → factor 2 (OK); global 16Q/1KV → factor 16
  (q_len×16 ≤ 32 ⇒ fused-vector eligibility caps at q_len 2 even at supported dims — γ>2
  MTP verify shapes exit the vector path regardless; K1/K2 kernels must cover verify shapes).
- `sinks` is a first-class SDPA param; masks: "causal" string or explicit array;
  softmax fp32 always. Training mode + CPU always fall back.
- 0.32.0 note "improved SDPA vector kernel for asymmetric head dims" = the 192/128 case,
  not 256/512. Re-read dispatch source at any pin bump (0.32.1 exists).

## Quantization (CONFIRMED from source)

- Modes: affine (default g64 b4; Metal kernels for bits {2,3,4,5,6,8}), mxfp4 (g32),
  nvfp4 (g16; **scale-encoding bug #2962**: signed-E4M3 vs NVIDIA UE4M3, 137× range loss —
  avoid), mxfp8 (g32). QQMM (quantized-in × quantized-weight) nvfp4/mxfp8 only.
- Dispatch qmv/qmm/qmv_quad/qmm_splitk + **qmv_wide (new in 0.32.0: small-batch quantized
  matvec — the MTP-verify shape;** plausibly the Ollama-contributed Gemma-4 kernel).
  Generation-aware batch limits (get_qmv_batch_limit) — never hardcode M thresholds.
- Mixed recipes in mlx-lm convert: mixed_2_6/3_4/3_6/4_6 (extra bits: down_proj, v_proj?,
  lm_head, first/last ⅛ layers — llama.cpp Q4_K_M heritage). Note 12B lm_head is TIED to
  the 262144×3840 embedding (~1.0 GB at extra bits — real budget cost).
- **No GGUF-quant ingestion anywhere:** core `mx.load(.gguf)` upcasts unsupported quant
  types to fp16 silently; mlx-lm has export-only, llama-arch-only GGUF code. QAT path =
  requantize from `-qat-q4_0-unquantized` safetensors.
- QuantizedKVCache exists (defaults 8-bit ctor / 4-bit via to_quantized; quantizes per
  decode step) but quantized SDPA is an UNFUSED python-level two-qmm composition, no sinks
  support — consistent with the parked KV-quant lane (O-6).

## mlx-lm 0.31.3 (the oracle)

- Gemma 4 complete since 0.31.2: layer pattern (`sliding_window_pattern=5`), per-type RoPE
  (proportional 0.25/θ1e6 global; default/θ1e4 local), KV sharing (`shared_kv` threading,
  sanitize drops shared-layer k/v weights, cache allocated only for owner layers),
  `attention_k_eq_v`, PLE (with the ~1 GB/2k-token bf16 intermediate warning), MoE
  (`SwitchGLU`; router FORCED to 8-bit under any quant — adopt), tool-call parser
  (`function_gemma4`, hyphenated-name fix in 0.31.3). Text-only (vision/audio sanitized out).
- Serving bits worth knowing (not carried — Python): continuous batching
  (BatchGenerator; speculative and batching mutually exclusive), PromptTrie/LRUPromptCache
  (radix-style prefix cache, eviction priority assistant>user>system), prompt-cache
  save/load to safetensors, `prefill_step_size` 2048 (512 in speculative path).
- Speculative: generic --draft-model only; **no native Gemma-4 MTP in mlx-lm**; Ollama's
  "Gemma 4 MTP" is their own engine's draft-model marketing.
- RotatingKVCache trim-after-wrap correctness: historically buggy for hybrid models
  (issue #980 class) — hyperion's ring is bespoke; validate restore paths independently.

## Memory APIs (C++-reachable)

Top-level `mx::` (metal:: wrappers deprecated): set_memory_limit / set_cache_limit /
**set_wired_limit** / get_active_memory / get_peak_memory / reset_peak_memory /
clear_cache / device_info. `recommendedMaxWorkingSetSize` measured 12,713,115,648 B on
m5-16g (bonsai ADR 0005). `sysctl iogpu.wired_limit_mb` exists (macOS 15+; unsupported by
Apple) — document, don't require. Community usable-GPU rule ~75% of RAM (16 GB point
extrapolated ≈ 10.9–12 GB — treat the device-read value as truth).

## Threads/streams

Default stream is thread-local; cross-thread implicit-stream use crashes ("no Stream(gpu,N)
in current thread"; mlx#2133/#3078 open). 0.31.2 "multi-thread" ≠ naive thread-pool safety.
0.32.0 adds `new_thread_unsafe_stream` (semantics not fully documented). Rule: ONE engine
thread owns MLX; explicit `stream=` on every op; no MLX calls from server tasks.

## M5 hardware (CONFIRMED Apple unless tagged)

- Base M5 (16 GB MBP14 config): 10-core CPU (4P+6E), 10-core GPU, **one Neural Accelerator
  per GPU core**, 16-core ANE (Core-ML-only; MLX does not use it), **153 GB/s** (LPDDR5X;
  M4 was 120). Memory options 16/24/32 GB.
- Apple MLX-on-M5 measurements (24 GB rig): **prefill/TTFT 3.33–4.06× vs M4** (bf16 AND
  4-bit both benefit), **decode +19–27%** (≈ bandwidth ratio) — prefill is compute-bound
  and NAX-accelerated; decode is bandwidth-bound; spec-decode is the only through-ceiling
  decode lever. **Requires macOS 26.2+.**
- NAX programming: Metal 4 TensorOps / Metal Performance Primitives (C++ templates,
  simdgroup-scope cooperative tensors); NOT plain MSL — naive compute shaders get
  ~2–6 TFLOPS-class, not NAX throughput (community roofline, LIKELY tier). Tile sweet spot
  ~32×32; dtypes FP16 (fp16/fp32 accum) + INT8 (int32 accum) confirmed on A19 proxy;
  **BF16 NAX support UNCONFIRMED** — probe on-device early (affects whether bf16
  activations hit NAX or shader ALUs in custom kernels; stock MLX quantized matmul already
  routes to NAX per Apple's own benchmark inference).
- MLX NAX status: "NAX with JIT" landed 0.30.1; op-level dispatch coverage undocumented —
  probe empirically at M0 (a 3840×3840×{1,64,512,2048} matmul sweep bf16-vs-fp16-vs-q4
  tells you what routes where).
- Sustained bandwidth ~122 GB/s measured (M5 Air, LIKELY; Pro chassis with fan should sit
  closer to peak). Base-M5 MBP14 sustained-load reviews show no throttling at Cinebench
  class; long-decode thermal profile unmeasured anywhere — collect at M1.
- Decode ceiling arithmetic (orientation): 153 GB/s ÷ ~7 GB working set ≈ 22 tok/s; at
  sustained 122 ≈ 17 tok/s. Helios/bonsai measured ~14 on M-class; the M1 stock baseline
  will set the real bar. M5 Pro/Max (shipped 2026-03): 307/614 GB/s — same engine, better
  ceilings; no code changes.

## Authoring paths for custom kernels

`mx::fast::metal_kernel` (C++ API exists; JIT, template-specialized, forward-only, GPU-only,
math-mode option since 0.32.0) → the K1/K2/K3 vehicle. Full C++ `Primitive` (eval_gpu +
vjp/vmap) only if graph-fusion demands it — decide at M4 exit. mlx-c exists as a C binding
layer; **header coverage of fast:: unverified** — hyperion binds MLX C++ directly through
its own narrow C shim (we own the ABI; mlx-c optional convenience, verify before adopting).
