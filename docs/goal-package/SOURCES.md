# SOURCES — evidence map (compiled 2026-07-21)

## Primary — model (CONFIRMED tier)

- `config.json`, all five sizes (fetched directly): [12B](https://huggingface.co/google/gemma-4-12B-it/resolve/main/config.json) · [E2B](https://huggingface.co/google/gemma-4-E2B-it/resolve/main/config.json) · [E4B](https://huggingface.co/google/gemma-4-E4B/resolve/main/config.json) · [26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B/resolve/main/config.json) · [31B](https://huggingface.co/google/gemma-4-31B-it/resolve/main/config.json)
- Gemma 4 technical report: [arXiv 2607.02770](https://arxiv.org/html/2607.02770v1) (K=V "up to 37.5%" global-KV cut; KV-share 20/35, 18/42; drafter Table 1; memory Table 3; QAT encoder numbers)
- Model card: [ai.google.dev/gemma/docs/core/model_card_4](https://ai.google.dev/gemma/docs/core/model_card_4) (sampling t=1.0/top-p 0.95/top-k 64 — single-source)
- MTP drafter: [ai.google.dev/gemma/docs/mtp/overview](https://ai.google.dev/gemma/docs/mtp/overview) + [blog.google MTP post](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/) ("up to 3×"; "~2.2× locally" batch 4–8) + [vLLM PR #41745](https://github.com/vllm-project/vllm/pull/41745) (Q-only shared-KV attach; H100 numbers) + [llama.cpp PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398) (DGX Spark 1.9–3.1×, accept 0.588; no MoE speedup — conflicts with vLLM MoE result, unresolved)
- Prompt format / thinking / tools: [prompt-formatting-gemma4](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4) + [thinking](https://ai.google.dev/gemma/docs/capabilities/thinking) + [function-calling](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4)
- 12B unified/encoder-free: [dev guide](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/) (35M projector, 48-px pooled patches, audio 640/40 ms) + [intro post](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/)
- License: [opensource.googleblog.com Apache-2.0 announcement](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html) + [gemma/apache_2](https://ai.google.dev/gemma/apache_2) (PUP incorporation UNCONFIRMED — check per-checkpoint frontmatter)
- QAT collection: [gemma-4-qat-q4-0](https://huggingface.co/collections/google/gemma-4-qat-q4-0); [12B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized); GGUF 6.98 GB file size ([card](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)); q4_0 block format ([llama.cpp wiki](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)); [Unsloth QAT notes](https://unsloth.ai/docs/models/gemma-4/qat) (bf16-vs-f16 scale conversion loss — third-party, distinct scheme)

## Primary — MLX / Apple (CONFIRMED tier)

- MLX releases: [v0.32.0](https://github.com/ml-explore/mlx/releases/tag/v0.32.0) (qmv_wide; SDPA asymmetric-head-dim vector improvement; math-mode for custom kernels; new_thread_unsafe_stream); 0.30.1 "NAX with JIT"; 0.31.2 multi-thread
- SDPA dispatch source (the head-dim gap): [scaled_dot_product_attention.cpp](https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/scaled_dot_product_attention.cpp) — fused-full {64,80,128}; vector {64,96,128,256}+192/128
- Quantized kernels: [quantized.h](https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/quantized.h) (bits {2,3,4,5,6,8}; qmv_wide_impl); [nn quantized.py](https://raw.githubusercontent.com/ml-explore/mlx/main/python/mlx/nn/layers/quantized.py) (mode defaults affine 64/4, mxfp4 32/4, nvfp4 16/4, mxfp8 32/8); [nvfp4 scale bug #2962](https://github.com/ml-explore/mlx/issues/2962)
- mlx-lm: [gemma4_text.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/gemma4_text.py) (reference wiring: pattern, KV-share, p-RoPE, PLE, MoE router 8-bit rule); [cache.py](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/cache.py) (RotatingKVCache, QuantizedKVCache, PromptTrie/LRUPromptCache); releases 0.31.2 (Gemma 4 + tool parser) / 0.31.3 (thread-local stream, hyphenated tool names)
- Thread-safety: [mlx#2133](https://github.com/ml-explore/mlx/issues/2133), [#3078](https://github.com/ml-explore/mlx/issues/3078) (default stream thread-local; crashes)
- M5: [Apple newsroom M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/) (153 GB/s; NAX per GPU core; >4× M4 AI compute); [Apple ML Research MLX-on-M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) (prefill 3.33–4.06×, decode +19–27%, macOS 26.2 requirement, 24 GB rig); [MBP 14" M5 specs](https://support.apple.com/en-us/125405) (16/24/32 GB; 10C/10G)
- M5 Pro/Max (shipped 2026-03): [Apple newsroom](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/) (307/614 GB/s)
- Community (LIKELY tier, labeled): [tzakharko NAX microbench](https://tzakharko.github.io/apple-neural-accelerators-benchmark/) (A19 proxy: FP16/INT8 tiles 32×32, ~7.4 TFLOPS/5 cores; BF16 unconfirmed); [M5 roofline](https://www.michaelstinkerings.org/apple-m5-gpu-roofline-analysis/) (~122 GB/s sustained, Air; non-TensorOps shaders 2–6 TFLOPS class); [~75% wired rule-of-thumb](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html) (16 GB point extrapolated)

## Internal (MEASURED tier — the two retired engines)

- Helios @ c5f4b06: BENCHMARKS.md (native 16K prefill +52.269% / peak 21.874→7.638 GB;
  decode p50 ~70 ms; chat p99 510.888 ms; code_review_8k p99 2161.658 ms; XR86 MTP
  +19.969% / accept 0.706 / verifier 85.762%; XR07 prefix-reuse parity fail; XR09 KV-quant
  0.000% active relief); docs/xr-*.md; native/gemma4_mlx (K=V, per-type RoPE, softcap —
  reference math); spec/03 (C ABI shape)
- mlx-bonsai @ e99e2ee(+T0): ADR 0003 (M5 floor), 0005 (governor: recommendedMaxWorkingSetSize
  12,713,115,648 B; predictive transient admission), 0006/0007 (server/tool contracts),
  0009 (agent-eval); bench-results/t0-reprefill-tax.md (79.077% avoidable prefill;
  −81.6% TTFT live probe; 2048-boundary bitwise-exact; REJECTED on G1); EVAL-RESULTS.md
  (bonsai-Qwen agentic verdict); _goals/ab-verification-protocol.md (G1–G4 law)
- Kernel dossier (Inference/kernel-consolidation, 2026-07): gemma-challenge playbook
  (head-dim fix biggest win; FP8 logit-saturation warning; eval-prompts ADOPT; QAT base
  ADOPT), BitLinear/ternary prior art (dropped with Bonsai), TileLang note (CUDA-port lane)

## External-adjacent (NunSpark — v0.2 exploration, 2026-07-21)

`github.com/sharma-open-source/NunSpark` (v0.12, MIT, ~9k LOC Python/MLX) — an independent
Apple-Silicon MLX engine for running oversized LLMs via disk-streaming + deep-K speculative
decoding. Explored for adoptable ideas; see `references/nunspark-adoptions.md` for the full
provenance and where each finding lands. Label discipline: **MEASURED** = their gated A/B
(M4 16GB unless noted), **COMMUNITY** = external hardware, **CLAIM** = design thesis.

- Key MEASURED findings hyperion adopted: the fp16 near-tie argmax-flip (`docs/plan5-m2-mismatch-
  investigation.md`, ~1/110 tokens → A1); the memory cliff (16GB budget sweep 4/5/6/8/10 GB →
  1.79/2.52/3.23/2.84/1.33 tok/s → A4); spec-slower-than-greedy on MoE across four scales
  (`docs/plan4-m5-gate-summary.md`, `docs/backlog.md` → A5); MoE expert-streaming recipe
  (Qwen3-30B-A3B 0.5→3.2 tok/s, persistent scatter buffers `docs/backlog.md` #10, two-region
  cache `docs/plan4-m2/m3-gate-summary.md` → M9/C); the trim-rollback latent bug (`docs/backlog.md`
  #4 → A2); the n-gram adaptive drafter (`src/nunspark/ngram_drafter.py` → B1); the archspec/
  mask-by-kind seam (`src/nunspark/archspec.py` → A3/B2).
- **Standing caveat (do not launder):** NunSpark has **zero measured Gemma numbers** — Gemma is
  registered but never load-tested there; its Gemma-MTP path carries the trim bug (A2), its tree
  and EAGLE paths are unvalidated. Its correctness *lessons* are load-bearing; its Gemma *code* is
  a reference, not a proven path. hyperion's Gemma-4 numbers remain first-of-kind.
- Drafter architecture CONFIRMED from `config.json`: [12B assistant](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized-assistant) (4L, hidden 1024, 16Q/8KV, hd256) · [E4B assistant](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant) (4L, hidden 256, 4Q/2KV, `backbone_hidden_size: 2560`, `num_centroids: 2048`) → B3.

## Confidence discipline

References carry CONFIRMED/LIKELY/UNCONFIRMED tags inline (references/*.md). Standing
UNCONFIRMED items an implementing agent must re-verify at build time: NAX op-level dispatch
coverage in MLX (undocumented; probe empirically), exact fused-SDPA sets in the pinned MLX
build (re-read dispatch source at pin time), QAT q4_0 unquantized-tensor exclusions
(inspect checkpoint), Gemma 4 PUP incorporation, WWDC26 session 232 content (unpublished at
research time), mlx-c header coverage if the FFI binds via mlx-c instead of direct C++.
