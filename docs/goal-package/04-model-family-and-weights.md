# 04 — Model family & weights

Authoritative per-size table lives in `references/gemma4-family-facts.md` (extracted from the
five official `config.json` files 2026-07-21; CONFIRMED tier). This file is the engine-facing
digest + weight logistics.

## Family matrix (engine-relevant)

| | E2B | E4B | **12B (primary)** | 26B-A4B | 31B |
|---|---|---|---|---|---|
| Layers (local:global) | 35 (4:1) | 42 (5:1) | **48 (5:1 → 40/8)** | 30 (5:1) | 60 (5:1) |
| hidden / intermediate | 1536/6144 | 2560/10240 | **3840/15360** | 2816 (+MoE 704×128) | 5376/21504 |
| Q heads / local KV / global KV | 8/1/1 | 8/2/2 | **16/8/1** | 16/8/2 | 32/16/4 |
| head_dim local/global | 256/512 all sizes | | | | |
| K=V global (`attention_k_eq_v`) | no | no | **yes** | yes | yes |
| Cross-layer KV sharing | 20/35 | 18/42 | **0** | 0 | 0 |
| Sliding window | 512 | 512 | **1024** | 1024 | 1024 |
| PLE (`hidden_size_per_layer_input`) | 256 | 256 | **0 (none)** | 0 | 0 |
| Context | 128K | 128K | **256K** | 256K | 256K |
| v1 status | geometry only (O-3) | **M8 gated** | **primary** | geometry only (O-8) | geometry only (O-8) |

Constants shared by all: vocab 262144, RoPE local θ=1e4 full-rotation, global θ=1e6
partial 0.25, `gelu_pytorch_tanh`, RMSNorm eps 1e-6, pre+post-norm sandwich + QK-norm,
`final_logit_softcapping=30.0`, **no attention-logit softcap** (that was Gemma 2 — do not
add one), tied embeddings, last layer always global.

## 12B specifics the graph must honor

- Global layers at indices where `(layer+1) % 6 == 0` → {5, 11, 17, 23, 29, 35, 41, 47}.
- Global attention: 16 Q-heads share **1** KV head of head_dim **512**; K projection output
  IS the value tensor (no v_proj weight exists); p-RoPE rotates 128 of 512 dims.
- Local attention: 8 KV heads, head_dim 256, full-dim RoPE θ=1e4, causal within a
  1024-token trailing window.
- `model_type: gemma4_unified` / `Gemma4UnifiedForConditionalGeneration` — the 12B is the
  encoder-free unified multimodal variant; its text stack is what v1 implements. Weight
  sanitize: drop `vision_embedder`/audio projector tensors in v1 (keep the manifest aware of
  them for the O-5 lane).

## E4B deltas that M8 must implement (not cosmetic)

1. **Cross-layer KV sharing:** the last 18 of 42 layers compute NO k/v projections; each
   reuses the KV of the nearest earlier layer of the same attention type (mlx-lm
   `gemma4_text.py` is the reference wiring; exact index map is config-derived, not
   hardcoded). Cache allocation exists only for the 24 non-shared layers.
2. **PLE (per-layer embeddings):** 256-dim per-layer input embedding gated into the residual
   stream; a second embedding lookup path, cheap but structural.
3. **K≠V in global layers** (`attention_k_eq_v=false`, 2 global KV heads) — the K=V fusion
   from K1 must be geometry-gated, not assumed.
4. Sliding window 512 (not 1024); 128K context; 4:1→5:1 pattern comes from `layer_types`.

## MoE hooks (26B-A4B; geometry-validated in v1, runtime O-8)

128 experts, top-8 routing, `moe_intermediate_size=704`; router precision is
quality-critical (mlx-lm forces router to 8-bit under any global quant — adopt that rule).
No MoE kernels in v1; the `Geometry`/manifest layer must parse and validate these fields so
the 32 GB+ port is config-work, not architecture-work.

## Weights logistics

| Artifact | Use | Note |
|---|---|---|
| `gemma-4-12B-it-qat-q4_0-unquantized` | Quantization base | bf16-out-of-QAT; requantize to MLX affine (03 §Quantization; E1 ablation picks g32-grid vs g64) |
| `gemma-4-12B-it` (bf16) | Off-device parity / QAT-delta spot-check | ≈24 GB (11.95B×2B) — does NOT fit 16 GB resident; on-device oracle & native arm both run the SAME Q4 (parity isolates the engine, not the quant) |
| `gemma-4-12B-it-qat-q4_0-unquantized-assistant` | MTP drafter (M7) | ~400M, 4-layer (3 local + 1 global), dim 1024, shares target embedding table, Q-only cross-attn to target KV |
| `gemma-4-E4B-it-qat-q4_0-unquantized` (+assistant) | M8 tier | E4B drafter is ~78M |
| GGUF q4_0 releases | **Not ingested** | No MLX GGUF-quant import path; documented dead end |

License: Gemma 4 is Apache-2.0 (Google's first OSI-licensed Gemma release; confirmed via
Google open-source blog + ai.google.dev/gemma/apache_2). One open flag: whether a Prohibited
Use Policy still binds Gemma 4 on top of Apache-2.0 was UNCONFIRMED at research time — check
the license frontmatter of each checkpoint at download and record in the ledger before any
redistribution decision (affects O-2, not v1 engineering).

## Tokenizer & template

SentencePiece-family, vocab 262144, digits split, whitespace preserved; load via HF
`tokenizers` (Rust) from the checkpoint's tokenizer files — in-process, no Python
(Helios shelled to `mlx_lm` per request; banned). Chat template: Gemma 4's own
(`<|turn|>`-family tokens; tool declaration/call/response tokens; `<|think|>` toggle;
empty-thinking stabilization token present in 12B-it template). Render via minijinja from
the checkpoint's `chat_template` with golden-fixture tests vs `transformers`
`apply_chat_template` output (fixtures generated once by a pinned script, frozen). Training
cutoff Jan 2025 ⇒ domain vocab (Haystack 4.0 tags etc.) always in-context — a serving-layer
reality baked into agent-eval fixtures, not an engine feature.
