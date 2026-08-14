# 04 — Model family & weights

Normative dual-family scope: [ADR 0006](../decisions/0006-dual-family-v1-scope-and-gates.md).
The Gemma facts below remain the implemented regression baseline.

## Qwen3.8 text target

- Source: `Qwen/Qwen3.8-27B` at immutable revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Architecture discriminator: outer `qwen3_5`, text `qwen3_5_text`;
  `Qwen3_5ForConditionalGeneration`.
- Text graph: 64 dense layers in a repeated three-Gated-DeltaNet/one-full-attention pattern
  (48 recurrent + 16 full GQA), hidden 5120, intermediate 17408, vocabulary 248320.
- Full attention: 24 Q / 4 KV heads, head dimension 256, gated Q/output, q/k RMSNorm, partial
  RoPE. Gated DeltaNet has explicit convolution and FP32 recurrent state.
- Embedding and LM head are untied. Vision and bundled MTP are separate optional components and
  are omitted from the initial text artifact.
- Native context is 262144, but the first local deployment profile is 16K total cached tokens.
- The checkpoint tokenizer/template, two stop IDs, thinking/effort behavior, tool grammar, and
  generation defaults form an exact conversation profile; they are not Gemma conventions.

At ADR 0006, this identity is pinned but not downloaded, converted, native-loadable, or accepted.
The Rust dispatcher recognizes it only to fail before any Gemma-specific path.

## Gemma 4 family matrix

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
| current status | geometry only | converted baseline; acceptance pending | **implemented regression; acceptance pending** | geometry only | geometry only |

Constants shared by all Gemma rows: vocab 262144, RoPE local θ=1e4 full-rotation, global θ=1e6
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
  encoder-free unified multimodal variant; its text stack is what the current native graph
  implements. Weight
  sanitize: drop `vision_embedder`/audio projector tensors in v1 (keep the manifest aware of
  them for a separately gated P8 vision lane).

## E4B deltas for any future executable tier (legacy Gemma breadth)

1. **Cross-layer KV sharing:** the last 18 of 42 layers compute NO k/v projections; each
   reuses the KV of the nearest earlier layer of the same attention type (mlx-lm
   `gemma4_text.py` is the reference wiring; exact index map is config-derived, not
   hardcoded). Cache allocation exists only for the 24 non-shared layers.
2. **PLE (per-layer embeddings):** 256-dim per-layer input embedding gated into the residual
   stream; a second embedding lookup path, cheap but structural.
3. **K≠V in global layers** (`attention_k_eq_v=false`, 2 global KV heads) — the K=V fusion
   from K1 must be geometry-gated, not assumed.
4. Sliding window 512 (not 1024); 128K context; 4:1→5:1 pattern comes from `layer_types`.

## MoE hooks (26B-A4B legacy geometry capability; not a P0–P7 gate)

128 experts, top-8 routing, `moe_intermediate_size=704`; router precision is
quality-critical (mlx-lm forces router to 8-bit under any global quant — adopt that rule).
The existing `Geometry`/manifest layer retains parsing and validation for these fields. There are
no MoE kernels or executable 26B-A4B acceptance requirement in P0–P7; any future runtime tier gets
its own scope, artifact, and evidence gate.

## Weights logistics

| Artifact | Use | Note |
|---|---|---|
| `gemma-4-12B-it-qat-q4_0-unquantized` | Current Gemma quantization base | bf16-out-of-QAT; requantized to the accepted M0 MLX-affine artifact; the E1 ablation is legacy work |
| `gemma-4-12B-it` (bf16) | Off-device parity / QAT-delta spot-check | ≈24 GB (11.95B×2B) — does NOT fit 16 GB resident; on-device oracle & native arm both run the SAME Q4 (parity isolates the engine, not the quant) |
| `gemma-4-12B-it-qat-q4_0-unquantized-assistant` | Post-V1 Gemma MTP source | ~400M, 4-layer (3 local + 1 global), dim 1024, shares target embedding table, Q-only cross-attn to target KV |
| `gemma-4-E4B-it-qat-q4_0-unquantized` (+assistant) | Legacy E4B baseline / post-V1 MTP source | The converted E4B baseline is retained; its ~78M assistant is not a P0–P7 dependency |
| GGUF q4_0 releases | **Not ingested** | No MLX GGUF-quant import path; documented dead end |
| `Qwen/Qwen3.8-27B` | P2 oracle/conversion source | Pinned BF16 source; not downloaded or accepted; text output is selected by an explicit component inventory |

License: Gemma 4 is Apache-2.0 (Google's first OSI-licensed Gemma release; confirmed via
Google open-source blog + ai.google.dev/gemma/apache_2). One open flag: whether a Prohibited
Use Policy still binds Gemma 4 on top of Apache-2.0 was UNCONFIRMED at research time — check
the license frontmatter of each checkpoint at download and record in the ledger before any
redistribution decision. Hyperion's repository license covers Hyperion-authored source only;
model redistribution remains a separate decision and does not block V1 engineering.

## Gemma tokenizer & template

SentencePiece-family, vocab 262144, digits split, whitespace preserved; load via HF
`tokenizers` (Rust) from the checkpoint's tokenizer files — in-process, no Python
(Helios shelled to `mlx_lm` per request; banned). Chat template: Gemma 4's own
(`<|turn|>`-family tokens; tool declaration/call/response tokens; `<|think|>` toggle;
empty-thinking stabilization token present in 12B-it template). Render via minijinja from
the checkpoint's `chat_template` with golden-fixture tests vs `transformers`
`apply_chat_template` output (fixtures generated once by a pinned script, frozen). Training
cutoff Jan 2025 ⇒ domain vocab (Haystack 4.0 tags etc.) always in-context — a serving-layer
reality baked into agent-eval fixtures, not an engine feature.
