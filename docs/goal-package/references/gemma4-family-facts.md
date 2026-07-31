# Gemma 4 family — engine-relevant facts (CONFIRMED from config.json ×5 unless tagged; 2026-07-21)

## Master config table

| Field | E2B | E4B | **12B (unified)** | 26B-A4B (MoE) | 31B |
|---|---|---|---|---|---|
| model_type | gemma4 | gemma4 | **gemma4_unified** | gemma4 | gemma4 |
| hidden_size | 1536 | 2560 | **3840** | 2816 | 5376 |
| intermediate_size | 6144 | 10240 | **15360** | 2112 dense + 704 moe | 21504 |
| num_hidden_layers | 35 | 42 | **48** | 30 | 60 |
| local:global (layer_types) | 4:1 (28/7) | 5:1 (35/7) | **5:1 (40/8)** | 5:1 (25/5) | 5:1 (50/10) |
| Last layer | always full/global (all sizes) | | | | |
| num_attention_heads | 8 | 8 | **16** | 16 | 32 |
| num_key_value_heads (local) | 1 | 2 | **8** | 8 | 16 |
| head_dim local / global | 256 / 512 (all sizes) | | | | |
| num_global_key_value_heads | 1 | 2 | **1** | 2 | 4 |
| attention_k_eq_v (global only) | false | false | **true** | true | true |
| num_kv_shared_layers | **20** | **18** | 0 | 0 | 0 |
| sliding_window | 512 | 512 | **1024** | 1024 | 1024 |
| rope local / global | θ=1e4 default / θ=1e6 proportional, partial_rotary_factor=0.25 (all sizes) | | | | |
| vocab_size | 262144 (all) | | | | |
| max_position_embeddings | 131072 | 131072 | **262144** | 262144 | 262144 |
| tie_word_embeddings | true (all) | | | | |
| final_logit_softcapping | 30.0 (all); NO attention-logit softcap field exists | | | | |
| hidden_activation | gelu_pytorch_tanh (all); rms_norm_eps 1e-6; attention_bias false | | | | |
| hidden_size_per_layer_input (PLE) | **256** | **256** | 0 | 0 | 0 |
| use_double_wide_mlp | true (E2B only) | false | false | false | false |
| MoE | — | — | — | 128 experts, top-8, moe_inter 704 | — |
| audio_config | 12-layer conformer | same | encoder-free linear (640-dim, 40 ms/16 kHz) | **null** | **null** |
| vision path | ~150M ViT | ~150M ViT | **encoder-free 35M linear projector** (16-px patches, 3×3 pooled → 48 px; soft tokens {70,140,280,560,1120}, default 280) | ~550M ViT | ~550M ViT |
| Params total | ~4.2–5.1B (sources conflict) | ~6.8–8B (conflict) | **11.95B** | 25.2–26B / 3.8B active | ~29.3–30.7B |

Global-layer indices (5:1 sizes): `(layer+1) % 6 == 0`. 12B: {5,11,17,23,29,35,41,47}.

## Mechanics that drive kernel/KV design

- **K=V (12B/26B/31B, global layers only):** the key projection output IS the value tensor;
  no v_proj weight exists in those layers. Report: "reduce the global KV cache footprint by
  up to 37.5%" (arXiv). Engine: single K=V tensor per global layer; kernels read it once.
- **Cross-layer KV sharing (E2B/E4B only):** last N layers of each attention type reuse the
  KV of the nearest earlier layer of the same type; shared layers have NO k/v projections
  (mlx-lm `gemma4_text.py` reference wiring; exact index map config-derived).
  ~57% of E4B layers allocate no cache.
- **p-RoPE:** global layers rotate only `0.25 × 512 = 128` dims at θ=1e6; local layers
  full-dim θ=1e4. Two RoPE paths, selected by layer type. No YaRN branch.
- **Norms:** pre+post RMSNorm sandwich + QK-norm (arXiv prose). Final logits:
  `30·tanh(x/30)` once after lm_head (fp32).
- **Tokenizer:** SentencePiece-family, 262144, digits split, whitespace preserved;
  training cutoff Jan 2025.

## MTP drafter (per-size `-assistant` checkpoints, Apache-2.0)

4-layer (3 local + 1 global), drafter dim 256 (E-series) / 1024 (large); params 76–78M
(E2B/E4B), **~400M (12B)**, 430M (26B), 500M (31B). Shares target input-embedding table;
concatenates target final-layer activations with token embeddings, down-projects; **Q-only
attention layers cross-attend the TARGET's KV cache** (drafter holds no KV); draft-layer ↔
"last non-KV-shared target layer of same attention type" mapping (vLLM PR). Acceptance
protocol behaviorally consistent with exact greedy verify; formal rejection-sampling
guarantee UNCONFIRMED in public sources. Published: "up to 3×" (Google), "~2.2× locally"
Apple silicon batch 4–8 (26B), llama.cpp DGX Spark 1.9–3.1× accept 0.588 (MoE showed no
speedup there — conflicting with vLLM's MoE result; unresolved).

## Chat / thinking / tools (wire format)

- Turn wrapper `<|turn|>role … <|turn|>`-family; system/user/assistant roles supported.
- Thinking: `<|think|>` in system turn enables; trace emitted as `<|channel|>thought …`
  block; **strip prior-turn thoughts; KEEP thoughts across tool-call chains within the
  current turn**; 12B-it template includes an empty-thinking stabilization token
  (ghost-thought suppression). Community templates have known interleave bugs (HF #115) —
  trust Google docs + golden fixtures, not third-party templates.
- Tools: `<|tool|>declaration:…`, `<|tool_call>call:name{args}<tool_call|>`,
  `<|tool_response|>response:name{data}`; string literals delimited `<|"|>`. Parallel calls
  emitted in practice; duplicate-identical-call bug reported in the wild (LM Studio
  tracker) — dedupe at the server.
- Sampling defaults (model card, single-source): t=1.0, top_p 0.95, top_k 64.

## QAT q4_0

Per-size triplets: `-qat-q4_0-unquantized` (bf16 weights out of QAT — the custom-compilation
base), `-qat-q4_0-gguf` (packed; 12B file 6.98 GB), `-qat-w4a16-ct` (compressed-tensors).
q4_0 = symmetric block-32, one fp16 scale, `w = s·(q−8)` → representable exactly by MLX
affine g32/b4 with bias −8s. Which tensors stay unquantized in Google's packed release:
UNCONFIRMED — inspect the checkpoint (expect embeddings/norms high-precision). Google
QAT-vs-bf16 quality deltas: not published; third-party (Unsloth) numbers measure a
different scheme — do not import. arXiv Table 3 runtime footprint @32K quantized: 12B
7.65 GB, E4B 2.3 GB, E2B 0.8 GB, 26B 16.2 (2.8 active), 31B 19.2.

## Conflicts / UNCONFIRMED ledger (engine-relevant)

1. E2B/E4B total-param arithmetic doesn't reconcile across arXiv Table 1 vs model-card
   roundings (engine impact: none; manifest uses tensor-count truth).
2. MoE MTP speedup: llama.cpp none vs vLLM healthy (impacts O-8 only).
3. Exact E-series shared-KV index wiring: config + mlx-lm derivation, no prose spec —
   validate M8 parity against mlx-lm.
4. Formal MTP acceptance guarantee: verify empirically via the M7 exactness invariant.
5. QAT unquantized-tensor exclusions: inspect at download.
6. Video modality mechanics; PUP-on-Apache question — outside v1 engineering.
