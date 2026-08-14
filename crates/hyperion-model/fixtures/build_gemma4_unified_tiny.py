#!/usr/bin/env python3
"""Generate the committed synthetic gemma4_unified-tiny artifact.

A 6-layer dense-unified Gemma 4 model (5:1: layers 0-4 sliding, layer 5 global) with
REAL (tiny, deterministic, quantized g64/b4-affine) safetensors weight shards — so the
native forward-pass test (M2-2.3a) and the Rust manifest tests can exercise the real
load + math without the ~6.7 GB git-ignored 12B. The geometry is g64-compatible (every
quantized in-feature divides 64) and mirrors the 12B's structure exactly: the 4-norm
sandwich, QK-norm (q/k scaled, v parameterless), K=V global (no v_proj on layer 5),
dual RoPE (local full θ=1e4 / global proportional 0.25 θ=1e6), tied embedding,
per-layer layer_scalar.

The dims here MUST match the tiny Geometry constructed in
``native/hyperion_mlx/tests/forward_test.cc`` (``make_tiny_geometry``) — a mismatch is a
test bug.

Run: python3 crates/hyperion-model/fixtures/build_gemma4_unified_tiny.py
"""

import json
from pathlib import Path

import mlx.core as mx

# ---- tiny geometry (g64-compatible; must match forward_test.cc make_tiny_geometry) ----
HIDDEN = 128
INTERMEDIATE = 256
NUM_LAYERS = 6
GLOBAL_LAYERS = {5}  # 5:1, last global
N_HEADS = 4
HEAD_DIM_LOCAL = 64
HEAD_DIM_GLOBAL = 128
N_KV_HEADS_LOCAL = 2
N_KV_HEADS_GLOBAL = 1
VOCAB = 128
GROUP_SIZE = 64
BITS = 4

OUT = Path(__file__).parent / "gemma4-unified-tiny"
SHARD_A = "model-00001-of-00002.safetensors"
SHARD_B = "model-00002-of-00002.safetensors"

# Deterministic PRNG so the committed fixture is reproducible.
_RNG = mx.random.key(0xC0FFEE)


def normal(shape):
    """A deterministic small-magnitude random array."""
    global _RNG
    _RNG, sub = mx.random.split(_RNG)
    return mx.astype(0.02 * mx.random.normal(shape, key=sub), mx.bfloat16)


def quantize_proj(weight_bf16):
    """Quantize a [out, in] bf16 weight to MLX affine g64/b4 -> (w, scales, biases)."""
    w, scales, biases = mx.quantize(weight_bf16, group_size=GROUP_SIZE, bits=BITS, mode="affine")
    return w, scales, biases


def put_proj(tensors, base, weight_bf16):
    w, s, b = quantize_proj(weight_bf16)
    tensors[f"{base}.weight"] = w
    tensors[f"{base}.scales"] = s
    tensors[f"{base}.biases"] = b


def layer_tensors(layer, tensors):
    is_global = layer in GLOBAL_LAYERS
    head_dim = HEAD_DIM_GLOBAL if is_global else HEAD_DIM_LOCAL
    n_kv = N_KV_HEADS_GLOBAL if is_global else N_KV_HEADS_LOCAL
    pfx = f"language_model.model.layers.{layer}"
    attn = f"{pfx}.self_attn"

    put_proj(tensors, f"{attn}.q_proj", normal((N_HEADS * head_dim, HIDDEN)))
    put_proj(tensors, f"{attn}.k_proj", normal((n_kv * head_dim, HIDDEN)))
    if not is_global:
        put_proj(tensors, f"{attn}.v_proj", normal((n_kv * head_dim, HIDDEN)))  # K≠V local
    put_proj(tensors, f"{attn}.o_proj", normal((HIDDEN, N_HEADS * head_dim)))

    # QK-norm: scaled RMSNorm over head_dim (no v_norm weight — parameterless).
    tensors[f"{attn}.q_norm.weight"] = normal((head_dim,))
    tensors[f"{attn}.k_norm.weight"] = normal((head_dim,))

    # Gemma 2 4-norm sandwich.
    for norm in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        tensors[f"{pfx}.{norm}.weight"] = normal((HIDDEN,))

    put_proj(tensors, f"{pfx}.mlp.gate_proj", normal((INTERMEDIATE, HIDDEN)))
    put_proj(tensors, f"{pfx}.mlp.up_proj", normal((INTERMEDIATE, HIDDEN)))
    put_proj(tensors, f"{pfx}.mlp.down_proj", normal((HIDDEN, INTERMEDIATE)))

    tensors[f"{pfx}.layer_scalar"] = normal((1,))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shard_a = {}
    shard_b = {}

    # Shared embedding (tied LM head) + final norm in shard B.
    put_proj(shard_b, "language_model.model.embed_tokens", normal((VOCAB, HIDDEN)))
    shard_b["language_model.model.norm.weight"] = normal((HIDDEN,))

    weight_map = {}
    for layer in range(NUM_LAYERS):
        target = shard_a if layer < 4 else shard_b
        layer_tensors(layer, target)
        # record the shard for every tensor this layer added
        pfx = f"language_model.model.layers.{layer}."
        for name in target:
            if name.startswith(pfx):
                weight_map[name] = SHARD_A if layer < 4 else SHARD_B

    for name in shard_b:
        if name.startswith("language_model.model.embed_tokens") or name == "language_model.model.norm.weight":
            weight_map[name] = SHARD_B

    # add the shard-A layer names to the weight_map (layer 0-3)
    for name in shard_a:
        if name.startswith("language_model.model.layers."):
            layer = int(name.split(".")[3])
            weight_map[name] = SHARD_A if layer < 4 else SHARD_B

    mx.save_safetensors(str(OUT / SHARD_A), shard_a)
    mx.save_safetensors(str(OUT / SHARD_B), shard_b)

    total = sum(t.nbytes for t in list(shard_a.values()) + list(shard_b.values()))
    config = {
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4_unified",
        "tie_word_embeddings": True,
        "quantization": {"group_size": GROUP_SIZE, "bits": BITS, "mode": "affine"},
        "quantization_config": {"group_size": GROUP_SIZE, "bits": BITS, "mode": "affine"},
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_k_eq_v": True,
            "bos_token_id": 2,
            "dtype": "bfloat16",
            "enable_moe_block": False,
            "eos_token_id": 1,
            "final_logit_softcapping": 30.0,
            "global_head_dim": HEAD_DIM_GLOBAL,
            "head_dim": HEAD_DIM_LOCAL,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": HIDDEN,
            "hidden_size_per_layer_input": 0,
            "initializer_range": 0.02,
            "intermediate_size": INTERMEDIATE,
            "layer_types": [
                "full_attention" if i in GLOBAL_LAYERS else "sliding_attention"
                for i in range(NUM_LAYERS)
            ],
            "max_position_embeddings": 256,
            "model_type": "gemma4_unified_text",
            "moe_intermediate_size": None,
            "num_attention_heads": N_HEADS,
            "num_experts": None,
            "num_global_key_value_heads": N_KV_HEADS_GLOBAL,
            "num_hidden_layers": NUM_LAYERS,
            "num_key_value_heads": N_KV_HEADS_LOCAL,
            "num_kv_shared_layers": 0,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "rope_theta": 10_000.0,
                    "rope_type": "default",
                },
            },
            "sliding_window": 8,
            "tie_word_embeddings": True,
            "top_k_experts": None,
            "use_bidirectional_attention": "vision",
            "use_cache": True,
            "use_double_wide_mlp": False,
            "vocab_size": VOCAB,
            "vocab_size_per_layer_input": 0,
        },
    }
    index = {
        "metadata": {"total_size": total, "total_parameters": total},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {OUT}: {SHARD_A} ({len(shard_a)} tensors), {SHARD_B} ({len(shard_b)} tensors), total {total} B")


if __name__ == "__main__":
    main()
