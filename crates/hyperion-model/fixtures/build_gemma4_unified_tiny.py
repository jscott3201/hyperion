#!/usr/bin/env python3
"""Generate the committed synthetic gemma4_unified-tiny weight manifest.

A 6-layer dense-unified Gemma 4 artifact (5:1 layout: layers 0-4 sliding,
layer 5 global) with the *correct schema* but tiny/zero byte counts — so the
model-free Rust tier can exercise ``WeightManifest`` without the ~6.7 GB
git-ignored real 12B weights. The schema honors the load-bearing invariant:
global layers store no ``v_proj`` (``attention_k_eq_v`` — K IS V), which is how
``WeightManifest::layer_kind`` derives attention kind from the weights.

Run: python3 crates/hyperion-model/fixtures/build_gemma4_unified_tiny.py
"""

import json
from pathlib import Path

NUM_LAYERS = 6
GLOBAL_LAYERS = {5}  # 5:1, last is global
OUT = Path(__file__).parent / "gemma4-unified-tiny"

PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJS = ("gate_proj", "up_proj", "down_proj")
SHARD_A = "model-00001-of-00002.safetensors"
SHARD_B = "model-00002-of-00002.safetensors"


def layer_tensors(layer: int, weight_map: dict, shard: str) -> None:
    projs = [p for p in PROJS if not (layer in GLOBAL_LAYERS and p == "v_proj")]
    pfx = f"language_model.model.layers.{layer}"
    for proj in projs:
        for kind in ("weight", "scales", "biases"):
            weight_map[f"{pfx}.self_attn.{proj}.{kind}"] = shard
    for norm in ("q_norm", "k_norm"):
        weight_map[f"{pfx}.self_attn.{norm}.weight"] = shard
    for proj in MLP_PROJS:
        for kind in ("weight", "scales", "biases"):
            weight_map[f"{pfx}.mlp.{proj}.{kind}"] = shard
    for norm in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
    ):
        weight_map[f"{pfx}.{norm}.weight"] = shard
    weight_map[f"{pfx}.layer_scalar"] = shard


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}

    # Shared embedding (tied LM head) + final norm in shard B.
    for kind in ("weight", "scales", "biases"):
        weight_map[f"language_model.model.embed_tokens.{kind}"] = SHARD_B
    weight_map["language_model.model.norm.weight"] = SHARD_B

    for layer in range(NUM_LAYERS):
        # Layers 0-3 in shard A; 4-5 in shard B (exercises shard_for splitting).
        shard = SHARD_A if layer < 4 else SHARD_B
        layer_tensors(layer, weight_map, shard)

    config = {
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4_unified",
        "tie_word_embeddings": True,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"},
        "text_config": {
            "model_type": "gemma4_unified",
            "num_hidden_layers": NUM_LAYERS,
            "layer_types": [
                "full_attention" if i in GLOBAL_LAYERS else "sliding_attention"
                for i in range(NUM_LAYERS)
            ],
        },
    }
    index = {
        "metadata": {"total_size": 4096, "total_parameters": 4096},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    print(f"wrote {OUT} ({len(weight_map)} tensors, {NUM_LAYERS} layers)")


if __name__ == "__main__":
    main()
