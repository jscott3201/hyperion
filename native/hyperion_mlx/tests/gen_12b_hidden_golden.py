#!/usr/bin/env python3
"""Generate the committed golden hidden-state fixture for M2-2.3b (real-12B oracle parity).

Loads the M1-locked 12B Q4 artifact via the pinned mlx-lm oracle, runs a fixed short
prompt through ``model.model(ids)`` (the post-final-RMSNorm hidden state, PRE lm_head —
exactly what the native ``ForwardPass.forward`` returns), and dumps the token ids + the
hidden state to a committed safetensors golden. The native ``forward_12b_test`` then
loads the SAME 12B via the native loader, runs the SAME ids, and compares — sealing the
2.3a math against the oracle (scale=1.0, QK-norm-before-rope, v_norm-no-scale, the
proportional RoPE inf-freq trick, and the k_eq_v V=v_norm(k_proj) path).

Run on the M5 (needs the git-ignored ~6.7 GB 12B artifact):
    HYPERION_12B_ARTIFACT=$repo/artifacts/models/gemma4-12b-qat-mlx-g64-b4 \
      oracle/.venv/bin/python native/hyperion_mlx/tests/gen_12b_hidden_golden.py

Self-skips (exit 0) when the artifact is absent. The golden is tiny (L × 3840 bf16) and
committed; regenerate it only when the prompt or the math changes.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = REPO / "artifacts/models/gemma4-12b-qat-mlx-g64-b4"
GOLDEN = Path(__file__).parent / "fixtures/12b_hidden_golden.safetensors"

# A fixed short prompt (>= 6 tokens so the first global layer at index 5 fires).
PROMPT = "The quick brown fox jumps over the lazy dog."

def main() -> int:
    import mlx.core as mx
    from mlx_lm import load

    artifact = os.environ.get("HYPERION_12B_ARTIFACT", str(DEFAULT_ARTIFACT))
    if not Path(artifact).is_dir():
        print(f"gen_12b_hidden_golden: {artifact} absent; skipping (M5-gated)", file=sys.stderr)
        return 0

    print(f"loading 12B oracle from {artifact} ...", file=sys.stderr)
    model, tokenizer = load(artifact)
    ids = tokenizer.encode(PROMPT)
    print(f"prompt: {PROMPT!r}  ids ({len(ids)}): {ids}", file=sys.stderr)

    # Re-implement the Gemma4TextModel loop to capture per-layer intermediates, so the
    # native test can localize the first divergence. cache=None -> offset 0 prefill; with
    # N<=window the mask is "causal" for every layer (matches the native offset-0 path).
    m = model.language_model.model  # Gemma4TextModel
    inputs = mx.array(ids)[None]
    h = m.embed_tokens(inputs) * m.embed_scale
    states = {"ids": mx.array(ids, mx.int32), "embed": mx.contiguous(mx.astype(h, mx.bfloat16))}
    for i, layer in enumerate(m.layers):
        if i in (0, 5):
            # Capture attention internals to localize the divergence (sliding L0 vs global L5).
            attn_out, (kv_k, kv_v), _ = layer.self_attn(h, "causal", None, None, 0)
            states[f"layer_{i:02d}_attn"] = mx.contiguous(mx.astype(attn_out, mx.bfloat16))
            states[f"layer_{i:02d}_k"] = mx.contiguous(mx.astype(kv_k, mx.bfloat16))
            states[f"layer_{i:02d}_v"] = mx.contiguous(mx.astype(kv_v, mx.bfloat16))
            # Raw projections (pre-norm) to isolate the matmul from the norms/rope.
            a = layer.self_attn
            states[f"layer_{i:02d}_qproj"] = mx.contiguous(mx.astype(a.q_proj(h), mx.bfloat16))
            states[f"layer_{i:02d}_kproj"] = mx.contiguous(mx.astype(a.k_proj(h), mx.bfloat16))
        h, _, _ = layer(h, "causal", None, None, None, 0)
        states[f"layer_{i:02d}"] = mx.contiguous(mx.astype(h, mx.bfloat16))
    h = m.norm(h)
    states["final"] = mx.contiguous(mx.astype(h, mx.bfloat16))
    mx.eval(states)
    print(f"captured {len(states)} states: embed + {len(m.layers)} layers + final", file=sys.stderr)

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(GOLDEN), states)
    print(f"wrote {GOLDEN} ({len(states)} tensors)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
