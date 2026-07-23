#!/usr/bin/env python3
"""Generate the committed FAULTED golden fixture for the G1 logit fault-boundary gate.

The companion to gen_12b_greedy_golden.py (the CLEAN golden). This injects a
single-layer structural fault into the oracle, then greedy-generates N tokens from
the SAME chat-templated prompt the clean golden uses. Dumps:

  - ``ids``              — the templated prompt (int32), identical to the clean golden.
  - ``prefill_logits``   — the post-softcap last-position logits from the FAULTED
                           prefill step (the faulted G1 logit frame; [vocab] bf16).
  - ``greedy_tokens``    — the N (1 prefill + N-1 decode) greedy tokens under the fault.

FAULT CHOICE (derive, don't port — 08-correctness-and-verification.md §21). The 08
"e.g." fault is "RoPE offset +1 on one layer"; measured on the Gemma-4 12B that
produces a logit delta AT OR BELOW the engine noise floor (sig_rel ~0.019 vs clean
~0.021 — no separation; the residual stream + QK-norm absorb a one-position
rotation). The SEPARATING fault is a per-layer residual-scale amplification:
``layer_scalar × LAYER_SCALAR_FACTOR`` at FAULT_LAYER. × 5 at layer 0 yields
sig_rel ~0.41 (~20× the noise floor, argmax-stable) — the derived fault. This is a
real single-layer structural perturbation (amplify the layer's contribution to the
residual stream), NOT a quantization-noise proxy (08:23-25). It mirrors the native
``ForwardPass::forward_faulted(..., fault_layer=0, layer_scalar_factor=5.0)``
exactly (decoder_layer multiplies that layer's layer_scalar by the factor).

FAULT INJECTION SEAM: the gemma4 DecoderLayer exposes ``layer_scalar`` (the per-
layer residual scale, an mx.array). We multiply FAULT_LAYER's layer_scalar by
LAYER_SCALAR_FACTOR for the duration of the generate loop (prefill + decode). The
fault persists across the loop, so the faulted greedy tokens diverge from the
clean tokens — confirming the fault moved the output (the gate's false-negative
guard). Both arms run the SAME fault on the SAME quant (08:23-25), so the faulted
delta (native_faulted vs oracle_faulted) isolates the engine's FAITHFULNESS to the
fault — a correct engine reproduces the fault; a broken one that silently absorbs
it does not.

Run on the M5 (needs the git-ignored ~6.7 GB 12B artifact):
    HYPERION_12B_ARTIFACT=$repo/artifacts/models/gemma4-12b-qat-mlx-g64-b4 \
      oracle/.venv/bin/python native/hyperion_mlx/tests/gen_12b_faulted_golden.py

Self-skips (exit 0) when the artifact is absent. The golden is tiny (~0.5 MB) and
committed; regenerate it only when the prompt, N, the fault, or the epilogue math
changes. Re-derive the gate THRESHOLD (in forward_12b_fault_boundary_test.cc) when
MLX/quant/kernels change (08:21).
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = REPO / "artifacts/models/gemma4-12b-qat-mlx-g64-b4"
GOLDEN = Path(__file__).parent / "fixtures/12b_faulted_greedy_golden.safetensors"

# MUST match gen_12b_greedy_golden.py so the clean + faulted goldens are directly
# comparable (same prompt, same N, same fault-free layers).
PROMPT = "The capital of France is"
N_TOKENS = 24
# The derived separating fault: layer 0's per-layer residual scale × 5. See the
# module docstring for the derivation (RoPE-offset rejected — no separation on Gemma-4).
FAULT_LAYER = 0
LAYER_SCALAR_FACTOR = 5.0


def main() -> int:
    import mlx.core as mx
    from mlx_lm import load

    artifact = os.environ.get("HYPERION_12B_ARTIFACT", str(DEFAULT_ARTIFACT))
    if not Path(artifact).is_dir():
        print(f"gen_12b_faulted_golden: {artifact} absent; skipping (M5-gated)",
              file=sys.stderr)
        return 0

    print(f"loading 12B oracle from {artifact} ...", file=sys.stderr)
    model, tokenizer = load(artifact)

    # Same chat-templated prompt as the clean golden (the 12B is instruct/thinking;
    # raw-prompt greedy is degenerate — see gen_12b_greedy_golden.py).
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer.encode(formatted)
    print(f"prompt: {PROMPT!r}  templated ids ({len(ids)}): {ids}", file=sys.stderr)

    # ── Fault injection: scale FAULT_LAYER's layer_scalar by LAYER_SCALAR_FACTOR. ──
    target = model.language_model.model.layers[FAULT_LAYER]
    original_scalar = target.layer_scalar
    target.layer_scalar = original_scalar * LAYER_SCALAR_FACTOR
    print(f"  fault: layer {FAULT_LAYER} layer_scalar *= {LAYER_SCALAR_FACTOR}",
          file=sys.stderr)

    try:
        cache = model.make_cache()
        tokens = []

        # Prefill: model(prompt, cache) -> logits [1, L, vocab]; take the last position.
        logits = model(mx.array(ids)[None], cache=cache)
        last = logits[:, -1, :]
        prefill_logits = mx.contiguous(mx.astype(last, mx.bfloat16))
        next_tok = int(mx.argmax(last, axis=-1).item())
        tokens.append(next_tok)
        mx.async_eval(prefill_logits)

        # Decode N-1 more: model(last_tok[None], cache) -> logits [1, 1, vocab]; argmax.
        for _ in range(N_TOKENS - 1):
            logits = model(mx.array([next_tok])[None], cache=cache)
            last = logits[:, -1, :]
            next_tok = int(mx.argmax(last, axis=-1).item())
            tokens.append(next_tok)
        mx.eval(prefill_logits)
    finally:
        target.layer_scalar = original_scalar

    states = {
        "ids": mx.array(ids, mx.int32),
        "prefill_logits": prefill_logits,
        "greedy_tokens": mx.array(tokens, mx.int32),
    }
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(GOLDEN), states)
    print(f"wrote {GOLDEN}: faulted greedy_tokens={tokens}", file=sys.stderr)
    print(f"  decoded: {tokenizer.decode(tokens)!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
