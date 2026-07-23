#!/usr/bin/env python3
"""Generate the committed greedy-token golden fixture for M2-2.7 (G1 token-exact seal).

The 2.3b hidden-state golden proved the per-layer math; this is the PARITY seal the
milestone is named for. Loads the M1-locked 12B Q4 artifact via the pinned mlx-lm
oracle, greedy-generates N tokens from the SAME 10-token prompt used by the 2.3b
golden ("The quick brown fox jumps over the lazy dog."), and dumps:

  - ``ids``            — the 10-token prompt (int32), so the native test feeds the
                         exact same ids (no tokenizer drift).
  - ``prefill_logits`` — the post-softcap last-position logits from the prefill step
                         (the G1 logit frame; [vocab] bf16). The native test reports
                         max|Δ| vs this — the kernel-change-tier G1 metric (fault-
                         boundary calibration is a G1 refinement; 2.7 reports the
                         number, the hard gate is token-exact).
  - ``greedy_tokens``  — the N (1 prefill + N-1 decode) greedy token ids (int32) the
                         native hyp_prefill_chunk/hyp_decode_block must match exactly.

Greedy = ``argmax(logits[:, -1, :])`` (logsumexp is monotonic → skip it). The model's
top-level ``__call__`` applies final_norm → tied lm_head → ``tanh(x/30)*30`` softcap
internally, so the logits here are already post-softcap (the native epilogue mirrors
this). cache = model.make_cache() (the heterogeneous sliding+global cache; offset is
threaded per-step by the cache objects themselves, mirroring the native cursor).

Run on the M5 (needs the git-ignored ~6.7 GB 12B artifact):
    HYPERION_12B_ARTIFACT=$repo/artifacts/models/gemma4-12b-qat-mlx-g64-b4 \
      oracle/.venv/bin/python native/hyperion_mlx/tests/gen_12b_greedy_golden.py

Self-skips (exit 0) when the artifact is absent. The golden is tiny (~0.5 MB) and
committed; regenerate only when the prompt, N, or the epilogue math changes.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = REPO / "artifacts/models/gemma4-12b-qat-mlx-g64-b4"
GOLDEN = Path(__file__).parent / "fixtures/12b_greedy_golden.safetensors"

PROMPT = "The capital of France is"
N_TOKENS = 24  # 1 prefill + 23 decode (enough to reach the answer through the thought channel)


def main() -> int:
    import mlx.core as mx
    from mlx_lm import load

    artifact = os.environ.get("HYPERION_12B_ARTIFACT", str(DEFAULT_ARTIFACT))
    if not Path(artifact).is_dir():
        print(f"gen_12b_greedy_golden: {artifact} absent; skipping (M5-gated)", file=sys.stderr)
        return 0

    print(f"loading 12B oracle from {artifact} ...", file=sys.stderr)
    model, tokenizer = load(artifact)

    # The 12B is ``Gemma4UnifiedForConditionalGeneration`` — an instruct/thinking model
    # that REQUIRES the chat template. Raw-prompt greedy is degenerate (argmax lands on
    # near-vocab-edge byte tokens); the chat-templated prompt produces coherent tokens
    # through the ``<|channel>thought`` channel. The native test feeds the SAME templated
    # ids, so token-exact parity holds (the forward math is faithful, proven by 2.3b).
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer.encode(formatted)
    print(f"prompt: {PROMPT!r}  templated ids ({len(ids)}): {ids}", file=sys.stderr)

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

    states = {
        "ids": mx.array(ids, mx.int32),
        "prefill_logits": prefill_logits,
        "greedy_tokens": mx.array(tokens, mx.int32),
    }
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(GOLDEN), states)
    print(f"wrote {GOLDEN}: greedy_tokens={tokens}", file=sys.stderr)
    print(f"  decoded: {tokenizer.decode(tokens)!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
