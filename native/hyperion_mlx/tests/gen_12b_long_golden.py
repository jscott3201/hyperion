#!/usr/bin/env python3
"""Generate the committed long-context greedy golden for M2-2.6a (chunked prefill + rotation).

The 2.7 golden sealed a 21-token prompt (single chunk, no rotation). This seals LONG
context: a ~2.5K-token chat-templated prompt that crosses one 2048 chunk boundary AND
pushes the sliding ring past the 1024 window (so the chunk-2 forward fires the rotation
read). The oracle prefill is CHUNKED (2048, mirroring the native ``hyp_prefill_chunk``) —
bitwise-invariant with a single-call prefill (each token's hidden attends causally to
[0, token], chunk-split-invariant; 05 §Prefill chunking) — and matches the native path
exactly. Decodes N tokens; asserts token-exact vs the native ``forward_12b_long_test``.

Dumps ``ids`` (the templated prompt), ``prefill_logits`` (post-softcap last-position, the
G1 logit frame), ``greedy_tokens`` (N tokens), and ``prompt_len`` (sanity). Self-skips
when the artifact is absent.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = REPO / "artifacts/models/gemma4-12b-qat-mlx-g64-b4"
GOLDEN = Path(__file__).parent / "fixtures/12b_long_greedy_golden.safetensors"

# A long public-domain-style user message (a fixed descriptive passage repeated) → ≥ 2048
# templated tokens (crosses one 2048 chunk boundary + the 1024 sliding window). Committed
# verbatim; regenerate only when the passage, N, or the epilogue math changes.
BASE = (
    "The hyperion engine is a clean-sheet Rust and MLX inference runtime for the Gemma 4 "
    "family on Apple Silicon. It targets the 16 gigabyte MacBook Pro first, with a "
    "device-derived memory budget and a predictive governor that admits work before "
    "submission. The heterogeneous key-value cache pairs a sliding local ring, capacity "
    "window plus gamma, with a capacity-stepped global cache. Attention reads the cached "
    "prefix at offset greater than zero, and the rotation-aware read reconstructs logical "
    "order once the ring has turned past its window. "
)
PROMPT = BASE * 22  # ~2.4K templated tokens (> 2048: crosses one chunk boundary; > 1024: fires the rotation read)
PREFILL_STEP = 2048
N_TOKENS = 8


def main() -> int:
    import mlx.core as mx
    from mlx_lm import load

    artifact = os.environ.get("HYPERION_12B_ARTIFACT", str(DEFAULT_ARTIFACT))
    if not Path(artifact).is_dir():
        print(f"gen_12b_long_golden: {artifact} absent; skipping (M5-gated)", file=sys.stderr)
        return 0

    print(f"loading 12B oracle from {artifact} ...", file=sys.stderr)
    model, tokenizer = load(artifact)

    # The 12B is an instruct/thinking model — apply the chat template (raw greedy is
    # degenerate; see gen_12b_greedy_golden.py).
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer.encode(formatted)
    print(f"templated ids: {len(ids)} tokens (need > {PREFILL_STEP} to cross a chunk boundary)", file=sys.stderr)
    assert len(ids) > PREFILL_STEP, f"prompt too short: {len(ids)} ≤ {PREFILL_STEP}"

    cache = model.make_cache()
    tokens = []

    # Chunked prefill (mirrors the native hyp_prefill_chunk): 2048-token chunks, mx.eval
    # the cache after each. The last chunk's final-position logits → token 1.
    offset = 0
    last_logits = None
    while offset < len(ids):
        take = min(PREFILL_STEP, len(ids) - offset)
        chunk = ids[offset:offset + take]
        logits = model(mx.array(chunk)[None], cache=cache)
        mx.eval([c.state for c in cache])
        if offset + take == len(ids):
            last_logits = logits[:, -1, :]
        offset += take
        mx.clear_cache()

    prefill_logits = mx.contiguous(mx.astype(last_logits, mx.bfloat16))
    next_tok = int(mx.argmax(last_logits, axis=-1).item())
    tokens.append(next_tok)
    mx.async_eval(prefill_logits)

    # Decode N-1 more.
    for _ in range(N_TOKENS - 1):
        logits = model(mx.array([next_tok])[None], cache=cache)
        last = logits[:, -1, :]
        next_tok = int(mx.argmax(last, axis=-1).item())
        tokens.append(next_tok)
    mx.eval(prefill_logits)

    states = {
        "ids": mx.array(ids, mx.int32),
        "prompt_len": mx.array([len(ids)], mx.int32),
        "prefill_logits": prefill_logits,
        "greedy_tokens": mx.array(tokens, mx.int32),
    }
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(GOLDEN), states)
    print(f"wrote {GOLDEN}: {len(ids)}-token prompt, greedy_tokens={tokens}", file=sys.stderr)
    print(f"  decoded: {tokenizer.decode(tokens)!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
