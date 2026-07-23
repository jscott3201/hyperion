#!/usr/bin/env python3
"""LEAN stock-mlx-lm decode throughput measurement for the M2-2.6d 0.9x gate.

The M1-stock arm: times mlx-lm's generate_step (the same API the M1 worker uses)
with the SAME timing model as native_decode_bench.cc — monotonic offsets_ns per token
(offsets[0] = TTFT/prefill, offsets[1..] = decode). Same prompt (the golden's
chat-templated ids), same N, same machine. NOT ledger-compliant (no A-C-C-A, no
manifest SHAs) — a defensible 0.9x SIGNAL only.

Self-skips when HYPERION_12B_ARTIFACT is unset.
"""
import os, sys, time, json
from pathlib import Path

PROMPT = "The capital of France is"
N_TOKENS = 24
TRIALS = 5
WARMUPS = 2

def main() -> int:
    artifact = os.environ.get("HYPERION_12B_ARTIFACT")
    if not artifact or not Path(artifact).is_dir():
        print("stock_decode_bench: HYPERION_12B_ARTIFACT unset/absent; skipping", file=sys.stderr)
        return 0
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    print(f"loading 12B stock from {artifact} ...", file=sys.stderr)
    model, tokenizer = load(artifact)
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer.encode(formatted)
    L = len(ids)
    prompt = mx.array(ids, dtype=mx.int32)

    def generate(n):
        mx.clear_cache()
        mx.synchronize()
        cache = make_prompt_cache(model)
        start = time.perf_counter_ns()
        offsets = []
        for token, _logprobs in generate_step(prompt, model, max_tokens=n, prompt_cache=cache, prefill_step_size=2048):
            mx.synchronize()
            offsets.append(time.perf_counter_ns() - start)
            del _logprobs
        return offsets

    for _ in range(WARMUPS):
        generate(N_TOKENS)

    prefill_tok_s, decode_tok_s = [], []
    all_itls = []
    for _ in range(TRIALS):
        o = generate(N_TOKENS)
        prefill_tok_s.append(L * 1e9 / o[0])
        decode_dur = o[-1] - o[0]
        decode_tok_s.append((N_TOKENS - 1) * 1e9 / decode_dur)
        for i in range(1, len(o)):
            all_itls.append(o[i] - o[i-1])
    all_itls.sort()
    def nr(pct):
        idx = max(1, (len(all_itls) * pct + 99) // 100)
        return all_itls[min(idx, len(all_itls))-1]

    med = lambda v: sorted(v)[len(v)//2]
    print(f"stock_decode_bench mode=stock-mlx-lm prompt_tokens={L} generated_tokens={N_TOKENS} trials={TRIALS} warmups={WARMUPS}")
    print(f"  prefill_tok_s median={med(prefill_tok_s):.3f} min={min(prefill_tok_s):.3f} max={max(prefill_tok_s):.3f}")
    print(f"  decode_tok_s  median={med(decode_tok_s):.3f} min={min(decode_tok_s):.3f} max={max(decode_tok_s):.3f}")
    print(f"  itl_ns p50={nr(50)} p95={nr(95)} p99={nr(99)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
