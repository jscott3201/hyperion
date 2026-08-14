# Hyperion parity oracle

This environment is benchmark tooling only. It is never linked, spawned, or imported by
`hyperion-server` or any request-path crate.

The exact VCS pin is required because released `mlx-lm` 0.31.3 predates the upstream
`gemma4_unified` compatibility mapping used by the official Gemma 4 checkpoint. The selected
commit is the first reviewed upstream fix and remains within the contract's `>=0.31.3` floor.

Create the isolated environment with `scripts/setup-oracle.sh`. `uv.lock` is authoritative;
do not install from mutable `main`.

## Qwen3.8 P2 contract

[`qwen38/`](qwen38/) contains a separate, model-free contract for the future dual-reference
Qwen3.8 oracle. It pins source and implementation identities plus first-party case inputs, but it
does not reuse this Gemma/MLX environment as proof that two Qwen references ran. The contract is
explicitly unexecuted; weights, real traces, tolerances, agreement, native execution, and support
all remain false.
