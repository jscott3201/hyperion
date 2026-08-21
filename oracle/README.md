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
Qwen3.8 oracle. It pins source and implementation identities, first-party case inputs, the raw
trace grammar, and two independent producer semantic modes. It does not reuse this Gemma/MLX
environment as proof that two Qwen references ran; the Transformers arm requires its own lock.
The contract remains explicitly unexecuted: producer environments, weights, real traces,
tolerances, agreement, native execution, and support all remain false. The frozen
atomic-publication requirement is implemented as a model-free primitive in
[`qwen38_evidence_io.py`](qwen38_evidence_io.py) with hostile controls in
[`test_qwen38_evidence_io.py`](test_qwen38_evidence_io.py); the semantic verifier and both
producer entrypoints remain unbuilt.
