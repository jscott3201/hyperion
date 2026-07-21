# Hyperion parity oracle

This environment is benchmark tooling only. It is never linked, spawned, or imported by
`hyperion-server` or any request-path crate.

The exact VCS pin is required because released `mlx-lm` 0.31.3 predates the upstream
`gemma4_unified` compatibility mapping used by the official Gemma 4 checkpoint. The selected
commit is the first reviewed upstream fix and remains within the contract's `>=0.31.3` floor.

Create the isolated environment with `scripts/setup-oracle.sh`. `uv.lock` is authoritative;
do not install from mutable `main`.
