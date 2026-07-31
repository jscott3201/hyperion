# Hyperion

**M5-first local Gemma 4 inference in Rust + MLX.**

Hyperion is a clean-sheet inference runtime for people who want Gemma 4 to feel native on
an Apple M5-class Mac. Rust owns serving, scheduling, policy, and tokenization; a C++/MLX
engine owns model execution behind one narrow C ABI. The project favors correct model math,
bounded memory, real streaming, and reproducible evidence over a broad compatibility matrix.

> [!IMPORTANT]
> Hyperion is pre-1.0 and under active development. It requires Apple GPU family 10 (M5
> generation) or newer, macOS 26.2 or newer, and MLX 0.32.0 exactly. M1-M4 Macs and non-Apple
> platforms are intentionally unsupported. Model weights are not included.

## Why Hyperion

- **Gemma 4-native:** config-driven hybrid sliding/global attention, per-type RoPE, QAT-aware
  loading, and one model family instead of a collection of loosely integrated backends.
- **M5-first:** the runtime targets Metal, unified memory, and the M5 generation directly;
  the platform canary fails loudly on unsupported hosts and there is no slower fallback backend.
- **Local API, native request path:** OpenAI- and Anthropic-compatible HTTP subsets, real SSE,
  an in-process tokenizer, and no Python process in the serving path.
- **Evidence before claims:** performance changes are baseline-gated, correctness outranks
  throughput, and accepted measurements land in an append-only ledger.

## Architecture

![Hyperion architecture: API clients flow through the Rust serving plane and a narrow C ABI into the C++ MLX engine on Apple M5, while the oracle and evidence ledger remain off the request path.](docs/assets/hyperion-architecture.svg)

The serving process owns one MLX engine thread and admits one generation at a time. That
deliberate single-flight design keeps stream ownership, cancellation, KV state, and memory
admission explicit while the runtime matures.

## What works today

| Area | Current state |
|---|---|
| Native runtime | Gemma 4 12B QAT, converted to MLX affine Q4 / group size 64, with a custom Metal canary and a narrow C ABI |
| Serving | OpenAI-style chat completions and model discovery; Anthropic-style messages and token counting; streaming and non-streaming responses |
| Safety | Loopback-first bind policy, required bearer token off-loopback, request cancellation, bounded channels, single-flight admission, and fail-closed memory governance |
| Correctness | Pinned `mlx-lm` oracle, tokenizer/template parity fixtures, native model tests, contract tests, and append-only benchmark evidence |

The following are still active work, not shipped capabilities:

- Tool/function calling. Requests carrying tools currently return HTTP 400 while the
  incremental parser, deduplication, repair, and telemetry slice is completed.
- Thinking-lane SSE framing and transcript policy beyond the currently supported text path.
- Stable installation and release packaging.
- Runtime qualification beyond M5-or-newer Apple GPUs and the exact pinned MLX version.
- The M5 constrained-JSON lane, later speculative decoding, and larger-model tiers.

See the [milestone gates](docs/goal-package/10-milestones-and-gates.md) for the build order and
the [risk register](docs/goal-package/11-risk-register.md) for known failure modes.

## Quick start

### Prerequisites

- An Apple M5-generation-or-newer Mac running macOS 26.2+
- Xcode Command Line Tools, CMake 3.25 or newer, and the Metal toolchain
- MLX **0.32.0 exactly**, including its CMake package and `libmlx.dylib`
- Rust 1.95.0; [`rust-toolchain.toml`](rust-toolchain.toml) pins the toolchain and components

Hyperion looks for MLX at `/opt/homebrew/opt/mlx` by default. Set `MLX_ROOT` to a different
installation prefix when needed.

Clone the repository and run the model-free platform canary:

```sh
git clone https://github.com/jscott3201/hyperion.git
cd hyperion
cargo run --locked -p hyperion-bench -- canary
```

This compiles the Rust workspace, native C++/MLX library, and Metal canary, then verifies the
platform and pinned runtime before any model is loaded.

### Prepare the 12B model

Model preparation additionally requires `uv`, Python 3.12.13, `jq`, and the Hugging Face
`hf` CLI.
The source checkpoint is about 23.9 GB and the converted artifact is about 6.3 GiB. Plan for
at least 35 GB of free space, and potentially more if another Hugging Face cache retains a
second copy.

```sh
scripts/setup-oracle.sh
scripts/download-m0-model.sh
scripts/convert-m0-model.sh
scripts/hash-m0-model.sh
scripts/verify-m0-models.sh
```

The scripts pin the upstream revision and conversion recipe, then verify the exact source and
converted inventories. Payloads stay under the gitignored `artifacts/models/` tree.

### Start the server

```sh
cargo run --locked --release -p hyperion-server -- \
  --model artifacts/models/gemma4-12b-qat-mlx-g64-b4
```

Hyperion listens on `127.0.0.1:8080` by default. Try a streaming OpenAI-style request:

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "gemma4-12b-qat-mlx-g64-b4",
    "messages": [{"role": "user", "content": "Explain unified memory in two sentences."}],
    "max_tokens": 96,
    "temperature": 0,
    "stream": true
  }'
```

A non-loopback bind is rejected unless a bearer token is configured:

```sh
cargo run --locked --release -p hyperion-server -- \
  --model artifacts/models/gemma4-12b-qat-mlx-g64-b4 \
  --addr 0.0.0.0:8080 \
  --token "$HYPERION_BEARER_TOKEN"
```

## Development

The fast, model-free pull-request gate is:

```sh
scripts/setup-oracle.sh  # one-time, no model weights required
scripts/ci-pr.sh
```

The complete gate also expects `jq` and `ripgrep`, matching the CI environment.

For the native model-free suite alone:

```sh
scripts/test-native.sh --model-free
```

Both commands build native and Rust artifacts. On a space-constrained machine, prefer the
narrowest relevant test target and use `cargo clean` after the artifacts are no longer useful.
Real-model and benchmark gates are intentionally separate from ordinary pull-request CI.

## Repository map

| Path | Purpose |
|---|---|
| [`crates/`](crates/) | Six Rust crates for core policy, model geometry, tokenization, FFI, serving, and benchmarking |
| [`native/hyperion_mlx/`](native/hyperion_mlx/) | C++17 MLX graph, KV/cache machinery, platform policy, and Metal kernels |
| [`docs/goal-package/`](docs/goal-package/INDEX.md) | Goal contract, architecture, milestones, gates, and risk register |
| [`docs/decisions/`](docs/decisions/) | Numbered architectural decision records |
| [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) | Append-only evidence ledger; only `MEASURED` rows support benchmark claims |
| [`artifacts/models/MANIFEST.md`](artifacts/models/MANIFEST.md) | Pinned model revisions, hashes, conversion identity, sizes, and license reviews |

## Contributing

Hyperion is opening while the runtime contract is still being hardened. Issues and focused
pull requests are welcome. Before changing runtime behavior, read the
[goal contract](docs/goal-package/01-goal-contract.md), the current
[milestone gate](docs/goal-package/10-milestones-and-gates.md), and the repository's
[agent/build law](docs/goal-package/AGENTS.md). Keep changes small, pair performance work with
correctness evidence, and do not weaken a frozen gate to make a result pass.

## License

Unless otherwise noted, Hyperion's project-authored source code and documentation are
available under either the [MIT License](LICENSE-MIT) or the
[Apache License 2.0](LICENSE-APACHE), at your option.

This repository does not bundle or relicense Gemma model weights. Model artifacts and
third-party dependencies remain governed by their own terms; review the pinned model card and
preserve any applicable license or notice material before redistribution. Hyperion is not
affiliated with or endorsed by Google, Apple, or the MLX project.
