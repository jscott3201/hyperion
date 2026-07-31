# 02 — Architecture

## Workspace (lean; every crate on the live path — anti-pattern: Helios shipped 5 crates no
other crate imported)

```
hyperion/
├── Cargo.toml                  # workspace, resolver 2, rust-version 1.95
├── crates/
│   ├── hyperion-core/          # engine loop, request lifecycle, scheduler, governor policy
│   ├── hyperion-model/         # Gemma 4 geometry/config validation, weights manifest, family matrix
│   ├── hyperion-tokenizer/     # in-process tokenizer (HF tokenizers), chat template (minijinja),
│   │                           # tool-call wire format, thinking-mode transcript policy
│   ├── hyperion-ffi/           # C ABI bindings + unsafe confinement (only crate with unsafe)
│   ├── hyperion-server/        # axum: OpenAI + Anthropic surfaces, SSE, tool_call parser,
│   │                           # constrained-JSON lane, auth, error taxonomy
│   └── hyperion-bench/         # A/B harness, parity-oracle client, agent-eval, ledger tools
├── native/hyperion_mlx/        # C++17 MLX graph, kernels (metal/), governor mechanics, probes
│   ├── include/hyperion_mlx.h  # the C ABI (single header)
│   ├── src/
│   ├── metal/                  # custom kernels (K1/K2/K3 lanes)
│   └── tests/
├── scripts/                    # verify.sh, bench entry points, oracle venv setup
├── docs/decisions/             # ADRs, numbered, append-only
├── docs/goal-package/          # THIS package
├── benchmarks/ + eval-results/ # ledgers (append-only) + raw outputs (gitignored)
└── artifacts/                  # models, traces (gitignored)
```

Rationale: 6 crates vs Helios's 13. `gemma4d-{engine,sampler,router,chat,tokenizer}` were
islands imported by nothing on the serve path; hyperion forbids that by construction — CI
includes a "no orphan crates" check (every workspace member reachable from `hyperion-server`
or `hyperion-bench`).

## Threading model (from mlx-bonsai, hardened by MLX thread-safety reality)

- **One dedicated engine thread owns all MLX/native state.** Server tasks talk to it over a
  bounded mpsc channel (capacity 8), cancel-on-drop, engine resets generation state around
  every request. This sidesteps the documented MLX footgun: the default stream is
  thread-local and cross-thread implicit-stream use crashes (`no Stream(gpu, N) in current
  thread` class; mlx#2133, #3078). Native code passes explicit streams; nothing relies on
  the ambient default from a foreign thread.
- Single-flight generation: `Semaphore(1)` → second concurrent generation gets **429**
  immediately (bonsai contract). A small accept-queue (depth 2) MAY be added post-v1; not a
  v1 gate.
- Chunked prefill (default chunk 2048) yields between chunks: cancellation checks, governor
  re-admission per chunk, and (post-v1) decode interleaving points.

## Native boundary (C ABI — Helios shape, slimmed)

Carried verbatim from the proven Helios/bonsai pattern:

- Opaque handles: `HypModel`, `HypKvState`, `HypDrafter`, `HypStepResult` (magic-tagged).
- Every function returns `HypStatus` (OK / INVALID_ARG / NOT_FOUND / IO / OOM_GOVERNOR /
  UNSUPPORTED / INTERNAL / CANCELLED); `hyp_last_error()` returns a thread-local message.
  No C++ exception crosses; no Rust panic crosses; every handle has create/free lifecycle
  tests including double-free and null.
- Synchronous pull-step calls (`hyp_prefill_chunk`, `hyp_decode_block`) filling caller-owned
  out-params. No callbacks across the ABI. Streaming is composed Rust-side from step results.
- Rich telemetry struct per step (Helios `Gemma4DecodeProfileInfo` heritage, slimmed):
  peak/active MLX bytes, OS phys_footprint, per-layer-type KV bytes, kv-eval ms split
  (global vs local), governor state, and (M7) draft/accept counts.
- Target ABI surface ≤ 25 functions. Additions require an ADR.

## Startup canary (fail loudly)

At `hyp_model_load`: assert Apple GPU generation ≥ M5 family and macOS ≥ 26.2 (NAX
availability per Apple's MLX-on-M5 publication), assert MLX version == pinned 0.32.0, run a
1-token smoke forward, record `recommendedMaxWorkingSetSize` + derive the budget, and emit a
single MEASURED canary line. Unsupported hardware exits nonzero with a one-line reason. No
fallback ladder.

## Config-driven geometry (family support without family sprawl)

`hyperion-model` parses HF `config.json` into a validated `Geometry` struct that covers all
five Gemma 4 sizes (see references/gemma4-family-facts.md):

```
layers, hidden, intermediate, vocab=262144, heads_q, kv_heads_local,
head_dim_local=256, head_dim_global=512, kv_heads_global, attention_k_eq_v,
layer_types[..] (sliding|full, from pattern; last layer always full),
sliding_window (512|1024), rope_local{theta=1e4, full-dim},
rope_global{theta=1e6, partial=0.25}, final_logit_softcapping=30.0,
num_kv_shared_layers (E-series), ple{hidden_per_layer_input} (E-series),
moe{num_experts=128, top_k=8, moe_intermediate} (26B-A4B), max_position_embeddings
```

Unsupported/unknown config fields fail loudly (no silent default-through — Helios lesson).
The native graph is built FROM `Geometry`; there is no per-model C++ fork. MoE + PLE +
shared-KV wiring have geometry-level tests in v1 (M8 runs E4B end-to-end; 26B/31B stay
geometry-validated only, O-8).

Per-layer-kind dispatch (mask / cache-kind / layer-runner) is resolved **once per kind**, not
per token — the pattern NunSpark's `ArchSpec`/`MaskPlan`/`LayerRunner` seam independently
validates (B2). Two disciplines are load-bearing for Gemma's hybrid layout and are adopted
explicitly:

- **Masks are built per layer *kind*, from that kind's own cache (A3).** Build one mask for the
  sliding kind and one for the global kind; source each from the **first layer of that kind**,
  never layer 0 unconditionally. A rotating (sliding) cache clamps its offset to `window-1`, so
  building a global-attention mask from a sliding layer's cache silently truncates it once the
  sequence exceeds the window — a real crash (`broadcast_shapes`) NunSpark hit twice. This is a
  correctness rule for the mask-construction code, tested at M2.
- **Stash cross-layer KV only when a downstream layer or the MTP drafter actually consumes it**
  — Gemma shares K/V across many layers; materializing and pinning every layer's KV wastes the
  16 GB budget. The layer runner captures a producer layer's `(k,v)` only when `Geometry` says
  something reads it (the drafter's `store_full_length_kv` target, per attention type).

## Determinism posture

Greedy mode must be run-to-run deterministic on the same machine/build (single stream, no
batch-size-dependent reduction switches — we control the kernel dispatch). Sampled mode uses
a seeded RNG recorded per request. This is the runtime's reproducibility posture; document any
MLX-internal nondeterminism found rather than papering over it.
