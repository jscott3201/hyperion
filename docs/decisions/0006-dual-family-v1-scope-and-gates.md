# ADR 0006: dual-family V1 scope and architecture-first sequence

- **Status:** accepted
- **Date:** 2026-08-14
- **Owners:** Hyperion team
- **Supersedes:** the Gemma-only forward scope and fixed M0–M9 future sequence in the v0.2 goal package
- **Preserves:** ADRs 0001–0005, accepted historical evidence, public API contracts, and all still-applicable correctness/security rules
- **Related:** `docs/goal-package/01-goal-contract.md`, `docs/goal-package/10-milestones-and-gates.md`, `artifacts/models/MANIFEST.md`

## Context

Hyperion began as a Gemma 4-only runtime. The implemented Gemma graph, native boundary,
serving core, and evidence discipline remain useful, but the owner has selected Qwen's hybrid
`qwen3_5` architecture—beginning with the official Qwen3.8-27B text checkpoint—as a second
first-class family.

This is not an artifact-only extension. Qwen3.8 has 48 Gated DeltaNet recurrent layers and
16 gated full-attention layers, untied embedding/head tensors, a different cache topology,
different model math, and a checkpoint-specific thinking/tool protocol. Routing it through
Gemma geometry, KV, or conversation code would create false support and corrupt state.

The project is pre-1.0 and greenfield internally. Private Rust types, the C ABI, artifact
schemas, state layouts, and milestone names may change rather than accumulate compatibility
shims. Public request/response behavior, security boundaries, artifact integrity, licensing,
and evidence claims remain contracts.

## Current gate truth

This decision does not retroactively accept work:

- M0 is accepted.
- ADR 0002 accepts the M1 measurement **protocol**, not its measurement outcome. M1 remains
  deferred and unaccepted.
- Substantial Gemma M2/M3 code exists, but neither milestone has an acceptance ADR.
- ADRs 0003 and 0004 accept component-level ABI/streaming decisions, not M3 as a whole.
- The latest recorded governor calibration fails G4. There is no accepted native performance
  claim.
- Qwen is research/design only at this decision point and is not loadable by the native runtime.

Implementation status and milestone acceptance remain separate in public and internal docs.

## Decision

### 1. V1 families and target profile

Gemma 4 and Qwen's `qwen3_5` hybrid architecture are first-class families behind one serving
runtime. The initial Qwen checkpoint is:

- repository: `Qwen/Qwen3.8-27B`;
- immutable revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- component: text-only; vision and bundled MTP are omitted initially;
- workload: batch one, interactive generation;
- local envelope: 16K total cached tokens on the 16 GB M5, including rendered prompt,
  retained reasoning, answer, tool-loop history, and reserved output.

Gemma 4 remains the current executable regression path; native/product acceptance beyond M0 is
still pending. The existing pinned 12B QAT Q4 artifact is the first real model driven through the
new architecture seam.

### 2. Separate identities and semantics

Hyperion separates:

- an **architecture adapter**: config validation, tensor plan, graph, state topology, and
  capabilities;
- a **conversation profile**: tokenizer/template hashes, stops, thinking/tool grammar,
  parsing, and generation defaults for an exact checkpoint;
- a **deployment profile**: transformed artifact, weight/cache formats, context/chunk limits,
  residency, and admission reserve;
- an immutable **artifact manifest**, an **evidence record**, and a **state/snapshot identity**.

Shared code owns request lifecycle, scheduling, sampling policy, cancellation, streaming,
telemetry, and evidence plumbing. Family code owns model math and state semantics. No optional-
field superset geometry may force Qwen and Gemma through one graph description.

### 3. Memory and optimization posture

- Ordinary dense decode keeps the selected model artifact resident. It does not reread full
  model weights from SSD per token.
- Qwen Gated DeltaNet recurrent matrices remain FP32 initially and all recurrent/KV mutation is
  transactional: a failed or cancelled evaluation cannot publish partial state.
- BF16 KV is the Qwen correctness control. Cache compression is conditional on measured fit,
  quality, and target-device latency.
- Native affine Q2 and sensitivity-driven mixed scalar precision are the initial capacity and
  quality controls. Learned scalar methods follow; custom Metal weight formats are conditional.
- Conversion must be shard/layer bounded, resumable, hash-verifying, and crash-atomic because
  the source checkpoint cannot coexist casually with all outputs on the target machine.
- MLX 0.32.0 remains the production control. Newer commits are separately pinned experiments.

### 4. Dependency sequence

The active program is:

1. **P0 — contract and evidence reset:** land this scope, preserve gate truth, and freeze the
   current Gemma regression identity.
2. **P1 — architecture seam and Gemma regression:** establish family-aware load dispatch and the
   minimal architecture/conversation/deployment identity contracts; route Gemma unchanged. Qwen's
   concrete conversation and deployment implementations land only when their P2/P3 inputs exist.
3. **P2 — differential Qwen oracle and streamed converter:** may begin after P0 in parallel
   with P1; pin independent Transformers and mlx-lm references and build bounded conversion.
4. **P3 — native Qwen correctness:** requires P1 and P2; implement the exact hybrid graph and
   transactional recurrent/KV state before extreme compression.
5. **P4 — scalar quantization and measured fit:** select a resident artifact on a declared
   quality/bytes/latency frontier and prove the complete 16K envelope.
6. **P5 — optional custom kernels:** a bounded experiment after P4, opened only for a separately
   declared improvement target. Whether unopened, promoted, or parked after a losing experiment,
   P5 acceptance is not a V1 dependency; an opened lane requires a recorded disposition.
7. **P6 — context/cache promotion:** BF16 first; compressed K/V only if evidence requires it.
   Session-scoped in-memory exact-prefix reuse may be evaluated here; durable/persistent snapshots
   remain post-V1.
8. **P7 — dual-family hardening and acceptance:** real-artifact suites, switching/isolation,
   hostile-input tests, documentation, and evidence closure.

Runtime SSD prefill/partial-residency experiments, durable/persistent state snapshots, vision,
MTP, routed adapters, and contexts beyond the accepted 16K Qwen profile are post-V1 lanes.

P1's first slice deliberately changes only Rust load dispatch. It recognizes the exact
`qwen3_5` discriminator but returns a typed recognized-but-unavailable error before tokenizer,
conversation, or native initialization. The current `HypGeometryParams` and C++ graph remain
Gemma-specific until Qwen geometry and state traces are frozen. Recognition is not support.

## Promotion gates

At minimum, the program requires:

- pinned real-Gemma token/logit/template/state regression through the family seam;
- fail-closed architecture detection with no cross-family semantic leakage;
- pinned Transformers-versus-mlx-lm Qwen BF16 traces for input IDs, selected layer/state
  slices, logits, and free-running generation;
- checkpoint-specific thinking, stop, tool-loop, consecutive-result, and parser-split fixtures;
- converter inventories, hashes, atomic publication, bounded memory, and hostile-manifest tests;
- GDN step/chunk parity, commit/abort/cancel/replay, finite-state, and memory-slope checks;
- direct consumption of packed quantized weights without full/layer FP reconstruction;
- measured admission of the full 16K envelope, including reserved output, with zero
  uncontrolled OOM or reliance on swap; and
- evidence bound to artifact, conversation, deployment, state-layout, runtime, kernel, oracle,
  harness, corpus, and machine identities.

Qwen performance needs its own preregistered A-arm. The deferred Gemma M1 outcome blocks Gemma
relative performance claims, not correctness-oriented implementation work.

## Explicit non-goals for V1

- Qwen vision or bundled MTP as a dependency.
- Gemma MTP as a V1 dependency.
- Dynamic routed adapters or MixLoRA serving.
- A Qwen 32K, 262K, or 1M local support promise.
- Transparent macOS swap or per-token full-model SSD reads.
- Simultaneous residency of both large model artifacts on the 16 GB target.
- Compatibility parsing for obsolete private experimental schemas or ABI layouts.

## Consequences

- Living goal-package files and `AGENTS.md` are updated to this dependency sequence. The old
  M0–M9 text remains historical evidence only where explicitly labeled.
- Existing accepted ADRs remain immutable and valid within their scopes.
- Family recognition, executable support, and accepted evidence are reported separately.
- Conversation-profile and family-specific artifact-manifest work follow the load dispatcher;
  this ADR does not claim they are implemented.
- Losing experimental formats are removed rather than retained as compatibility burden.
