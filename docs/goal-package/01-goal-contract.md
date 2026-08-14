# 01 — Goal contract

Normative scope: [ADR 0006](../decisions/0006-dual-family-v1-scope-and-gates.md). Historical
Gemma evidence remains valid; the old Gemma-only forward scope does not.

## North star

**A native, evidence-driven Apple-silicon inference runtime where Gemma 4 and Qwen's hybrid
`qwen3_5` architecture are first-class model families rather than accidental variants of one
graph.**

Hyperion targets demanding local agentic workloads on the 16 GB MacBook Pro M5: real streaming,
strict tool behavior, bounded memory, checkpoint-correct conversations, and honest quality/
latency evidence. Rust owns serving and policy; one C++/MLX runtime owns model execution behind a
narrow C ABI. Python and third-party reference runtimes remain oracle/conversion tools only.

## Why clean-sheet

Helios/gemma4d proved Gemma math but its re-traced execution, concatenate-grown global KV,
deferred evaluation, and hot-loop peak resets produced severe decode tails and a 16K memory
cliff. mlx-bonsai contributed strong serving/governor/evidence machinery, while its tested
ternary Qwen3.6 artifact failed the target agentic quality bar. That artifact result does not
rule out the independently released Qwen3.8 dense hybrid architecture.

The dual-family direction keeps proven mechanisms and rejects inherited family assumptions.
Because the project is pre-1.0, private geometry, ABI, state, and artifact layouts are replaced
when a cleaner design wins; no migration shims are required for experimental internals.

## V1 scope

| Family/profile | V1 role | Evidence state at ADR 0006 |
|---|---|---|
| Gemma 4 12B QAT Q4 | Current executable regression path | M0 accepted; native/product acceptance beyond M0 pending |
| Gemma 4 E4B and other sizes | Preserve config-driven family capability; gate only with real evidence | Existing geometry/converted E4B evidence retained; no blanket acceptance |
| `Qwen/Qwen3.8-27B` text at pinned revision | New native target: batch one, 16K total cached tokens, 16 GB M5 | Research/design and family recognition only initially |
| Qwen vision and bundled MTP | Optional component graph | Post-V1 |

For Qwen, 16K means rendered prompt + retained reasoning + answer + tool-loop history + reserved
output. A request is admitted only after the output reserve and full live-memory equation fit.

## Hard constraints

1. **Platform floor:** Apple M5 generation, macOS 26.2+, MLX 0.32.0 exactly. No M1–M4 fallback
   ladder. Any upgrade is an independently pinned, evidence-gated decision.
2. **One native serving path:** Rust policy and one C++/MLX runtime; one MLX-owning engine thread;
   narrow C ABI; no Python or helper subprocess on the request path; test doubles stay in tests.
3. **Family semantics are separate:** architecture adapters own config/tensor/graph/state math;
   checkpoint conversation profiles own tokenization/templates/stops/thinking/tools/defaults;
   deployment profiles own transformed artifacts, cache formats, limits, and residency.
4. **16 GB is binding:** use the device-derived effective budget and a fail-closed governor.
   Predictions include weights, committed and staged state, KV, scratch, compiled resources,
   allocator/server footprint, and reserve. Zero uncontrolled OOM is a standing gate.
5. **Resident dense decode:** ordinary token-by-token decode cannot reread the full model from
   SSD or rely on macOS swap. SSD is allowed for bounded conversion and cold components;
   durable snapshots and named prefill/offline experiments are separately gated P8 work.
6. **Transactional state:** Qwen recurrent and KV mutation, Gemma KV mutation, cancellation,
   speculative work, and reload publish state only after successful evaluation. Identity mismatch
   resets or recomputes state; no cross-family/profile reuse.
7. **Correctness before speed:** pinned reference traces, real artifacts where feasible,
   two-sided numeric gates, free-running behavior, and hostile-input tests outrank throughput.
8. **Baseline before claims:** performance gates use preregistered A-arms, A-C-C-A ordering, and
   non-overlapping promotion evidence. The deferred Gemma M1 outcome and future Qwen A-arm are
   reported honestly.
9. **Evidence identities:** every result binds artifact, conversation, deployment, state layout,
   runtime/kernel, oracle, harness, corpus, machine state, and exact command.
10. **Public contracts remain contracts:** greenfield freedom does not weaken API behavior,
    security, artifact integrity, licensing, privacy, or user-data boundaries.

## Family-specific initial posture

### Gemma 4

Preserve the implemented sliding/global attention graph, K=V and shared-KV capabilities, PLE/MoE
geometry, Gemma tokenizer/template/tool behavior, and current regression fixtures. These are
Gemma capabilities, not universal fields every architecture must implement.

### Qwen3.8 / `qwen3_5`

- text graph: 48 Gated DeltaNet recurrent layers + 16 gated full-GQA layers;
- recurrent state: FP32 initially, explicit convolution state, transactional commit;
- cache: KV only for the 16 full-attention layers, BF16 correctness control;
- weights: resident affine Q2/mixed scalar controls first; learned scalar quantization next;
- conversation: exact pinned Qwen tokenizer/template, thinking/effort/stops/tool grammar;
- conversion: shard/layer bounded, resumable, hash-verifying, and atomically published.

No Qwen field is defaulted into the Gemma geometry and no Gemma parser interprets Qwen output.

## Non-goals for V1

- Private ABI, geometry, cache, or artifact backward compatibility.
- Qwen vision or either family's MTP as a V1 dependency.
- Dynamic routed adapters/MixLoRA.
- A Qwen 32K, 262K, or 1M local promise.
- Transparent macOS swap or full-model SSD reads during normal dense decode.
- Simultaneous residence of both large model artifacts on the 16 GB target.
- Production internet-facing multi-tenancy, continuous batching, TUI, CUDA, or every model family.
- Treating file-size arithmetic, aggregate perplexity, implementation, or family recognition as
  accepted support/performance evidence.

## Definition of done

V1 is complete when P0–P4 and P6–P7 in `10-milestones-and-gates.md` are accepted. P5 acceptance
is never required. If its optional custom-kernel lane is opened, a promote-or-park disposition
with evidence is required before P7 closes; an unopened P5 needs no record.

- pinned Gemma and Qwen text artifacts execute through clean family adapters behind the shared
  serving/runtime policy;
- checkpoint-specific conversation profiles reproduce stops, thinking, tools, and transcripts;
- the selected Qwen artifact fits the complete 16K target under measured M5 admission with no
  uncontrolled OOM or swap dependence;
- promoted packed weight/cache formats are consumed directly by production kernels;
- transactional state, cancellation, switching, isolation, and hostile-input gates pass;
- performance and quality claims have preregistered, repeatable, identity-bound evidence; and
- tracked specifications, ADRs, public documentation, and release behavior match implementation.

A custom weight format, compressed KV, speculative decoding, vision, routed adapters, and
longer contexts are not required if the simpler resident path meets these gates.
