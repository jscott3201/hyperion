# Kickoff prompt (Claude Code / Codex goal)

Use after reading [ADR 0006](../decisions/0006-dual-family-v1-scope-and-gates.md),
`AGENTS.md`, and `01-goal-contract.md`. Follow the dependency DAG in
`10-milestones-and-gates.md`; implementation is not acceptance.

```text
/goal Implement Hyperion, a clean-sheet Rust 1.95 + MLX 0.32.0 inference runtime for Gemma 4
and Qwen's qwen3_5 hybrid architecture on Apple M5/macOS 26.2+, with the 16 GB MacBook Pro M5
as the binding profile. Preserve the implemented Gemma path through a family-specific adapter;
add the pinned Qwen/Qwen3.8-27B text path as a separate 48-Gated-DeltaNet/16-full-GQA graph.

Keep one native C++/MLX serving runtime, one MLX-owning engine thread, a narrow C ABI, no Python
or helper subprocess on the request path, real incremental SSE, bounded single-flight admission,
and a fail-closed memory governor. Architecture adapters own config/tensors/graph/state;
checkpoint conversation profiles own tokenizer/template/stops/thinking/tools/defaults;
deployment profiles own transformed artifacts/cache/limits/residency. Never pass Qwen through
Gemma geometry or conversation code.

For Qwen V1, target text-only batch-one inference with 16K total cached tokens—including prompt,
reasoning, answer, tool history, and reserved output—on 16 GB. Keep Gated DeltaNet recurrent
matrices FP32 and state transactional. Use BF16 KV as the correctness control and resident affine
Q2/mixed scalar weights as the capacity controls. Custom weight formats and compressed KV are
conditional on measured quality, memory, and latency. Stream conversion by shard/layer with
hashes, resumability, and atomic publication; do not stream the full dense model from SSD per
decode token or rely on macOS swap.

Correctness outranks speed. Pin independent Transformers and mlx-lm Qwen references, compare
state/layer/logit/free-generation traces, and bind evidence to artifact, conversation,
deployment, state-layout, runtime/kernel, oracle, harness, machine, and command identities.
Use real weights where feasible, preserve append-only evidence, and never relabel implementation
or family recognition as accepted support. Start with the next dependency-ready phase in
10-milestones-and-gates.md and end accepted phases with a numbered ADR.
```

## Current pickup checklist

1. Verify branch/status, read ADRs and the current gate-truth table in `10`.
2. Confirm M5/macOS/MLX/Rust pins before native or measured work.
3. Preserve the pinned Gemma 12B regression artifact in `artifacts/models/MANIFEST.md`.
4. Do not download Qwen3.8's 55.56 GB source without checking storage; use an external volume or
   remote/high-memory conversion host and a bounded converter.
5. Keep oracle/conversion environments isolated from serving.
6. Run the narrowest relevant model-free checks on PRs and the complete real-model/M5 gates only
   in the protected release/measurement environment.
