# 10 — Active phases and gates

[ADR 0006](../decisions/0006-dual-family-v1-scope-and-gates.md) supersedes the old Gemma-only
M0–M9 forward plan. Historical outcomes remain valid; they do not silently become acceptance
for this program.

Rules: follow the dependency DAG below. A phase is accepted only on concrete evidence—tests,
ledger rows, and a numbered decision record. Real weights/tensors are required wherever
feasible. Implementation does not imply acceptance.

```text
P0 contract/evidence reset
 ├── P1 family seam + Gemma regression ──┐
 └── P2 Qwen oracle + converter ─────────┴── P3 native Qwen correctness
                                                   |
                                             P4 scalar quant + fit
                                                   |
                                      P6 context/cache ──── P7 dual-family acceptance

Optional side lane after P4: P5 custom kernels. If opened, promote or park before P7 closes.

P8 storage/optional components is post-V1.
```

P2 may proceed in parallel with P1 after P0. P3 requires both. P5 acceptance is never mandatory;
it may open only after P4 for a separately declared improvement target. P7 requires a P5
promote-or-park disposition only when that lane was opened.

## Current gate truth

| Historical item | Status after ADR 0006 |
|---|---|
| M0 bootstrap/canary | **Accepted**; retained as platform/reproducibility baseline. |
| M1 | ADR 0002 protocol accepted; measurement outcome deferred and **not accepted**. |
| Gemma M2/M3 implementation | Substantial code exists; no milestone-acceptance ADR. |
| ADRs 0003/0004 | Component ABI/streaming decisions accepted; not whole-M3 acceptance. |
| Latest governor evidence | Fails G4; no accepted native performance claim. |
| Old M4–M9 future plan | No longer the active V1 sequence; useful Gemma work is rescheduled below. |
| Qwen | Research/design and architecture recognition only; not native-loadable. |

The pinned Gemma 12B artifact remains the regression baseline. Existing append-only evidence is
not edited or relabeled.

## P0 — Contract and evidence reset

Outcome: contributors see one owner-approved scope and one honest acceptance account.

- Accept ADR 0006 and reconcile the operating contract, goal package, public README, model
  manifest, and risk register.
- Preserve the pinned Gemma 12B artifact and its accepted M0 identities.
- Pin Qwen3.8 source identity without claiming download, conversion, or support.
- Define architecture, conversation, deployment, artifact, evidence, and state identities as
  separate concerns.
- Preserve one native backend, one MLX-owning engine thread, bounded single-flight serving,
  fail-closed admission, exact dependency pins, and append-only evidence.

Gate: scope files agree; current status matches accepted ADRs/ledger; no Qwen support claim is
made by recognition alone.

## P1 — Architecture seam and Gemma regression

Outcome: shared policy no longer assumes one family, while Gemma behavior is unchanged.

- Add strict top-level family dispatch. Recognized-but-unimplemented families fail before
  tokenizer/template loading or native initialization.
- Route the pinned Gemma artifact through a family-specific config/load branch.
- Keep the current C ABI and native graph explicitly Gemma-specific until Qwen traces are frozen.
- Define the minimal architecture, conversation, and deployment identity contracts now. Keep the
  current Gemma implementations concrete; add Qwen's concrete conversation/deployment consumers
  only after their pinned P2/P3 inputs exist. Do not prebuild unused feature abstractions.
- Define model switching as drain → synchronize → unload → validate/load → publish, with one
  large model resident and clean recovery on failure. Reload implementation may be a later P1
  slice; its semantics are fixed here.

Gate: strict classifier tests cover known/missing/unknown/conflicting types; the separate identity
contracts are explicit; ABI surface is unchanged; model-free suites pass; pinned real-Gemma
tokens/logits/template/state match their pre-seam controls. Qwen returns a typed unavailable
result before any Gemma semantics run.

## P2 — Differential Qwen oracle and streamed conversion

Outcome: exact, reproducible Qwen references and artifacts can be produced without whole-model
local residency.

- Freeze `Qwen/Qwen3.8-27B` at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and verify source identities.
- Pin exact Transformers and mlx-lm revisions and execution modes; neither is infallible.
- Compare input IDs, selected layer/state slices, logits/top-k traces, and free-running output.
- Cover thinking on/off and effort, both stop IDs, preserved-thinking multi-turn history, tools,
  consecutive tool results, and parser boundary cases.
- Build shard/layer-at-a-time conversion: verify → transform → write → hash → atomically publish
  manifest. Record consumed, omitted, shared, and rejected tensors; resume safely after failure.
- Keep conversion/calibration Python and third-party code outside the serving path.

Gate: both references agree within frozen tolerances on the declared corpus; a bounded smoke
artifact round-trips with exact inventories/hashes; incomplete output is never accepted.

## P3 — Native Qwen correctness

Outcome: the native text graph matches P2 before extreme compression.

- Implement the 48 Gated DeltaNet plus 16 gated-GQA layers, partial RoPE, q/k norms, SiLU MLP,
  untied embedding/head, and exact family-specific tensor mapping.
- Keep recurrent matrices FP32 and convolution state explicit.
- Stage recurrent and KV changes and publish only after successful MLX evaluation. Cancellation,
  failure, or rejected work leaves the last committed state reusable.
- Implement the pinned Qwen conversation profile independently of Gemma rendering/parsing.
- Test token-step versus chunked prefill, reset/commit/abort/replay, long-run finite/state-norm
  envelopes, memory slope, layer outputs, logits, and free-running generation.

Gate: P2 tolerances hold; state is transactionally correct; no unintended state-memory slope or
uncontrolled memory event; tools/thinking/stops match the pinned profile.

## P4 — Scalar quantization and measured fit

Outcome: select the lowest-risk resident artifact on a measured quality/bytes/latency frontier.

- Controls: BF16 teacher remotely, Q3 group-128 diagnostic, affine Q2 group 64/128 capacity arms.
- Candidates: per-tensor mixed Q2/Q3/Q4/Q5/Q6, sensitivity/DWQ/GPTQ/AWQ signals, then
  GSQ-compatible scalar Q2/Q3.
- Initially preserve norms, `A_log`, `dt_bias`, convolution vectors, and recurrent runtime state.
- Use disjoint pinned calibration/evaluation sets and pair likelihood/logit metrics with
  free-running reasoning, code, instruction, structured-output, tool, and long-context behavior.
- Measure artifact + recurrent/cache + scratch + compiled resources + allocator/server + reserve.

Gate: selected packed artifact is consumed directly; no full/layer FP reconstruction; complete
16K request including reserved output remains inside the measured governor budget without swap
dependence; zero uncontrolled OOM; cold/compile/warm/thermal results and paired quality bounds are
recorded. Qwen performance has a preregistered A-arm.

## P5 — Optional custom weight kernels

Outcome: an optional bounded experiment either beats the accepted scalar path or is removed.

- First spike a scale-only scalar Q2 Metal QMV on the byte-weighted Qwen matrix mix.
- Benchmark `M=1,2,3,4,8`, embedding lookup, and the LM head independently.
- Only if needed, compare a bounded codebook shortlist. Keep per-tensor scalar fallback and never
  reconstruct full/layer floating weights.

Disposition, only if opened: promote only when a predeclared end-to-end target and quality bound
are both met within bounded scratch. One synthetic spike plus one full conversion round may fail;
record and remove a losing format. A parked or unopened P5 does not block V1.

## P6 — Context and cache

Outcome: promote only cache mechanisms that materially improve the accepted envelope.

- BF16 KV is the correctness control and may be the V1 choice.
- If required, compare delayed MLX Q4, then fused independently selectable K8/V4 and K8/V3;
  more aggressive key/predictor codecs come later.
- Keep GDN recurrent state outside KV codecs.
- Admission counts rendered prompt, retained reasoning/history, requested output, staged state,
  scratch, allocator/cache, server, and reserve.
- Any session-scoped in-memory exact-prefix reuse binds artifact, conversation, deployment, and
  state-layout identities; owner scope and whole-state invalidation are mandatory. It is distinct
  from durable or SSD-backed snapshots.

Gate: full 16K behavior fits and preserves paired instruction, reasoning, retrieval, tool, and
security outcomes. A compressed codec promotes only with direct packed attention consumption and
measured memory/latency value.

## P7 — Dual-family hardening and acceptance

Outcome: supported claims match evidence for both families.

- Run family-specific real-artifact correctness suites and shared serving, auth, cancellation,
  reload, hostile-input, memory-governor, and long-run suites.
- Prove atomic model switching, no cross-family state/profile reuse, and bounded unload/reload.
- Bind every result to artifact, conversation, deployment, state-layout, runtime/kernel, oracle,
  harness, corpus, and machine identities.
- Update public docs and release packaging only for implemented, accepted capabilities.

Gate: each advertised artifact passes its own suite and all shared operational gates; a numbered
acceptance ADR and evidence rows exist; no deferred feature is implied by family recognition.

## P8 — Post-V1 storage and optional components

Named lanes: SSD layer-major prefill/partial residency for explicit offline or batched workloads,
durable/persistent state snapshots, Qwen vision, bundled Qwen MTP, Gemma MTP, routed adapters,
32K+ Qwen profiles, continuous batching, and additional platforms.

Ordinary dense token-by-token decode must not depend on rereading the full model from SSD. Each
lane needs its own workload hypothesis, evidence protocol, and decision record before promotion.
