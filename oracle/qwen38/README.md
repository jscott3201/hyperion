# Qwen3.8 oracle contract

This directory contains the model-free P2 oracle foundation. It freezes inputs, raw trace
semantics, verifier-owned state normalization, and the two producer recipes without claiming that
either reference has run.

Current state: **contract frozen, unexecuted; model-free I/O foundation implemented**.

- No Qwen weights have been acquired or locally verified.
- No complete Transformers environment or producer entrypoint exists. The existing `oracle`
  environment is only an unexecuted mlx-lm candidate and must not impersonate both arms.
- No Transformers or mlx-lm real-weight trace has been produced.
- No numeric tolerance or cross-reference agreement has been measured.
- The v1 bundle schema is raw evidence only. It deliberately has no comparison-report artifact;
  tolerances and a sealed exhaustive comparison report require clean repeats, fault injection, and
  a reviewed superseding contract.
- Qwen remains unavailable to the native runtime and is not an accepted artifact or supported
  model.

## Files

- `source-manifest.json` records the complete 32-file expected inventory of
  `Qwen/Qwen3.8-27B` at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Small Git-backed files were content-hashed;
  large-file hashes and sizes come from the revision-pinned Hugging Face API. This is an expected
  identity, not evidence that the 55,586,114,863-byte tree has been downloaded and rehashed.
- `cases.jsonl` contains only first-party prompts, standard Transformers function/tool wrappers,
  transcript inputs, bounded 16-step generation cases, and parser probes. It contains no
  rendered template output, model tokenization, logits, states, or generated text.
- `trace-schema.json` freezes the bounded run matrix, prediction/cache frame semantics, selected
  layers and positions, native tensor layouts, top-32 diagnostics, metric formulas, closed event
  and payload records, and headerless little-endian payload grammar. It also promotes selected
  render inputs into bounded free-running thinking, tool, and preserved-history probes without
  inventing expected model output. Parser outcomes remain verifier-owned outside both raw arm
  roots rather than being self-attested by either model producer.
- `producer-contracts.json` freezes the direct-forward modes, loader/tokenizer provenance,
  parameter-closure algorithm, and closed receipt value contracts. Transformers uses the pinned
  pure-Torch GDN/conv fallbacks with optional kernel packages absent; mlx-lm verifies the full
  unmodified checkpoint tree, then uses its pinned sanitizer and stock Metal GDN kernel. Their
  environment and executable identities remain unbuilt.
- `contract.json` supersedes and hash-binds the reviewed v1 input-only contract, both new files,
  the two source implementations, and all non-claim statuses.
- `../qwen38_contract.py` validates the committed contract and can verify a future complete source
  tree. It uses only the Python standard library and fails on missing inputs rather than skipping.
- `../qwen38_evidence_io.py` implements the frozen publication requirements as model-free
  infrastructure: canonical content-addressed tree inventory, identity-bound descriptor hashing,
  exact final-marker creation, file and bottom-up directory fsync, and same-filesystem atomic
  no-replace publication (`renameatx_np(RENAME_EXCL)` on macOS, `renameat2(RENAME_NOREPLACE)` on
  Linux, typed failure elsewhere). It is a publication primitive only. It is not the raw-bundle
  semantic verifier, emits no detached seal, and authors no pass, agreement, acceptance, or
  support field.

The two pinned semantic modes are:

- Hugging Face Transformers commit
  `95940bf8775059a42f047256f076e4f607bc43ec`, with hub kernels disabled before import, FLA,
  `causal_conv1d`, `kernels`, FlashAttention, and xFormers absent, eager full attention,
  local-only BF16 loading, and direct fixed-step greedy calls.
- mlx-lm commit `8239c72de5a0e42c539e30489021db73c7fe258c` on MLX 0.32.0, using its stock Qwen3.5 text
  graph and fixed-step greedy forward calls.

Stock calls cannot expose every selected intermediate site. Instrumented Transformers runs use
temporary read-only hooks that are removed on every exit; instrumented mlx-lm runs use an explicit
pinned text-graph orchestration adapter because its public model call returns only logits. A fresh
stock-versus-instrumented control binds generation frames and termination before either route may
contribute evidence. Neither adapter has been implemented or executed yet.

These are frozen recipes, not executed or accepted environment identities. Strict parameter
closure and the independent raw-bundle verifier are specified but not implemented, so execution
remains forbidden until those controls exist. The atomic no-replace publisher they require is now
implemented and tested as model-free infrastructure. The Transformers arm
still needs an independently hashed Python/Torch/CUDA lock and an 80 GB-class device; the mlx-lm
candidate still needs its selected MLX/Metal wheel, high-memory Apple host, package-tree receipt,
and producer entrypoint. The local 16 GB M5 cannot host the resident BF16 oracle. Both arms use the
checkpoint tokenizer and template, so matching rendered prompts are a shared-dependency check—not
an independent vote on model math.

Each arm is an orchestrator receipt plus one input-preparation child and twelve fresh model-run
children. The receipt inventories a static 128-leaf cache topology; call-specific offsets,
capacities, shapes, and bytes belong only to the joined event and payload records. Environment
receipts close the canonical site-packages and standard-library trees, import origins, static
forward signature, per-call/per-layer kernel routes, and source-to-runtime parameter mapping.
This is provenance on a trusted isolated runner, not remote hardware attestation: the source and
environment must be immutable while each child runs, and the external verifier recomputes every
identity it can observe.

The three stop-related case coverage labels describe configured stop-policy inputs, not observed
termination. Actual stop-token versus 16-step-bound behavior is recorded per call and run. Memory
and swap receipts are likewise diagnostic and arm-local: they use each accelerator allocator plus
host-global swap sampling and are never treated as cross-arm acceptance metrics.

Raw state stays runtime-native. The independent verifier must transpose mlx-lm recurrent state,
reduce Transformers' four-token convolution cache to the comparable three-token history, and
slice mlx-lm's allocated KV backing to its receipted logical offset. The backing capacity is not
assumed to be a multiple of 256 after nonaligned chunk appends; it is recomputed from the exact
call schedule using mlx-lm's step-256 growth/truncation recurrence. Qwen's equal 128-wide key and
value features make the recurrent transpose invisible to shape checks; asymmetric synthetic
controls make that mapping testable.

## Model-free verification

```sh
python3 -B oracle/qwen38_contract.py
python3 -B oracle/test_qwen38_contract.py
python3 -B oracle/test_qwen38_evidence_io.py
```

On a high-memory machine with the complete, cache-free source tree, verify every source byte with:

```sh
python3 -B oracle/qwen38_contract.py --source-root /absolute/path/to/Qwen3.8-27B
```

That command rejects extra or missing files, path aliases, symlinks, hard links, size drift, and
hash drift. A Hugging Face transport cache must stay outside the verified payload root.

## Next protected slice

Build the independent raw-bundle verifier on top of the accepted I/O foundation: it must validate
arm layout, events, payload manifests, and receipts, publish verifier-owned parser outcomes, and
create the detached seal outside both arm roots, publishing only fully valid bundles through the
tested atomic no-replace primitive. The separate Transformers environment and both producer
entrypoints follow. Each producer must verify the source before and after execution, load locally
with no mutable network fallback, write its own raw outputs and receipt, and never consume the
other producer's output or normalization. The verifier will derive clean-versus-fault tolerances
after repeat runs. The bundle is valid only after an exact detached inventory digest covers both
raw arms and shared verifier-owned parser results, followed by tested atomic publication;
producer-authored pass or agreement fields are forbidden.
