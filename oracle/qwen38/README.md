# Qwen3.8 oracle contract

This directory is the model-free first slice of P2. It freezes the inputs and evidence shape for
the future Qwen3.8 differential oracle without claiming that either reference has run.

Current state: **contract only, unexecuted**.

- No Qwen weights have been acquired or locally verified.
- No Transformers or mlx-lm real-weight trace has been produced.
- No numeric tolerance or cross-reference agreement has been measured.
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
- `contract.json` pins the two implementation sources and preregisters required trace-channel
  classes, boundaries, evidence identities, and anti-self-attestation rules. Exact tensor
  selections, layouts, frames, and serialization remain intentionally unfrozen.
- `../qwen38_contract.py` validates the committed contract and can verify a future complete source
  tree. It uses only the Python standard library and fails on missing inputs rather than skipping.

The two pinned implementation candidates have this partial execution intent:

- Hugging Face Transformers commit
  `95940bf8775059a42f047256f076e4f607bc43ec`, with hub kernels disabled, eager full attention,
  local-only loading, BF16 weights, and fixed-step greedy forward calls.
- mlx-lm commit `8239c72de5a0e42c539e30489021db73c7fe258c` on MLX 0.32.0, using its stock Qwen3.5 text
  graph and fixed-step greedy forward calls.

These are source pins, not accepted execution modes or environment identities. Device/runtime,
cache and chunk policy, optional kernel availability, deterministic controls, producer call
shape, and executable/package-tree receipts still need to be frozen independently. Both arms use
the checkpoint tokenizer and template, so matching rendered prompts are a shared-dependency
check—not an independent vote on model math.

## Model-free verification

```sh
python3 -B oracle/qwen38_contract.py
python3 -B oracle/test_qwen38_contract.py
```

On a high-memory machine with the complete, cache-free source tree, verify every source byte with:

```sh
python3 -B oracle/qwen38_contract.py --source-root /absolute/path/to/Qwen3.8-27B
```

That command rejects extra or missing files, path aliases, symlinks, hard links, size drift, and
hash drift. A Hugging Face transport cache must stay outside the verified payload root.

## Next protected slice

First freeze the exact producer modes and comparable trace schema: selected layers/positions,
tensor axes/dtypes/shapes, full-logit frames, top-k width, prefill chunk boundaries, and
serialization. Then build separate Transformers and mlx-lm environments and producers. Each must
verify the source before and after execution, load locally with no mutable network fallback,
write its own raw outputs and receipt, and never consume the other producer's normalization. A
third project-owned verifier will compare the raw arms and derive clean-versus-fault tolerances.
The bundle is valid only after an exact external inventory digest and atomic publication; a
producer-authored `pass` field has no authority.
