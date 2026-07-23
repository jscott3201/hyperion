# 0003 — M3 ABI: step-result read accessor

- Status: accepted
- Date: 2026-07-23
- Owners: Hyperion team

## Context

The M3 serving layer drives the native generation loop from Rust: prefill,
then repeated `hyp_decode_block` steps, reading each sampled token back to
emit SSE deltas and to decide stop. Today there is **no public way to read a
`HypStepResult`'s fields across the C ABI**. The only reader is
`step_result_read(HypStepResult)` in `native/hyperion_mlx/src/step_result_access.h`,
a **private C++ helper** (no `extern "C"`, not counted by
`check-abi-surface.sh`) that exists so the M2 native parity tests can assert
token-exact equality vs the oracle. The serving path cannot call a private
C++ symbol.

The 02-architecture ABI contract caps the surface at 25 functions and
requires an ADR for any addition. Before this decision the surface stood at 13
functions, `abi_version == 2` (the M3 sampler pair
`hyp_prefill_chunk_sampled` / `hyp_decode_block_sampled`).

## Decision

1. Add exactly one ABI function — `hyp_step_result_fields` — that copies a
   step result's fields into a caller-owned `HypStepResultFields`:
   ```c
   HypStatus hyp_step_result_fields(HypStepResult result,
                                    HypStepResultFields* out_fields);
   ```
   Surface 13 → 14, `abi_version` 2 → 3. Well under the 25-function cap.

2. **Copy, not borrow.** The function writes `*out_fields = result->fields`
   and returns. It does **not** hand back a pointer into native memory, so
   there is no aliasing or lifetime hazard across the Rust/C seam: the caller
   owns its own `HypStepResultFields` copy and still owns the handle, which it
   must free separately with `hyp_step_result_free`. The struct is
   trivially copyable (POD; the FFI already pins its size/offsets in a layout
   test).

3. **Validate the handle on the ABI side, do not double-trust.** The function
   checks `out_fields != nullptr`, `result != nullptr`, and
   `result->magic == kStepResultMagic`, returning `HYP_STATUS_INVALID_ARGUMENT`
   on any failure. This mirrors the private `step_result_read` check rather
   than calling it and trusting a default-constructed struct on a bad handle —
   the ABI boundary validates for itself.

4. The private `step_result_read` helper stays. The M2 native parity tests
   still use it; it is not an ABI function and is not counted. The new public
   accessor is the public-face twin of the same field read.

5. `check-abi-surface.sh` ratchets to `expected_count=14` and records the ADR
   in its comment; `kAbiVersion` in `runtime.mm` bumps to 3 in lockstep, and
   the canary test asserts `abi_version == 3`. The two are bumped together so
   a client probing `hyp_runtime_canary` sees a version consistent with the
   function count the surface check enforces.

## Consequences

- The serving generation loop can be written entirely in Rust: call
  `hyp_decode_block` / `hyp_decode_block_sampled`, then
  `hyp_step_result_fields` to read the sampled `token_id`, `logit`, and the
  top-k logprob sidecar. No C++ reader is needed on the serving path.
- One new ABI function, no handle-lifetime coupling, no new opaque type. The
  trivially-copyable `HypStepResultFields` (136 bytes; pinned by the FFI
  layout test) is the only thing crossing out.
- The private `step_result_read` remains a test-only convenience; if a later
  slice wants to collapse the two, it must not regress the M2 parity tests.
- Future ABI additions continue to require an ADR and a lockstep
  `check-abi-surface.sh` / `kAbiVersion` bump.

## Acceptance evidence

- Native build clean; `hyperion_abi_contract` and `hyperion_runtime_canary`
  ctest cases pass with the surface at 14 and `abi_version == 3`.
- `scripts/check-abi-surface.sh` reports `abi-surface: 14/25 functions`.
- This decision text is immutable; later evidence may only be appended.
