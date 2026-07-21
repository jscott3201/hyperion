# 0001 — M0 runtime and evidence boundaries

- Status: accepted
- Date: 2026-07-21
- Owners: Hyperion team

## Context

The goal package intentionally puts model-independent platform bootstrap in M0 and the real
native Gemma forward graph in M2, but `02-architecture.md` describes a one-token smoke at
`hyp_model_load`. The eventual tier-1 and G4 lists also mention surfaces that do not exist until
M2, M3, or M8. Session zero names five repositories whose combined payload plus a converted
Q4 artifact would nearly consume the currently available local disk.

Released `mlx-lm` 0.31.3 also predates the official checkpoint's `gemma4_unified` model-type
mapping. Upstream added that compatibility in commit
`8239c72de5a0e42c539e30489021db73c7fe258c`.

## Decision

1. M0 exposes exactly two native functions: stateless `hyp_runtime_canary` and
   `hyp_last_error`. It does not fabricate `hyp_model_load` or a stub model. The canary uses
   public Metal family support to require Apple GPU family 10, requires macOS 26.2+, compares
   both MLX compile-time and runtime versions to exactly 0.32.0, loads and executes the built
   Hyperion metallib, and evaluates a real MLX GPU tensor. M2's real `hyp_model_load` must call
   this canary and then perform the real Gemma one-token forward.
2. The M5 predicate is `supportsFamily(MTLGPUFamilyApple10)`. Marketing names and `hw.model`
   strings are diagnostics only. Pure policy tests reject Apple9/M4, macOS 26.1, MLX version
   drift, and a missing device.
3. The M5-16G effective ceiling is
   `min(12 GiB, floor(recommendedMaxWorkingSetSize × 0.949))`; the soft watermark is 90% of
   that result. Live/free OS memory never changes admission policy.
4. Tier 1 grows cumulatively with real surfaces: M0 runs the workspace build, clippy, current
   unit/ABI tests, no-orphan check, unsafe-confinement check, ABI count, native CMake tests, and
   metallib build. M3 adds template/parser contracts; M8 adds all-five geometry. Empty future
   test placeholders do not count.
5. G4 also grows cumulatively. M0 proves the device-derived ceiling and a real oracle run
   without uncontrolled OOM. M2/M3 activate the 8K/32K peak and governor-prediction sentinels.
6. The isolated oracle pins `mlx==0.32.0` and package version `mlx-lm==0.31.3` from the exact
   upstream compatibility commit above. Mutable branches and checkpoint-config rewrites are
   forbidden. Python remains bench/oracle-only and cannot be selected by serving code.
7. M0 downloads, verifies, converts, and runs the primary 12B QAT source at immutable revision
   `b6ed86275a6a5735884e208bfed95b445a684ca2`. The E4B QAT baseline is acquired in M1, the
   BF16 reference in M2, and the assistant checkpoints in M7/M8, after their immutable
   revisions are recorded. This avoids exhausting the 16 GB development machine's local disk
   without weakening the M0 real-oracle gate or deferring an owning milestone's prerequisite.
8. Development PR CI is model-free and may compile on GitHub's macOS 26 Apple-Silicon host;
   it never treats that M1 host as a supported runtime and never bypasses the production
   canary. Full milestone/release CI targets a labeled self-hosted M5 and runs only after a
   reviewed main push or an authorized manual dispatch through the `hyperion-m5-release`
   environment—not automatically for pull-request code. Checkout remains clean; the locked
   oracle is recreated, and `HYPERION_M0_ORACLE_MODEL` must point outside the checkout to the
   content-addressed model store. The gate verifies that store before native and oracle phases
   run sequentially.

## Consequences

- The M0 gate remains real-tensor and fail-loud while avoiding a fake model API.
- There is one production native path and two ABI functions, well below the 25-function cap.
- The hosted PR job catches portable compile/static/ABI regressions; only M5 evidence can make
  runtime or performance claims.
- Future milestone tests are additions to the same tier-1 entry point, not parallel definitions
  of “tier 1.”
- The model-download exception is a storage schedule, not a scope deletion; each deferred
  artifact remains a blocking prerequisite for its owning milestone.

## Acceptance evidence

- `HYP-M0-GATE-002` records a complete `PASS` from the clean, post-review M5 gate on exact
  source commit `90e154efae439638ffde6402d640ac3266e88522`.
- Mandatory adversarial review found thirteen total issues across the initial and focused
  passes; every actionable finding was addressed and the reviewer approved exact evidence
  head `22cf74f8246d2c2e4e9940b217248e2ba670a448` for the non-draft PR.
- GitHub PR `#1` tier-1 workflow run `29862998459` passed at that exact evidence head with no
  review threads or requested changes.
- Verdict: accepted for merge into `development`. This existing decision text is now
  immutable; later evidence may only be appended.
