# Hyperion evidence ledger

This file is append-only after rows are accepted. Raw traces belong in `benchmarks/raw/` and
are intentionally ignored. Every claim is classified as `MEASURED`, `DECIDED`, or
`ASPIRATIONAL`; only `MEASURED` rows may support benchmark claims.

## Safe machine-state profile: `m5-16g-local-2026-07-21`

- Hardware class: MacBook Pro, base Apple M5, 16 GB unified memory, arm64
- OS: macOS 26.6
- Public Metal family: Apple10 (`1010`)
- MLX: 0.32.0 headers and dylib
- Rust: 1.95.0
- CMake: 4.4.0
- Privacy rule: never record serial numbers, platform UUIDs, account names, or credentials

## Row schema

Each row records an immutable ID, classification, source revision/state, exact command,
machine-state profile, inputs and hashes, complete metrics/output, trial count, and verdict.
Performance promotions additionally require A-C-C-A ordering, five measured trials after one
warmup, candidate-min greater than baseline-max, and G1–G4 evidence appropriate to the current
milestone.

## M0 rows

M0 evidence rows bind exact commits. Pre-review rows remain immutable; the milestone verdict
is appended only after the reviewed commit and real oracle gate are rerun.

### `HYP-M0-CANARY-001` — native startup canary

- Classification: `MEASURED`
- Source commit: `6ec89b974486a0eb22df46702f80dfb1d8bb6438`
- Worktree: clean for tracked files; model and build artifacts ignored by policy
- Machine state: `m5-16g-local-2026-07-21`
- Exact command: `cargo run --locked -p hyperion-bench -- canary`
- Cargo lock SHA-256: `63bf01240e9139a52d71d7bf6d8e8ee76a4ad28246dd2e6f27c5794242df98e8`
- Loaded sidecar SHA-256: `cc978c2159c7b7eabdda39b256932d3108841a6d0521f7d5855684bf162a7f30`
- Trials: one functional platform/native canary; this is not a performance row
- Exit status: `0`
- Output:

```text
MEASURED {"schema":"hyperion.canary.v1","model_scope":"gemma-4","abi_version":1,"macos":"26.6.0","gpu_family":1010,"gpu_name":"Apple M5","mlx_compile":"0.32.0","mlx_runtime":"0.32.0","recommended_working_set_bytes":12713115648,"budget_formula":"min(12GiB,floor(recommended*0.949))","effective_budget_bytes":12064746749,"soft_watermark_bytes":10858272074,"mlx_probe_value":4.0,"metallib_probe_value":42.0}
```

- Verdict: `PASS` — public M5/macOS/MLX predicates, device-derived budget, custom metallib
  load/dispatch, and a real MLX tensor evaluation all passed.

### `HYP-M0-NAX-PROBE-001` — 3840-wide MLX capability sweep

- Classification: `MEASURED`
- Source commit: `6ec89b974486a0eb22df46702f80dfb1d8bb6438`
- Worktree: clean for tracked files
- Machine state: `m5-16g-local-2026-07-21`; MLX device architecture diagnostic
  `applegpu_g17g`; recommended working set `12,713,115,648` bytes
- Exact command: `scripts/nax-probe.sh`
- Oracle lock SHA-256: `5e6e51756f1420e078f09badc0748e010eaaa7a5f1b81adfebbe9ca24a0e8883`
- Method: `3840×3840×M`, one discarded warmup and five trials per shape/mode; timings below
  are nanoseconds; nominal TFLOP/s is arithmetic throughput, not proof of MLX's internal NAX
  dispatch choice.

| M | Mode | Trial nanoseconds | Median ns | Nominal TFLOP/s | Checksum |
|---:|---|---|---:|---:|---:|
| 1 | bfloat16 | `683125,552250,539708,487958,526000` | 539708 | 0.054643 | 55.0 |
| 1 | float16 | `499500,543042,441333,478250,506708` | 499500 | 0.059041 | -8.2734375 |
| 1 | q4-affine-g64 | `276833,312125,451584,347791,318500` | 318500 | 0.092594 | 9.109375 |
| 64 | bfloat16 | `812042,709083,697917,696333,694458` | 697917 | 2.704386 | 14.625 |
| 64 | float16 | `746417,707458,815542,686167,760875` | 746417 | 2.528663 | 139.875 |
| 64 | q4-affine-g64 | `1165167,1121375,1120792,1088750,1151834` | 1121375 | 1.683145 | 82.8125 |
| 512 | bfloat16 | `1905042,1901833,2009792,1750792,1933250` | 1905042 | 7.926069 | 21.875 |
| 512 | float16 | `1902250,1880833,1965375,1921667,1973666` | 1921667 | 7.857498 | 8.703125 |
| 512 | q4-affine-g64 | `1962000,2139250,1976125,1914083,2000250` | 1976125 | 7.640961 | -6.5234375 |
| 2048 | bfloat16 | `4188667,4284417,4523333,4664958,4300167` | 4300167 | 14.045496 | -37.5 |
| 2048 | float16 | `4238750,4284125,4252959,4300750,4254125` | 4254125 | 14.197509 | -104.6875 |
| 2048 | q4-affine-g64 | `4422500,4419958,4415125,4408458,4468541` | 4419958 | 13.664831 | 103.5625 |

- Verdict: `PASS (capability observation)` — all required dtype/shape cells executed on the
  M5. No kernel promotion or NAX-route claim is made from this probe.

### `HYP-M0-ORACLE-001` — pinned Gemma 4 generation smoke

- Classification: `MEASURED`
- Source commit: `094664919bc2e28fd0d93b3c1034db7511d1aca3`
- Worktree: clean for tracked files; source and converted model payloads ignored by policy
- Machine state: `m5-16g-local-2026-07-21`
- Exact parent command: `scripts/ci-release.sh`
- Exact oracle stage: `scripts/oracle-smoke.sh`
- Oracle lock SHA-256: `5e6e51756f1420e078f09badc0748e010eaaa7a5f1b81adfebbe9ca24a0e8883`
- Source repository/revision: `google/gemma-4-12B-it-qat-q4_0-unquantized` at
  `b6ed86275a6a5735884e208bfed95b445a684ca2`
- Source SHA-256 manifest digest:
  `6a07a92df9260b71117b113a8ad0b305432a48f895abd850a7616241a636ebed`
- Converted identity: affine Q4, group size 64, 4 bits; converted SHA-256 manifest digest
  `9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144`
- Generation controls: greedy temperature `0.0`, seed `0`, maximum 128 tokens, checkpoint
  chat template with thinking explicitly disabled, exact full-response comparison
- Trials: one functional generation smoke; this is not a performance row
- Exit status: `0`
- Output:

```text
MEASURED {"schema":"hyperion.oracle-smoke.v1","model":"gemma-4-12B-QAT-Q4-g64-affine","prompt":"HYPERION_M0_OK","generated":true}
HYPERION_M0_OK
```

- Verdict: `PASS` — the hash-bound converted checkpoint performed real generation and its
  complete response exactly matched the frozen marker.

### `HYP-M0-GATE-001` — pre-review full release gate

- Classification: `MEASURED`
- Source commit: `094664919bc2e28fd0d93b3c1034db7511d1aca3`
- Worktree: clean for tracked files; model and build artifacts ignored by policy
- Machine state: `m5-16g-local-2026-07-21`
- Exact command: `scripts/ci-release.sh`
- Cargo lock SHA-256: `63bf01240e9139a52d71d7bf6d8e8ee76a4ad28246dd2e6f27c5794242df98e8`
- Oracle lock SHA-256: `5e6e51756f1420e078f09badc0748e010eaaa7a5f1b81adfebbe9ca24a0e8883`
- Gate order: fast PR checks; all native CMake tests; ignored Rust real-M5 canary; native
  canary executable; strict Gemma oracle generation. The native executable exited before the
  Python oracle process started.
- Fast checks: locked workspace build, Clippy with warnings denied, Rust tests, six-crate
  no-orphan graph, unsafe confinement, two-symbol C ABI surface, append-only ledger guard,
  and model-free CMake build/tests all passed.
- Full native checks: all three CMake tests passed, including the runtime canary; the ignored
  Rust real-M5 canary passed; the standalone canary reported MLX `0.32.0`, Apple10 family,
  a device-derived `12,064,746,749`-byte effective budget, MLX result `4.0`, and custom
  metallib result `42.0`.
- Oracle output: strict full response `HYPERION_M0_OK` from the hash-bound converted Gemma 4
  artifact.
- Trials: one complete functional release-gate run; this is not a performance row
- Exit status: `0`
- Verdict: `PASS (pre-review)` — every M0 fast and heavy gate passed sequentially on the real
  target machine. Final milestone acceptance remains withheld until the mandatory adversarial
  review is addressed and the resulting commit is rerun.

### `HYP-M0-REVIEW-001` — adversarial rejection of the pre-review gate

- Classification: `DECIDED`
- Reviewed commit: `81f891178f390188e7e322056cb03695ddaad616`
- Base: `development` at `0ee828a00bcd0b2990af581542a9258540765970`
- Exact review scope: `git diff development...81f891178f390188e7e322056cb03695ddaad616`
- Review mode: fresh read-only adversarial pass against the M0 goal package, source, tests,
  CI, artifacts, oracle, ABI, and claimed evidence
- Finding: Release-mode `NDEBUG` removed the C++ `assert` expressions, so platform-policy
  rejection and ABI tests had passed without executing their checks.
- Finding: The release workflow's clean checkout removed ignored model/oracle prerequisites;
  automatic pull-request code on a persistent self-hosted runner was also unsafe.
- Finding: The oracle generation command did not mechanically verify the locked environment,
  source/converted checksum manifests, model geometry, or quantization identity before printing
  its model label; the metallib override was not bound to the freshly built sidecar hash.
- Finding: The MLX probe relied on ambient stream resolution; the development push guard
  compared `HEAD` to itself; E4B QAT was assigned to M8 despite being an M1 baseline; a local
  `SHA256SUMS` made acquisition non-rerunnable; and an uncontrolled allocation failure was
  mislabeled as predictive governor rejection.
- Disposition: all prior MEASURED values remain immutable observations, but
  `HYP-M0-GATE-001` is insufficient for milestone acceptance. A corrected commit must make
  tests release-active with a negative control, enforce explicit streams and hash-bound
  prerequisites, harden CI/acquisition/ABI failure paths, pass focused re-review, and rerun the
  complete gate before an acceptance row may be appended.
- Verdict: `REJECTED FOR ACCEPTANCE` — do not merge the reviewed commit.

### `HYP-M0-GATE-002` — post-review full release gate

- Classification: `MEASURED`
- Source commit: `90e154efae439638ffde6402d640ac3266e88522`
- Review state: the mandatory adversarial review rejected the earlier candidate, verified all
  remediations, and returned `READY` for this exact source commit before this gate ran
- Worktree: clean for tracked files; model and build artifacts ignored by policy
- Machine state: `m5-16g-local-2026-07-21`
- Exact command: `scripts/ci-release.sh`
- Build isolation: fresh temporary Cargo target and CMake build directories; no prior build
  output reused
- Cargo lock SHA-256: `63bf01240e9139a52d71d7bf6d8e8ee76a4ad28246dd2e6f27c5794242df98e8`
- Oracle lock SHA-256: `5e6e51756f1420e078f09badc0748e010eaaa7a5f1b81adfebbe9ca24a0e8883`
- Oracle package identity: Python 3.12.13, MLX 0.32.0, and `mlx-lm` package version 0.31.3
  from exact source commit `8239c72de5a0e42c539e30489021db73c7fe258c`
- Source repository/revision: `google/gemma-4-12B-it-qat-q4_0-unquantized` at
  `b6ed86275a6a5735884e208bfed95b445a684ca2`
- Source SHA-256 manifest digest:
  `6a07a92df9260b71117b113a8ad0b305432a48f895abd850a7616241a636ebed`
- Converted identity: affine Q4, group size 64, 4 bits; converted SHA-256 manifest digest
  `9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144`
- Fresh CMake metallib SHA-256:
  `99553a7d5388eff0ea81e4346276a945733c1e9ee90f18b5b568fb4706f8c7d2`
- Fresh Cargo sidecar SHA-256:
  `cc978c2159c7b7eabdda39b256932d3108841a6d0521f7d5855684bf162a7f30`;
  the hash was unchanged after canary execution
- Fast checks: locked workspace build, Clippy with warnings denied, Rust tests, six-crate
  no-orphan graph, unsafe confinement, two-symbol C ABI surface, append-only policy regression
  and real-base guard, and model-free native build/tests all passed
- Full native checks: Release-active platform, deliberate negative-control, ABI, and runtime
  tests passed; the ignored Rust real-M5 test passed; both native probe values were finite and
  exact
- Phase order: the native executable and Rust canary exited before the isolated Python oracle
  process started; source and converted manifests were independently verified before both
  native and oracle phases
- Trials: one complete functional post-review release-gate run; this is not a performance row
- Exit status: `0`
- Native output:

```text
MEASURED {"schema":"hyperion.canary.v1","model_scope":"gemma-4","abi_version":1,"macos":"26.6.0","gpu_family":1010,"gpu_name":"Apple M5","mlx_compile":"0.32.0","mlx_runtime":"0.32.0","recommended_working_set_bytes":12713115648,"budget_formula":"min(12GiB,floor(recommended*0.949))","effective_budget_bytes":12064746749,"soft_watermark_bytes":10858272074,"mlx_probe_value":4.0,"metallib_probe_value":42.0}
MEASURED {"schema":"hyperion.metallib.v1","source":"fresh-cargo-sidecar","sha256":"cc978c2159c7b7eabdda39b256932d3108841a6d0521f7d5855684bf162a7f30","stable_during_canary":true}
```

- Oracle output:

```text
MEASURED {"schema":"hyperion.oracle-smoke.v1","model":"gemma-4-12B-QAT-Q4-g64-affine","prompt":"HYPERION_M0_OK","generated":true}
HYPERION_M0_OK
```

- Verdict: `PASS (post-review release gate)` — every M0 fast and heavy gate passed
  sequentially on the reviewed target commit. Milestone acceptance remains pending the
  non-draft feature PR's required CI and merge into `development`.

### `HYP-M1-DEFER-001` — scored baseline run deferred for implementation progress

- Classification: `DECIDED`
- Source commit: `c782f6feb8a3d90aff71f4c4b9cffb1e9ab21f06`
- Run ID: `m1-c782f6f-scored`
- Run status: intentionally interrupted during the 6 GiB discovery cell; local raw evidence
  preserved, not archived, and permanently invalid for acceptance
- Valid engineering signal: the separate unscored `m1-pilot-c782f6f` completed one warmup
  and five measured-shape trials for both 12B and E4B at 512 input / 129 output IDs with
  deterministic outputs and independently valid traces
- Invalid for claims: the interrupted run's completed core and partial discovery cells are
  diagnostic only and are not MEASURED ledger rows
- Owner decision: prioritize native implementation progress and defer, rather than weaken,
  the full five-trial matrix, budget curve, A-C-C-A confirmation, server smoke, archive, and
  evidence review
- Consequence: M2 implementation may proceed, but M1 remains unaccepted; no throughput-optimal
  cap or relative M2 decode-performance gate may be claimed until a fresh compliant M1 run
- Verdict: `DEFERRED (NO PERFORMANCE CLAIM)`
