# 0002 — M1 baseline measurement protocol

- Status: accepted; measurement outcome pending
- Date: 2026-07-21
- Owners: Hyperion team

## Context

M1 establishes the permanent stock `mlx-lm` A-arm for the 12B and E4B Gemma 4 QAT Q4
checkpoints and locates the M5-16GB resident-memory throughput cliff. The milestone contract
requires five-trial rows and an A-C-C-A self-check, but it does not define C for a
baseline-only milestone. Stock `mlx_lm.benchmark` also cannot produce the required evidence:
it uses random token IDs, reports averages instead of raw trial distributions, does not expose
TTFT or token-level ITL, and does not separate per-trial memory sources.

The high-level generation wrappers are unsuitable for the resident-budget sweep because they
temporarily replace the caller's wired limit with the device recommendation. A project-owned
measurement controller is therefore necessary, but it must not become a second model or
generation implementation.

## Decision

1. **Immutable stock boundary.** The measured engine is the locked oracle environment:
   Python 3.12.13, MLX 0.32.0, and `mlx-lm` 0.31.3 from exact upstream commit
   `8239c72de5a0e42c539e30489021db73c7fe258c`. Hyperion may call unchanged
   `mlx_lm.load`, the returned model, `make_prompt_cache`, and `generate_step`; it may supply
   exact token arrays, timing, public MLX memory controls, OS sampling, and evidence writing.
   It may not patch, vendor, monkeypatch, or reimplement the package, model, cache, sampler, or
   generator. `stream_generate`, `BatchGenerator`, and `mlx_lm.server` are excluded from the
   direct-engine and budget-sweep rows. The unmodified server is used only for the separate
   loopback smoke.
2. **Pinned model identities.** The primary arm is
   `google/gemma-4-12B-it-qat-q4_0-unquantized` at
   `b6ed86275a6a5735884e208bfed95b445a684ca2`, converted affine Q4/group-64/bits-4 with
   manifest SHA-256 `9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144`.
   The second-tier arm is `google/gemma-4-E4B-it-qat-q4_0-unquantized` at
   `476025a01dbf99361c062bbeca3d6a76bb4c4566`, converted with the same recipe and
   manifest SHA-256 `9ba65423d3b2bab1e7c52ea88a1a2b0a33c1f51909b1df66330bf872b7a6c2b0`.
   Both tokenizer files hash to
   `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
   Their checkpoint chat-template hashes differ—12B
   `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`, E4B
   `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`—so rendered
   fixtures and hashes are model-specific. Remote template updates never alter an M1 row.
3. **Frozen workload corpus.** A versioned manifest binds source records, model-facing
   rendered bytes, little-endian `u32` token IDs, exact token counts, tokenizer/template
   hashes, and generator version. Every prompt is a deterministic round-robin composition of
   eight smart-building families: (1) fault-detection-and-diagnostics explanation, (2) energy
   recommendation, (3) point tagging, (4) operator copilot, (5) long transcript, (6) document
   QA, (7) short chat, and (8) compound parallel tool-chain. Inputs use diverse, indexed
   points, timestamps, rules, documents, and tool records; repeated-token padding and random
   token IDs are forbidden. A scored run fails if regenerated render/token hashes differ.
4. **Matrix and cardinality.** The gating matrix is both models at exactly 1,024, 4,096,
   8,192, 16,384, and 32,768 post-template input tokens. Each cell runs one same-shape
   discarded warmup followed by five measured trials. The standard `512/1024` compatibility
   row is also recorded: 512 input tokens and 1,025 generated token IDs, yielding 1,024
   post-first-token decode intervals. The future nightly `512/128` row uses the same 512-token
   fixture and 129 generated token IDs but is not an M1 gate. All gating cells likewise request
   1,025 generated IDs. A 128K stretch row may be added and labeled `low_n`; no 1K–32K row,
   including 32K, is gate-eligible with fewer than five successful measured trials.
5. **Trial isolation and generation.** Each model/context/arm runs in a fresh process. The
   model is loaded eagerly after the wired limit is established. Every trial starts with a
   synchronized device, cleared allocator cache, one boundary-only `reset_peak_memory`, a
   fresh prompt cache, the exact frozen token array, prefill chunks of 2,048, greedy sampling,
   and seed 0. The cache, iterator, outputs, and log-probability references are released after
   the synchronized trial. No peak reset or cache clear occurs inside a generation step.
6. **Timing definitions.** Model load, tokenization, fixture verification, and process startup
   are outside direct-engine latency. Trial time starts immediately before the first iterator
   advance. TTFT ends when the first token ID is materially available to the caller.
   `prefill_tok_s = input_tokens / ttft_seconds`. Decode excludes the first output token from
   both numerator and denominator:
   `decode_tok_s = (generated_ids - 1) / (last_token_time - first_token_time)`. ITLs are the
   adjacent materialization-time differences; p50/p95/p99 use nearest rank
   `ceil(percentile * n)` over the sorted raw values and record `n` (normally 1,024). Raw
   nanosecond timestamps and an output-token SHA-256 are retained for every trial.
7. **Memory definitions.** Model residency is included. Every trial records MLX active bytes,
   cache bytes, and the high-water value returned by `get_peak_memory`; the controller samples
   macOS `proc_pid_rusage` at 25 ms and records resident size, wired size, `phys_footprint`,
   interval/lifetime maximum footprint, and sample timestamps. Settled pre-trial and
   post-trial values remain distinct from maxima. No single field is relabeled as total peak;
   summaries report each source independently and use
   `max(MLX active + cache, phys_footprint, sampled resident size)` only as an explicitly
   derived working-set observation.
8. **Preregistered budget discovery.** Discovery uses the 12B model, the frozen 4,096-token
   composite prompt, 1,025 generated IDs, and one warmup plus five trials per point. Coarse
   requested wired caps are exactly 4, 5, 6, 7, 8, 9, 10, and 11 GiB, where
   `GiB = 1,073,741,824` bytes. The cap is set before model load and set a second time; the
   second call must return the requested value, proving it was effective. After the coarse
   sweep, the two in-range half-GiB neighbors of the best coarse median are measured with fresh
   processes and the same cardinality. Failures and pressure diagnostics stay in the curve.
9. **Optimum and tie rule.** Only points with five successful trials, finite metrics, matching
   output hashes, and no uncontrolled OOM are eligible. The empirical best is the highest
   median decode tok/s. The operational plateau contains every eligible point whose median is
   at least 99% of that best median and whose trial range overlaps the best point's range; the
   selected C is the smallest byte cap in that set. If no other point qualifies, C is the
   empirical best. A unique speed optimum may be claimed only when its minimum exceeds every
   other point's maximum; otherwise the report says `plateau` or `uncertain maximum` and makes
   no unique-speed claim. The conservative selected cap is still recorded so the governor has
   a reproducible operating point.
10. **A-C-C-A confirmation.** A is the stock/default cap returned by
    `max_recommended_working_set_size` (expected `12,713,115,648` bytes on the named M5
    profile). C is selected once from discovery by rule 9. Discovery trials are never reused.
    Confirmation runs fresh A, C, C, A blocks in that order; each block is a fresh process with
    one discarded warmup and five measured trials on the same 12B/4K workload. A selected-cap
    speedup or promotion requires the minimum of all ten C trials to exceed the maximum of all
    ten A trials. If not, M1 records the curve and self-check but makes no speedup claim and
    retains A as the operational default.
11. **Server agent smoke.** Each model is served by an unmodified fresh
    `mlx_lm.server` loopback process on `127.0.0.1`, temperature 0, seed 0, thinking disabled,
    decode/prompt concurrency 1, prefill step 2,048, and prompt-cache size 0. After one
    discarded successful request to absorb model-load races, a streamed OpenAI-compatible
    turn must produce a schema-valid `get_points` tool call, receive one fixed canonical tool
    result, and produce a final response in a second turn. Raw requests, chunks, server logs,
    canonicalized call IDs, usage, and inter-emission times are retained. HTTP inter-emission
    time is not called token ITL. This smoke establishes interoperability and determinism only;
    it does not seed the M5 quality floor.
12. **Evidence and failure policy.** Measurement requires a clean tracked worktree and binds
    the exact Git commit, Cargo/oracle locks, executable, worker, installed `mlx-lm` source,
    model manifests/configs/tokenizers/templates, corpus manifest, commands, machine profile,
    environment allowlist, and raw files by SHA-256. Raw JSONL includes trial markers, every
    token timestamp, memory samples, failures, and process exit state; derived JSON/CSV and the
    append-only ledger are reproducible from it. Signals, allocator errors, mismatched hashes,
    missing samples, partial cells, and invalid schemas fail closed and remain visible.
    Development PR CI validates corpus hashes, schemas, statistics, schedules, and negative
    controls without loading a model. Heavy M5 measurement is manual/protected and never runs
    automatically on pull-request code.
13. **Measurement order.** No scored inference may run until this accepted protocol and the
    harness/corpus it names are committed. After the premeasurement refute pass, all scored
    rows run from a clean commit. M1 acceptance then requires a second adversarial review of
    the exact evidence-bound commit, a complete M5 gate, passing non-draft PR CI, append-only
    evidence, and an appended outcome section below. Existing M0 evidence and decision bytes
    remain unchanged.

## Consequences

- M1 remains a stock-runtime baseline even though Hyperion owns the measurement process.
- The budget curve cannot be selected post hoc, and discovery cannot leak into confirmation.
- TTFT, decode, ITL, and memory fields have one reproducible interpretation across later
  native A/B comparisons.
- Template drift is visible rather than silently folded into model comparisons.
- An honest plateau or failure to beat A is a valid M1 outcome; it cannot be advertised as a
  performance promotion.

## Acceptance evidence

Pending. Append-only evidence and the final M1 verdict will be added after the preregistered
measurement, mandatory adversarial review, and reviewed-commit rerun.

## Protocol clarification 1 — stock allocator-cache behavior

The prohibition in decision 5 applies to project-owned orchestration: Hyperion adds no peak
reset or allocator-cache clear inside the iterator. The pinned, unchanged `generate_step`
implementation itself calls `mx.clear_cache()` at its stock prefill-chunk and 256-token
boundaries. Those upstream calls remain part of the immutable A-arm and must not be removed,
patched, or relabeled as Hyperion behavior.

## Protocol clarification 2 — fail-closed oracle identity

The oracle project now narrows its resolver range to Python 3.12 and the setup and runtime
checks require patch release 3.12.13 exactly. Regenerating `oracle/uv.lock` under that narrower
range removed irrelevant Python-version alternatives without changing the Python 3.12 package
realization; its accepted SHA-256 is
`b3603b4ebbc7f5883afe3d8cc10fc1767239837f5256985bd591b777c993dbaf`.
The immutable boundary is checked by canonical hashes over every installed non-bytecode
payload in the `mlx`, `mlx-metal`, and `mlx-lm` packages and their immutable distribution
metadata, not only three entry-point sources. The accepted tree hashes are MLX
`bacebd4f46680155a129301ffefc516402142183584f2b47673bc91b561f0cd9`, mlx-metal
`628a99548b65855148fb03f71cac83ce46eae42140f119fa8d1b51285c2abefd`, and mlx-lm
`40dc49399a07cdf22e3516070cfe222e89ec2f0ff29cd6e257e1b069edc3472f`.
The run manifest separately hashes the exact benchmark executable, its loaded native MLX
dylib, and the compiled canary metallib at both run creation and the final master gate.

## Protocol clarification 3 — causal boundaries and authoritative schedule

Every trial uses a bidirectional controller handshake. The worker blocks after constructing
the fresh prompt cache until the controller takes the settled pre-trial OS sample, and blocks
again after synchronized cleanup until the controller takes the post-cleanup sample. Periodic
samples are attributed only when the trial stage is unchanged across the OS sampling syscall;
cross-boundary samples remain unattributed. The committed
`benchmarks/m1/schedule.json` is the machine-readable schedule. The master gate requires the
exact core matrix, all eight coarse budget outcomes, both refinement outcomes derived from the
best eligible coarse point, fresh chronological A-C-C-A blocks bound to the saved selection,
and both server smokes. Controlled failed budget points remain in the curve as ineligible
outcomes; missing points, uncontrolled OOM, or failures in core/confirmation remain fatal.

## Protocol clarification 4 — low-N stretch and durable evidence

The optional 128K sentinel is an explicit non-gating `stretch-128k` row with 129 generated IDs,
one warmup, and one to four measured trials. Its model-specific 131,072-token fixtures are
hash-frozen with the corpus, and every such row is labeled `low_n`; it cannot satisfy a core
gate or promotion. Accepted M1 raw evidence is archived as an immutable, content-addressed
GitHub release asset named from the measured commit. The protected workflow downloads the
asset after upload, compares SHA-256, and emits a retrieval receipt. The finite-retention
Actions artifact is only a convenience copy and is not the permanent A-arm record.

## Protocol clarification 5 — executable startup and exact model payloads

Oracle entry points execute only as `python -I -S` through the committed isolated launcher.
Before adding the locked site-packages directory to `sys.path`, that launcher verifies Python
3.12.13's executable SHA-256, the complete uv-managed runtime tree, and the complete
site-packages tree. The latter covers every regular non-bytecode file and rejects symlinks;
only `.DS_Store` and wheel `RECORD` installer receipts are excluded because they are
non-executable and `RECORD` entry-script rows embed the recreated environment path. Every
referenced distribution payload, `_virtualenv.py`, every `.pth` file, and any otherwise
untracked importable or startup-hook file remains covered. The accepted startup identities
are executable `01564940172b2811e1f39a4dc90e84c7a26a19cf071bbc5de67e456d82627bec`,
runtime tree `01a580d385a91f4b8bc195c8b2f56c4c2d156f6c1e1ad8768fc4501987c4e12f`
over 1,897 entries, and site-packages tree
`db258e22404a3937d46d72ff44083400aafcf34636b8444a91a29c858b297006`
over 5,470 entries. Oracle setup builds a fresh staging environment, swaps it into place only
after verification, and restores the prior generated environment on failure.

Each source and converted model verification also requires an exact manifest inventory,
rejects tree/root symlinks, and hashes every listed payload byte. Hugging Face transport cache
metadata is ignored only for source snapshots and is never part of a loaded converted tree.
The benchmark worker rechecks the complete converted tree immediately before its one model
load; the server-smoke controller repeats that check immediately before each fresh server
load. An extra `model*.safetensors` shard therefore fails before `mlx_lm.load` can glob it.

## Protocol clarification 6 — authenticated failures, streams, and publication boundary

Every trace verifier resolves the controller-declared sibling stderr filename, hashes its
actual bytes, validates the exact exit-code/signal shape, and independently recomputes the OOM
classification. A controlled budget failure requires exit code 1, no signal, exactly one
structured capacity/allocation failure, and a nonempty controller error set. Signal 9 is always
uncontrolled. Missing/tampered stderr, forged exit state, and SIGKILL relabeling are negative
controls and cannot remove a discovery point from eligibility.

Server SSE evidence requires one consistent response identity, one schema-valid choice per
non-usage chunk, exactly one finish reason, and exactly one terminal usage chunk; arbitrary
JSON or unknown fields fail verification. Shutdown may end cleanly or by the controller's
SIGTERM. A server that requires SIGKILL fails the smoke and retains a failure record.

The durable archive has a preregistered whole-run filename allowlist, rejects symlinks and
extra files, scans every candidate evidence byte for machine paths and credential patterns,
and compares the allowlist to the content manifest before upload. The release contains two
permanent assets: the content-addressed evidence ZIP and its retrieval receipt. Repository
write permission exists only in the protected, manually dispatched archive job; checkout
never persists credentials, and `GH_TOKEN` is exposed only to the final publishing step.

## Protocol clarification 7 — deterministic startup and guarded load boundaries

This premeasurement clarification supersedes the startup and bytecode wording in clarification
5. Every oracle process now starts from an empty environment containing only the five committed
worker variables and uses Python `-B -S -s -P -X pycache_prefix=/dev/null`. This permits the
committed `PYTHONHASHSEED=0` to take effect while retaining no-site, no-user-site, safe-path, and
inert bytecode-cache behavior. The launcher rejects any other environment or flag state and
requires the cross-process hash probe `hash("hyperion-m1-fixed-hash-probe")` to equal
`1244036990071903237`. The uv runtime receipt now hashes every regular file, including all
existing standard-library bytecode, as tree
`63c25fabba8839ccb349e3554fedf9c46011d9e414c76912448a19869f666cac`
over 2,122 entries. A clean oracle recreation removes site-package bytecode, and the launcher
rejects any later `.pyc`, `__pycache__`, symlink, or special-file insertion before importing
MLX; the non-bytecode site tree remains
`db258e22404a3937d46d72ff44083400aafcf34636b8444a91a29c858b297006`
over 5,470 entries.

Source-model transport metadata is no longer blindly excluded. Its complete regular-file tree
is recursively hashed while every link, special file, and model-payload suffix is rejected.
The accepted transport-cache trees are 12B
`09b457cc0d497d5603265bea079d1c534ec1d54779ea0280844d9ba7a958fbf0`
and E4B `bffa1f7553e9bbc094174133554c6bc8df2cfad630a0dd7c1a00805baaf17027`,
each over 21 files. Converted trees still permit no cache or extra file. The direct benchmark
uses a verify-load-verify wrapper whose first verification is the operation immediately before
the unchanged `mlx_lm.load`. Server smoke uses a project-owned `ModelProvider` subclass only to
perform the same guard immediately inside the pinned provider's `_load` boundary; the upstream
model, tokenizer, cache, generation, response, and HTTP implementations remain unchanged. Each
server log must contain exactly one successful in-process load receipt.

A controlled discovery failure is now limited to `MemoryError`, `RuntimeError`, or `OSError`
with a bounded capacity phrase, an exact seven-field failure event, and a traceback made only of
nonempty strings whose first and final lines reproduce the exception. Broad `alloc` substring
matching is prohibited. SSE tool-call objects require exactly `index`, `id`, `type`, and
`function`; the nested function requires exactly string `name` and string `arguments` fields.
Both live validation and raw-journal replay enforce these shapes. Every workflow checkout,
including model-free PR CI, disables persisted credentials.

## Protocol clarification 8 — exact transport topology and post-load checks

This premeasurement clarification supersedes clarification 7's source-cache tree identities.
The canonical transport-cache digest now includes a typed entry for every nested directory as
well as the hash, size, and path of every regular file. Traversal errors fail closed, so an
empty or unreadable auxiliary directory cannot disappear from the identity. The accepted 12B
tree is `8ee7b68d0ece0fd7a1d281f3c1e9c9d82ece65bcb1cacfbaa04c5864baba9be7` and
the accepted E4B tree is
`2ee372adf9573c9e7037dd5c4a740d8dc14112b37eabe831b5a58a0e6f702eb7`, each
with 21 regular files and two nested directories.

The direct and server load guards perform their second identity check in a `finally` boundary,
including when the underlying loader raises. Model-free mutation controls require both guards
to reject a payload change made inside the synthetic loader and prohibit a successful server
load receipt for that changed tree.
