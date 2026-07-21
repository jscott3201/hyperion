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
