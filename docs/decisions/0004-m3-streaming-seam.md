# ADR 0004: the M3 streaming seam — `Engine::stream` + cancel-on-drop

- **Status:** accepted
- **Date:** 2026-07-23
- **Owners:** Hyperion team
- **Supersedes:** —
- **Related:** ADR 0003 (`hyp_step_result_fields`, the read accessor the
  streaming loop uses per step)

## Context

The 06-serving spec mandates **real token-incremental SSE** ("decode loop emits
per-step deltas through the bounded channel; cancel-on-client-drop propagates
to the engine thread between steps/chunks") and explicitly **bans** post-hoc
"streaming" ("Helios's 3-frame post-hoc streaming is banned"). PR A landed a
synchronous `Engine::drive -> Vec<u32>` — it decodes the whole generation and
returns the token vec at the end. That can't drive real SSE: the handler would
receive all tokens at once and emit them in a tight loop (the banned post-hoc
pattern), with no mid-stream cancel.

The serving layer (PR B) needs the engine to **yield one token per decode
step**, over a bounded channel, so the axum handler can flush an SSE frame per
token and so a client disconnect can cancel the engine between steps.

## Decision

Add `Engine::stream` as the streaming entry point; `drive` becomes a
non-streaming collector wrapper over it.

- **`Engine::stream(&self, request, cancel, tx: tokio::sync::mpsc::Sender<StepEvent>) -> Result<Usage, EngineError>`**
  — prefills, then decodes one token at a time, yielding a `StepEvent::Token`
  per step on `tx` and a terminal `StepEvent::Done(Usage)`. The decode loop
  checks the explicit `CancelToken` between steps.
- **`tokio::sync::mpsc` (cap 8)** — the bounded engine channel (06 §Concurrency).
  The engine calls `tx.blocking_send` (sync; it runs on a plain thread, not a
  tokio runtime — `blocking_send` needs no runtime). The cap is the
  **backpressure mechanism**: a slow client throttles the engine rather than
  dropping events.
- **Cancel-on-client-drop** — when the axum response future drops (client
  disconnect), the channel receiver drops, and the engine's next
  `blocking_send` returns `SendError`, mapped to `EngineError::Cancelled`. No
  explicit `CancelToken` fire is needed — the dropped channel IS the signal.
  This is the spec's "cancel-on-client-drop propagates to the engine thread
  between steps."
- **`Engine::drive`** — the non-streaming collector: it creates a channel,
  runs `stream` inline, and collects the `Token` events into a `Vec`. Keeps
  PR A's `#[ignore]` tiny-fixture test (which calls `drive`) byte-identical.

The axum layer depends on a `trait EngineDriver` (the handler-facing seam, `Send
+ Sync`), not the `!Send` `Engine` directly. The engine thread owns the
`Engine` and runs `Engine::stream`; the handler holds a `MailboxEngine` (an
`EngineDriver` that posts jobs to the engine thread) or a `StubEngine` (in
tests). This keeps the `!Send` invariant (02 §Threading) while letting the
async handler drive a generation.

## Consequences

- **One streaming primitive, two entry points.** `stream` is the real loop;
  `drive` is sugar. The non-streaming HTTP path collects the stream into a
  single JSON response (no separate code path through the engine).
- **Backpressure is structural.** The cap-8 channel means a client that stops
  reading pauses the engine after 8 buffered tokens — no unbounded queue, no
  OOM from a slow consumer.
- **Cancel is dual-pathed.** Explicit `CancelToken` (for `/control/shutdown`
  or a future hard-cancel) + dropped-receiver (client disconnect). Both
  surface `EngineError::Cancelled` → HTTP 499 (pre-stream) or an SSE
  terminal/error event (mid-stream).
- **The `!Send` engine stays on one thread.** `MailboxEngine` (Send + Sync) is
  the bridge; the engine thread loads the `Engine` on-thread (it can't be
  moved — raw pointers), drains the mailbox, and runs `stream` per job.

## Why not std `mpsc`?

`std::sync::mpsc` would work on the engine side (sync `send`), but the receiver
is drained by the async axum handler — `tokio::sync::mpsc` gives an async
`recv().await` + a `Send` receiver. The engine's `blocking_send` on a tokio
channel is safe from a non-async thread (it's a blocking call). One channel
type crosses both worlds cleanly.

## What this does NOT decide

- The tool-call parser, the thinking-lane `<|channel|>thought` SSE policy, and
  the governor v1 calibration report are separate M3 sub-slices (out of PR B).
- Continuous batching is explicitly post-v1 (06 §Concurrency); the
  single-flight `Semaphore(1)` serializes requests in PR B.
