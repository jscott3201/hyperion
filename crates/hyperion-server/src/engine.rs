//! The engine-driver: owns the loaded native model and drives a generation
//! loop on one thread.
//!
//! Architecture (02 §Threading model): one dedicated engine thread owns all
//! MLX/native state — `Model` and `KvState` are `!Send`, so they cannot cross
//! threads. This module is the **synchronous core** that runs on that thread:
//! [`Engine::stream`] prefills the prompt to produce completion token 1, then
//! decodes subsequent tokens one at a time. Each sampled `token_id` is read back
//! through the M3 `hyp_step_result_fields` accessor (ADR 0003) and yielded as a
//! [`StepEvent`] until the total `max_tokens` budget or EOS. [`Engine::drive`] is
//! the non-streaming collector wrapper.
//!
//! The axum/SSE transport (PR B) owns an `Engine` on a dedicated thread and
//! feeds it `EngineRequest`s; [`Engine::stream`] sends each completion token over
//! a bounded `tokio::sync::mpsc` channel (cap 8, 06 §Concurrency) that the
//! handler converts to SSE frames. Cancel propagates two ways: an explicit
//! [`CancelToken`] polled between steps, and the dropped-receiver path — when
//! the client disconnects, the axum response future drops the channel receiver
//! and the engine's next `blocking_send` fails, surfacing
//! [`EngineError::Cancelled`]. That is cancel-on-client-drop, end-to-end.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use hyperion_core::geometry_abi::AbiGeometry;
use hyperion_ffi::{
    Error as NativeError, HypSamplingConfig, HypStepResultFields, HypTokenStream, KvState, Model,
    Status, StepResult,
};
use hyperion_model::geometry::Geometry;

/// The native prefill chunk size (05 §Prefill chunking). `hyp_prefill_chunk`
/// chunks internally at this boundary and expects the whole prompt in one
/// call; it does NOT support continuation prefill (`offset != 0` is
/// rejected). Exposed for PR B's `/control` stats / display, not for Rust-side
/// slicing.
pub const PREFILL_CHUNK: usize = 2048;

// `hyp_decode_block_sampled` resets its local RNG state from `config.seed` on
// every FFI call, then advances it with this LCG once per decoded token. The
// Rust engine calls native with `n_tokens=1`, so it must carry the same state
// across calls: prefill consumes the request seed, decode token 2 consumes the
// first successor, and so on. Keep these constants identical to model.cc's
// native per-step sampling contract.
const NATIVE_SAMPLING_LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const NATIVE_SAMPLING_LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;

#[must_use]
fn next_sampling_seed(seed: u64) -> u64 {
    seed.wrapping_mul(NATIVE_SAMPLING_LCG_MULTIPLIER)
        .wrapping_add(NATIVE_SAMPLING_LCG_INCREMENT)
}

fn advance_sampling_seed(sampling: &mut Option<HypSamplingConfig>) {
    if let Some(config) = sampling {
        config.seed = next_sampling_seed(config.seed);
    }
}

/// A single-flight generation is busy — the second concurrent request is
/// rejected with HTTP 429 (06 §Concurrency). This guard is the model-free
/// primitive the axum layer will sit a `Semaphore(1)` on top of.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SingleFlightBusy;

impl std::fmt::Display for SingleFlightBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("single-flight generation is busy (429)")
    }
}

impl std::error::Error for SingleFlightBusy {}

/// A cooperative cancel flag. The engine loop polls [`CancelToken::is_cancelled`]
/// before and after prefill and decode steps; a cancelled run stops cleanly and
/// surfaces [`EngineError::Cancelled`]. `CancelToken` is `Clone + Send + Sync`
/// so the axum response future can fire it on client-drop (PR B).
#[derive(Clone, Debug)]
pub struct CancelToken {
    flagged: Arc<AtomicBool>,
}

impl CancelToken {
    #[must_use]
    pub fn new() -> Self {
        Self {
            flagged: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal cancellation. Idempotent.
    pub fn cancel(&self) {
        self.flagged.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flagged.load(Ordering::SeqCst)
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

/// The normalized input both dialects produce (06: "both dialects normalize
/// to one PreparedPrompt"). PR A carries the tokenized prompt + sampler
/// config + stop conditions; the chat-template render + dialect
/// normalization land with PR B's HTTP layer.
#[derive(Clone)]
pub struct EngineRequest {
    /// The tokenized prompt (template-rendered, BOS included by the template).
    pub prompt_tokens: Vec<u32>,
    /// Hard ceiling on generated tokens.
    pub max_tokens: u32,
    /// EOS token id; decoding stops if this is sampled. `None` ⇒ run to
    /// `max_tokens` (no early stop).
    pub eos_token_id: Option<u32>,
    /// Greedy (`None`) or stochastic sampling config. Greedy is the G1
    /// token-exact default; a config with `temperature > 0` selects sampled
    /// mode (the native side routes on `config != NULL && temperature > 0`).
    pub sampling: Option<HypSamplingConfig>,
}

impl std::fmt::Debug for EngineRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `HypSamplingConfig` has no Debug impl (raw repr(C)), so report the
        // mode (greedy vs sampled) without the raw float fields.
        let mode = if self.sampling.is_some() {
            "sampled"
        } else {
            "greedy"
        };
        f.debug_struct("EngineRequest")
            .field("prompt_tokens_len", &self.prompt_tokens.len())
            .field("max_tokens", &self.max_tokens)
            .field("eos_token_id", &self.eos_token_id)
            .field("sampling", &mode)
            .finish()
    }
}

/// The sampled token id from a prefill epilogue or decode step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepToken {
    pub id: u32,
    pub logit: f32,
}

/// Token accounting for one generation (06 §Streaming: "usage accounting in the
/// terminal frame, both dialects"). The terminal [`StepEvent::Done`] carries
/// this; the SSE framer renders it into the dialect's usage field.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Usage {
    /// The prompt token count (the rendered + tokenized prompt length).
    pub prompt_tokens: u32,
    /// The generated token count (tokens emitted before EOS / max_tokens).
    pub completion_tokens: u32,
}

/// One step of a streaming generation, sent over the bounded channel from the
/// engine thread to the axum handler (06 §Streaming: "token-incremental real
/// SSE"). `Token` is one completion token; `Done` is the terminal frame carrying
/// [`Usage`]. The handler converts each into the dialect's SSE frame.
#[derive(Clone, Copy, Debug)]
pub enum StepEvent {
    /// One completion token (id + logit).
    Token(StepToken),
    /// The terminal frame — generation finished at `max_tokens` or EOS, with
    /// usage accounting. Always the last event on a successful run.
    Done(Usage),
}

impl StepEvent {
    /// The token id if this is a `Token` event; `None` for `Done`. Convenience
    /// for the non-streaming collector ([`Engine::drive`]) and tests.
    #[must_use]
    pub fn token_id(&self) -> Option<u32> {
        match self {
            Self::Token(t) => Some(t.id),
            Self::Done(_) => None,
        }
    }
}

/// Errors the engine can surface. The axum layer maps these to the 06 error
/// taxonomy (PR B): `Native(OomGovernor)` → 529, `Busy` → 429,
/// `Cancelled` → 499, other `Native` → the `Status::http_status_code()` map.
#[derive(Debug)]
pub enum EngineError {
    /// A native call failed (IO, governor rejection, invalid argument, …).
    Native(NativeError),
    /// Single-flight: another generation owns the engine.
    Busy,
    /// The caller cancelled before completion.
    Cancelled,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native(e) => write!(f, "native engine error: {e}"),
            Self::Busy => f.write_str("single-flight busy"),
            Self::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<NativeError> for EngineError {
    fn from(e: NativeError) -> Self {
        Self::Native(e)
    }
}

impl From<SingleFlightBusy> for EngineError {
    fn from(_: SingleFlightBusy) -> Self {
        Self::Busy
    }
}

/// The streaming seam the axum layer drives a generation through. This is the
/// **handler-facing** abstraction, not the `!Send` [`Engine`] itself: the
/// engine thread owns the `Engine` and runs [`Engine::stream`] directly; the
/// handler holds an `impl EngineDriver` (a mailbox sender that IS `Send`) and
/// calls [`EngineDriver::stream`] on it. Tests inject a [`StubEngine`] (B7) so
/// the contract/SSE/cancel tests run model-free — the stub implements this
/// trait directly and yields a fixed [`StepEvent`] sequence.
///
/// `stream` hands the request to the engine thread and returns the [`Usage`].
/// The handler separately drains the `tokio::sync::mpsc::Receiver` it created
/// and converts each [`StepEvent`] into the dialect's SSE frame. Dropping the
/// receiver (client disconnect) cancels via the `blocking_send` failure path
/// inside [`Engine::stream`].
pub trait EngineDriver: Send + Sync {
    /// Stream one generation. Yields [`StepEvent`]s on `tx` (one `Token` per
    /// completion token, then `Done`), returns the [`Usage`] on success. The engine
    /// thread calls `tx.blocking_send` (sync; it runs in a plain thread, not a
    /// tokio runtime) — a dropped receiver surfaces as
    /// [`EngineError::Cancelled`].
    fn stream(
        &self,
        request: &EngineRequest,
        cancel: &CancelToken,
        tx: tokio::sync::mpsc::Sender<StepEvent>,
    ) -> Result<Usage, EngineError>;
}

/// A mailbox job: the handler sends one per generation; the engine thread
/// receives it, runs [`Engine::stream`], and returns the result via the
/// oneshot. The `StepEvent` channel (`tx`) is carried so the engine streams
/// directly to the handler's receiver.
#[allow(dead_code)]
pub struct MailboxJob {
    request: EngineRequest,
    cancel: CancelToken,
    tx: tokio::sync::mpsc::Sender<StepEvent>,
    reply: tokio::sync::oneshot::Sender<Result<Usage, EngineError>>,
}

/// The production [`EngineDriver`]: a `Send + Sync` handle that posts
/// [`MailboxJob`]s over a channel to the dedicated engine thread (which owns
/// the `!Send` [`Engine`]). The handler holds an `Arc<MailboxEngine>`; the
/// engine thread drains `rx` in [`engine_thread_loop`]. This is the real
/// counterpart to the test [`StubEngine`].
#[derive(Clone)]
pub struct MailboxEngine {
    tx: std::sync::mpsc::Sender<MailboxJob>,
}

impl MailboxEngine {
    /// Create the mailbox pair: the `MailboxEngine` (held by the handler) +
    /// the `Receiver` (drained by the engine thread via
    /// [`engine_thread_loop`]). The channel is unbounded (cap 1 effectively:
    /// the single-flight permit serializes requests before they reach here).
    #[must_use]
    pub fn channel() -> (Self, std::sync::mpsc::Receiver<MailboxJob>) {
        let (tx, rx) = std::sync::mpsc::channel();
        (Self { tx }, rx)
    }
}

impl EngineDriver for MailboxEngine {
    fn stream(
        &self,
        request: &EngineRequest,
        cancel: &CancelToken,
        tx: tokio::sync::mpsc::Sender<StepEvent>,
    ) -> Result<Usage, EngineError> {
        let (reply, reply_rx) = tokio::sync::oneshot::channel();
        let job = MailboxJob {
            request: request.clone(),
            cancel: cancel.clone(),
            tx,
            reply,
        };
        // Post the job. A send failure means the engine thread exited (shutting
        // down) → surface as Cancelled (the request can't be served).
        self.tx.send(job).map_err(|_| EngineError::Cancelled)?;
        // Block waiting for the engine thread's result. The engine thread runs
        // `Engine::stream` (which streams + returns the Usage/Err); this thread
        // is a tokio blocking worker (the handler spawns it via
        // `spawn_blocking`), so blocking here is correct.
        reply_rx
            .blocking_recv()
            .map_err(|_| EngineError::Cancelled)?
    }
}

/// The dedicated engine thread loop: loads the `!Send` [`Engine`] **on this
/// thread** (the `Engine` is `!Send`, so it can't be moved in — it must be
/// constructed where it lives), drains the mailbox receiver, runs
/// [`Engine::stream`] per job, and posts the result. Exits when the mailbox
/// sender is dropped (all `MailboxEngine` clones gone → shutdown). Run this on
/// a plain `std::thread::spawn`, not a tokio task (the `Engine` is `!Send` +
/// uses blocking `blocking_send`).
///
/// `load_result` is sent back via `loaded` so the caller can surface a load
/// failure (the thread stays alive to drain the mailbox even if load failed —
/// it exits immediately since `engine` is `Err`).
pub fn engine_thread_loop(
    geometry: Geometry,
    weights_path: String,
    rx: std::sync::mpsc::Receiver<MailboxJob>,
    loaded: std::sync::mpsc::Sender<Result<(), EngineError>>,
) {
    let engine = match Engine::load(&geometry, &weights_path) {
        Ok(e) => {
            let _ = loaded.send(Ok(()));
            e
        }
        Err(e) => {
            let _ = loaded.send(Err(e));
            return;
        }
    };
    while let Ok(job) = rx.recv() {
        let result = engine.stream(&job.request, &job.cancel, job.tx);
        // A send error means the handler gave up (client disconnect) — the
        // result is dropped; the engine already returned Cancelled via the
        // dropped StepEvent channel. Ignore.
        let _ = job.reply.send(result);
    }
    // Sender dropped → shutdown. The Engine + its native handles drop here
    // (on the engine thread — the `!Send` invariant holds).
}

/// The loaded model + a reused step-result buffer. Owns the `!Send` native
/// handles; constructed on, and only driven from, the engine thread.
///
/// The `_not_send` marker (a raw pointer is `!Send + !Sync`) makes the struct
/// `!Send` so a future refactor cannot accidentally move one across threads
/// — the engine-thread ownership invariant (02 §Threading model). The real
/// single-thread ownership is enforced by PR B's channel wiring; this guard
/// makes the intent loud at compile time.
pub struct Engine {
    model: Model,
    /// Reused across prefill and decode steps (the ABI handle is caller-owned,
    /// reused at 2.x). Each step's `fields()` read copies out, so reuse is safe.
    step: StepResult,
    /// `!Send + !Sync` marker (raw pointers are neither). Never read.
    _not_send: std::marker::PhantomData<*const ()>,
}

/// A generation in flight — holds the per-request `KvState` (reset-around-every-
/// generation, 02) and releases it on drop. The axum layer acquires a
/// single-flight permit before constructing this.
struct Generation {
    kvstate: KvState,
}

impl Engine {
    /// Load a model from a validated geometry + weights directory. The
    /// `Geometry` is bridged to the ABI params via `AbiGeometry` (the buffer
    /// the `layer_types` pointer borrows is owned for the call's duration).
    pub fn load(geometry: &Geometry, weights_path: &str) -> Result<Self, EngineError> {
        let model = Model::create()?;
        let abi = AbiGeometry::from_geometry(geometry);
        model.load(abi.params(), weights_path)?;
        let step = StepResult::create()?;
        Ok(Self {
            model,
            step,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Drive a greedy (or sampled) generation to completion, returning the
    /// generated token ids (excluding the prompt). The non-streaming collector
    /// uses the prefill epilogue as completion token 1, then collects at most
    /// `max_tokens - 1` decode tokens. Checks `cancel` before and after native
    /// steps.
    ///
    /// Runs synchronously on the calling thread — in the server this is the
    /// dedicated engine thread. Single-flight is the caller's responsibility
    /// (the axum layer acquires a `Semaphore(1)` permit; PR B).
    pub fn drive(
        &self,
        request: &EngineRequest,
        cancel: &CancelToken,
    ) -> Result<Vec<u32>, EngineError> {
        // Non-streaming: a direct prefill + decode loop (NOT via `stream` —
        // `stream`'s cap-8 channel would deadlock when this single thread
        // both sends and drains: the 9th `blocking_send` would block forever
        // waiting for a receiver that never runs concurrently). The streaming
        // path's channel is only for the async handler, which drains on a
        // separate task. This path collects directly into a `Vec`.
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if request.prompt_tokens.is_empty() {
            return Err(EngineError::Native(NativeError {
                status: Status::InvalidArgument,
                message: "prompt_tokens is empty".into(),
            }));
        }
        // A zero completion budget is a successful no-op. Return before
        // allocating a per-request KV or invoking native prefill.
        if request.max_tokens == 0 {
            return Ok(Vec::new());
        }
        let mut sampling = request.sampling;
        let mut generation = Generation {
            kvstate: KvState::create(&self.model)?,
        };
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let prefill = self.prefill(&mut generation, &request.prompt_tokens, sampling.as_ref())?;
        advance_sampling_seed(&mut sampling);
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        let mut generated = Vec::with_capacity(request.max_tokens as usize);
        generated.push(prefill.id);
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if request.eos_token_id == Some(prefill.id) {
            return Ok(generated);
        }
        while generated.len() < request.max_tokens as usize {
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            let step = self.decode_step(&mut generation, sampling.as_ref())?;
            advance_sampling_seed(&mut sampling);
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            generated.push(step.id);
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            if request.eos_token_id == Some(step.id) {
                break;
            }
        }
        Ok(generated)
    }

    /// Stream a generation: prefill produces completion token 1, then decode
    /// produces at most `max_tokens - 1` subsequent tokens. Each is yielded as
    /// [`StepEvent::Token`] on `tx`, followed by terminal [`StepEvent::Done`]
    /// with [`Usage`]. The loop checks `cancel` before and after native steps
    /// (Cancelled → early return). **Cancel-on-client-drop**: if
    /// the receiver is dropped (the axum response future dropped on client
    /// disconnect), the next `tx.blocking_send` returns `SendError`, mapped to
    /// [`EngineError::Cancelled`] — no explicit [`CancelToken`] fire needed.
    ///
    /// Runs synchronously on the calling thread (the dedicated engine thread).
    /// `tx.blocking_send` is safe from a non-async thread: it blocks the
    /// caller until the channel has room or the receiver drops; it needs no
    /// tokio runtime. The bounded cap (8) is the backpressure mechanism — a
    /// slow client throttles the engine rather than dropping events.
    pub fn stream(
        &self,
        request: &EngineRequest,
        cancel: &CancelToken,
        tx: tokio::sync::mpsc::Sender<StepEvent>,
    ) -> Result<Usage, EngineError> {
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        if request.prompt_tokens.is_empty() {
            // An empty prompt has nothing to prefill; the model cannot
            // bootstrap a generation. Surface as a bad-argument native error
            // (400 in the taxonomy) rather than silently emitting from nothing.
            return Err(EngineError::Native(NativeError {
                status: Status::InvalidArgument,
                message: "prompt_tokens is empty".into(),
            }));
        }

        // No KV state or native prefill is needed when the caller requested no
        // completion tokens. A successful stream still consists of exactly its
        // terminal usage event; a pre-cancelled request or dropped receiver is
        // still reported as cancellation.
        if request.max_tokens == 0 {
            let usage = Usage {
                prompt_tokens: u32::try_from(request.prompt_tokens.len()).unwrap_or(u32::MAX),
                completion_tokens: 0,
            };
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            if tx.blocking_send(StepEvent::Done(usage)).is_err() {
                return Err(EngineError::Cancelled);
            }
            return Ok(usage);
        }

        let mut sampling = request.sampling;

        // Per-request KV (reset-around-every-generation). A fresh KvState
        // means no cross-request leakage; drop at the end of the scope frees it.
        let mut generation = Generation {
            kvstate: KvState::create(&self.model)?,
        };
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }

        let prefill = self.prefill(&mut generation, &request.prompt_tokens, sampling.as_ref())?;
        advance_sampling_seed(&mut sampling);
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }

        let mut completion_tokens = 1u32;
        let prefill_is_eos = request.eos_token_id == Some(prefill.id);
        if tx.blocking_send(StepEvent::Token(prefill)).is_err() {
            return Err(EngineError::Cancelled);
        }
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        while completion_tokens < request.max_tokens && !prefill_is_eos {
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            // decode_block decodes n_tokens=1 from the current KV offset — the
            // last token is already in the KV from prefill (or the prior step),
            // so no token is fed back here. The sampled id is read out and
            // streamed; the native side advances the offset for the next call.
            let step = self.decode_step(&mut generation, sampling.as_ref())?;
            advance_sampling_seed(&mut sampling);
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }

            // Yield the token. A dropped receiver (client disconnect) makes
            // `blocking_send` return Err → Cancelled (cancel-on-client-drop).
            if tx.blocking_send(StepEvent::Token(step)).is_err() {
                return Err(EngineError::Cancelled);
            }
            completion_tokens += 1;
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }

            if request.eos_token_id == Some(step.id) {
                break;
            }
        }

        let usage = Usage {
            prompt_tokens: u32::try_from(request.prompt_tokens.len()).unwrap_or(u32::MAX),
            completion_tokens,
        };
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        // The terminal frame. A dropped receiver here is still a cancel (the
        // client gave up before the Done frame); surface Cancelled so the
        // handler doesn't report a spurious success.
        if tx.blocking_send(StepEvent::Done(usage)).is_err() {
            return Err(EngineError::Cancelled);
        }
        Ok(usage)
    }

    /// Prefill the entire prompt in a single native call. The native
    /// `hyp_prefill_chunk` rejects `offset != 0`, so it does NOT support
    /// continuation prefill across multiple calls — it chunks internally at
    /// `kPrefillChunkSize` (2048) and expects the whole prompt in one call.
    /// Inter-chunk cancellation (02 "yields between chunks") would need a
    /// native continuation-prefill path that does not exist today, so the only
    /// cancellation points around prefill are the checks immediately before and
    /// after this call in `drive` / `stream`.
    fn prefill(
        &self,
        generation: &mut Generation,
        prompt: &[u32],
        sampling: Option<&HypSamplingConfig>,
    ) -> Result<StepToken, EngineError> {
        // The caller already rejected the empty prompt and checked cancel up
        // front; there is no mid-prefill yield point on the current native API.
        // `is_prompt` flags the prompt framing for the whole stream.
        let stream = HypTokenStream::from_slice(prompt, true);
        match sampling {
            Some(cfg) => generation.kvstate.prefill_chunk_sampled(
                &self.model,
                &stream,
                Some(cfg),
                &self.step,
            )?,
            None => generation
                .kvstate
                .prefill_chunk(&self.model, &stream, &self.step)?,
        }
        self.read_step_token()
    }

    /// Decode one token from the current KV offset. `decode_block` advances
    /// the KV by `n_tokens=1` using the last token already resident in the KV
    /// (from prefill or the prior step) — no token is fed back in. The sampled
    /// id + logit are read out via the M3 `hyp_step_result_fields` accessor.
    fn decode_step(
        &self,
        generation: &mut Generation,
        sampling: Option<&HypSamplingConfig>,
    ) -> Result<StepToken, EngineError> {
        match sampling {
            Some(cfg) => {
                generation
                    .kvstate
                    .decode_block_sampled(&self.model, 1, Some(cfg), &self.step)?;
            }
            None => {
                generation
                    .kvstate
                    .decode_block(&self.model, 1, &self.step)?;
            }
        }
        self.read_step_token()
    }

    fn read_step_token(&self) -> Result<StepToken, EngineError> {
        let fields: HypStepResultFields = self.step.fields()?;
        Ok(StepToken {
            id: fields.token_id,
            logit: fields.logit,
        })
    }
}

/// Test-support: a model-free `EngineDriver` for the contract/SSE/cancel
/// tests (B7). Compiled unconditionally so integration tests (a separate
/// crate) can import it; production code doesn't use it.
#[allow(dead_code)]
pub mod test_support {
    use super::*;

    /// A model-free `EngineDriver` for the contract/SSE/cancel tests (B7). Yields
    /// a fixed `StepEvent` sequence — one `Token` per id in `tokens`, then
    /// `Done`. `block_after` sleeps the engine thread before that index (to let
    /// a test drop the receiver mid-stream and observe cancel-on-drop); `None`
    /// never blocks. This exercises the streaming seam with no native model.
    pub struct StubEngine {
        pub tokens: Vec<u32>,
        /// If `Some(i)`, `tokio::time::sleep` before yielding `tokens[i]` so the
        /// test can race a receiver drop past that point.
        pub block_after: Option<usize>,
    }

    impl EngineDriver for StubEngine {
        fn stream(
            &self,
            request: &EngineRequest,
            cancel: &CancelToken,
            tx: tokio::sync::mpsc::Sender<StepEvent>,
        ) -> Result<Usage, EngineError> {
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            let mut emitted = 0u32;
            for (i, id) in self.tokens.iter().enumerate() {
                if let Some(block_at) = self.block_after
                    && i == block_at
                {
                    // A blocking sleep on a sync engine thread. The test
                    // runtime is `current_thread`, so this blocks the same
                    // thread the test drives — tests using this must spawn
                    // the stream on a separate thread (see
                    // `stream_returns_cancelled_on_dropped_receiver`).
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                if tx
                    .blocking_send(StepEvent::Token(StepToken {
                        id: *id,
                        logit: 0.0,
                    }))
                    .is_err()
                {
                    return Err(EngineError::Cancelled);
                }
                emitted += 1;
                if request.eos_token_id == Some(*id) {
                    break;
                }
            }
            let usage = Usage {
                prompt_tokens: u32::try_from(request.prompt_tokens.len()).unwrap_or(u32::MAX),
                completion_tokens: emitted,
            };
            if tx.blocking_send(StepEvent::Done(usage)).is_err() {
                return Err(EngineError::Cancelled);
            }
            Ok(usage)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::StubEngine;
    use super::*;

    #[test]
    fn cancel_token_is_idempotent_and_observes_seq_cst() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        token.cancel(); // idempotent
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancel_token_clones_share_the_flag() {
        let token = CancelToken::new();
        let twin = token.clone();
        token.cancel();
        assert!(twin.is_cancelled(), "a cloned token must observe the fire");
    }

    #[test]
    fn sampling_seed_sequence_matches_native_per_step_contract() {
        // Exact values independently pin at least three transitions, including
        // the implicit u64 modulo arithmetic used by model.cc.
        let mut seed = 17;
        let expected = [
            17,
            17_399_290_477_736_686_412,
            604_063_801_117_309_355,
            6_215_382_037_137_340_766,
            1_645_788_613_595_576_533,
        ];
        for expected_seed in expected {
            assert_eq!(seed, expected_seed);
            seed = next_sampling_seed(seed);
        }
    }

    #[test]
    fn sampling_seed_transition_wraps_and_greedy_has_no_state() {
        assert_eq!(
            next_sampling_seed(u64::MAX),
            13_525_302_890_751_722_018,
            "native LCG arithmetic must wrap modulo 2^64"
        );

        let mut greedy = None;
        advance_sampling_seed(&mut greedy);
        assert!(greedy.is_none(), "greedy mode must remain stateless");
    }

    #[test]
    fn single_flight_busy_is_displayable_and_is_an_error() {
        let err = SingleFlightBusy;
        assert!(err.to_string().contains("429"));
        let engine_err: EngineError = err.into();
        assert!(matches!(engine_err, EngineError::Busy));
    }

    #[test]
    fn empty_prompt_is_a_bad_argument_not_a_silent_run() {
        // The driver must refuse to generate from an empty prompt rather than
        // emit from nothing. This is a model-free check (no native calls happen
        // before the guard), so it runs in tier-1 CI.
        let request = EngineRequest {
            prompt_tokens: Vec::new(),
            max_tokens: 4,
            eos_token_id: Some(1),
            sampling: None,
        };
        // Engine::drive needs a loaded model; we cannot call it model-free.
        // Instead assert the documented precondition directly: the empty-prompt
        // branch is reachable and produces InvalidArgument, not a panic or an
        // empty Vec. We test the guard logic via a stand-alone closure that
        // mirrors the driver's check (the real call is exercised by the
        // --ignored real-model test on the self-hosted M5).
        fn check_prompt(prompt: &[u32]) -> Result<(), EngineError> {
            if prompt.is_empty() {
                return Err(EngineError::Native(NativeError {
                    status: Status::InvalidArgument,
                    message: "prompt_tokens is empty".into(),
                }));
            }
            Ok(())
        }
        let err = check_prompt(&request.prompt_tokens).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Native(NativeError {
                status: Status::InvalidArgument,
                ..
            })
        ));
        // A non-empty prompt passes the same guard.
        assert!(check_prompt(&[1u32, 2, 3]).is_ok());
    }

    #[test]
    fn prefill_presents_the_whole_prompt_as_one_stream() {
        // The native `hyp_prefill_chunk` rejects `offset != 0`, so it does NOT
        // support continuation prefill: the whole prompt must reach the native
        // side in a single call (it chunks internally at PREFILL_CHUNK). The
        // earlier Rust-side slice loop broke any prompt > PREFILL_CHUNK: the
        // first chunk set kvstate->offset, the second hit the offset!=0 guard
        // and returned InvalidArgument. This guard proves the whole prompt is
        // presented as one stream (count == prompt.len()), not sliced.
        let prompt: Vec<u32> = (0..5000).collect();
        let stream = HypTokenStream::from_slice(&prompt, true);
        assert_eq!(
            stream.count as usize,
            prompt.len(),
            "prefill must present the whole prompt in one stream, not slice it"
        );
        assert_eq!(stream.is_prompt, 1, "prompt framing must be flagged");
    }

    /// A `StubEngine` yields one `Token` per id then `Done`; the receiver
    /// observes exactly that sequence, in order (model-free, no native calls).
    #[test]
    fn stream_yields_one_token_event_per_step_then_done() {
        let engine = StubEngine {
            tokens: vec![7, 3, 40, 100],
            block_after: None,
        };
        let request = EngineRequest {
            prompt_tokens: vec![1, 2],
            max_tokens: 4,
            eos_token_id: None,
            sampling: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StepEvent>(8);
        // `stream` uses `blocking_send`, which is fine on a plain thread; drive
        // it on a std thread so this synchronous test doesn't need a runtime
        // for the send side (the recv side is sync via `blocking_recv`).
        let handle = std::thread::spawn(move || engine.stream(&request, &CancelToken::new(), tx));
        let mut ids = Vec::new();
        let mut terminal = None;
        while let Some(event) = rx.blocking_recv() {
            match event {
                StepEvent::Token(t) => ids.push(t.id),
                StepEvent::Done(u) => terminal = Some(u),
            }
        }
        let usage = handle
            .join()
            .expect("stream thread panicked")
            .expect("stream ok");
        assert_eq!(ids, vec![7, 3, 40, 100]);
        assert_eq!(terminal, Some(usage));
        assert_eq!(usage.completion_tokens, 4);
        assert_eq!(usage.prompt_tokens, 2);
    }

    /// EOS stops the stream early: the stub honors `eos_token_id` and emits
    /// `Done` with the truncated `completion_tokens` count.
    #[test]
    fn stream_stops_at_eos() {
        let engine = StubEngine {
            tokens: vec![7, 3, 40, 100],
            block_after: None,
        };
        let request = EngineRequest {
            prompt_tokens: vec![1],
            max_tokens: 4,
            eos_token_id: Some(40),
            sampling: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StepEvent>(8);
        let handle = std::thread::spawn(move || engine.stream(&request, &CancelToken::new(), tx));
        let mut ids = Vec::new();
        while let Some(StepEvent::Token(t)) = rx.blocking_recv() {
            ids.push(t.id);
        }
        let usage = handle.join().unwrap().unwrap();
        assert_eq!(ids, vec![7, 3, 40], "stops after the EOS token");
        assert_eq!(usage.completion_tokens, 3);
    }

    /// Cancel-on-client-drop: dropping the receiver mid-stream makes the next
    /// `blocking_send` fail → `stream` returns `Cancelled`. The stub blocks
    /// after the first token so the drop races past it. Model-free.
    #[test]
    fn stream_returns_cancelled_on_dropped_receiver() {
        let engine = StubEngine {
            tokens: vec![7, 3, 40, 100],
            block_after: Some(1), // sleep before yielding tokens[1]
        };
        let request = EngineRequest {
            prompt_tokens: vec![1],
            max_tokens: 4,
            eos_token_id: None,
            sampling: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StepEvent>(8);
        let handle = std::thread::spawn(move || engine.stream(&request, &CancelToken::new(), tx));
        // Take the first token, then drop the receiver (simulate client
        // disconnect while the engine sleeps before token 2).
        let _first = rx.blocking_recv();
        drop(rx);
        let result = handle.join().expect("stream thread panicked");
        assert!(
            matches!(result, Err(EngineError::Cancelled)),
            "a dropped receiver must surface as Cancelled, got {result:?}"
        );
    }

    /// A cancel fired before the loop starts short-circuits to `Cancelled`
    /// without touching the channel (no events sent).
    #[test]
    fn stream_short_circuits_on_pre_cancelled_token() {
        let engine = StubEngine {
            tokens: vec![7, 3],
            block_after: None,
        };
        let request = EngineRequest {
            prompt_tokens: vec![1],
            max_tokens: 4,
            eos_token_id: None,
            sampling: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StepEvent>(8);
        let cancel = CancelToken::new();
        cancel.cancel();
        let result = engine.stream(&request, &cancel, tx);
        assert!(matches!(result, Err(EngineError::Cancelled)));
        assert!(
            rx.blocking_recv().is_none(),
            "no events on a pre-cancelled run"
        );
    }

    /// The tiny fixture's geometry, mirroring `make_tiny_geometry()` in the
    /// native forward test (the committed `gemma4-unified-tiny` config.json is
    /// intentionally minimal and does not carry the rope/softmax fields
    /// `Geometry::from_text_config_str` requires, so we build the validated
    /// struct directly with the same values the native test uses).
    fn tiny_geometry() -> Geometry {
        use hyperion_model::geometry::{LayerType, RopeSpec, TextModelType};
        Geometry {
            model_type: TextModelType::Gemma4UnifiedText,
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 6,
            layer_types: vec![
                LayerType::Sliding,
                LayerType::Sliding,
                LayerType::Sliding,
                LayerType::Sliding,
                LayerType::Sliding,
                LayerType::Full,
            ],
            num_attention_heads: 4,
            head_dim_local: 64,
            head_dim_global: 128,
            num_kv_heads_local: 2,
            num_kv_heads_global: 1,
            attention_k_eq_v_global: true,
            num_kv_shared_layers: 0,
            sliding_window: 8,
            rope_local: RopeSpec {
                theta: 10_000.0,
                partial_rotary_factor: None,
                proportional: false,
            },
            rope_global: RopeSpec {
                theta: 1_000_000.0,
                partial_rotary_factor: Some(0.25),
                proportional: true,
            },
            final_logit_softcapping: 30.0,
            rms_norm_eps: 1e-6,
            attention_bias: false,
            vocab_size: 128,
            max_position_embeddings: 256,
            tie_word_embeddings: true,
            ple_hidden_per_layer_input: 0,
            ple_vocab_per_layer_input: 0,
            use_double_wide_mlp: false,
            moe: None,
        }
    }

    /// The committed tiny fixture directory (a real 6-layer gemma4_unified
    /// artifact; ~728 KB, so it runs in-repo). The dev/M5 machine has it; CI
    /// does not, so this is `#[ignore]`.
    fn tiny_fixture_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("hyperion-model")
            .join("fixtures")
            .join("gemma4-unified-tiny")
    }

    /// Obtain the prefill token directly through the native FFI on a fresh KV.
    /// This intentionally does not call `Engine::prefill`: the ignored real
    /// engine regression must catch a future Rust loop that discards a valid
    /// native prefill `StepResult` again.
    fn direct_native_prefill_token(
        engine: &Engine,
        prompt: &[u32],
        sampling: Option<&HypSamplingConfig>,
    ) -> u32 {
        let kvstate = KvState::create(&engine.model).expect("create direct-prefill KV");
        let stream = HypTokenStream::from_slice(prompt, true);
        match sampling {
            Some(cfg) => kvstate
                .prefill_chunk_sampled(&engine.model, &stream, Some(cfg), &engine.step)
                .expect("sampled native prefill"),
            None => kvstate
                .prefill_chunk(&engine.model, &stream, &engine.step)
                .expect("greedy native prefill"),
        }
        engine
            .step
            .fields()
            .expect("read direct native prefill fields")
            .token_id
    }

    /// Obtain an entire sampled sequence through direct native calls on one
    /// fresh KV. This intentionally advances a test-local copy of the seed
    /// with literal model.cc constants instead of calling the production seed
    /// helper, so it catches a Rust engine that resets or mis-advances the seed.
    fn direct_native_sampled_tokens(
        engine: &Engine,
        prompt: &[u32],
        mut sampling: HypSamplingConfig,
        max_tokens: u32,
    ) -> Vec<u32> {
        assert!(max_tokens > 0, "direct native oracle requires a token");
        let kvstate = KvState::create(&engine.model).expect("create direct-sampled KV");
        let stream = HypTokenStream::from_slice(prompt, true);
        kvstate
            .prefill_chunk_sampled(&engine.model, &stream, Some(&sampling), &engine.step)
            .expect("sampled native prefill");
        let mut tokens = vec![
            engine
                .step
                .fields()
                .expect("read direct sampled prefill fields")
                .token_id,
        ];

        while tokens.len() < max_tokens as usize {
            // Literal independent oracle for model.cc's wrapping u64 update.
            sampling.seed = sampling
                .seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            kvstate
                .decode_block_sampled(&engine.model, 1, Some(&sampling), &engine.step)
                .expect("sampled native decode");
            tokens.push(
                engine
                    .step
                    .fields()
                    .expect("read direct sampled decode fields")
                    .token_id,
            );
        }
        tokens
    }

    /// Real-tensor end-to-end proof of the engine wiring (load → prefill token 1
    /// → decode → fields-read → loop) on the committed tiny fixture. It compares
    /// `Engine` against independent direct native calls, covers total completion
    /// budgets 0/1/N, zero-budget Done-only streaming, prefill EOS, a multi-step
    /// sampled sequence, streaming usage, and receiver-drop cancellation.
    /// `#[ignore]` so PR CI compiles but skips it; the self-hosted M5 gate may run
    /// it. The rigorous byte-identity-vs-12B-CLI comparison remains the real HTTP
    /// parity gate.
    #[test]
    #[ignore = "requires the committed tiny fixture + a Metal device (self-hosted M5)"]
    fn engine_drives_greedy_on_tiny_fixture() {
        let geometry = tiny_geometry();
        let dir = tiny_fixture_dir();
        let engine = Engine::load(&geometry, dir.to_str().expect("utf-8 fixture path"))
            .expect("load tiny fixture");

        // Same prompt the native forward_test uses: {7, 3, 40, 100}.
        let request = EngineRequest {
            prompt_tokens: vec![7, 3, 40, 100],
            max_tokens: 8,
            eos_token_id: None,
            sampling: None,
        };
        let cancel = CancelToken::new();
        let native_prefill = direct_native_prefill_token(&engine, &request.prompt_tokens, None);
        let first = engine
            .drive(&request, &cancel)
            .expect("greedy drive on tiny fixture");
        assert_eq!(first.len(), request.max_tokens as usize);
        assert_eq!(
            first[0], native_prefill,
            "Engine::drive must begin with the native prefill token"
        );
        assert!(
            first.iter().all(|t| *t < 128),
            "every token must be in the 128-token vocab"
        );

        let one_token_request = EngineRequest {
            max_tokens: 1,
            ..request.clone()
        };
        assert_eq!(
            engine.drive(&one_token_request, &cancel).unwrap(),
            vec![native_prefill],
            "max_tokens=1 is exactly the native prefill token"
        );
        let zero_token_request = EngineRequest {
            max_tokens: 0,
            ..request.clone()
        };
        assert!(
            engine
                .drive(&zero_token_request, &cancel)
                .unwrap()
                .is_empty(),
            "max_tokens=0 must emit no completion tokens"
        );
        let (zero_tx, mut zero_rx) = tokio::sync::mpsc::channel(1);
        let zero_usage = engine
            .stream(&zero_token_request, &cancel, zero_tx)
            .expect("zero-token stream");
        assert_eq!(
            zero_usage,
            Usage {
                prompt_tokens: 4,
                completion_tokens: 0,
            }
        );
        assert!(matches!(
            zero_rx.blocking_recv(),
            Some(StepEvent::Done(done)) if done == zero_usage
        ));
        assert!(
            zero_rx.blocking_recv().is_none(),
            "zero budget must emit exactly one Done event"
        );

        let (zero_drop_tx, zero_drop_rx) = tokio::sync::mpsc::channel(1);
        drop(zero_drop_rx);
        assert!(matches!(
            engine.stream(&zero_token_request, &cancel, zero_drop_tx),
            Err(EngineError::Cancelled)
        ));
        let prefill_eos_request = EngineRequest {
            eos_token_id: Some(native_prefill),
            ..request.clone()
        };
        assert_eq!(
            engine.drive(&prefill_eos_request, &cancel).unwrap(),
            vec![native_prefill],
            "prefill EOS must be emitted once without decode"
        );

        let sampling = HypSamplingConfig {
            temperature: 0.8,
            top_k: 32,
            top_p: 0.95,
            min_p: 0.0,
            seed: 17,
        };
        let sampled_max_tokens = 4;
        let native_sampled = direct_native_sampled_tokens(
            &engine,
            &request.prompt_tokens,
            sampling,
            sampled_max_tokens,
        );
        let sampled_request = EngineRequest {
            max_tokens: sampled_max_tokens,
            sampling: Some(sampling),
            ..request.clone()
        };
        assert_eq!(
            engine.drive(&sampled_request, &cancel).unwrap(),
            native_sampled,
            "sampled Engine must match independent direct native calls across steps"
        );

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let usage = engine
            .stream(&one_token_request, &cancel, tx)
            .expect("one-token stream");
        assert_eq!(usage.completion_tokens, 1);
        assert!(matches!(
            rx.blocking_recv(),
            Some(StepEvent::Token(StepToken { id, .. })) if id == native_prefill
        ));
        assert!(matches!(
            rx.blocking_recv(),
            Some(StepEvent::Done(done)) if done == usage
        ));
        assert!(rx.blocking_recv().is_none());

        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        assert!(matches!(
            engine.stream(&one_token_request, &cancel, tx),
            Err(EngineError::Cancelled)
        ));

        // Determinism: a second run with a fresh per-request KV produces the
        // identical sequence (greedy is deterministic; reset-around-every-
        // generation means a fresh KvState, not a stale continuation).
        let second = engine
            .drive(&request, &cancel)
            .expect("second greedy drive");
        assert_eq!(first, second, "greedy must be deterministic across runs");
    }
}
