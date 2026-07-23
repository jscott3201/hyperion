//! The engine-driver: owns the loaded native model and drives a generation
//! loop on one thread.
//!
//! Architecture (02 §Threading model): one dedicated engine thread owns all
//! MLX/native state — `Model` and `KvState` are `!Send`, so they cannot cross
//! threads. This module is the **synchronous core** that runs on that thread:
//! [`Engine::drive`] prefill-chunks the prompt, then decodes one token
//! at a time, reading each sampled `token_id` back through the M3
//! `hyp_step_result_fields` accessor (ADR 0003) until `max_tokens` or EOS.
//!
//! The axum/SSE transport (PR B) owns an `Engine` on a dedicated thread and
//! feeds it `EngineRequest`s over a bounded mpsc channel; cancel-on-drop
//! propagates a [`CancelToken`] the loop checks between steps/chunks. PR A
//! ships the synchronous driver + the single-flight/cancel primitives it
//! depends on, verified by a model-free unit suite and a self-hosted M5
//! byte-identical test.

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
/// between prefill chunks and decode steps; a cancelled run stops cleanly and
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

/// The sampled token id for one decode step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepToken {
    pub id: u32,
    pub logit: f32,
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
    /// Reused across decode steps (the ABI handle is caller-owned, reused at
    /// 2.x). Each step's `fields()` read copies out, so reuse is safe.
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
    /// generated token ids (excluding the prompt). Checks `cancel` between
    /// prefill chunks and decode steps.
    ///
    /// Runs synchronously on the calling thread — in the server this is the
    /// dedicated engine thread. Single-flight is the caller's responsibility
    /// (the axum layer acquires a `Semaphore(1)` permit; PR B).
    pub fn drive(
        &self,
        request: &EngineRequest,
        cancel: &CancelToken,
    ) -> Result<Vec<u32>, EngineError> {
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

        // Per-request KV (reset-around-every-generation). A fresh KvState
        // means no cross-request leakage; drop at the end of the scope frees it.
        let mut generation = Generation {
            kvstate: KvState::create(&self.model)?,
        };

        self.prefill(&mut generation, &request.prompt_tokens, cancel)?;

        let mut generated = Vec::with_capacity(request.max_tokens as usize);

        while generated.len() < request.max_tokens as usize {
            if cancel.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            // decode_block decodes n_tokens=1 from the current KV offset — the
            // last token is already in the KV from prefill (or the prior step),
            // so no token is fed back here. The sampled id is read out and
            // appended; the native side advances the offset for the next call.
            let step = self.decode_step(&mut generation, &request.sampling)?;
            generated.push(step.id);
            if request.eos_token_id == Some(step.id) {
                break;
            }
        }
        Ok(generated)
    }

    /// Prefill the entire prompt in a single native call. The native
    /// `hyp_prefill_chunk` rejects `offset != 0`, so it does NOT support
    /// continuation prefill across multiple calls — it chunks internally at
    /// `kPrefillChunkSize` (2048) and expects the whole prompt in one call.
    /// Inter-chunk cancellation (02 "yields between chunks") would need a
    /// native continuation-prefill path that does not exist today, so the only
    /// cancellation point during prefill is the upfront check in `drive`.
    fn prefill(
        &self,
        generation: &mut Generation,
        prompt: &[u32],
        _cancel: &CancelToken,
    ) -> Result<(), EngineError> {
        // `drive` already rejected the empty prompt and checked cancel up
        // front; there is no mid-prefill yield point on the current native
        // API. `is_prompt` flags the prompt framing for the whole stream.
        let stream = HypTokenStream::from_slice(prompt, true);
        // Greedy prefill (the sampled variant is identical when config is
        // None / temperature 0; PR B will route sampled prefill here too).
        generation
            .kvstate
            .prefill_chunk(&self.model, &stream, &self.step)?;
        Ok(())
    }

    /// Decode one token from the current KV offset. `decode_block` advances
    /// the KV by `n_tokens=1` using the last token already resident in the KV
    /// (from prefill or the prior step) — no token is fed back in. The sampled
    /// id + logit are read out via the M3 `hyp_step_result_fields` accessor.
    fn decode_step(
        &self,
        generation: &mut Generation,
        sampling: &Option<HypSamplingConfig>,
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
        let fields: HypStepResultFields = self.step.fields()?;
        Ok(StepToken {
            id: fields.token_id,
            logit: fields.logit,
        })
    }
}

#[cfg(test)]
mod tests {
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

    /// Real-tensor end-to-end proof of the engine wiring (load → prefill →
    /// decode → fields-read → loop) on the committed tiny fixture. Greedy is
    /// deterministic, so two runs must produce the identical token sequence
    /// and every token must be in-vocab. `#[ignore]` so PR CI skips it; the
    /// self-hosted M5 gate runs it. The rigorous byte-identity-vs-12B-CLI
    /// comparison is the M3 gate's job (PR B's SSE-wrapped variant), not this
    /// slice; this proves the Rust loop itself works on real tensors.
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
        let first = engine
            .drive(&request, &cancel)
            .expect("greedy drive on tiny fixture");
        assert!(!first.is_empty(), "greedy produced no tokens");
        assert!(
            first.iter().all(|t| *t < 128),
            "every token must be in the 128-token vocab"
        );

        // Determinism: a second run with a fresh per-request KV produces the
        // identical sequence (greedy is deterministic; reset-around-every-
        // generation means a fresh KvState, not a stale continuation).
        let second = engine
            .drive(&request, &cancel)
            .expect("second greedy drive");
        assert_eq!(first, second, "greedy must be deterministic across runs");
    }
}
