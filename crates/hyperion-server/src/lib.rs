//! M3 serving: the axum dual-dialect HTTP + real SSE surface on top of the
//! engine-driver (PR A). One single router, localhost-first, 32 MiB body limit
//! (06 §Surfaces). Both dialects normalize to one `PreparedPrompt`; dialect
//! only selects SSE framing + error envelope.
//!
//! Modules:
//! - [`engine`] — the `!Send` native engine + the `EngineDriver` streaming
//!   seam (PR A, landed in `70fa376`; PR B added `stream` + `StepEvent`).
//! - [`auth`] — the bind-gate (fail-closed on non-loopback without a bearer)
//!   + constant-time bearer middleware (B3).
//! - [`dialect`] + [`prepare`] — parse Anthropic/OpenAI bodies → `ChatMessage`
//!   list → render → tokenize → `PreparedPrompt` (B4).
//! - [`sse`] — the pure dialect SSE framers (Anthropic event sequence,
//!   OpenAI chunked + `[DONE]`) + the 529 mid-stream error event (B5).
//! - [`control`] — the `/control` ops surface (health/stats/reload+409/
//!   shutdown) (B6).
//! - [`tool_call`] — transport-neutral incremental Gemma 4 tool-call parsing,
//!   bounded dedupe, finite repair, and raw-fidelity telemetry.
//! - [`history`] — bounded OpenAI/Anthropic tool-history normalization into
//!   renderer-safe, transport-neutral message chains.
//! - [`tool_schema`] — bounded provider tool normalization and generated-call
//!   schema validation, kept independent of the HTTP adapters.
//! - [`server`] — the axum router, the single-flight `Semaphore(1)` → 429, the
//!   streaming handler, cancel-on-client-drop, 503 not-ready (B5).

pub mod auth;
pub mod control;
pub mod dialect;
pub mod engine;
pub mod history;
pub mod prepare;
pub(crate) mod response_adapter;
pub mod server;
pub mod sse;
pub mod tool_call;
pub mod tool_schema;

use hyperion_core::EngineIdentity;

/// Information used by the binary's build report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuildInfo {
    /// Engine identity consumed on the serving path.
    pub engine: EngineIdentity,
    /// Whether M3's HTTP surface is implemented.
    pub serving_available: bool,
}

/// Return build state. M3 ships the HTTP surface, so `serving_available` is
/// now `true` (the `--build-info` flag and the M0/M1 canary still report it).
#[must_use]
pub fn build_info() -> BuildInfo {
    let tokenizer = hyperion_tokenizer::contract();
    let engine = hyperion_core::identity();
    debug_assert_eq!(engine.tokenizer, tokenizer);
    BuildInfo {
        engine,
        serving_available: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m3_now_claims_a_serving_backend() {
        let info = build_info();
        assert_eq!(info.engine.native_backend_count, 1);
        assert!(
            info.serving_available,
            "M3 ships the HTTP surface; build_info must advertise it"
        );
    }
}
