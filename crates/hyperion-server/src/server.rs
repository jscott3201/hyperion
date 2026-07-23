//! B5: the axum router + the streaming/non-streaming handlers. One single
//! router, localhost-first, 32 MiB body limit (06 §Surfaces). Single-flight
//! `Semaphore(1)` → 429; cancel-on-client-drop via the dropped SSE receiver.
//!
//! The `Server` holds everything the handlers share: the `EngineDriver` (the
//! real mailbox on the engine thread, or a `StubEngine` in tests), the chat
//! template + tokenizer (for normalization), the ops [`ControlState`], the
//! single-flight permit, the optional bearer token, and the context window.
//!
//! Handlers are `async` and take `State` (an `Arc<Server>`). The streaming
//! handler returns an `Sse` body; the non-streaming handler returns JSON. Both
//! go through [`normalize`] (prepare) → single-flight → engine → frame.

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bytes::Bytes;
use tokio::sync::Semaphore;

use hyperion_tokenizer::TokenizerHandle;
use hyperion_tokenizer::renderer::ChatTemplate;

use crate::auth::{AuthError, verify_bearer};
use crate::control::{ControlState, ReloadVerdict};
use crate::dialect::{Dialect, ErrorEnvelope};
use crate::engine::{CancelToken, EngineDriver, EngineError, EngineRequest, StepEvent, Usage};
use crate::prepare::{
    ContextWindow, PrepareError, count_anthropic_tokens, prepare_anthropic, prepare_openai,
};
use crate::sse::{Framer, StopReason};

/// The shared server state, cheaply cloned (all `Arc`) into each handler.
#[derive(Clone)]
pub struct Server {
    /// The engine-driver seam (real mailbox on the engine thread, or a
    /// `StubEngine` in tests). `Arc<dyn EngineDriver>` so the handler doesn't
    /// monomorphize on the concrete engine.
    engine: Arc<dyn EngineDriver>,
    template: Arc<ChatTemplate>,
    tokenizer: Arc<TokenizerHandle>,
    control: ControlState,
    /// Single-flight: `Semaphore(1)` — a second concurrent request gets 429.
    permit: Arc<Semaphore>,
    /// The configured bearer token (`None` ⇒ auth off, loopback).
    bearer: Option<Arc<String>>,
    context: ContextWindow,
    model_id: String,
    default_max_tokens: u32,
}

/// The 32 MiB body limit (06 §Surfaces) → 413 on overflow. Distinct from the
/// context-overflow 413 (prepare).
const BODY_LIMIT: usize = 32 * 1024 * 1024;

/// The single-flight + in-flight guard: holds the permit + a control clone +
/// the cancel token for the lifetime of a generation's **stream** (not the
/// `run` call). Released on Drop — which happens when the SSE body is fully
/// consumed OR the response future is dropped (client disconnect). This is
/// the fix for the adversarial findings that the permit was released when
/// `run` returned (streaming is lazy) and that `set_in_flight(false)` lived
/// inside the stream generator (never reached on a dropped future).
struct Guard {
    /// The single-flight permit — dropped releases the Semaphore(1).
    _permit: tokio::sync::OwnedSemaphorePermit,
    /// The control state to flip in_flight(false) on Drop.
    control: ControlState,
    /// The cancel token — fired on Drop so a client disconnect cancels the
    /// engine promptly (the dropped-receiver path also cancels; this covers
    /// the prefill window + any path that doesn't hit a `blocking_send`).
    cancel: CancelToken,
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.control.set_in_flight(false);
        self.cancel.cancel();
    }
}

/// The configured server inputs, grouped so [`Server::new`] stays under the
/// argument limit + callers build this explicitly. The caller loads the model +
/// template + tokenizer + engine mailbox, then hands them here.
#[derive(Clone)]
pub struct ServerConfig {
    /// The engine-driver seam (real mailbox on the engine thread, or a
    /// `StubEngine` in tests).
    pub engine: Arc<dyn EngineDriver>,
    /// The compiled chat template (renders both dialects' messages).
    pub template: ChatTemplate,
    /// The in-process tokenizer (encode the rendered prompt; decode each step).
    pub tokenizer: TokenizerHandle,
    /// The ops state (health/stats/reload/shutdown counters).
    pub control: ControlState,
    /// The configured bearer token (`None` ⇒ auth off, loopback).
    pub bearer: Option<String>,
    /// The context window (tokens) for the 413 overflow check.
    pub context: ContextWindow,
    /// The model id echoed in `/v1/models` + SSE frames.
    pub model_id: String,
    /// The default `max_tokens` when a dialect's body omits it (OpenAI).
    pub default_max_tokens: u32,
}

impl Server {
    /// Build the server state from a [`ServerConfig`].
    #[must_use]
    pub fn new(cfg: ServerConfig) -> Self {
        Self {
            engine: cfg.engine,
            template: Arc::new(cfg.template),
            tokenizer: Arc::new(cfg.tokenizer),
            control: cfg.control,
            permit: Arc::new(Semaphore::new(1)),
            bearer: cfg.bearer.map(Arc::new),
            context: cfg.context,
            model_id: cfg.model_id,
            default_max_tokens: cfg.default_max_tokens,
        }
    }

    /// Build the axum router with all routes + the body limit. The bearer
    /// middleware is applied per-handler via [`check_auth`] (axum middleware
    /// on a sub-router would also work; per-handler is explicit + testable).
    pub fn router(&self) -> Router {
        Router::new()
            .route("/v1/messages", post(handle_messages))
            .route("/v1/messages/count_tokens", post(handle_count_tokens))
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/v1/models", get(handle_list_models))
            .route("/v1/models/{id}", get(handle_get_model))
            .route("/control/health", get(handle_health))
            .route("/control/stats", get(handle_stats))
            .route("/control/reload", post(handle_reload))
            .route("/control/shutdown", post(handle_shutdown))
            .layer(axum::extract::DefaultBodyLimit::max(BODY_LIMIT))
            .with_state(self.clone())
    }

    /// The model id (echoed in `/v1/models` + SSE frames).
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Check the bearer token (constant-time). Returns `Ok` if auth is off
/// (loopback) or the token matches; `Err(401)` otherwise. The dialect is
/// needed for the error envelope. The `Err` is `Box<Response>` because
/// `Response` is large (clippy `result_large_err`); the 401 path is rare.
fn check_auth(
    bearer: &Option<Arc<String>>,
    headers: &HeaderMap,
    dialect: Dialect,
) -> Result<(), Box<Response>> {
    let configured = bearer.as_deref().map(String::as_str);
    let header = headers.get("authorization").and_then(|v| v.to_str().ok());
    if verify_bearer(configured, header).is_err() {
        let body = ErrorEnvelope::new(401, AuthError.to_string()).to_json(dialect);
        return Err(Box::new((StatusCode::UNAUTHORIZED, body).into_response()));
    }
    Ok(())
}

/// A 503 not-ready response (the model isn't loaded yet, or shutting down).
fn not_ready(dialect: Dialect) -> Response {
    let body = ErrorEnvelope::new(503, "model not ready").to_json(dialect);
    (StatusCode::SERVICE_UNAVAILABLE, body).into_response()
}

/// The body-too-large response (the 32 MiB limit fired). axum's
/// `RequestBodyLimit` returns a 413 itself; this is a fallback for manual
/// checks.
fn body_too_large(dialect: Dialect) -> Response {
    let body = ErrorEnvelope::new(413, "request body too large").to_json(dialect);
    (StatusCode::PAYLOAD_TOO_LARGE, body).into_response()
}

/// Map a `PrepareError` to the dialect-correct error response.
fn prepare_error_response(err: PrepareError, dialect: Dialect) -> Response {
    let status = err.http_status();
    let body = ErrorEnvelope::new(status, err.to_string()).to_json(dialect);
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST),
        body,
    )
        .into_response()
}

/// Acquire the single-flight permit. `Ok(permit)` if acquired; `Err(429)` if
/// busy. Takes the `Arc<Semaphore>` so it can `try_acquire_owned` (an owned
/// permit that's `Send`). The `Err` is `Box<Response>` (`Response` is large).
fn acquire_permit(
    permit: &Arc<Semaphore>,
    control: &ControlState,
    dialect: Dialect,
) -> Result<tokio::sync::OwnedSemaphorePermit, Box<Response>> {
    match permit.clone().try_acquire_owned() {
        Ok(p) => Ok(p),
        Err(_) => {
            control.inc_single_flight_rejections();
            let body = ErrorEnvelope::new(429, "single-flight generation is busy").to_json(dialect);
            Err(Box::new(
                (StatusCode::TOO_MANY_REQUESTS, body).into_response(),
            ))
        }
    }
}

/// The stop reason for a finished/cancelled stream, inferred from the engine
/// result + the request's `max_tokens`.
fn stop_reason(usage: &Usage, request: &EngineRequest, cancelled: bool) -> StopReason {
    if cancelled {
        return StopReason::Cancelled;
    }
    if usage.completion_tokens >= request.max_tokens {
        return StopReason::MaxTokens;
    }
    StopReason::EndTurn
}

/// Map an `EngineError` to an HTTP response (the pre-stream + non-streaming
/// paths). The streaming path uses [`Framer::error`] once headers are flushed
/// (a mid-stream error becomes an SSE `error` event, not an HTTP change).
fn engine_error_response(err: EngineError, dialect: Dialect) -> Response {
    let (status, message) = match &err {
        EngineError::Busy => (429, "single-flight generation is busy".to_string()),
        EngineError::Cancelled => (499, "client closed the request".to_string()),
        EngineError::Native(n) => (n.status.http_status_code(), n.message.clone()),
    };
    let body = ErrorEnvelope::new(status, message).to_json(dialect);
    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (code, body).into_response()
}

// ── Handlers ──────────────────────────────────────────────────────────────

/// `POST /v1/messages` (Anthropic). Streaming or non-streaming.
async fn handle_messages(State(srv): State<Server>, headers: HeaderMap, body: String) -> Response {
    let dialect = Dialect::Anthropic;
    if let Err(r) = check_auth(&srv.bearer, &headers, dialect) {
        return *r;
    }
    if !srv.control.is_ready() {
        return not_ready(dialect);
    }
    if body.len() > BODY_LIMIT {
        return body_too_large(dialect);
    }
    srv.control.inc_requests();
    let prepared = match prepare_anthropic(&body, &srv.template, &srv.tokenizer, srv.context) {
        Ok(p) => p,
        Err(e) => return prepare_error_response(e, dialect),
    };
    run(srv, prepared).await
}

/// `POST /v1/chat/completions` (OpenAI). Streaming or non-streaming.
async fn handle_chat_completions(
    State(srv): State<Server>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let dialect = Dialect::OpenAi;
    if let Err(r) = check_auth(&srv.bearer, &headers, dialect) {
        return *r;
    }
    if !srv.control.is_ready() {
        return not_ready(dialect);
    }
    if body.len() > BODY_LIMIT {
        return body_too_large(dialect);
    }
    srv.control.inc_requests();
    let prepared = match prepare_openai(
        &body,
        &srv.template,
        &srv.tokenizer,
        srv.context,
        srv.default_max_tokens,
    ) {
        Ok(p) => p,
        Err(e) => return prepare_error_response(e, dialect),
    };
    run(srv, prepared).await
}

/// `POST /v1/messages/count_tokens` — render + tokenize, return the count (no
/// generation).
async fn handle_count_tokens(
    State(srv): State<Server>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let dialect = Dialect::Anthropic;
    if let Err(r) = check_auth(&srv.bearer, &headers, dialect) {
        return *r;
    }
    if !srv.control.is_ready() {
        return not_ready(dialect);
    }
    if body.len() > BODY_LIMIT {
        return body_too_large(dialect);
    }
    // count_tokens does NOT require max_tokens (it only counts input tokens);
    // use the dedicated path, not prepare_anthropic (which requires max_tokens
    // for a generation).
    let input_tokens = match count_anthropic_tokens(&body, &srv.template, &srv.tokenizer) {
        Ok(n) => n,
        Err(e) => return prepare_error_response(e, dialect),
    };
    let resp = serde_json::json!({"input_tokens": input_tokens});
    Json(resp).into_response()
}

/// `GET /v1/models` — the one loaded model.
async fn handle_list_models(State(srv): State<Server>, headers: HeaderMap) -> Response {
    let dialect = Dialect::OpenAi;
    if let Err(r) = check_auth(&srv.bearer, &headers, dialect) {
        return *r;
    }
    let body = serde_json::json!({
        "object": "list",
        "data": [{"id": srv.model_id, "object": "model", "owned_by": "hyperion"}]
    });
    Json(body).into_response()
}

/// `GET /v1/models/:id` — the one loaded model (404 if the id doesn't match).
async fn handle_get_model(
    State(srv): State<Server>,
    headers: HeaderMap,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let dialect = Dialect::OpenAi;
    if let Err(r) = check_auth(&srv.bearer, &headers, dialect) {
        return *r;
    }
    if id != srv.model_id {
        let body = ErrorEnvelope::new(404, format!("model '{id}' not found")).to_json(dialect);
        return (StatusCode::NOT_FOUND, body).into_response();
    }
    let body = serde_json::json!({"id": srv.model_id, "object": "model", "owned_by": "hyperion"});
    Json(body).into_response()
}

/// `GET /control/health`.
async fn handle_health(State(srv): State<Server>, headers: HeaderMap) -> Response {
    // /control is auth-gated the same as /v1 when a token is set. Use OpenAI
    // envelope shape for ops errors (arbitrary; ops isn't a dialect).
    if let Err(r) = check_auth(&srv.bearer, &headers, Dialect::OpenAi) {
        return *r;
    }
    Json(srv.control.health()).into_response()
}

/// `GET /control/stats`.
async fn handle_stats(State(srv): State<Server>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&srv.bearer, &headers, Dialect::OpenAi) {
        return *r;
    }
    Json(srv.control.stats()).into_response()
}

/// `POST /control/reload` — 409 if in flight, 501 if idle.
async fn handle_reload(State(srv): State<Server>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&srv.bearer, &headers, Dialect::OpenAi) {
        return *r;
    }
    match srv.control.reload_verdict() {
        ReloadVerdict::Conflict => {
            let body =
                ErrorEnvelope::new(409, "a generation is in flight").to_json(Dialect::OpenAi);
            (StatusCode::CONFLICT, body).into_response()
        }
        ReloadVerdict::NotImplemented => {
            let body = ErrorEnvelope::new(501, "reload not implemented").to_json(Dialect::OpenAi);
            (StatusCode::NOT_IMPLEMENTED, body).into_response()
        }
    }
}

/// `POST /control/shutdown` — graceful drain.
async fn handle_shutdown(State(srv): State<Server>, headers: HeaderMap) -> Response {
    if let Err(r) = check_auth(&srv.bearer, &headers, Dialect::OpenAi) {
        return *r;
    }
    srv.control.request_shutdown();
    Json(serde_json::json!({"shutting_down": true})).into_response()
}

/// The core run: single-flight → engine stream → frame (SSE or JSON). Shared
/// by both dialects' generation endpoints.
async fn run(srv: Server, prepared: crate::prepare::PreparedPrompt) -> Response {
    let dialect = prepared.dialect;
    let request = prepared.engine_request;
    let stream = prepared.stream;
    let prompt_tokens = prepared.prompt_tokens_len;

    // Acquire the single-flight permit + arm the guard. The guard owns the
    // permit + the control clone + the cancel token, and releases them on
    // Drop — which happens when the SSE body is fully consumed OR the response
    // future is dropped (client disconnect). This is the fix for the
    // adversarial findings that the permit was released when `run` returned
    // (streaming is lazy) and `set_in_flight(false)` lived inside the stream
    // generator (never reached on a dropped future).
    let permit = match acquire_permit(&srv.permit, &srv.control, dialect) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    srv.control.set_in_flight(true);
    let cancel = CancelToken::new();
    let guard = Guard {
        _permit: permit,
        control: srv.control.clone(),
        cancel: cancel.clone(),
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<StepEvent>(8);

    // The engine task: drive the engine's stream on a blocking thread (the
    // engine's `blocking_send` is sync) so the async runtime stays responsive.
    let engine = srv.engine.clone();
    let request_clone = request.clone();
    let cancel_clone = cancel.clone();
    let engine_task =
        tokio::task::spawn_blocking(move || engine.stream(&request_clone, &cancel_clone, tx));

    if stream {
        // SSE: emit the raw framer bytes directly as the body (NOT via axum's
        // `Sse`/`Event` — the framer produces exact, well-formed SSE byte
        // strings, verified by the `sse.rs` unit tests; repacking them through
        // axum's `Event::data` corrupts the `event:`/`data:` framing). Each
        // `Framer` method returns a complete `event: ...\ndata: ...\n\n` (or
        // `data: ...\n\n`) chunk; we yield the bytes verbatim.
        let mut framer = Framer::new(dialect, &srv.model_id, prompt_tokens);
        let tokenizer = srv.tokenizer.clone();
        let control = srv.control.clone();
        let request_for_stream = request.clone();
        // The guard moves into the stream so it lives as long as the stream
        // does (consumed lazily by axum). Dropping the stream (client
        // disconnect, or full consumption) drops the guard → releases the
        // permit + flips in_flight(false) + cancels.
        let guard = guard;
        let stream = async_stream::stream! {
            let _guard = guard; // held for the stream's lifetime
            // The opening frames (Anthropic message_start + content_block_start;
            // empty for OpenAI). `start` is a complete SSE chunk.
            yield Ok::<Bytes, Infallible>(Bytes::from(framer.start()));
            let mut usage = Usage { prompt_tokens, completion_tokens: 0 };
            let mut rx = rx;
            while let Some(event) = rx.recv().await {
                match event {
                    StepEvent::Token(t) => {
                        let piece = tokenizer.decode(&[t.id]);
                        let frame = framer.token(t, &piece);
                        yield Ok(Bytes::from(frame));
                    }
                    StepEvent::Done(u) => usage = u,
                }
            }
            // The engine finished: await its result for the terminal frame.
            let result = engine_task.await.unwrap_or(Err(EngineError::Native(
                hyperion_ffi::Error {
                    status: hyperion_ffi::Status::Internal,
                    message: "engine task failed".into(),
                },
            )));
            match result {
                Ok(_) => {
                    let stop = stop_reason(&usage, &request_for_stream, false);
                    let frame = framer.done(usage, stop);
                    yield Ok(Bytes::from(frame));
                }
                Err(EngineError::Native(n)) if n.status.http_status_code() == 529 => {
                    // 529 mid-stream: an SSE error event, NOT an HTTP change
                    // (headers already flushed). Bump the governor counter.
                    control.inc_governor_rejections();
                    let frame = framer.error(529, &n.message);
                    yield Ok(Bytes::from(frame));
                }
                Err(e) => {
                    let (status, message) = match &e {
                        EngineError::Busy => (429, "single-flight generation is busy".to_string()),
                        EngineError::Cancelled => (499, "client closed the request".to_string()),
                        EngineError::Native(n) => (n.status.http_status_code(), n.message.clone()),
                    };
                    let frame = framer.error(status, &message);
                    yield Ok(Bytes::from(frame));
                }
            }
        };
        // Build the raw-SSE response. The `Content-Type: text/event-stream`
        // + `Cache-Control: no-cache` (the SSE contract) + the byte stream.
        let body = Body::from_stream(stream);
        let mut resp = Response::new(body);
        *resp.status_mut() = StatusCode::OK;
        resp.headers_mut().insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        resp.headers_mut()
            .insert("cache-control", HeaderValue::from_static("no-cache"));
        resp
    } else {
        // Non-streaming: collect all tokens, then build a single JSON response.
        let mut text = String::new();
        let mut usage = Usage::default();
        let mut rx = rx;
        while let Some(event) = rx.recv().await {
            match event {
                StepEvent::Token(t) => {
                    let piece = srv.tokenizer.decode(&[t.id]);
                    text.push_str(&piece);
                }
                StepEvent::Done(u) => usage = u,
            }
        }
        // The guard (held here) releases the permit + flips in_flight(false) on
        // drop at the end of this scope — for the non-streaming path, the
        // generation is complete by the time we reach here, so dropping is
        // correct. (On a client disconnect during the await above, axum drops
        // the handler future → this scope drops → the guard releases.)
        let _ = guard;
        let engine_result =
            engine_task
                .await
                .unwrap_or(Err(EngineError::Native(hyperion_ffi::Error {
                    status: hyperion_ffi::Status::Internal,
                    message: "engine task failed".into(),
                })));
        match engine_result {
            Ok(_) => {
                let stop = stop_reason(&usage, &request, false);
                non_streaming_json(dialect, &srv.model_id, &text, usage, stop)
            }
            Err(e) => engine_error_response(e, dialect),
        }
    }
}

/// Build the non-streaming JSON response for a completed generation.
fn non_streaming_json(
    dialect: Dialect,
    model: &str,
    text: &str,
    usage: Usage,
    stop: StopReason,
) -> Response {
    match dialect {
        Dialect::Anthropic => {
            let body = serde_json::json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop.anthropic(),
                "stop_sequence": null,
                "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens}
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Dialect::OpenAi => {
            let body = serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": stop.openai()}],
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.prompt_tokens + usage.completion_tokens
                }
            });
            (StatusCode::OK, Json(body)).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: full router integration tests (oneshot into the router) live in B8's
    // contract-test suite. These are unit-level checks of the helpers.

    #[test]
    fn stop_reason_max_tokens_vs_end_turn() {
        let req = EngineRequest {
            prompt_tokens: vec![1],
            max_tokens: 10,
            eos_token_id: None,
            sampling: None,
        };
        assert_eq!(
            stop_reason(
                &Usage {
                    prompt_tokens: 1,
                    completion_tokens: 10
                },
                &req,
                false
            ),
            StopReason::MaxTokens
        );
        assert_eq!(
            stop_reason(
                &Usage {
                    prompt_tokens: 1,
                    completion_tokens: 3
                },
                &req,
                false
            ),
            StopReason::EndTurn
        );
        assert_eq!(
            stop_reason(
                &Usage {
                    prompt_tokens: 1,
                    completion_tokens: 3
                },
                &req,
                true
            ),
            StopReason::Cancelled
        );
    }

    // A stub-driven end-to-end streaming check that doesn't need a real model:
    // build a Server with a StubEngine, drive the router via oneshot, assert
    // the SSE body contains the stub's tokens. This needs a real ChatTemplate +
    // TokenizerHandle, so it's #[ignore] (M5) — the contract tests in B8 cover
    // the error paths model-free via direct handler calls where possible.
}
