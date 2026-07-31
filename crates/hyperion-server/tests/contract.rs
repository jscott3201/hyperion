//! B8: the error-taxonomy contract tests — one per HTTP status code in the 06
//! §Error taxonomy, for both dialects where applicable. These drive the axum
//! router via `tower::ServiceExt::oneshot` with a `StubEngine` + a minimal
//! inline tokenizer + a trivial chat template, so the suite is **model-free**
//! (tier-1 CI). The 529/499 mid-stream paths are unit-tested in `sse::` +
//! `engine::` (the framing + the cancel-on-drop); here we cover the
//! pre-stream HTTP statuses.
//!
//! Codes covered: 401 (auth), 404 (model not found), 413 (body too large +
//! context overflow), 429 (single-flight busy), 503 (not ready), 400
//! (malformed body / tools unsupported / missing max_tokens), 501 (reload
//! idle), 409 (reload in flight), 200 (health/stats/count_tokens/list models).

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use hyperion_server::auth::bind_gate;
use hyperion_server::control::ControlState;
use hyperion_server::engine::test_support::StubEngine;
use hyperion_server::engine::{
    CancelToken, EngineDriver, EngineError, EngineRequest, StepEvent, StepToken, Usage,
};
use hyperion_server::prepare::ContextWindow;
use hyperion_server::server::{Server, ServerConfig};
use hyperion_tokenizer::TokenizerHandle;
use hyperion_tokenizer::renderer::ChatTemplate;
use std::net::IpAddr;
use std::net::Ipv4Addr;
use std::sync::Mutex;
use tower::ServiceExt;

/// A minimal HuggingFace `tokenizer.json` with a tiny WordLevel vocab that
/// includes `<eos>` (so `token_to_id("<eos>")` resolves) + a few tokens. This
/// lets the contract tests run `prepare` (render + tokenize) without a fixture
/// file. The chat template below emits no special tokens, so encode is trivial.
const MINIMAL_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"<eos>": 0, "<bos>": 1, "hello": 2, "world": 3, "[UNK]": 4},
    "unk_token": "[UNK]"
  }
}"#;

/// A model-free BPE/ByteFallback tokenizer whose generated `<0xE5>` token is
/// unresolved by incremental `push` and becomes one replacement scalar only
/// when the response decoder is finished at EOF.
const EOF_FALLBACK_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {"type": "ByteFallback"},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": true,
    "ignore_merges": false,
    "vocab": {"<eos>": 0, "<bos>": 1, "<0xE5>": 2},
    "merges": []
  }
}"#;

/// A trivial chat template that just concatenates the messages' content. It
/// emits no special tokens (the contract tests don't need real gemma4 framing
/// — they exercise the HTTP/taxonomy layer, not the template).
const TRIVIAL_TEMPLATE: &str =
    "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}";

/// Build a `Server` for the contract tests with a `StubEngine` + the minimal
/// inline tokenizer/template. `bearer` sets the auth token; `control` +
/// `context` are tunable per test.
fn test_server(
    stub: StubEngine,
    bearer: Option<&str>,
    context: ContextWindow,
    control: &ControlState,
) -> Server {
    test_server_with_tokenizer(stub, bearer, context, control, MINIMAL_TOKENIZER_JSON)
}

fn test_server_with_tokenizer(
    stub: StubEngine,
    bearer: Option<&str>,
    context: ContextWindow,
    control: &ControlState,
    tokenizer_json: &str,
) -> Server {
    test_server_with_engine(Arc::new(stub), bearer, context, control, tokenizer_json)
}

fn test_server_with_engine(
    engine: Arc<dyn EngineDriver>,
    bearer: Option<&str>,
    context: ContextWindow,
    control: &ControlState,
    tokenizer_json: &str,
) -> Server {
    let tokenizer =
        TokenizerHandle::from_bytes(tokenizer_json.as_bytes()).expect("model-free tokenizer loads");
    let template = ChatTemplate::from_source(TRIVIAL_TEMPLATE, "<bos>", "<eos>")
        .expect("trivial template compiles");
    Server::new(ServerConfig {
        engine,
        template,
        tokenizer,
        control: control.clone(),
        bearer: bearer.map(str::to_string),
        context,
        model_id: "test-model".to_string(),
        default_max_tokens: 64,
    })
}

struct ControlledFailureEngine {
    cleanup_reached: std::sync::mpsc::SyncSender<()>,
    release_cleanup: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl EngineDriver for ControlledFailureEngine {
    fn stream(
        &self,
        request: &EngineRequest,
        cancel: &CancelToken,
        tx: tokio::sync::mpsc::Sender<StepEvent>,
    ) -> Result<Usage, EngineError> {
        for _ in 0..10_000 {
            if tx
                .blocking_send(StepEvent::Token(StepToken {
                    id: 99_999,
                    logit: 0.0,
                }))
                .is_err()
            {
                assert!(cancel.is_cancelled(), "decoder failure cancels engine");
                self.cleanup_reached
                    .send(())
                    .expect("test observes cleanup barrier");
                self.release_cleanup
                    .lock()
                    .expect("cleanup mutex")
                    .recv()
                    .expect("test releases cleanup barrier");
                return Err(EngineError::Cancelled);
            }
        }
        let usage = Usage {
            prompt_tokens: u32::try_from(request.prompt_tokens.len()).unwrap_or(u32::MAX),
            completion_tokens: 10_000,
        };
        tx.blocking_send(StepEvent::Done(usage))
            .map_err(|_| EngineError::Cancelled)?;
        Ok(usage)
    }
}

/// Send a request through the router, return the (status, body string).
async fn send(
    router: axum::Router,
    method: Method,
    uri: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> (StatusCode, String) {
    let mut builder = Request::builder().method(method).uri(uri);
    for (k, v) in headers {
        builder = builder.header(*k, *v);
    }
    let request = builder.body(Body::from(body.to_string())).unwrap();
    let response = router.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

fn fresh_control() -> ControlState {
    let c = ControlState::new("test-model");
    c.set_ready(true);
    c
}

fn streamed_text(body: &str, dialect: &str) -> String {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .filter_map(|value| match dialect {
            "anthropic" if value["type"] == "content_block_delta" => {
                value["delta"]["text"].as_str().map(str::to_owned)
            }
            "openai" => value["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn bind_gate_refuses_non_loopback_without_token() {
    let addr = IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0));
    assert!(
        bind_gate(addr, None).is_err(),
        "0.0.0.0 without token refuses"
    );
    assert!(bind_gate(addr, Some("t")).is_ok());
}

#[tokio::test]
async fn bind_gate_loopback_allows_no_token() {
    let addr = IpAddr::V4(Ipv4Addr::LOCALHOST);
    assert!(bind_gate(addr, None).is_ok());
}

#[tokio::test]
async fn auth_401_wrong_token_anthropic() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![2, 3],
            block_after: None,
        },
        Some("secret"),
        4096,
        &control,
    );
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[
            ("authorization", "Bearer wrong"),
            ("content-type", "application/json"),
        ],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(
        body.contains("\"type\":\"error\""),
        "Anthropic 401 envelope: {body}"
    );
    assert!(body.contains("authentication_error"), "401 type: {body}");
}

#[tokio::test]
async fn auth_401_missing_header_openai() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![2, 3],
            block_after: None,
        },
        Some("secret"),
        4096,
        &control,
    );
    let (status, _body) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_200_correct_token() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![2, 3],
            block_after: None,
        },
        Some("secret"),
        4096,
        &control,
    );
    // count_tokens: a 200 path that doesn't need the engine AND must NOT require
    // max_tokens (Anthropic's count_tokens only counts input tokens — the
    // adversarial review caught a spurious 400 here). No max_tokens in the body.
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages/count_tokens",
        &[
            ("authorization", "Bearer secret"),
            ("content-type", "application/json"),
        ],
        r#"{"messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "count_tokens without max_tokens: {body}"
    );
    assert!(
        body.contains("input_tokens"),
        "count_tokens response: {body}"
    );
}

#[tokio::test]
async fn not_ready_503_before_load() {
    let control = ControlState::new("test-model");
    // ready stays false (not loaded)
    let srv = test_server(
        StubEngine {
            tokens: vec![2],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body.contains("service_unavailable"), "503 type: {body}");
}

#[tokio::test]
async fn malformed_body_400_anthropic() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        "not json",
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("invalid_request_error"), "400 type: {body}");
}

#[tokio::test]
async fn missing_max_tokens_400_anthropic() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, _body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    // max_tokens missing → 400 (Anthropic requires it). (prepare_anthropic
    // checks max_tokens after parse; render happens first — but the trivial
    // template renders, then the missing-max_tokens gate fires.)
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn tools_unsupported_400_openai() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"tools":[]}"#,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("tool calling") || body.contains("invalid_request_error"),
        "tools 400: {body}"
    );
}

#[tokio::test]
async fn model_not_found_404() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(
        srv.router(),
        Method::GET,
        "/v1/models/no-such-model",
        &[],
        "",
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body.contains("not_found_error"), "404 type: {body}");
}

#[tokio::test]
async fn list_models_200() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(srv.router(), Method::GET, "/v1/models", &[], "").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body.contains("test-model"),
        "lists the loaded model: {body}"
    );
}

#[tokio::test]
async fn health_200() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(srv.router(), Method::GET, "/control/health", &[], "").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("\"ready\":true"), "health: {body}");
}

#[tokio::test]
async fn stats_200() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(srv.router(), Method::GET, "/control/stats", &[], "").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("requests_total"), "stats: {body}");
}

#[tokio::test]
async fn reload_501_when_idle() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, _body) = send(srv.router(), Method::POST, "/control/reload", &[], "").await;
    assert_eq!(
        status,
        StatusCode::NOT_IMPLEMENTED,
        "idle reload is 501 (honest stub)"
    );
}

#[tokio::test]
async fn reload_409_when_in_flight() {
    let control = fresh_control();
    control.set_in_flight(true);
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(srv.router(), Method::POST, "/control/reload", &[], "").await;
    assert_eq!(status, StatusCode::CONFLICT, "in-flight reload is 409");
    assert!(body.contains("conflict_error"), "409 type: {body}");
}

#[tokio::test]
async fn shutdown_200_sets_flag() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(srv.router(), Method::POST, "/control/shutdown", &[], "").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("shutting_down"));
    assert!(control.is_shutdown(), "the shutdown flag is set");
}

#[tokio::test]
async fn context_overflow_413_anthropic() {
    let control = fresh_control();
    // A tiny context window so prompt + max_tokens overflows easily.
    let srv = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        2, // context window of 2 tokens
        &control,
    );
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hello world hi there"}],"max_tokens":10}"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::PAYLOAD_TOO_LARGE,
        "context overflow → 413: {body}"
    );
    // The 413 envelope (context-overflow path, distinct from body-size 413).
    assert!(body.contains("request_too_large"), "413 type: {body}");
}

#[tokio::test]
async fn single_flight_429_when_busy() {
    // Hold the single-flight permit on a separate task so the test request
    // can't acquire it → 429. The permit is per-Server (a fresh Semaphore(1)),
    // so we acquire it via a request that blocks in the engine.
    let control = fresh_control();
    // A stub that blocks before yielding the first token, so its permit stays
    // held while the second request arrives.
    let srv = Arc::new(test_server(
        StubEngine {
            tokens: vec![2, 3],
            block_after: Some(0),
        },
        None,
        4096,
        &control,
    ));
    let srv1 = srv.clone();
    // First request: acquire the permit + block in the engine.
    let first = tokio::spawn(async move {
        send(
            srv1.router(),
            Method::POST,
            "/v1/messages",
            &[("content-type", "application/json")],
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}"#,
        )
        .await
    });
    // Let the first request acquire the permit + enter the engine.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    // Second request: single-flight busy → 429.
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}"#,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::TOO_MANY_REQUESTS,
        "second concurrent request → 429"
    );
    assert!(body.contains("rate_limit_error"), "429 type: {body}");
    // The control counter bumped.
    assert!(control.stats().single_flight_rejections >= 1);
    // Let the first finish (cleanup).
    first.abort();
}

// ── Adversarial-review regression guards ─────────────────────────────────
//
// These pin the fixes for the PR-B adversarial findings: the raw-SSE byte
// stream (not axum-Event-repacked, which corrupted the event:/data: framing),
// the in_flight flag cleared on client disconnect (the Guard's Drop), and
// count_tokens not requiring max_tokens (covered by auth_200_correct_token
// above).

/// The streaming response body must be raw, well-formed SSE: the framer's
/// `event: <type>\ndata: {...}\n\n` chunks emitted verbatim (NOT repacked
/// through axum's `Event::data`, which would produce `data: event: ...` —
/// garbage). Drives an Anthropic stream through the router + asserts the body
/// contains the canonical event sequence with the right framing prefixes.
#[tokio::test]
async fn streaming_emits_raw_sse_bytes_anthropic() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![2, 3],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
        ))
        .unwrap();
    let response = srv.router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    // The canonical Anthropic sequence — each event has its own `event:` line +
    // `data:` line (NOT `data: event: message_start`).
    assert!(
        body.contains("event: message_start\n"),
        "message_start frame: {body}"
    );
    assert!(
        body.contains("event: content_block_start\n"),
        "content_block_start: {body}"
    );
    assert!(
        body.contains("event: content_block_delta\n"),
        "content_block_delta: {body}"
    );
    assert!(
        body.contains("\"text_delta\""),
        "text_delta payload: {body}"
    );
    assert!(
        body.contains("event: content_block_stop\n"),
        "content_block_stop: {body}"
    );
    assert!(
        body.contains("event: message_delta\n"),
        "message_delta: {body}"
    );
    assert!(
        body.contains("event: message_stop\n"),
        "message_stop: {body}"
    );
    // The corruption signature: `data: event:` must NOT appear (that's the
    // axum-Event repacking bug the review caught).
    assert!(
        !body.contains("data: event:"),
        "the SSE framing is raw (no `data: event:` repacking): {body}"
    );
    // Each frame ends with the SSE separator `\n\n`.
    assert!(body.contains("\n\n"), "frames are \\n\\n-separated");
}

/// The OpenAI stream ends with a distinct `data: [DONE]\n\n` event (NOT merged
/// into the final chunk's data field — the repacking bug the review caught).
#[tokio::test]
async fn streaming_emits_raw_sse_bytes_openai_done() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![2],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        ))
        .unwrap();
    let response = srv.router().oneshot(request).await.unwrap();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    // [DONE] is its own `data:` line (a distinct SSE event), not merged.
    assert!(
        body.contains("data: [DONE]\n\n"),
        "OpenAI [DONE] terminator: {body}"
    );
    assert!(
        body.contains("\"delta\":{\"content\""),
        "OpenAI delta chunk: {body}"
    );
}

#[tokio::test]
async fn streaming_and_non_streaming_share_incremental_decode_semantics() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![2, 3],
            block_after: None,
        },
        None,
        4096,
        &control,
    );

    let (status, non_streaming) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let response: serde_json::Value = serde_json::from_str(&non_streaming).unwrap();
    let expected = response["content"][0]["text"].as_str().unwrap();
    assert_eq!(expected, "hello world");

    let (status, anthropic) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(streamed_text(&anthropic, "anthropic"), expected);

    let (status, openai) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(streamed_text(&openai, "openai"), expected);
}

#[tokio::test]
async fn eof_only_fallback_text_precedes_both_dialects_terminal_frames() {
    let control = fresh_control();
    let srv = test_server_with_tokenizer(
        StubEngine {
            tokens: vec![2],
            block_after: None,
        },
        None,
        4096,
        &control,
        EOF_FALLBACK_TOKENIZER_JSON,
    );

    let (status, anthropic_json) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let response: serde_json::Value = serde_json::from_str(&anthropic_json).unwrap();
    let anthropic_expected = response["content"][0]["text"].as_str().unwrap();
    assert_eq!(anthropic_expected, "�");

    let (status, anthropic_sse) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        streamed_text(&anthropic_sse, "anthropic"),
        anthropic_expected
    );
    assert_eq!(anthropic_sse.matches('�').count(), 1, "{anthropic_sse}");
    let suffix = anthropic_sse.find("\"text\":\"�\"").unwrap();
    let message_delta = anthropic_sse.find("event: message_delta\n").unwrap();
    let message_stop = anthropic_sse.find("event: message_stop\n").unwrap();
    assert!(suffix < message_delta, "EOF text precedes message_delta");
    assert!(
        message_delta < message_stop,
        "message_delta precedes message_stop"
    );

    let (status, openai_json) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let response: serde_json::Value = serde_json::from_str(&openai_json).unwrap();
    let openai_expected = response["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    assert_eq!(openai_expected, anthropic_expected);

    let (status, openai_sse) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(streamed_text(&openai_sse, "openai"), openai_expected);
    assert_eq!(openai_sse.matches('�').count(), 1, "{openai_sse}");
    let suffix = openai_sse.find("\"content\":\"�\"").unwrap();
    let terminal = openai_sse.find("\"finish_reason\":\"stop\"").unwrap();
    let done = openai_sse.find("data: [DONE]\n\n").unwrap();
    assert!(suffix < terminal, "EOF text precedes terminal chunk");
    assert!(terminal < done, "terminal chunk precedes [DONE]");
}

#[tokio::test]
async fn streaming_decoder_failure_is_one_opaque_error_without_terminal_frame() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            // Unknown IDs decode to no text in the fixture tokenizer, creating
            // an adversarial unresolved run with substantial generation left
            // when the 257th ID exceeds the decoder ceiling.
            tokens: vec![99_999; 10_000],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":300,"stream":true}"#,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "SSE headers are already committed");
    assert_eq!(body.matches("event: error\n").count(), 1, "{body}");
    assert_eq!(body.matches("internal server error").count(), 1, "{body}");
    assert!(
        !body.contains("message_delta"),
        "no successful terminal: {body}"
    );
    assert!(
        !body.contains("message_stop"),
        "no successful terminal: {body}"
    );
    assert!(
        !body.contains("retained token limit"),
        "details stay opaque: {body}"
    );
    assert!(!control.is_in_flight(), "failed stream releases its permit");
    let (reload, _) = send(srv.router(), Method::POST, "/control/reload", &[], "").await;
    assert_eq!(reload, StatusCode::NOT_IMPLEMENTED, "permit is not stuck");
}

#[tokio::test]
async fn failed_stream_holds_permit_until_engine_cleanup_finishes() {
    let control = fresh_control();
    let (cleanup_tx, cleanup_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let srv = Arc::new(test_server_with_engine(
        Arc::new(ControlledFailureEngine {
            cleanup_reached: cleanup_tx,
            release_cleanup: Mutex::new(release_rx),
        }),
        None,
        4096,
        &control,
        MINIMAL_TOKENIZER_JSON,
    ));

    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":300,"stream":true}"#,
        ))
        .unwrap();
    let response = srv.router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body_task = tokio::spawn(async move {
        axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap()
    });

    tokio::task::spawn_blocking(move || {
        cleanup_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("engine reaches controlled cleanup")
    })
    .await
    .unwrap();
    let body_pending_during_cleanup = !body_task.is_finished();
    let in_flight_during_cleanup = control.is_in_flight();

    let (concurrent_status, concurrent_body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
    )
    .await;

    // Release before assertions so a regression cannot strand a blocking test
    // engine during panic unwinding.
    release_tx.send(()).expect("release engine cleanup");
    let bytes = tokio::time::timeout(std::time::Duration::from_secs(2), body_task)
        .await
        .expect("failed stream completes after cleanup")
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(
        body_pending_during_cleanup,
        "error waits for engine cleanup"
    );
    assert!(
        in_flight_during_cleanup,
        "permit remains held during cleanup"
    );
    assert_eq!(
        concurrent_status,
        StatusCode::TOO_MANY_REQUESTS,
        "{concurrent_body}"
    );
    assert_eq!(body.matches("event: error\n").count(), 1, "{body}");
    assert_eq!(body.matches("internal server error").count(), 1, "{body}");
    assert!(
        !body.contains("message_delta"),
        "no success terminal: {body}"
    );
    assert!(
        !body.contains("message_stop"),
        "no success terminal: {body}"
    );
    assert!(!control.is_in_flight(), "permit releases after cleanup");
}

#[tokio::test]
async fn non_streaming_decoder_failure_uses_opaque_dialect_envelope() {
    let control = fresh_control();
    let srv = test_server(
        StubEngine {
            tokens: vec![99_999; 10_000],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":300,"stream":false}"#,
    )
    .await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(response["error"]["type"], "internal_server_error");
    assert_eq!(response["error"]["message"], "internal server error");
    assert!(
        !body.contains("retained token limit"),
        "details stay opaque: {body}"
    );
    assert!(
        !control.is_in_flight(),
        "failed request releases its permit"
    );
    let (reload, _) = send(srv.router(), Method::POST, "/control/reload", &[], "").await;
    assert_eq!(reload, StatusCode::NOT_IMPLEMENTED, "permit is not stuck");
}

/// The in_flight flag clears when a stream is dropped (client disconnect).
/// The Guard's Drop flips in_flight(false) — the fix for the review's finding
/// that set_in_flight(false) lived inside the stream generator (never reached
/// on a dropped future). We start a stream, drop the response (mid-stream),
/// and assert in_flight is false afterward.
#[tokio::test]
async fn in_flight_clears_on_stream_drop() {
    let control = fresh_control();
    // A stub that blocks before the first token so the stream is mid-flight
    // when we drop it.
    let srv = Arc::new(test_server(
        StubEngine {
            tokens: vec![2, 3],
            block_after: Some(0),
        },
        None,
        4096,
        &control,
    ));
    let srv_clone = srv.clone();
    // Start the stream; don't await the body.
    let handle = tokio::spawn(async move {
        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
            ))
            .unwrap();
        srv_clone.router().oneshot(request).await
    });
    // Let it acquire the permit + enter the engine (set in_flight true).
    tokio::time::sleep(std::time::Duration::from_millis(60)).await;
    assert!(control.is_in_flight(), "in_flight is true mid-stream");
    // Simulate client disconnect: abort the task + drop the JoinHandle (which
    // holds the Response → the SSE body → the stream → the Guard). The Guard's
    // Drop flips in_flight(false). `abort()` drops a pending future; awaiting
    // the handle (completed or aborted) drops the stored Response. Both paths
    // release the guard.
    handle.abort();
    let _ = handle.await; // drops the Response (lazy SSE body + the Guard)
    // Give the runtime a tick to run any pending Drop.
    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
    assert!(
        !control.is_in_flight(),
        "in_flight cleared after the stream dropped (Guard::drop)"
    );
    // And a subsequent reload is now 501 (not stuck 409).
    let (status, _) = send(srv.router(), Method::POST, "/control/reload", &[], "").await;
    assert_eq!(
        status,
        StatusCode::NOT_IMPLEMENTED,
        "reload works after disconnect (not stuck 409)"
    );
}

// ── The self-hosted M5 real-model gate (#[ignore]) ──────────────────────
//
// The M3 gate line "streamed greedy run byte-identical to M2 CLI on fixtures"
// is sealed by the NATIVE forward_12b_test (token-exact vs the committed
// `12b_greedy_golden.safetensors`). PR B's job is the transport: prove the
// Rust engine + SSE layer doesn't corrupt the token stream. This #[ignore]
// test loads the real 12B, runs a streamed greedy, and asserts:
//   1. the stream is non-empty + every token in-vocab,
//   2. greedy is deterministic across two runs (a streaming regression guard),
//   3. the STREAMED tokens equal the NON-STREAMED tokens for the same prompt
//      (the SSE layer doesn't drop/dup/reorder — the load-bearing PR B invariant).
// Run on the M5: `HYPERION_12B_ARTIFACT=$repo/artifacts/models/gemma4-12b-qat-mlx-g64-b4 \
//   cargo test -p hyperion-server --test contract -- --ignored streamed_greedy`.

#[tokio::test]
#[ignore = "requires the 12B artifact + a Metal device (self-hosted M5)"]
async fn streamed_greedy_preserves_token_stream() {
    use hyperion_server::engine::{Engine, EngineRequest};
    use hyperion_tokenizer::renderer::{ChatMessage, RenderOptions};

    let dir = std::env::var("HYPERION_12B_ARTIFACT").unwrap_or_else(|_| {
        // Default to the canonical git-ignored path.
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../artifacts/models/gemma4-12b-qat-mlx-g64-b4"
        )
        .to_string()
    });
    let artifact = std::path::Path::new(&dir);
    let config =
        std::fs::read_to_string(artifact.join("config.json")).expect("12B config.json readable");
    let geometry = hyperion_model::geometry::Geometry::from_text_config_str(&config)
        .expect("12B geometry parses");

    let tokenizer =
        TokenizerHandle::from_file(&artifact.join("tokenizer.json")).expect("tokenizer loads");
    let template =
        ChatTemplate::from_artifact(artifact, Some(&tokenizer)).expect("chat template loads");

    // A templated prompt (raw-prompt greedy is degenerate per the chat-template
    // discipline; the templated prompt produces coherent tokens).
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(serde_json::json!("Say hello in one word.")),
        tool_calls: None,
        tool_responses: None,
        reasoning: None,
        reasoning_content: None,
        tool_call_id: None,
        name: None,
    }];
    let rendered = template
        .render(
            &messages,
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: Some(false),
                preserve_thinking: None,
                tools: Vec::new(),
            },
        )
        .expect("render");
    let prompt = tokenizer.encode(&rendered, false);
    assert!(!prompt.is_empty(), "prompt tokenizes to a non-empty vec");

    // The `!Send` `Engine` must be loaded + driven on one dedicated thread.
    // Spawn that thread, load the 12B there, and run both the non-streaming
    // (drive, twice for determinism) + streaming (stream) paths there. The
    // results come back over a oneshot.
    let (tx_tokens, rx_tokens) =
        std::sync::mpsc::channel::<(Vec<u32>, Vec<u32>, Vec<u32>, String)>();
    let engine_geometry = geometry.clone();
    let engine_dir = dir.clone();
    let engine_prompt = prompt.clone();
    let engine_eos = tokenizer.token_to_id("<eos>");
    let _engine_thread = std::thread::Builder::new()
        .name("test-engine".into())
        .spawn(move || {
            let engine = match Engine::load(&engine_geometry, &engine_dir) {
                Ok(e) => e,
                Err(e) => {
                    let _ =
                        tx_tokens.send((Vec::new(), Vec::new(), Vec::new(), format!("load: {e}")));
                    return;
                }
            };
            let request = EngineRequest {
                prompt_tokens: engine_prompt.clone(),
                max_tokens: 16,
                eos_token_id: engine_eos,
                sampling: None, // greedy
            };
            let cancel = hyperion_server::engine::CancelToken::new();
            // 1+2: non-streaming, twice (non-empty + deterministic).
            let first = match engine.drive(&request, &cancel) {
                Ok(v) => v,
                Err(e) => {
                    let _ =
                        tx_tokens.send((Vec::new(), Vec::new(), Vec::new(), format!("drive: {e}")));
                    return;
                }
            };
            let second = match engine.drive(&request, &cancel) {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx_tokens.send((
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        format!("drive2: {e}"),
                    ));
                    return;
                }
            };
            // 3: streaming — `engine.stream` runs INLINE on this engine thread
            // (the `!Send` engine can't move to another thread), sending to a
            // channel a DRAINER thread collects concurrently. Two threads → no
            // deadlock (the engine sends, the drainer receives).
            let (step_tx, mut step_rx) = tokio::sync::mpsc::channel(8);
            let drainer = std::thread::spawn(move || {
                let mut out = Vec::new();
                while let Some(event) = step_rx.blocking_recv() {
                    if let hyperion_server::engine::StepEvent::Token(t) = event {
                        out.push(t.id);
                    }
                }
                out
            });
            let stream_result = engine.stream(&request, &cancel, step_tx);
            let drained = drainer.join().unwrap_or_default();
            match stream_result {
                Ok(_) => {
                    let _ = tx_tokens.send((first, second, drained, String::new()));
                }
                Err(e) => {
                    let _ = tx_tokens.send((first, second, drained, format!("stream: {e}")));
                }
            }
        })
        .expect("spawn engine thread");

    let (first, second, streamed, err) = rx_tokens.recv().expect("engine thread reported");
    assert!(err.is_empty(), "engine error: {err}");
    assert!(!first.is_empty(), "greedy produced tokens");
    assert!(
        first.iter().all(|t| (*t as usize) < tokenizer.vocab_size()),
        "every token in-vocab"
    );
    assert_eq!(first, second, "greedy is deterministic across runs");
    assert_eq!(
        streamed, first,
        "the streamed tokens must equal the non-streamed tokens (no SSE corruption)"
    );
}
