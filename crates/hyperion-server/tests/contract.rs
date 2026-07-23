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
use hyperion_server::prepare::ContextWindow;
use hyperion_server::server::{Server, ServerConfig};
use hyperion_tokenizer::TokenizerHandle;
use hyperion_tokenizer::renderer::ChatTemplate;
use std::net::IpAddr;
use std::net::Ipv4Addr;
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
    let tokenizer = TokenizerHandle::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes())
        .expect("minimal tokenizer loads");
    let template = ChatTemplate::from_source(TRIVIAL_TEMPLATE, "<bos>", "<eos>")
        .expect("trivial template compiles");
    Server::new(ServerConfig {
        engine: Arc::new(stub),
        template,
        tokenizer,
        control: control.clone(),
        bearer: bearer.map(str::to_string),
        context,
        model_id: "test-model".to_string(),
        default_max_tokens: 64,
    })
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
    // count_tokens: a 200 path that doesn't need the engine. (The body carries
    // max_tokens because count_tokens reuses the prepare path, which validates
    // it; the count itself ignores max_tokens.)
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages/count_tokens",
        &[
            ("authorization", "Bearer secret"),
            ("content-type", "application/json"),
        ],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
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
