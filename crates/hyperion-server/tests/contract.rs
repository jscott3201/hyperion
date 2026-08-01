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
use http_body_util::BodyExt;
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
use std::sync::atomic::{AtomicUsize, Ordering};
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

/// A model-free BPE tokenizer whose Fuse decoder concatenates native tool
/// controls and call material exactly as Gemma 4 emits them. Opener, closer,
/// and native quote markers must survive selective response decoding;
/// channel/EOS are marked special so the live adapter can prove they remain
/// hidden.
const TOOL_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 5, "content": "<|tool_call>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 6, "content": "<tool_call|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 8, "content": "<|\"|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 9, "content": "<|channel>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {"type": "Fuse"},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "[UNK]",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {
      "<eos>": 0,
      "<bos>": 1,
      "hello": 2,
      "world": 3,
      "[UNK]": 4,
      "<|tool_call>": 5,
      "<tool_call|>": 6,
      "call:lookup{query:": 7,
      "<|\"|>": 8,
      "<|channel>": 9,
      "hi": 10,
      "bye": 11,
      "}": 12,
      "call:lookup{query:7}": 13
    },
    "merges": []
  }
}"#;

/// A model-free BPE/ByteFallback tokenizer whose incomplete `<0xE5>` would
/// make the dependency synthesize a replacement at EOF, which Hyperion rejects.
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
    "vocab": {"<eos>": 0, "<bos>": 1, "<0xE5>": 2, "hello": 3},
    "merges": []
  }
}"#;

/// A deterministic Gemma-like decoder fixture: ByteFallback assembles UTF-8
/// runs before SentencePiece-style Metaspace converts `▁` boundaries to spaces.
/// The IDs include multilingual text, split emoji bytes, and added specials.
const GEMMA_LIKE_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 28, "content": "<tool>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "ByteFallback"},
      {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}
    ]
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "[UNK]",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": true,
    "ignore_merges": false,
    "vocab": {
      "<eos>": 0, "<bos>": 1, "▁Hello": 2, "▁world": 3,
      "▁SentencePiece": 4, "▁spacing": 5, "▁こんにちは": 6,
      "▁世界": 7, "▁مرحبا": 8, "ASCII": 9,
      "<0xF0>": 10, "<0x9F>": 11, "<0x98>": 12, "<0x80>": 13,
      "▁bytes": 14, "<0x62>": 15, "<0x79>": 16, "<0x74>": 17,
      "<0x65>": 18, "<0x73>": 19, "<0xC3>": 20, "<0xA9>": 21,
      "<0xE7>": 22, "<0x95>": 23, "<0x8C>": 24, "<0xE2>": 25,
      "<0x96>": 26, "<0x81>": 27, "<tool>": 28, "[UNK]": 29
    },
    "merges": []
  }
}"#;

fn mixed_256_ids() -> Vec<u32> {
    let pattern = [2, 3, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 20, 21, 28, 9];
    pattern.repeat(16)
}

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

struct PrefixBarrierEngine {
    release_tail: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl EngineDriver for PrefixBarrierEngine {
    fn stream(
        &self,
        request: &EngineRequest,
        _cancel: &CancelToken,
        tx: tokio::sync::mpsc::Sender<StepEvent>,
    ) -> Result<Usage, EngineError> {
        tx.blocking_send(StepEvent::Token(StepToken { id: 3, logit: 0.0 }))
            .map_err(|_| EngineError::Cancelled)?;
        self.release_tail
            .lock()
            .expect("tail barrier mutex")
            .recv()
            .map_err(|_| EngineError::Cancelled)?;
        tx.blocking_send(StepEvent::Token(StepToken { id: 2, logit: 0.0 }))
            .map_err(|_| EngineError::Cancelled)?;
        let usage = Usage {
            prompt_tokens: u32::try_from(request.prompt_tokens.len()).unwrap_or(u32::MAX),
            completion_tokens: 2,
        };
        tx.blocking_send(StepEvent::Done(usage))
            .map_err(|_| EngineError::Cancelled)?;
        Ok(usage)
    }
}

struct ControlledDisconnectEngine {
    started: std::sync::mpsc::SyncSender<()>,
    cancelled_and_closed: std::sync::mpsc::SyncSender<()>,
    release_cleanup: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl EngineDriver for ControlledDisconnectEngine {
    fn stream(
        &self,
        _request: &EngineRequest,
        cancel: &CancelToken,
        tx: tokio::sync::mpsc::Sender<StepEvent>,
    ) -> Result<Usage, EngineError> {
        self.started.send(()).expect("test observes live engine");
        loop {
            if tx
                .blocking_send(StepEvent::Token(StepToken { id: 2, logit: 0.0 }))
                .is_err()
            {
                assert!(cancel.is_cancelled(), "response drop cancels engine");
                self.cancelled_and_closed
                    .send(())
                    .expect("test observes cancellation and receiver close");
                self.release_cleanup
                    .lock()
                    .expect("cleanup mutex")
                    .recv()
                    .expect("test releases cleanup barrier");
                return Err(EngineError::Cancelled);
            }
        }
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

/// Send a request through the router without lossy response decoding. The
/// real-model parity gate uses this so invalid transport bytes cannot be
/// replaced before comparison.
async fn send_exact(
    router: axum::Router,
    method: Method,
    uri: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> (StatusCode, axum::http::HeaderMap, String) {
    let mut builder = Request::builder().method(method).uri(uri);
    for (key, value) in headers {
        builder = builder.header(*key, *value);
    }
    let request = builder
        .body(Body::from(body.to_string()))
        .expect("exact-body request builds");
    let response = router
        .oneshot(request)
        .await
        .expect("exact-body router responds");
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("exact-body response is readable");
    let body = String::from_utf8(bytes.to_vec()).expect("response body is valid UTF-8");
    (status, headers, body)
}

fn fresh_control() -> ControlState {
    let c = ControlState::new("test-model");
    c.set_ready(true);
    c
}

fn openai_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "look something up",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": false
            },
            "strict": false
        }
    })
}

fn anthropic_tool() -> serde_json::Value {
    serde_json::json!({
        "name": "lookup",
        "description": "look something up",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": false
        },
        "strict": false
    })
}

struct CountingEngine {
    calls: Arc<AtomicUsize>,
    tokens: Vec<u32>,
}

impl EngineDriver for CountingEngine {
    fn stream(
        &self,
        request: &EngineRequest,
        cancel: &CancelToken,
        tx: tokio::sync::mpsc::Sender<StepEvent>,
    ) -> Result<Usage, EngineError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        StubEngine {
            tokens: self.tokens.clone(),
            block_after: None,
        }
        .stream(request, cancel, tx)
    }
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

fn sse_json_frames(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE data frame contains valid JSON"))
        .collect()
}

async fn observe_prefix_before_malformed_tail(
    uri: &str,
    request_body: &str,
    dialect: &str,
) -> (String, String) {
    let control = fresh_control();
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let srv = test_server_with_engine(
        Arc::new(PrefixBarrierEngine {
            release_tail: Mutex::new(release_rx),
        }),
        None,
        4096,
        &control,
        EOF_FALLBACK_TOKENIZER_JSON,
    );
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(request_body.to_string()))
        .expect("streaming request builds");
    let response = match srv.router().oneshot(request).await {
        Ok(response) => response,
        Err(error) => {
            let _ = release_tx.send(());
            panic!("streaming request failed before barrier release: {error}");
        }
    };
    let status = response.status();
    let mut body = response.into_body();
    let mut pre_release_bytes = Vec::new();
    let pre_release_result = loop {
        let pre_release = String::from_utf8_lossy(&pre_release_bytes);
        if streamed_text(&pre_release, dialect) == "hello" {
            break Ok(());
        }
        match tokio::time::timeout(std::time::Duration::from_secs(2), body.frame()).await {
            Ok(Some(Ok(frame))) => {
                if let Ok(data) = frame.into_data() {
                    pre_release_bytes.extend_from_slice(&data);
                }
            }
            Ok(Some(Err(error))) => break Err(format!("body read failed: {error}")),
            Ok(None) => break Err("body ended before the prefix arrived".to_string()),
            Err(_) => break Err("timed out before the prefix arrived".to_string()),
        }
    };

    let release_result = release_tx.send(());
    let remaining_result = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        axum::body::to_bytes(body, 1024 * 1024),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        release_result.is_ok(),
        "engine dropped the malformed-tail barrier"
    );
    assert!(
        pre_release_result.is_ok(),
        "{}",
        pre_release_result.unwrap_err()
    );
    let remaining = remaining_result
        .expect("stream completes after malformed-tail release")
        .expect("remaining SSE body is readable");
    let pre_release = String::from_utf8_lossy(&pre_release_bytes).into_owned();
    let mut full_body = pre_release.clone();
    full_body.push_str(&String::from_utf8_lossy(&remaining));
    (pre_release, full_body)
}

fn anthropic_stream_content(body: &str) -> serde_json::Value {
    let events = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str::<serde_json::Value>(data).unwrap());
    let mut blocks = Vec::<serde_json::Value>::new();
    let mut partial_inputs = Vec::<String>::new();

    for event in events {
        let Some(index) = event["index"].as_u64().map(|index| index as usize) else {
            continue;
        };
        while blocks.len() <= index {
            blocks.push(serde_json::Value::Null);
            partial_inputs.push(String::new());
        }
        match event["type"].as_str() {
            Some("content_block_start") => {
                blocks[index] = event["content_block"].clone();
            }
            Some("content_block_delta") if event["delta"]["type"] == "text_delta" => {
                let fragment = event["delta"]["text"].as_str().unwrap();
                let block = blocks[index].as_object_mut().unwrap();
                let mut text = block
                    .get("text")
                    .and_then(|value| value.as_str())
                    .unwrap_or("")
                    .to_owned();
                text.push_str(fragment);
                block.insert("text".to_owned(), serde_json::Value::String(text));
            }
            Some("content_block_delta") if event["delta"]["type"] == "input_json_delta" => {
                partial_inputs[index].push_str(event["delta"]["partial_json"].as_str().unwrap());
            }
            _ => {}
        }
    }

    for (index, partial) in partial_inputs.into_iter().enumerate() {
        if !partial.is_empty() {
            blocks[index]
                .as_object_mut()
                .unwrap()
                .insert("input".to_owned(), serde_json::from_str(&partial).unwrap());
        }
    }
    assert!(blocks.iter().all(|block| !block.is_null()));
    serde_json::Value::Array(blocks)
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
    // max_tokens missing → 400 before template rendering (Anthropic requires
    // it for generation).
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
    let request_body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [openai_tool()]
    })
    .to_string();
    let (status, body) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        &request_body,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("tool calling") || body.contains("invalid_request_error"),
        "tools 400: {body}"
    );
}

#[tokio::test]
async fn explicit_none_with_resolved_history_reaches_engine_for_both_dialects() {
    let control = fresh_control();
    let openai_calls = Arc::new(AtomicUsize::new(0));
    let openai_server = test_server_with_engine(
        Arc::new(CountingEngine {
            calls: openai_calls.clone(),
            tokens: vec![2],
        }),
        None,
        4096,
        &control,
        MINIMAL_TOKENIZER_JSON,
    );
    let openai_body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"query\":\"hi\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "world"}
        ],
        "tools": [openai_tool()],
        "tool_choice": "none"
    })
    .to_string();
    let (status, body) = send(
        openai_server.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        &openai_body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "OpenAI history: {body}");
    assert_eq!(openai_calls.load(Ordering::SeqCst), 1);

    let anthropic_calls = Arc::new(AtomicUsize::new(0));
    let anthropic_server = test_server_with_engine(
        Arc::new(CountingEngine {
            calls: anthropic_calls.clone(),
            tokens: vec![2],
        }),
        None,
        4096,
        &control,
        MINIMAL_TOKENIZER_JSON,
    );
    let anthropic_body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup",
                "input": {"query": "hi"}
            }]},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "world"
            }]}
        ],
        "max_tokens": 4,
        "tools": [anthropic_tool()],
        "tool_choice": {"type": "none"}
    })
    .to_string();
    let (status, body) = send(
        anthropic_server.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        &anthropic_body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "Anthropic history: {body}");
    assert_eq!(anthropic_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn omitted_or_auto_tool_choice_reaches_engine_for_both_dialects() {
    let control = fresh_control();
    let openai_calls = Arc::new(AtomicUsize::new(0));
    let openai_server = test_server_with_engine(
        Arc::new(CountingEngine {
            calls: openai_calls.clone(),
            tokens: vec![2],
        }),
        None,
        4096,
        &control,
        TOOL_TOKENIZER_JSON,
    );
    let openai_body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [openai_tool()]
    })
    .to_string();
    let (status, body) = send(
        openai_server.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        &openai_body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "OpenAI auto response: {body}");
    assert_eq!(openai_calls.load(Ordering::SeqCst), 1);

    let anthropic_calls = Arc::new(AtomicUsize::new(0));
    let anthropic_server = test_server_with_engine(
        Arc::new(CountingEngine {
            calls: anthropic_calls.clone(),
            tokens: vec![2],
        }),
        None,
        4096,
        &control,
        TOOL_TOKENIZER_JSON,
    );
    let anthropic_body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "tools": [anthropic_tool()],
        "tool_choice": {"type": "auto"}
    })
    .to_string();
    let (status, body) = send(
        anthropic_server.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        &anthropic_body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "Anthropic auto response: {body}");
    assert_eq!(anthropic_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn auto_without_native_quote_fails_before_engine() {
    let control = fresh_control();
    let calls = Arc::new(AtomicUsize::new(0));
    let tokenizer_without_quote = TOOL_TOKENIZER_JSON
        .replace(
            "    {\"id\": 8, \"content\": \"<|\\\"|>\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": true},\n",
            "",
        )
        .replace("      \"<|\\\"|>\": 8,\n", "");
    let server = test_server_with_engine(
        Arc::new(CountingEngine {
            calls: calls.clone(),
            tokens: vec![2],
        }),
        None,
        4096,
        &control,
        &tokenizer_without_quote,
    );
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [openai_tool()],
        "tool_choice": "auto"
    })
    .to_string();
    let (status, response) = send(
        server.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        &body,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let envelope: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(envelope["error"]["type"], "invalid_request_error");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn non_streaming_native_tool_shapes_are_provider_compatible() {
    let openai_control = fresh_control();
    let openai = test_server_with_tokenizer(
        StubEngine {
            tokens: vec![9, 5, 7, 8, 10, 8, 12, 6],
            block_after: None,
        },
        None,
        4096,
        &openai_control,
        TOOL_TOKENIZER_JSON,
    );
    let openai_body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [openai_tool()],
        "tool_choice": "auto"
    })
    .to_string();
    let (status, body) = send(
        openai.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        &openai_body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "OpenAI response: {body}");
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    let choice = &response["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert!(choice["message"]["content"].is_null());
    let call = &choice["message"]["tool_calls"][0];
    let id = call["id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("call_") && id.ends_with("_1"));
    assert!(
        call.get("index").is_none(),
        "non-streaming tool calls remain valid history without an index"
    );
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "lookup");
    let arguments: serde_json::Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(arguments, serde_json::json!({"query": "hi"}));
    assert!(!body.contains("<|channel>"));

    let replay_body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "hi"},
            choice["message"].clone(),
            {"role": "tool", "tool_call_id": id, "content": "ok"},
        ],
        "tools": [openai_tool()],
        "tool_choice": "none"
    })
    .to_string();
    let (status, replay) = send(
        openai.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        &replay_body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "OpenAI response replay: {replay}");

    let anthropic_control = fresh_control();
    let anthropic = test_server_with_tokenizer(
        StubEngine {
            tokens: vec![2, 5, 7, 8, 10, 8, 12, 6, 3],
            block_after: None,
        },
        None,
        4096,
        &anthropic_control,
        TOOL_TOKENIZER_JSON,
    );
    let anthropic_body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "tools": [anthropic_tool()],
        "tool_choice": {"type": "auto"}
    })
    .to_string();
    let (status, body) = send(
        anthropic.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        &anthropic_body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "Anthropic response: {body}");
    let response: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(response["stop_reason"], "tool_use");
    assert_eq!(response["content"].as_array().unwrap().len(), 3);
    assert_eq!(response["content"][0]["text"], "hello");
    let call = &response["content"][1];
    assert_eq!(call["type"], "tool_use");
    let id = call["id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("toolu_") && id.ends_with("_1"));
    assert_eq!(call["name"], "lookup");
    assert_eq!(call["input"], serde_json::json!({"query": "hi"}));
    assert_eq!(response["content"][2]["text"], "world");

    let replay_body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": response["content"].clone()},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": id, "content": "ok"}],
            },
        ],
        "max_tokens": 8,
        "tools": [anthropic_tool()],
        "tool_choice": {"type": "none"}
    })
    .to_string();
    let (status, replay) = send(
        anthropic.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        &replay_body,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "Anthropic response replay: {replay}"
    );
}

#[tokio::test]
async fn streaming_native_tool_shapes_preserve_mixed_order_and_indices() {
    let openai_control = fresh_control();
    let openai = test_server_with_tokenizer(
        StubEngine {
            tokens: vec![2, 5, 7, 8, 10, 8, 12, 6, 3, 5, 7, 8, 11, 8, 12, 6],
            block_after: None,
        },
        None,
        4096,
        &openai_control,
        TOOL_TOKENIZER_JSON,
    );
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "tools": [openai_tool()]
    })
    .to_string();
    let (status, openai_sse) = send(
        openai.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        &body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "OpenAI SSE: {openai_sse}");
    let chunks: Vec<serde_json::Value> = openai_sse
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let tool_deltas: Vec<&serde_json::Value> = chunks
        .iter()
        .filter(|chunk| chunk["choices"][0]["delta"]["tool_calls"].is_array())
        .collect();
    assert_eq!(tool_deltas.len(), 2);
    let response_prefix = tool_deltas[0]["choices"][0]["delta"]["tool_calls"][0]["id"]
        .as_str()
        .unwrap()
        .strip_suffix("_1")
        .unwrap()
        .to_owned();
    for (index, chunk) in tool_deltas.iter().enumerate() {
        let call = &chunk["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["index"].as_u64(), Some(index as u64));
        assert_eq!(call["id"], format!("{response_prefix}_{}", index + 1));
        assert_eq!(call["function"]["name"], "lookup");
        serde_json::from_str::<serde_json::Value>(call["function"]["arguments"].as_str().unwrap())
            .unwrap();
    }
    assert!(
        chunks.iter().any(|chunk| {
            chunk["choices"][0]["finish_reason"] == serde_json::json!("tool_calls")
        })
    );
    assert_eq!(streamed_text(&openai_sse, "openai"), "helloworld");

    let anthropic_control = fresh_control();
    let anthropic = test_server_with_tokenizer(
        StubEngine {
            tokens: vec![2, 5, 7, 8, 10, 8, 12, 6, 3, 5, 7, 8, 11, 8, 12, 6],
            block_after: None,
        },
        None,
        4096,
        &anthropic_control,
        TOOL_TOKENIZER_JSON,
    );
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "stream": true,
        "tools": [anthropic_tool()]
    })
    .to_string();
    let (status, anthropic_sse) = send(
        anthropic.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        &body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "Anthropic SSE: {anthropic_sse}");
    let events: Vec<serde_json::Value> = anthropic_sse
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let starts: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| event["type"] == "content_block_start")
        .collect();
    assert_eq!(starts.len(), 4);
    for (index, event) in starts.iter().enumerate() {
        assert_eq!(event["index"].as_u64(), Some(index as u64));
    }
    let response_prefix = starts[1]["content_block"]["id"]
        .as_str()
        .unwrap()
        .strip_suffix("_1")
        .unwrap();
    assert!(response_prefix.starts_with("toolu_"));
    assert_eq!(
        starts[3]["content_block"]["id"],
        format!("{response_prefix}_2")
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event["delta"]["type"] == "input_json_delta")
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event["type"] == "content_block_stop")
            .count(),
        4
    );
    assert!(events.iter().any(|event| {
        event["type"] == "message_delta" && event["delta"]["stop_reason"] == "tool_use"
    }));
    assert_eq!(streamed_text(&anthropic_sse, "anthropic"), "helloworld");

    let content = anthropic_stream_content(&anthropic_sse);
    let results = content
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["type"] == "tool_use")
        .map(|block| {
            serde_json::json!({
                "type": "tool_result",
                "tool_use_id": block["id"].clone(),
                "content": "ok",
            })
        })
        .collect::<Vec<_>>();
    let replay_body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": content},
            {"role": "user", "content": results},
        ],
        "max_tokens": 16,
        "tools": [anthropic_tool()],
        "tool_choice": {"type": "none"}
    })
    .to_string();
    let (status, replay) = send(
        anthropic.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        &replay_body,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "streamed Anthropic response replay: {replay}"
    );
}

#[tokio::test]
async fn invalid_calls_round_trip_and_duplicate_stats_aggregate_once() {
    let invalid_control = fresh_control();
    let invalid = test_server_with_tokenizer(
        StubEngine {
            tokens: vec![5, 13, 6],
            block_after: None,
        },
        None,
        4096,
        &invalid_control,
        TOOL_TOKENIZER_JSON,
    );
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [openai_tool()]
    })
    .to_string();
    let (status, response) = send(
        invalid.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        &body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "invalid call response: {response}");
    let response: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "<|tool_call>call:lookup{query:7}<tool_call|>"
    );
    assert!(response["choices"][0]["message"]["tool_calls"].is_null());

    let duplicate_control = fresh_control();
    let duplicate = test_server_with_tokenizer(
        StubEngine {
            tokens: vec![5, 7, 8, 10, 8, 12, 6, 5, 7, 8, 10, 8, 12, 6],
            block_after: None,
        },
        None,
        4096,
        &duplicate_control,
        TOOL_TOKENIZER_JSON,
    );
    let (status, response) = send(
        duplicate.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        &body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "duplicate response: {response}");
    let response: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(
        response["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let (status, stats) = send(duplicate.router(), Method::GET, "/control/stats", &[], "").await;
    assert_eq!(status, StatusCode::OK);
    let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(stats["tool_calls_parsed"], 2);
    assert_eq!(stats["tool_calls_wellformed"], 2);
    assert_eq!(stats["tool_calls_repaired"], 0);
    assert_eq!(stats["tool_calls_deduped"], 1);
    assert_eq!(stats["tool_call_candidate_overflows"], 0);
    assert_eq!(stats["tool_call_limit_exceeded"], 0);
}

#[tokio::test]
async fn malformed_tool_schema_and_history_are_fixed_non_reflective_400s() {
    let control = fresh_control();
    let calls = Arc::new(AtomicUsize::new(0));
    let server = test_server_with_engine(
        Arc::new(CountingEngine {
            calls: calls.clone(),
            tokens: vec![2],
        }),
        None,
        4096,
        &control,
        MINIMAL_TOKENIZER_JSON,
    );
    let schema_body = serde_json::json!({
        "messages": [],
        "tools": [{"SENTINEL_SCHEMA_KEY": "SENTINEL_SCHEMA_VALUE"}],
        "tool_choice": "none"
    })
    .to_string();
    let (status, body) = send(
        server.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        &schema_body,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("invalid tool schema"), "schema 400: {body}");
    assert!(!body.contains("SENTINEL"), "schema source leaked: {body}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let history_body = serde_json::json!({
        "messages": [
            {"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_SENTINEL_ID",
                "name": "SENTINEL_HISTORY_NAME",
                "input": {"query": "SENTINEL_HISTORY_VALUE"}
            }]},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_SENTINEL_ID",
                "content": "SENTINEL_RESULT_VALUE"
            }]}
        ],
        "max_tokens": 4,
        "tools": [anthropic_tool()],
        "tool_choice": {"type": "none"}
    })
    .to_string();
    let (status, body) = send(
        server.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        &history_body,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("invalid tool history"), "history 400: {body}");
    assert!(!body.contains("SENTINEL"), "history source leaked: {body}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn anthropic_count_tokens_accepts_auto_tools_and_resolved_history_without_max_tokens() {
    let control = fresh_control();
    let server = test_server(
        StubEngine {
            tokens: vec![],
            block_after: None,
        },
        None,
        4096,
        &control,
    );
    let body = serde_json::json!({
        "messages": [
            {"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup",
                "input": {"query": "hi"}
            }]},
            {"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "world"
            }]}
        ],
        "tools": [anthropic_tool()],
        "tool_choice": {"type": "auto"}
    })
    .to_string();
    let (status, response) = send(
        server.router(),
        Method::POST,
        "/v1/messages/count_tokens",
        &[("content-type", "application/json")],
        &body,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "count_tokens: {response}");
    let response: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert!(response["input_tokens"].is_number());
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
async fn mixed_256_token_stream_matches_one_shot_for_both_sse_dialects() {
    let ids = mixed_256_ids();
    assert_eq!(ids.len(), 256, "fixture must exercise exactly 256 IDs");
    assert!(!ids.contains(&0), "fixture must not terminate on EOS");
    let tokenizer = TokenizerHandle::from_bytes(GEMMA_LIKE_TOKENIZER_JSON.as_bytes())
        .expect("Gemma-like tokenizer loads");
    let expected = tokenizer.decode(&ids);
    assert!(
        !expected.contains('\u{FFFD}'),
        "one-shot reference is valid"
    );

    let control = fresh_control();
    let srv = test_server_with_tokenizer(
        StubEngine {
            tokens: ids,
            block_after: None,
        },
        None,
        8192,
        &control,
        GEMMA_LIKE_TOKENIZER_JSON,
    );

    let (status, anthropic) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":256,"stream":true}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{anthropic}");
    let anthropic_text = streamed_text(&anthropic, "anthropic");
    assert_eq!(anthropic_text, expected);
    assert!(!anthropic_text.contains('\u{FFFD}'));
    let anthropic_frames = sse_json_frames(&anthropic);
    assert!(
        anthropic_frames
            .iter()
            .all(|frame| frame["type"] != "error"),
        "successful stream has no error frame: {anthropic}"
    );
    let content_delta_indices = anthropic_frames
        .iter()
        .enumerate()
        .filter_map(|(index, frame)| (frame["type"] == "content_block_delta").then_some(index))
        .collect::<Vec<_>>();
    let message_delta_indices = anthropic_frames
        .iter()
        .enumerate()
        .filter_map(|(index, frame)| (frame["type"] == "message_delta").then_some(index))
        .collect::<Vec<_>>();
    let message_stop_indices = anthropic_frames
        .iter()
        .enumerate()
        .filter_map(|(index, frame)| (frame["type"] == "message_stop").then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(message_delta_indices.len(), 1);
    assert_eq!(message_stop_indices.len(), 1);
    assert!(
        content_delta_indices
            .iter()
            .all(|index| *index < message_delta_indices[0])
    );
    assert!(message_delta_indices[0] < message_stop_indices[0]);

    let (status, openai) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":256,"stream":true}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{openai}");
    let openai_text = streamed_text(&openai, "openai");
    assert_eq!(openai_text, expected);
    assert!(!openai_text.contains('\u{FFFD}'));
    let openai_frames = sse_json_frames(&openai);
    assert!(
        openai_frames
            .iter()
            .all(|frame| frame.get("error").is_none()),
        "successful stream has no error frame: {openai}"
    );
    let finish_indices = openai_frames
        .iter()
        .enumerate()
        .filter_map(|(index, frame)| {
            frame["choices"][0]
                .get("finish_reason")
                .is_some_and(|reason| !reason.is_null())
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(finish_indices.len(), 1);
    let content_indices = openai_frames
        .iter()
        .enumerate()
        .filter_map(|(index, frame)| {
            frame["choices"][0]["delta"]["content"]
                .as_str()
                .map(|_| index)
        })
        .collect::<Vec<_>>();
    assert!(
        content_indices
            .iter()
            .all(|index| *index < finish_indices[0])
    );
    assert_eq!(finish_indices[0], openai_frames.len() - 1);
    let data_payloads = openai
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect::<Vec<_>>();
    assert_eq!(
        data_payloads
            .iter()
            .filter(|payload| **payload == "[DONE]")
            .count(),
        1
    );
    assert_eq!(data_payloads.last(), Some(&"[DONE]"));
}

#[tokio::test]
async fn malformed_eof_fallback_fails_closed_for_both_dialects() {
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
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let response: serde_json::Value = serde_json::from_str(&anthropic_json).unwrap();
    assert_eq!(
        response,
        serde_json::json!({
            "type": "error",
            "error": {
                "type": "internal_server_error",
                "message": "internal server error"
            }
        })
    );
    assert!(!anthropic_json.contains('\u{FFFD}'));
    assert!(!anthropic_json.contains("replacement character"));

    let (status, anthropic_sse) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let expected_error = serde_json::json!({
        "type": "error",
        "error": {
            "type": "internal_server_error",
            "message": "internal server error"
        }
    });
    let frames = sse_json_frames(&anthropic_sse);
    assert_eq!(
        frames
            .iter()
            .filter(|frame| frame["type"] == "error")
            .count(),
        1
    );
    assert_eq!(
        frames
            .iter()
            .filter(|frame| **frame == expected_error)
            .count(),
        1
    );
    assert_eq!(frames.last(), Some(&expected_error));
    assert_eq!(anthropic_sse.matches("event: error\n").count(), 1);
    assert!(!anthropic_sse.contains('\u{FFFD}'));
    assert!(!anthropic_sse.contains("replacement character"));
    assert!(!anthropic_sse.contains("message_delta"));
    assert!(!anthropic_sse.contains("message_stop"));

    let (status, openai_json) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}"#,
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let response: serde_json::Value = serde_json::from_str(&openai_json).unwrap();
    assert_eq!(
        response,
        serde_json::json!({
            "error": {
                "message": "internal server error",
                "type": "internal_server_error",
                "code": null
            }
        })
    );
    assert!(!openai_json.contains('\u{FFFD}'));
    assert!(!openai_json.contains("replacement character"));

    let (status, openai_sse) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let expected_error = serde_json::json!({
        "error": {
            "message": "internal server error",
            "type": "internal_server_error",
            "code": 500
        }
    });
    assert_eq!(sse_json_frames(&openai_sse), vec![expected_error]);
    assert!(!openai_sse.contains('\u{FFFD}'));
    assert!(!openai_sse.contains("replacement character"));
    assert!(!openai_sse.contains("finish_reason"));
    assert!(!openai_sse.contains("data: [DONE]"));
}

#[tokio::test]
async fn malformed_eof_after_valid_prefix_emits_prefix_then_one_opaque_error() {
    let (anthropic_pre_release, anthropic) = observe_prefix_before_malformed_tail(
        "/v1/messages",
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
        "anthropic",
    )
    .await;
    assert_eq!(streamed_text(&anthropic_pre_release, "anthropic"), "hello");
    let pre_release_frames = sse_json_frames(&anthropic_pre_release);
    assert!(
        pre_release_frames
            .iter()
            .all(|frame| frame["type"] != "error")
    );
    assert!(!anthropic_pre_release.contains("message_delta"));
    assert!(!anthropic_pre_release.contains("message_stop"));
    assert_eq!(streamed_text(&anthropic, "anthropic"), "hello");
    let anthropic_error = serde_json::json!({
        "type": "error",
        "error": {
            "type": "internal_server_error",
            "message": "internal server error"
        }
    });
    let anthropic_frames = sse_json_frames(&anthropic);
    let anthropic_error_indices = anthropic_frames
        .iter()
        .enumerate()
        .filter_map(|(index, frame)| (frame == &anthropic_error).then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(anthropic_error_indices.len(), 1);
    assert_eq!(
        anthropic_frames
            .iter()
            .filter(|frame| frame["type"] == "error")
            .count(),
        1
    );
    assert_eq!(anthropic_error_indices[0], anthropic_frames.len() - 1);
    let text_index = anthropic_frames
        .iter()
        .position(|frame| frame["delta"]["text"] == "hello")
        .expect("valid prefix has a text-delta frame");
    assert!(text_index < anthropic_error_indices[0]);
    assert_eq!(anthropic.matches("event: error\n").count(), 1);
    assert!(!anthropic.contains('\u{FFFD}'));
    assert!(!anthropic.contains("replacement character"));
    assert!(!anthropic.contains("message_delta"));
    assert!(!anthropic.contains("message_stop"));

    let (openai_pre_release, openai) = observe_prefix_before_malformed_tail(
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
        "openai",
    )
    .await;
    assert_eq!(streamed_text(&openai_pre_release, "openai"), "hello");
    let pre_release_frames = sse_json_frames(&openai_pre_release);
    assert!(
        pre_release_frames
            .iter()
            .all(|frame| frame.get("error").is_none())
    );
    assert!(pre_release_frames.iter().all(|frame| {
        frame["choices"][0]
            .get("finish_reason")
            .is_none_or(serde_json::Value::is_null)
    }));
    assert!(!openai_pre_release.contains("data: [DONE]"));
    assert_eq!(streamed_text(&openai, "openai"), "hello");
    let openai_error = serde_json::json!({
        "error": {
            "message": "internal server error",
            "type": "internal_server_error",
            "code": 500
        }
    });
    let openai_frames = sse_json_frames(&openai);
    let openai_error_indices = openai_frames
        .iter()
        .enumerate()
        .filter_map(|(index, frame)| (frame == &openai_error).then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(openai_error_indices.len(), 1);
    assert_eq!(
        openai_frames
            .iter()
            .filter(|frame| frame.get("error").is_some())
            .count(),
        1
    );
    assert_eq!(openai_error_indices[0], openai_frames.len() - 1);
    let text_index = openai_frames
        .iter()
        .position(|frame| frame["choices"][0]["delta"]["content"] == "hello")
        .expect("valid prefix has a content chunk");
    assert!(text_index < openai_error_indices[0]);
    assert_eq!(openai.matches("internal server error").count(), 1);
    assert!(openai_frames.iter().all(|frame| {
        frame["choices"][0]
            .get("finish_reason")
            .is_none_or(serde_json::Value::is_null)
    }));
    assert!(!openai.contains('\u{FFFD}'));
    assert!(!openai.contains("replacement character"));
    assert!(!openai.contains("data: [DONE]"));
}

#[tokio::test]
async fn non_streaming_malformed_eof_discards_collected_prefix_for_both_dialects() {
    let control = fresh_control();
    let srv = test_server_with_tokenizer(
        StubEngine {
            tokens: vec![3, 2],
            block_after: None,
        },
        None,
        4096,
        &control,
        EOF_FALLBACK_TOKENIZER_JSON,
    );

    let (status, anthropic) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}"#,
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&anthropic).unwrap(),
        serde_json::json!({
            "type": "error",
            "error": {
                "type": "internal_server_error",
                "message": "internal server error"
            }
        })
    );
    assert!(!anthropic.contains("hello"));
    assert!(!anthropic.contains('\u{FFFD}'));
    assert!(!anthropic.contains("replacement character"));

    let (status, openai) = send(
        srv.router(),
        Method::POST,
        "/v1/chat/completions",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}"#,
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&openai).unwrap(),
        serde_json::json!({
            "error": {
                "message": "internal server error",
                "type": "internal_server_error",
                "code": null
            }
        })
    );
    assert!(!openai.contains("hello"));
    assert!(!openai.contains('\u{FFFD}'));
    assert!(!openai.contains("replacement character"));
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

/// A dropped SSE response cancels immediately but retains its generation lease
/// until the detached blocking engine closure finishes cleanup.
#[tokio::test]
async fn disconnect_holds_lease_until_engine_cleanup_exits() {
    let control = fresh_control();
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
    let (cancelled_tx, cancelled_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let srv = Arc::new(test_server_with_engine(
        Arc::new(ControlledDisconnectEngine {
            started: started_tx,
            cancelled_and_closed: cancelled_tx,
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
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
        ))
        .unwrap();
    let response = srv.router().oneshot(request).await.unwrap();
    let initial_status = response.status();
    let mut body = response.into_body();
    let opening_frame = body.frame().await.expect("opening SSE frame").unwrap();

    tokio::task::spawn_blocking(move || {
        started_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("engine is live before disconnect")
    })
    .await
    .unwrap();
    drop(body);
    tokio::task::spawn_blocking(move || {
        cancelled_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("engine observes cancellation and receiver close")
    })
    .await
    .unwrap();

    let in_flight_during_cleanup = control.is_in_flight();
    let (concurrent_status, concurrent_body) = send(
        srv.router(),
        Method::POST,
        "/v1/messages",
        &[("content-type", "application/json")],
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
    )
    .await;
    let (reload_during_cleanup, _) =
        send(srv.router(), Method::POST, "/control/reload", &[], "").await;

    release_tx.send(()).expect("release engine cleanup");
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while control.is_in_flight() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("generation lease clears after engine cleanup");
    let (reload_after_cleanup, _) =
        send(srv.router(), Method::POST, "/control/reload", &[], "").await;

    assert_eq!(initial_status, StatusCode::OK);
    assert!(
        opening_frame.is_data(),
        "stream was polled before disconnect"
    );
    assert!(
        in_flight_during_cleanup,
        "engine lease keeps in_flight true during cleanup"
    );
    assert_eq!(
        concurrent_status,
        StatusCode::TOO_MANY_REQUESTS,
        "{concurrent_body}"
    );
    assert_eq!(reload_during_cleanup, StatusCode::CONFLICT);
    assert_eq!(
        reload_after_cleanup,
        StatusCode::NOT_IMPLEMENTED,
        "lease clears only after engine cleanup exits"
    );
}

// ── The self-hosted M5 real-model gate (#[ignore]) ──────────────────────

#[derive(Debug)]
struct GreedyGolden {
    prompt_ids: Vec<u32>,
    greedy_tokens: Vec<u32>,
}

/// Read one I32 vector from a safetensors file without adding a test-only
/// dependency. Offsets are relative to the byte immediately after the padded
/// JSON header, as defined by the safetensors format.
fn read_greedy_golden(path: &std::path::Path) -> Result<GreedyGolden, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() < 8 {
        return Err("header is shorter than the 8-byte length prefix".to_string());
    }
    let mut length_bytes = [0_u8; 8];
    length_bytes.copy_from_slice(&bytes[..8]);
    let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
        .map_err(|_| "header length does not fit usize".to_string())?;
    if header_len == 0 {
        return Err("header length is empty".to_string());
    }
    let data_start = 8_usize
        .checked_add(header_len)
        .ok_or_else(|| "header byte range overflows usize".to_string())?;
    let header_bytes = bytes
        .get(8..data_start)
        .ok_or_else(|| "header byte range exceeds fixture length".to_string())?;
    let header_text = std::str::from_utf8(header_bytes)
        .map_err(|error| format!("header is not UTF-8: {error}"))?;
    let header: serde_json::Value = serde_json::from_str(header_text)
        .map_err(|error| format!("header is not valid JSON: {error}"))?;
    let tensors = header
        .as_object()
        .ok_or_else(|| "header root is not an object".to_string())?;

    let read_vector = |name: &str| -> Result<Vec<u32>, String> {
        let descriptor = tensors
            .get(name)
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("missing tensor descriptor {name:?}"))?;
        if descriptor.get("dtype").and_then(serde_json::Value::as_str) != Some("I32") {
            return Err(format!("tensor {name:?} is not I32"));
        }
        let shape = descriptor
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("tensor {name:?} has no shape array"))?;
        if shape.len() != 1 {
            return Err(format!("tensor {name:?} is not one-dimensional"));
        }
        let element_count = shape[0]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("tensor {name:?} has an invalid element count"))?;
        let offsets = descriptor
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("tensor {name:?} has no data_offsets array"))?;
        if offsets.len() != 2 {
            return Err(format!(
                "tensor {name:?} data_offsets does not have two entries"
            ));
        }
        let start = offsets[0]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("tensor {name:?} has an invalid start offset"))?;
        let end = offsets[1]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("tensor {name:?} has an invalid end offset"))?;
        if start > end || start % std::mem::size_of::<i32>() != 0 {
            return Err(format!("tensor {name:?} has invalid or unaligned offsets"));
        }
        let expected_bytes = element_count
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| format!("tensor {name:?} byte length overflows usize"))?;
        if end - start != expected_bytes {
            return Err(format!(
                "tensor {name:?} byte length {} does not match shape byte length {expected_bytes}",
                end - start
            ));
        }
        let absolute_start = data_start
            .checked_add(start)
            .ok_or_else(|| format!("tensor {name:?} start offset overflows usize"))?;
        let absolute_end = data_start
            .checked_add(end)
            .ok_or_else(|| format!("tensor {name:?} end offset overflows usize"))?;
        let tensor_bytes = bytes
            .get(absolute_start..absolute_end)
            .ok_or_else(|| format!("tensor {name:?} byte range exceeds fixture length"))?;
        tensor_bytes
            .chunks_exact(std::mem::size_of::<i32>())
            .enumerate()
            .map(|(index, chunk)| {
                let mut value_bytes = [0_u8; 4];
                value_bytes.copy_from_slice(chunk);
                let value = i32::from_le_bytes(value_bytes);
                u32::try_from(value).map_err(|_| {
                    format!("tensor {name:?} contains negative ID {value} at index {index}")
                })
            })
            .collect()
    };

    Ok(GreedyGolden {
        prompt_ids: read_vector("ids")?,
        greedy_tokens: read_vector("greedy_tokens")?,
    })
}

fn command_failure(label: &str, output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    format!(
        "{label} failed with status {}: stdout={stdout:?} stderr={stderr:?}",
        output.status
    )
}

/// Independently reject unknown/bad artifact identity when the ignored test is
/// selected directly. The evidence runner performs the same checks before and
/// after the test; keeping this guard here prevents a direct cargo invocation
/// from bypassing C4/C6.
fn verify_test_artifact_identity(
    artifact: &std::path::Path,
    manifest_name: &str,
) -> Result<(), String> {
    const HISTORICAL_MANIFEST_SHA256: &str =
        "9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144";
    const OWNER_PAYLOAD_MANIFEST_SHA256: &str =
        "3cee7e9c21051eb6e6857ef485b355b4a2e847901930d64620348cbc7f56c806";

    if manifest_name == "SHA256SUMS" {
        let verifier =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/model_identity.py");
        let output = std::process::Command::new("python3")
            .args(["-I", "-S"])
            .arg(verifier)
            .args(["--model"])
            .arg(artifact)
            .args(["--manifest-sha256", HISTORICAL_MANIFEST_SHA256])
            .output()
            .map_err(|error| format!("historical identity verifier did not start: {error}"))?;
        if !output.status.success() {
            return Err(command_failure("historical artifact identity", &output));
        }
        return Ok(());
    }

    let manifest = artifact.join("PAYLOAD_SHA256SUMS");
    let digest_output = std::process::Command::new("shasum")
        .args(["-a", "256"])
        .arg(&manifest)
        .output()
        .map_err(|error| format!("owner manifest digest command did not start: {error}"))?;
    if !digest_output.status.success() {
        return Err(command_failure("owner manifest digest", &digest_output));
    }
    let digest_stdout = std::str::from_utf8(&digest_output.stdout)
        .map_err(|error| format!("owner manifest digest is not UTF-8: {error}"))?;
    let actual_digest = digest_stdout
        .split_whitespace()
        .next()
        .ok_or_else(|| "owner manifest digest output is empty".to_string())?;
    if actual_digest != OWNER_PAYLOAD_MANIFEST_SHA256 {
        return Err(format!(
            "owner manifest digest mismatch: expected {OWNER_PAYLOAD_MANIFEST_SHA256}, found {actual_digest}"
        ));
    }

    let expected_names: std::collections::BTreeSet<_> = [
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    let manifest_text = std::fs::read_to_string(&manifest)
        .map_err(|error| format!("owner manifest is not readable UTF-8: {error}"))?;
    let mut actual_names = std::collections::BTreeSet::new();
    for (index, line) in manifest_text.lines().enumerate() {
        let (digest, raw_name) = line
            .split_once("  ")
            .ok_or_else(|| format!("owner manifest line {} is malformed", index + 1))?;
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "owner manifest line {} has an invalid SHA-256",
                index + 1
            ));
        }
        let without_binary_marker = raw_name.strip_prefix('*').unwrap_or(raw_name);
        let name = without_binary_marker
            .strip_prefix("./")
            .unwrap_or(without_binary_marker);
        let relative = std::path::Path::new(name);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
            || !actual_names.insert(name.to_string())
        {
            return Err(format!(
                "owner manifest line {} has a non-normal or duplicate path",
                index + 1
            ));
        }
    }
    if actual_names != expected_names {
        return Err(format!(
            "owner manifest must bind exactly the eight runtime files: {actual_names:?}"
        ));
    }
    for name in &actual_names {
        let path = artifact.join(name);
        if !path.is_file() || path.is_symlink() {
            return Err(format!(
                "owner payload file is missing, special, or symlinked: {name}"
            ));
        }
    }
    let checksum_output = std::process::Command::new("shasum")
        .args(["-a", "256", "-c", "PAYLOAD_SHA256SUMS"])
        .current_dir(artifact)
        .output()
        .map_err(|error| format!("owner payload checksum command did not start: {error}"))?;
    if !checksum_output.status.success() {
        return Err(command_failure("owner payload checksum", &checksum_output));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct RecordedHttpRun {
    prompt_ids: Vec<u32>,
    token_ids: Vec<u32>,
    max_tokens: u32,
    greedy: bool,
    done_usage: Option<Usage>,
    done_events: usize,
    done_was_last: bool,
    returned_usage: Option<Usage>,
    forwarded_all_events: bool,
}

/// Test-only observation seam around the real mailbox. The one-slot relay
/// forwards every event synchronously and in order; if the router-side bounded
/// channel closes, forwarding cancels the same request and the relay receiver
/// closes so the native engine observes backpressure/cancellation normally.
struct RecordingMailboxEngine {
    inner: hyperion_server::engine::MailboxEngine,
    runs: Arc<Mutex<Vec<RecordedHttpRun>>>,
}

struct RealServerHarness {
    server: Option<Server>,
    engine_thread: Option<std::thread::JoinHandle<()>>,
}

impl RealServerHarness {
    fn router(&self) -> axum::Router {
        self.server
            .as_ref()
            .expect("lifecycle: real server is present")
            .router()
    }

    fn shutdown(mut self) -> Result<(), String> {
        self.server.take();
        let engine_thread = self
            .engine_thread
            .take()
            .ok_or_else(|| "lifecycle: real engine thread handle is missing".to_string())?;
        engine_thread
            .join()
            .map_err(|_| "lifecycle: real engine thread panicked".to_string())
    }
}

impl Drop for RealServerHarness {
    fn drop(&mut self) {
        self.server.take();
        if let Some(engine_thread) = self.engine_thread.take() {
            let _ = engine_thread.join();
        }
    }
}

impl EngineDriver for RecordingMailboxEngine {
    fn stream(
        &self,
        request: &EngineRequest,
        cancel: &CancelToken,
        tx: tokio::sync::mpsc::Sender<StepEvent>,
    ) -> Result<Usage, EngineError> {
        let (relay_tx, mut relay_rx) = tokio::sync::mpsc::channel::<StepEvent>(1);
        let relay_cancel = cancel.clone();
        let relay = std::thread::Builder::new()
            .name("m3-recording-relay".to_string())
            .spawn(move || {
                let mut token_ids = Vec::new();
                let mut done_usage = None;
                let mut done_events = 0_usize;
                let mut done_was_last = false;
                while let Some(event) = relay_rx.blocking_recv() {
                    match event {
                        StepEvent::Token(token) => {
                            token_ids.push(token.id);
                            done_was_last = false;
                        }
                        StepEvent::Done(usage) => {
                            done_usage = Some(usage);
                            done_events = done_events.saturating_add(1);
                            done_was_last = true;
                        }
                    }
                    if tx.blocking_send(event).is_err() {
                        relay_cancel.cancel();
                        return (token_ids, done_usage, done_events, done_was_last, false);
                    }
                }
                (token_ids, done_usage, done_events, done_was_last, true)
            })
            .expect("lifecycle: recording relay thread spawns");

        let result = self.inner.stream(request, cancel, relay_tx);
        let returned_usage = result.as_ref().ok().copied();
        let (token_ids, done_usage, done_events, done_was_last, forwarded_all_events) = relay
            .join()
            .expect("lifecycle: recording relay thread joins cleanly");
        self.runs
            .lock()
            .expect("lifecycle: recording mutex is not poisoned")
            .push(RecordedHttpRun {
                prompt_ids: request.prompt_tokens.clone(),
                token_ids,
                max_tokens: request.max_tokens,
                greedy: request.sampling.is_none(),
                done_usage,
                done_events,
                done_was_last,
                returned_usage,
                forwarded_all_events,
            });
        if !forwarded_all_events {
            return Err(EngineError::Cancelled);
        }
        result
    }
}

// Run only through scripts/run-m3-stream-parity.sh on the self-hosted M5. The
// runner binds the complete artifact inventory before selecting this exact
// ignored test.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires the exact 12B artifact + a Metal device (self-hosted M5)"]
async fn real_http_sse_matches_m2_golden() {
    use hyperion_server::engine::{MailboxEngine, engine_thread_loop};

    const PROMPT: &str = "The capital of France is";
    const COMPLETION_TOKENS: usize = 24;
    const HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

    let artifact_os = std::env::var_os("HYPERION_12B_ARTIFACT")
        .filter(|value| !value.is_empty())
        .expect("artifact identity: HYPERION_12B_ARTIFACT must name the pinned real artifact");
    let artifact = std::path::PathBuf::from(artifact_os);
    assert!(
        artifact.is_dir() && !artifact.is_symlink(),
        "artifact identity: root must be a real directory, not a symlink: {}",
        artifact.display()
    );
    let historical_manifest = artifact.join("SHA256SUMS");
    let owner_payload_manifest = artifact.join("PAYLOAD_SHA256SUMS");
    let has_historical_manifest = historical_manifest.exists() || historical_manifest.is_symlink();
    let has_owner_payload_manifest =
        owner_payload_manifest.exists() || owner_payload_manifest.is_symlink();
    assert_ne!(
        has_historical_manifest, has_owner_payload_manifest,
        "artifact identity: exactly one recognized manifest must exist"
    );
    let selected_manifest = if has_historical_manifest {
        historical_manifest
    } else {
        owner_payload_manifest
    };
    assert!(
        selected_manifest.is_file() && !selected_manifest.is_symlink(),
        "artifact identity: selected manifest must be a real regular file"
    );
    let selected_manifest_name = selected_manifest
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .expect("artifact identity: selected manifest filename is UTF-8");
    verify_test_artifact_identity(&artifact, selected_manifest_name)
        .unwrap_or_else(|error| panic!("artifact identity: {error}"));
    for required in [
        "config.json",
        "chat_template.jinja",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        let path = artifact.join(required);
        assert!(
            path.is_file() && !path.is_symlink(),
            "artifact identity: required regular file is missing or a symlink: {}",
            path.display()
        );
    }

    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../native/hyperion_mlx/tests/fixtures/12b_greedy_golden.safetensors");
    assert!(
        fixture_path.is_file() && !fixture_path.is_symlink(),
        "fixture: committed M2 golden is missing or a symlink: {}",
        fixture_path.display()
    );
    let golden = read_greedy_golden(&fixture_path)
        .unwrap_or_else(|error| panic!("fixture: invalid M2 safetensors golden: {error}"));
    assert_eq!(
        golden.prompt_ids.len(),
        21,
        "fixture: expected the frozen 21-ID prompt"
    );
    assert_eq!(
        golden.greedy_tokens.len(),
        COMPLETION_TOKENS,
        "fixture: expected the frozen 24-token completion"
    );

    let config = std::fs::read_to_string(artifact.join("config.json"))
        .expect("artifact identity: 12B config.json is readable");
    let geometry = hyperion_model::geometry::Geometry::from_text_config_str(&config)
        .expect("artifact identity: 12B geometry parses");
    let tokenizer = TokenizerHandle::from_file(&artifact.join("tokenizer.json"))
        .expect("artifact identity: tokenizer loads");
    let template = ChatTemplate::from_artifact(&artifact, Some(&tokenizer))
        .expect("artifact identity: chat template loads");
    let expected_text = tokenizer.decode(&golden.greedy_tokens);
    assert!(
        !expected_text.is_empty(),
        "fixture: golden completion must decode to non-empty text"
    );
    let artifact_string = artifact
        .to_str()
        .expect("artifact identity: artifact path is valid UTF-8")
        .to_string();
    let context = geometry.max_position_embeddings;

    let (mailbox, mailbox_rx) = MailboxEngine::channel();
    let (loaded_tx, loaded_rx) = std::sync::mpsc::channel();
    let engine_thread = std::thread::Builder::new()
        .name("m3-real-engine".to_string())
        .spawn(move || engine_thread_loop(geometry, artifact_string, mailbox_rx, loaded_tx))
        .expect("lifecycle: real engine thread spawns");
    match loaded_rx.recv_timeout(std::time::Duration::from_secs(300)) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            drop(mailbox);
            engine_thread
                .join()
                .expect("lifecycle: failed engine load thread joins");
            panic!("artifact identity/model loading: real engine failed to load: {error}");
        }
        Err(error) => {
            drop(mailbox);
            engine_thread
                .join()
                .expect("lifecycle: timed-out engine load thread joins");
            panic!("lifecycle: real engine load did not report before timeout: {error}");
        }
    }

    let runs = Arc::new(Mutex::new(Vec::new()));
    let control = fresh_control();
    let harness = RealServerHarness {
        server: Some(Server::new(ServerConfig {
            engine: Arc::new(RecordingMailboxEngine {
                inner: mailbox,
                runs: runs.clone(),
            }),
            template,
            tokenizer: tokenizer.clone(),
            control,
            bearer: None,
            context,
            model_id: "gemma4-12b".to_string(),
            default_max_tokens: COMPLETION_TOKENS as u32,
        })),
        engine_thread: Some(engine_thread),
    };

    let non_streaming_request = serde_json::json!({
        "model": "gemma4-12b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": COMPLETION_TOKENS,
        "temperature": 0,
        "stream": false
    })
    .to_string();
    let non_streaming_result = tokio::time::timeout(
        HTTP_TIMEOUT,
        send_exact(
            harness.router(),
            Method::POST,
            "/v1/chat/completions",
            &[("content-type", "application/json")],
            &non_streaming_request,
        ),
    )
    .await;

    let streaming_request = serde_json::json!({
        "model": "gemma4-12b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": COMPLETION_TOKENS,
        "temperature": 0,
        "stream": true
    })
    .to_string();
    let streaming_result = if non_streaming_result.is_ok() {
        Some(
            tokio::time::timeout(
                HTTP_TIMEOUT,
                send_exact(
                    harness.router(),
                    Method::POST,
                    "/v1/chat/completions",
                    &[("content-type", "application/json")],
                    &streaming_request,
                ),
            )
            .await,
        )
    } else {
        None
    };

    harness.shutdown().unwrap_or_else(|error| panic!("{error}"));
    let non_streaming = non_streaming_result
        .expect("lifecycle: non-streaming HTTP request completes before timeout");
    let streaming = streaming_result
        .expect("lifecycle: streaming HTTP request skipped after non-streaming timeout")
        .expect("lifecycle: streaming HTTP request completes before timeout");

    let recorded = runs
        .lock()
        .expect("lifecycle: recording mutex is not poisoned")
        .clone();
    assert_eq!(
        recorded.len(),
        2,
        "lifecycle: exactly two HTTP generations reach the real mailbox"
    );
    for (index, run) in recorded.iter().enumerate() {
        assert_eq!(
            run.prompt_ids, golden.prompt_ids,
            "prompt: HTTP run {index} request IDs differ from the M2 golden"
        );
        assert!(
            run.greedy,
            "prompt: HTTP run {index} was not explicit greedy"
        );
        assert_eq!(
            run.max_tokens, COMPLETION_TOKENS as u32,
            "prompt: HTTP run {index} max_tokens differs from the M2 golden"
        );
        assert_eq!(
            run.token_ids, golden.greedy_tokens,
            "token: HTTP run {index} generated IDs differ from the M2 golden"
        );
        assert!(
            run.forwarded_all_events,
            "lifecycle: HTTP run {index} did not forward every real engine event"
        );
        let expected_usage = Usage {
            prompt_tokens: golden.prompt_ids.len() as u32,
            completion_tokens: COMPLETION_TOKENS as u32,
        };
        assert_eq!(
            run.done_usage,
            Some(expected_usage),
            "token: HTTP run {index} terminal StepEvent usage differs"
        );
        assert_eq!(
            run.done_events, 1,
            "token: HTTP run {index} must emit exactly one terminal StepEvent"
        );
        assert!(
            run.done_was_last,
            "token: HTTP run {index} terminal StepEvent must be last"
        );
        assert_eq!(
            run.returned_usage,
            Some(expected_usage),
            "token: HTTP run {index} returned usage differs"
        );
    }

    let (json_status, json_headers, json_body) = non_streaming;
    assert_eq!(json_status, StatusCode::OK, "JSON: {json_body}");
    assert_eq!(
        json_headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json"),
        "JSON: response content type"
    );
    let json: serde_json::Value =
        serde_json::from_str(&json_body).expect("JSON: response body is valid JSON");
    assert!(
        json["error"].is_null(),
        "JSON: response contains an error: {json}"
    );
    assert_eq!(
        json["choices"][0]["message"]["content"].as_str(),
        Some(expected_text.as_str()),
        "JSON: assistant content differs from decoding the M2 golden tokens"
    );
    assert_eq!(
        json["choices"][0]["finish_reason"], "length",
        "JSON: terminal reason"
    );
    assert_eq!(
        json["usage"]["prompt_tokens"],
        golden.prompt_ids.len(),
        "JSON: prompt usage differs"
    );
    assert_eq!(
        json["usage"]["completion_tokens"], COMPLETION_TOKENS,
        "JSON: completion usage differs"
    );
    assert_eq!(
        json["usage"]["total_tokens"],
        golden.prompt_ids.len() + COMPLETION_TOKENS,
        "JSON: total usage differs"
    );

    let (sse_status, sse_headers, sse_body) = streaming;
    assert_eq!(sse_status, StatusCode::OK, "SSE: {sse_body}");
    assert_eq!(
        sse_headers
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream"),
        "SSE: response content type"
    );
    assert!(
        !sse_body.contains("event: error"),
        "SSE: response contains an error event: {sse_body}"
    );
    assert_eq!(
        sse_body.matches("data: [DONE]\n\n").count(),
        1,
        "SSE: terminal [DONE] framing must occur exactly once"
    );
    assert!(
        sse_body.ends_with("data: [DONE]\n\n"),
        "SSE: [DONE] must be the final bytes"
    );
    for block in sse_body.split("\n\n").filter(|block| !block.is_empty()) {
        assert_eq!(
            block.lines().count(),
            1,
            "SSE: every OpenAI block must contain exactly one data line: {block:?}"
        );
        assert!(
            block.starts_with("data: "),
            "SSE: every OpenAI block must be a data line: {block:?}"
        );
    }
    let frames = sse_json_frames(&sse_body);
    assert!(
        frames.iter().all(|frame| frame["error"].is_null()),
        "SSE: JSON data frames contain an error"
    );
    let terminal_frames: Vec<_> = frames
        .iter()
        .filter(|frame| !frame["choices"][0]["finish_reason"].is_null())
        .collect();
    assert_eq!(
        terminal_frames.len(),
        1,
        "SSE: exactly one terminal usage frame is required"
    );
    let terminal = terminal_frames[0];
    assert!(
        std::ptr::eq(terminal, frames.last().expect("SSE: terminal frame exists")),
        "SSE: usage/finish frame must be the final JSON frame"
    );
    assert_eq!(
        terminal["choices"][0]["finish_reason"], "length",
        "SSE: terminal reason"
    );
    assert_eq!(
        terminal["usage"]["prompt_tokens"],
        golden.prompt_ids.len(),
        "SSE: prompt usage differs"
    );
    assert_eq!(
        terminal["usage"]["completion_tokens"], COMPLETION_TOKENS,
        "SSE: completion usage differs"
    );
    assert_eq!(
        terminal["usage"]["total_tokens"],
        golden.prompt_ids.len() + COMPLETION_TOKENS,
        "SSE: total usage differs"
    );
    let sse_text = streamed_text(&sse_body, "openai");
    assert_eq!(
        sse_text, expected_text,
        "SSE: concatenated content differs from decoding the M2 golden tokens"
    );
    assert_eq!(
        sse_text,
        json["choices"][0]["message"]["content"]
            .as_str()
            .expect("JSON: assistant content is text"),
        "SSE: concatenated content is not byte-identical to JSON assistant content"
    );
}
