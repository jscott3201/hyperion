//! The two API dialects (06 §Surfaces): Anthropic-native (`/v1/messages`) and
//! OpenAI-native (`/v1/chat/completions`). Both normalize to one `PreparedPrompt`
//! (same template render, same `EngineRequest`); the dialect only selects the
//! SSE framing ([`crate::sse`]) + the error envelope shape ([`ErrorEnvelope`]).
//!
//! This module owns the `Dialect` tag + the error-envelope renderer. The
//! request-body *parsing* (body → `ChatMessage` list + sampler config) lives in
//! [`crate::prepare`]; the SSE *framing* lives in [`crate::sse`].

use serde::Serialize;

/// Which API dialect a request uses. Selects SSE framing + error envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Dialect {
    /// `POST /v1/messages` — Anthropic-native, real SSE event sequence
    /// (message_start → content_block_* → message_delta → message_stop).
    Anthropic,
    /// `POST /v1/chat/completions` — OpenAI-native, chunked SSE + `[DONE]`.
    OpenAi,
}

impl Dialect {
    /// The `Content-Type` for a non-streaming JSON response in this dialect.
    #[must_use]
    pub const fn json_content_type(self) -> &'static str {
        "application/json"
    }

    /// The `Content-Type` for an SSE stream in this dialect.
    #[must_use]
    pub const fn sse_content_type(self) -> &'static str {
        "text/event-stream"
    }
}

/// The Anthropic error-type string for an HTTP status code (the `error.type`
/// field). OpenAI uses a `type` too; both are rendered by
/// [`ErrorEnvelope::to_json`].
#[must_use]
fn error_type(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        404 => "not_found_error",
        409 => "conflict_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        499 => "client_closed_request",
        500 => "internal_server_error",
        503 => "service_unavailable",
        529 => "overloaded_error",
        _ => "api_error",
    }
}

/// The dialect-correct error envelope (06 §Error taxonomy). A single struct
/// rendered two ways: Anthropic wraps `{type:"error", error:{...}}`; OpenAI
/// uses `{error:{message, type, code}}`. Both carry the same `message`.
#[derive(Clone, Debug)]
pub struct ErrorEnvelope {
    /// The HTTP status code (the taxonomy: 400/401/404/409/413/429/499/500/
    /// 503/529).
    pub status: u16,
    /// The human-facing message (06:24 "500 internal, opaque" — for 500 the
    /// message is generic; the detail stays server-side).
    pub message: String,
}

impl ErrorEnvelope {
    /// Build an envelope. For 500/503/529 the message is genericized (the spec
    /// says internal errors are opaque); the caller may still pass a specific
    /// message for non-opaque codes.
    #[must_use]
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        let mut message = message.into();
        // 06:24 — internal errors are opaque (detail server-side only).
        if matches!(status, 500) {
            message = "internal server error".to_string();
        }
        Self { status, message }
    }

    /// Render the dialect-correct JSON body. The `status` is the HTTP code;
    /// the body shape is dialect-selected.
    #[must_use]
    pub fn to_json(self, dialect: Dialect) -> String {
        let typ = error_type(self.status);
        match dialect {
            Dialect::Anthropic => {
                // {"type":"error","error":{"type":"<type>","message":"<msg>"}}
                let body = AnthropicErrorBody {
                    r#type: "error",
                    error: AnthropicError {
                        r#type: typ,
                        message: self.message,
                    },
                };
                serde_json::to_string(&body).unwrap_or_else(|_| Self::fallback(dialect))
            }
            Dialect::OpenAi => {
                // {"error":{"message":"<msg>","type":"<type>","code":null}}
                let body = OpenAiErrorBody {
                    error: OpenAiError {
                        message: self.message,
                        r#type: typ,
                        code: serde_json::Value::Null,
                    },
                };
                serde_json::to_string(&body).unwrap_or_else(|_| Self::fallback(dialect))
            }
        }
    }

    /// A minimal fallback envelope if serialization fails (never expected —
    /// the structs are plain Serialize; kept honest per the degrade-never
    /// discipline).
    #[must_use]
    fn fallback(dialect: Dialect) -> String {
        match dialect {
            Dialect::Anthropic => {
                r#"{"type":"error","error":{"type":"api_error","message":"error"}}"#.to_string()
            }
            Dialect::OpenAi => {
                r#"{"error":{"message":"error","type":"api_error","code":null}}"#.to_string()
            }
        }
    }
}

#[derive(Serialize)]
struct AnthropicErrorBody<'a> {
    r#type: &'a str,
    error: AnthropicError<'a>,
}

#[derive(Serialize)]
struct AnthropicError<'a> {
    r#type: &'a str,
    message: String,
}

#[derive(Serialize)]
struct OpenAiErrorBody {
    error: OpenAiError,
}

#[derive(Serialize)]
struct OpenAiError {
    message: String,
    r#type: &'static str,
    code: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_envelope_shape() {
        let body = ErrorEnvelope::new(400, "bad tool schema").to_json(Dialect::Anthropic);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["message"], "bad tool schema");
    }

    #[test]
    fn openai_envelope_shape() {
        let body = ErrorEnvelope::new(429, "busy").to_json(Dialect::OpenAi);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_error");
        assert_eq!(v["error"]["message"], "busy");
        assert!(v["error"]["code"].is_null());
    }

    #[test]
    fn error_type_covers_the_taxonomy() {
        // One per code in the 06 taxonomy.
        assert_eq!(error_type(400), "invalid_request_error");
        assert_eq!(error_type(401), "authentication_error");
        assert_eq!(error_type(404), "not_found_error");
        assert_eq!(error_type(409), "conflict_error");
        assert_eq!(error_type(413), "request_too_large");
        assert_eq!(error_type(429), "rate_limit_error");
        assert_eq!(error_type(499), "client_closed_request");
        assert_eq!(error_type(500), "internal_server_error");
        assert_eq!(error_type(503), "service_unavailable");
        assert_eq!(error_type(529), "overloaded_error");
    }

    #[test]
    fn internal_500_is_opaque() {
        // 06:24 — 500 is opaque; the caller's message is replaced.
        let body =
            ErrorEnvelope::new(500, "panic in layer 42 stack trace...").to_json(Dialect::Anthropic);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["message"], "internal server error");
    }

    #[test]
    fn unknown_status_falls_back_to_api_error() {
        assert_eq!(error_type(418), "api_error");
    }
}
