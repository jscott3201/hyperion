//! B5: the pure dialect SSE framers (06 §Streaming). Real token-incremental
//! SSE — one frame per decode step, flushed immediately (the spec bans Helios's
//! 3-frame post-hoc "streaming"). Both framers are **pure functions**: given a
//! [`StepEvent`] (or an error mid-stream), produce the exact SSE bytes for
//! the dialect. The handler flushes each frame as the engine yields it.
//!
//! - **Anthropic**: the event sequence `message_start` →
//!   `content_block_start` → `content_block_delta` (text_delta per step) →
//!   `content_block_stop` → `message_delta` (stop_reason + usage) →
//!   `message_stop`. Framing: `event: <type>\ndata: <json>\n\n`.
//! - **OpenAI**: chunked `data: {…}\n\n` with `choices[0].delta.content` per
//!   step, terminating `data: [DONE]\n\n`.
//! - **529 mid-stream**: once headers are flushed (200 OK sent), a governor
//!   rejection becomes an SSE `error` event, NOT an HTTP status change.
//!
//! The framers track just enough state (the content-block index, the
//! running usage) to emit well-formed sequences. The handler owns the
//! [`Framer`] and calls [`Framer::token`] per step + [`Framer::done`] (or
//! [`Framer::error`]) at the end.

use crate::dialect::Dialect;
use crate::engine::{StepToken, Usage};

/// A stop reason for the terminal Anthropic `message_delta` / OpenAI `finish`
/// field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    /// Hit `max_tokens`.
    MaxTokens,
    /// Sampled the EOS token.
    EndTurn,
    /// Cancelled (client dropped / explicit cancel) — the stream is cut short.
    Cancelled,
}

impl StopReason {
    /// The Anthropic `stop_reason` string.
    #[must_use]
    pub(crate) fn anthropic(self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::EndTurn => "end_turn",
            Self::Cancelled => "end_turn", // a cancelled stream surfaces as end_turn to the client
        }
    }

    /// The OpenAI `finish_reason` string.
    #[must_use]
    pub(crate) fn openai(self) -> &'static str {
        match self {
            Self::MaxTokens => "length",
            Self::EndTurn => "stop",
            Self::Cancelled => "stop",
        }
    }
}

/// The SSE framer state. Owns the dialect + the running token count + the
/// content-block index. The handler drives it: `start` → `token` × n →
/// `done`/`error`.
pub struct Framer {
    dialect: Dialect,
    /// True after the opening frames (`message_start` / the first chunk's
    /// role) have been emitted.
    started: bool,
    /// The running completion-token count (for the terminal usage frame).
    completion_tokens: u32,
    /// The prompt token count (set at start, for the terminal usage frame).
    prompt_tokens: u32,
    /// A model id echoed in the frames (e.g. "gemma4-12b").
    model: String,
}

impl Framer {
    /// Create a framer for `dialect`. `model` is echoed in the start/terminal
    /// frames; `prompt_tokens` seeds the usage accounting.
    #[must_use]
    pub fn new(dialect: Dialect, model: &str, prompt_tokens: u32) -> Self {
        Self {
            dialect,
            started: false,
            completion_tokens: 0,
            prompt_tokens,
            model: model.to_string(),
        }
    }

    /// Emit the opening frames for the dialect (the `message_start` +
    /// `content_block_start` for Anthropic; nothing for OpenAI — its first
    /// chunk carries the role). Returns the SSE bytes to flush.
    #[must_use]
    pub fn start(&mut self) -> String {
        debug_assert!(!self.started, "start called twice");
        self.started = true;
        match self.dialect {
            Dialect::Anthropic => {
                let message_start = serde_json::json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": null,
                        "usage": {"input_tokens": self.prompt_tokens, "output_tokens": 0}
                    }
                });
                let block_start = serde_json::json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                });
                format!(
                    "event: message_start\ndata: {message_start}\n\nevent: content_block_start\ndata: {block_start}\n\n"
                )
            }
            Dialect::OpenAi => String::new(), // OpenAI emits no start frame; the first chunk carries the role.
        }
    }

    /// Emit one token delta. For Anthropic a `content_block_delta` (text_delta);
    /// for OpenAI a `choices[0].delta.content` chunk. The token's **decoded
    /// text** is passed in (`text`) — the engine emits token ids; the handler
    /// decodes via the tokenizer it holds, then frames. Returns the SSE bytes
    /// to flush. The `StepToken` is carried for telemetry (the logit) but the
    /// frame content is the decoded text.
    #[must_use]
    pub fn token(&mut self, _token: StepToken, text: &str) -> String {
        self.completion_tokens += 1;
        match self.dialect {
            Dialect::Anthropic => {
                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text}
                });
                format!("event: content_block_delta\ndata: {delta}\n\n")
            }
            Dialect::OpenAi => {
                let chunk = serde_json::json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                });
                format!("data: {chunk}\n\n")
            }
        }
    }

    /// Emit the terminal frames for a successful completion with `usage` +
    /// `stop`. For Anthropic: `content_block_stop` → `message_delta` (usage) →
    /// `message_stop`. For OpenAI: a final chunk with `finish_reason` + usage,
    /// then `data: [DONE]`.
    #[must_use]
    pub fn done(&mut self, usage: Usage, stop: StopReason) -> String {
        match self.dialect {
            Dialect::Anthropic => {
                let block_stop = serde_json::json!({
                    "type": "content_block_stop",
                    "index": 0
                });
                let message_delta = serde_json::json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": stop.anthropic(), "stop_sequence": null},
                    "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens}
                });
                format!(
                    "event: content_block_stop\ndata: {block_stop}\n\nevent: message_delta\ndata: {message_delta}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
                )
            }
            Dialect::OpenAi => {
                let total = usage.prompt_tokens.saturating_add(usage.completion_tokens);
                let chunk = serde_json::json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": stop.openai()}],
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": total
                    }
                });
                format!("data: {chunk}\n\ndata: [DONE]\n\n")
            }
        }
    }

    /// Emit a mid-stream error event (529 governor rejection, or any
    /// `EngineError` after headers are flushed). This is an SSE `error` event,
    /// NOT an HTTP status change — the headers (200) already went out. The
    /// `status` is carried in the event body so the client can react.
    #[must_use]
    pub fn error(&self, status: u16, message: &str) -> String {
        match self.dialect {
            Dialect::Anthropic => {
                let body = serde_json::json!({
                    "type": "error",
                    "error": {"type": anthropic_error_type(status), "message": message}
                });
                format!("event: error\ndata: {body}\n\n")
            }
            Dialect::OpenAi => {
                let body = serde_json::json!({
                    "error": {"message": message, "type": openai_error_type(status), "code": status}
                });
                format!("data: {body}\n\n")
            }
        }
    }
}

/// Map an HTTP status to the Anthropic `error.type` string (mirrors
/// `dialect::error_type` but kept local so the SSE framer is self-contained).
#[must_use]
fn anthropic_error_type(status: u16) -> &'static str {
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

#[must_use]
fn openai_error_type(status: u16) -> &'static str {
    // OpenAI uses the same type vocabulary for streamed errors.
    anthropic_error_type(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_sequence_is_well_formed() {
        let mut f = Framer::new(Dialect::Anthropic, "gemma4-12b", 2);
        let mut out = f.start();
        out.push_str(&f.token(StepToken { id: 7, logit: 0.0 }, "Hel"));
        out.push_str(&f.token(StepToken { id: 3, logit: 0.0 }, "lo"));
        out.push_str(&f.done(
            Usage {
                prompt_tokens: 2,
                completion_tokens: 2,
            },
            StopReason::EndTurn,
        ));
        // The canonical event sequence.
        assert!(out.contains("event: message_start\n"));
        assert!(out.contains("event: content_block_start\n"));
        assert!(out.contains("event: content_block_delta\n"));
        assert!(out.contains("\"text_delta\""));
        assert!(out.contains("\"text\":\"Hel\""));
        assert!(out.contains("\"text\":\"lo\""));
        assert!(out.contains("event: content_block_stop\n"));
        assert!(out.contains("event: message_delta\n"));
        assert!(out.contains("\"stop_reason\":\"end_turn\""));
        assert!(out.contains("\"output_tokens\":2"));
        assert!(out.contains("event: message_stop\n"));
        // Every frame ends with \n\n (the SSE frame separator).
        assert_eq!(out.matches("\n\n").count(), 7);
    }

    #[test]
    fn openai_sequence_ends_with_done() {
        let mut f = Framer::new(Dialect::OpenAi, "gemma4-12b", 2);
        let mut out = f.start(); // empty for OpenAI
        assert!(out.is_empty(), "OpenAI emits no start frame");
        out.push_str(&f.token(StepToken { id: 7, logit: 0.0 }, "Hi"));
        out.push_str(&f.done(
            Usage {
                prompt_tokens: 2,
                completion_tokens: 1,
            },
            StopReason::EndTurn,
        ));
        assert!(out.contains("\"delta\":{\"content\":\"Hi\"}"));
        assert!(out.contains("\"finish_reason\":\"stop\""));
        assert!(out.contains("\"prompt_tokens\":2"));
        assert!(out.contains("\"completion_tokens\":1"));
        assert!(out.contains("\"total_tokens\":3"));
        assert!(out.contains("data: [DONE]\n\n"));
    }

    #[test]
    fn openai_max_tokens_finish_is_length() {
        let mut f = Framer::new(Dialect::OpenAi, "m", 1);
        let out = f.done(Usage::default(), StopReason::MaxTokens);
        assert!(out.contains("\"finish_reason\":\"length\""));
    }

    #[test]
    fn anthropic_error_event_on_governor_rejection() {
        let f = Framer::new(Dialect::Anthropic, "m", 1);
        let ev = f.error(529, "overloaded");
        assert!(ev.starts_with("event: error\n"));
        assert!(ev.contains("\"type\":\"overloaded_error\""));
        assert!(ev.contains("\"message\":\"overloaded\""));
        assert!(ev.ends_with("\n\n"));
    }

    #[test]
    fn openai_error_event_carries_code() {
        let f = Framer::new(Dialect::OpenAi, "m", 1);
        let ev = f.error(529, "overloaded");
        assert!(ev.contains("\"type\":\"overloaded_error\""));
        assert!(ev.contains("\"code\":529"));
        assert!(ev.ends_with("\n\n"));
    }

    #[test]
    fn text_is_json_escaped() {
        // A token with a quote/newline must be escaped, not emitted raw.
        let mut f = Framer::new(Dialect::Anthropic, "m", 1);
        let out = f.token(StepToken { id: 1, logit: 0.0 }, "a\"b\n");
        assert!(
            out.contains("\"text\":\"a\\\"b\\n\""),
            "raw quote/newline must be escaped: {out}"
        );
    }
}
