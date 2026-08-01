//! B5: the pure dialect SSE framers (06 §Streaming). Real token-incremental
//! SSE — one frame per safely decoded text fragment, flushed immediately (the
//! spec bans post-hoc "streaming"). Both framers are **pure functions**: given
//! decoded text (or an error mid-stream), produce the exact SSE bytes for the
//! dialect. The response decoder may briefly withhold an incomplete byte
//! fallback sequence until it can emit valid text.
//!
//! - **Anthropic**: `message_start`, then ordered text or `tool_use` content
//!   block lifecycles, followed by `message_delta` and `message_stop`.
//! - **OpenAI**: chunked text or indexed `delta.tool_calls`, then the terminal
//!   usage chunk and `data: [DONE]\n\n`.
//! - **529 mid-stream**: once headers are flushed (200 OK sent), a governor
//!   rejection becomes an SSE `error` event, NOT an HTTP status change.
//!
//! The framers track just enough state (the content-block index and response
//! metadata) to emit well-formed sequences. The handler owns the
//! [`Framer`] and calls [`Framer::text`] or [`Framer::tool_call`] for each
//! validated event, then [`Framer::done`] (or [`Framer::error`]) at the end.

use crate::dialect::Dialect;
use crate::engine::Usage;
use crate::response_adapter::AcceptedToolCall;

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

/// The SSE framer state. Owns the dialect, model metadata, and content-block
/// index. The handler drives it: `start` → `text` × n →
/// `done`/`error`.
pub struct Framer {
    dialect: Dialect,
    /// True after the opening frames (`message_start` / the first chunk's
    /// role) have been emitted.
    started: bool,
    /// The prompt token count (set at start, for the terminal usage frame).
    prompt_tokens: u32,
    /// A model id echoed in the frames (e.g. "gemma4-12b").
    model: String,
    /// Whether an Anthropic text block is currently open.
    anthropic_text_open: bool,
    /// Next monotonic Anthropic content-block index.
    next_content_index: usize,
    /// Next contiguous OpenAI tool-call index.
    next_tool_index: usize,
    /// Accepted call presence owns the successful terminal reason.
    has_tool_calls: bool,
}

impl Framer {
    /// Create a framer for `dialect`. `model` is echoed in the start/terminal
    /// frames; `prompt_tokens` seeds the usage accounting.
    #[must_use]
    pub fn new(dialect: Dialect, model: &str, prompt_tokens: u32) -> Self {
        Self {
            dialect,
            started: false,
            prompt_tokens,
            model: model.to_string(),
            anthropic_text_open: false,
            next_content_index: 0,
            next_tool_index: 0,
            has_tool_calls: false,
        }
    }

    /// Emit the opening frame for the dialect (`message_start` for Anthropic;
    /// nothing for OpenAI). Content blocks begin lazily so a call-only turn
    /// does not gain a synthetic empty text block.
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
                format!("event: message_start\ndata: {message_start}\n\n")
            }
            Dialect::OpenAi => String::new(), // OpenAI emits no start frame; the first chunk carries the role.
        }
    }

    /// Emit one safe decoded-text fragment. For Anthropic this is a
    /// `content_block_delta` (`text_delta`); for OpenAI it is a
    /// `choices[0].delta.content` chunk.
    #[must_use]
    pub fn text(&mut self, text: &str) -> String {
        match self.dialect {
            Dialect::Anthropic => {
                let index = self.next_content_index.saturating_sub(1);
                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text}
                });
                if self.anthropic_text_open {
                    format!("event: content_block_delta\ndata: {delta}\n\n")
                } else {
                    let index = self.next_content_index;
                    self.next_content_index = self.next_content_index.saturating_add(1);
                    self.anthropic_text_open = true;
                    let block_start = serde_json::json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""}
                    });
                    let delta = serde_json::json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text}
                    });
                    format!(
                        "event: content_block_start\ndata: {block_start}\n\nevent: content_block_delta\ndata: {delta}\n\n"
                    )
                }
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

    /// Emit one validated call using the provider-native streaming shape.
    #[must_use]
    pub(crate) fn tool_call(&mut self, call: &AcceptedToolCall) -> String {
        self.has_tool_calls = true;
        let arguments = serde_json::to_string(&call.arguments)
            .expect("serde_json::Value serialization cannot fail");
        match self.dialect {
            Dialect::Anthropic => {
                let mut output = String::new();
                if self.anthropic_text_open {
                    let index = self.next_content_index.saturating_sub(1);
                    let block_stop = serde_json::json!({
                        "type": "content_block_stop",
                        "index": index
                    });
                    output.push_str(&format!(
                        "event: content_block_stop\ndata: {block_stop}\n\n"
                    ));
                    self.anthropic_text_open = false;
                }
                let index = self.next_content_index;
                self.next_content_index = self.next_content_index.saturating_add(1);
                let block_start = serde_json::json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": {}
                    }
                });
                let delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": arguments
                    }
                });
                let block_stop = serde_json::json!({
                    "type": "content_block_stop",
                    "index": index
                });
                output.push_str(&format!(
                    "event: content_block_start\ndata: {block_start}\n\nevent: content_block_delta\ndata: {delta}\n\nevent: content_block_stop\ndata: {block_stop}\n\n"
                ));
                output
            }
            Dialect::OpenAi => {
                let index = self.next_tool_index;
                self.next_tool_index = self.next_tool_index.saturating_add(1);
                let chunk = serde_json::json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": arguments
                                }
                            }]
                        },
                        "finish_reason": null
                    }]
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
                let mut output = String::new();
                if self.anthropic_text_open {
                    let index = self.next_content_index.saturating_sub(1);
                    let block_stop = serde_json::json!({
                        "type": "content_block_stop",
                        "index": index
                    });
                    output.push_str(&format!(
                        "event: content_block_stop\ndata: {block_stop}\n\n"
                    ));
                    self.anthropic_text_open = false;
                }
                let stop_reason = if self.has_tool_calls {
                    "tool_use"
                } else {
                    stop.anthropic()
                };
                let message_delta = serde_json::json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                    "usage": {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens}
                });
                output.push_str(&format!(
                    "event: message_delta\ndata: {message_delta}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
                ));
                output
            }
            Dialect::OpenAi => {
                let total = usage.prompt_tokens.saturating_add(usage.completion_tokens);
                let chunk = serde_json::json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": if self.has_tool_calls { "tool_calls" } else { stop.openai() }}],
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
        out.push_str(&f.text("Hel"));
        out.push_str(&f.text("lo"));
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
        out.push_str(&f.text("Hi"));
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
        let out = f.text("a\"b\n");
        assert!(
            out.contains("\"text\":\"a\\\"b\\n\""),
            "raw quote/newline must be escaped: {out}"
        );
    }

    #[test]
    fn openai_tool_delta_is_complete_indexed_and_owns_finish_reason() {
        let mut f = Framer::new(Dialect::OpenAi, "m", 1);
        let call = AcceptedToolCall {
            id: "call_1".into(),
            name: "lookup".into(),
            arguments: serde_json::json!({"query": "hi"}),
        };
        let delta = f.tool_call(&call);
        assert!(delta.contains("\"tool_calls\":[{\"function\""));
        assert!(delta.contains("\"id\":\"call_1\""));
        assert!(delta.contains("\"index\":0"));
        assert!(delta.contains("\"arguments\":\"{\\\"query\\\":\\\"hi\\\"}\""));
        let done = f.done(Usage::default(), StopReason::MaxTokens);
        assert!(done.contains("\"finish_reason\":\"tool_calls\""));
    }

    #[test]
    fn anthropic_mixed_blocks_are_monotonic_and_closed() {
        let mut f = Framer::new(Dialect::Anthropic, "m", 1);
        let mut out = f.start();
        out.push_str(&f.text("before"));
        out.push_str(&f.tool_call(&AcceptedToolCall {
            id: "toolu_1".into(),
            name: "lookup".into(),
            arguments: serde_json::json!({"query": "hi"}),
        }));
        out.push_str(&f.text("after"));
        out.push_str(&f.done(Usage::default(), StopReason::EndTurn));
        assert!(out.contains("\"index\":0"));
        assert!(out.contains("\"index\":1"));
        assert!(out.contains("\"index\":2"));
        assert_eq!(out.matches("event: content_block_start\n").count(), 3);
        assert_eq!(out.matches("event: content_block_stop\n").count(), 3);
        assert!(out.contains("\"type\":\"tool_use\""));
        assert!(out.contains("\"type\":\"input_json_delta\""));
        assert!(out.contains("\"stop_reason\":\"tool_use\""));
    }
}
