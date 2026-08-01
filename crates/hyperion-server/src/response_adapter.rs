//! Request-local adaptation from native Gemma 4 output into validated response
//! events. Parsing, dedupe, repair, and call bounds remain owned by
//! [`ToolCallParser`]; this module adds registry validation and server-minted
//! provider call IDs without knowing about HTTP framing.

use std::sync::Arc;

use serde_json::Value;

use crate::dialect::Dialect;
use crate::tool_call::{ToolCallEvent, ToolCallParser, ToolCallStats};
use crate::tool_schema::{ToolMode, ToolRegistry};

/// One validated, provider-independent call ready for response framing.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AcceptedToolCall {
    /// Server-minted provider-safe correlation ID.
    pub id: String,
    /// Declared function name emitted by the model.
    pub name: String,
    /// Registry-validated object arguments.
    pub arguments: Value,
}

/// Ordered assistant response material after parsing and validation.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ResponseEvent {
    /// Exact assistant text, including raw invalid call blocks.
    Text(String),
    /// A unique, bounded, registry-valid call.
    Call(AcceptedToolCall),
}

/// One per-request native-output adapter.
pub(crate) struct ToolResponseAdapter {
    registry: Arc<ToolRegistry>,
    parser: Option<ToolCallParser>,
    dialect: Dialect,
    next_call_index: usize,
    accepted_calls: usize,
    finished: bool,
}

impl ToolResponseAdapter {
    /// Create an adapter. Explicit `none` bypasses parsing so ordinary decoded
    /// text retains the pre-tool-response behavior.
    #[must_use]
    pub(crate) fn new(registry: Arc<ToolRegistry>, dialect: Dialect) -> Self {
        let parser = (registry.mode() == ToolMode::Auto).then(ToolCallParser::new);
        Self {
            registry,
            parser,
            dialect,
            next_call_index: 0,
            accepted_calls: 0,
            finished: false,
        }
    }

    /// Feed one sequential decoded fragment through the request parser.
    pub(crate) fn push(&mut self, fragment: &str) -> Vec<ResponseEvent> {
        debug_assert!(!self.finished, "adapter push after finish");
        let Some(parser) = self.parser.as_mut() else {
            return if fragment.is_empty() {
                Vec::new()
            } else {
                vec![ResponseEvent::Text(fragment.to_owned())]
            };
        };
        adapt_events(
            &self.registry,
            self.dialect,
            &mut self.next_call_index,
            &mut self.accepted_calls,
            parser.push(fragment),
        )
    }

    /// Finish successful EOF exactly once, releasing incomplete candidates as
    /// exact text. Error paths intentionally do not call this method.
    pub(crate) fn finish(&mut self) -> Vec<ResponseEvent> {
        debug_assert!(!self.finished, "adapter finish called twice");
        self.finished = true;
        let Some(parser) = self.parser.as_mut() else {
            return Vec::new();
        };
        adapt_events(
            &self.registry,
            self.dialect,
            &mut self.next_call_index,
            &mut self.accepted_calls,
            parser.finish(),
        )
    }

    /// Snapshot the parser lifetime counters for one-time control aggregation.
    #[must_use]
    pub(crate) fn stats(&self) -> ToolCallStats {
        self.parser
            .as_ref()
            .map_or_else(ToolCallStats::default, ToolCallParser::stats)
    }

    /// Whether at least one validated unique call was surfaced.
    #[must_use]
    pub(crate) const fn has_calls(&self) -> bool {
        self.accepted_calls != 0
    }

    #[cfg(test)]
    fn with_limits(
        registry: Arc<ToolRegistry>,
        dialect: Dialect,
        max_tool_calls: usize,
        max_candidate_bytes: usize,
    ) -> Self {
        let parser = (registry.mode() == ToolMode::Auto)
            .then(|| ToolCallParser::with_limits(max_tool_calls, max_candidate_bytes));
        Self {
            registry,
            parser,
            dialect,
            next_call_index: 0,
            accepted_calls: 0,
            finished: false,
        }
    }
}

fn adapt_events(
    registry: &ToolRegistry,
    dialect: Dialect,
    next_call_index: &mut usize,
    accepted_calls: &mut usize,
    events: Vec<ToolCallEvent>,
) -> Vec<ResponseEvent> {
    events
        .into_iter()
        .map(|event| match event {
            ToolCallEvent::Text(text) => ResponseEvent::Text(text),
            ToolCallEvent::Call(call) => {
                if registry.validate_generated(&call).is_err() {
                    return ResponseEvent::Text(call.raw);
                }
                let id = match dialect {
                    Dialect::OpenAi => format!("call_{}", *next_call_index + 1),
                    Dialect::Anthropic => format!("toolu_{}", *next_call_index + 1),
                };
                *next_call_index = next_call_index.saturating_add(1);
                *accepted_calls = accepted_calls.saturating_add(1);
                ResponseEvent::Call(AcceptedToolCall {
                    id,
                    name: call.name,
                    arguments: call.arguments,
                })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_call::{TOOL_CALL_CLOSER, TOOL_CALL_OPENER};
    use crate::tool_schema::{OpenAiToolsInput, ToolRegistry};
    use serde_json::json;

    fn registry() -> Arc<ToolRegistry> {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": false
                }
            }
        }]);
        Arc::new(
            ToolRegistry::from_openai(OpenAiToolsInput {
                tools: Some(&tools),
                tool_choice: Some(&json!("auto")),
                parallel_tool_calls: None,
            })
            .unwrap(),
        )
    }

    fn collect(adapter: &mut ToolResponseAdapter, fragments: &[&str]) -> Vec<ResponseEvent> {
        let mut events = Vec::new();
        for fragment in fragments {
            events.extend(adapter.push(fragment));
        }
        events.extend(adapter.finish());
        events
    }

    fn coalesce_text(events: Vec<ResponseEvent>) -> Vec<ResponseEvent> {
        let mut output = Vec::new();
        for event in events {
            match event {
                ResponseEvent::Text(text) => {
                    if let Some(ResponseEvent::Text(previous)) = output.last_mut() {
                        previous.push_str(&text);
                    } else {
                        output.push(ResponseEvent::Text(text));
                    }
                }
                call => output.push(call),
            }
        }
        output
    }

    #[test]
    fn arbitrary_fragmentation_preserves_order_and_mints_stable_ids() {
        let source =
            format!("before {TOOL_CALL_OPENER}lookup{{\"query\":\"hi\"}}{TOOL_CALL_CLOSER} after");
        for split in 0..=source.len() {
            if !source.is_char_boundary(split) {
                continue;
            }
            let mut adapter = ToolResponseAdapter::new(registry(), Dialect::OpenAi);
            let events =
                coalesce_text(collect(&mut adapter, &[&source[..split], &source[split..]]));
            assert_eq!(
                events,
                vec![
                    ResponseEvent::Text("before ".into()),
                    ResponseEvent::Call(AcceptedToolCall {
                        id: "call_1".into(),
                        name: "lookup".into(),
                        arguments: json!({"query": "hi"}),
                    }),
                    ResponseEvent::Text(" after".into()),
                ],
                "split {split}"
            );
        }
    }

    #[test]
    fn validation_failure_round_trips_exact_raw_text() {
        let raw = format!("{TOOL_CALL_OPENER}lookup{{\"query\":7}}{TOOL_CALL_CLOSER}");
        let mut adapter = ToolResponseAdapter::new(registry(), Dialect::Anthropic);
        assert_eq!(
            collect(&mut adapter, &[&raw]),
            vec![ResponseEvent::Text(raw)]
        );
        assert!(!adapter.has_calls());
        assert_eq!(adapter.stats().parsed, 1);
    }

    #[test]
    fn duplicates_and_call_limit_remain_parser_owned() {
        let a = format!("{TOOL_CALL_OPENER}lookup{{\"query\":\"a\"}}{TOOL_CALL_CLOSER}");
        let b = format!("{TOOL_CALL_OPENER}lookup{{\"query\":\"b\"}}{TOOL_CALL_CLOSER}");
        let mut adapter =
            ToolResponseAdapter::with_limits(registry(), Dialect::Anthropic, 1, 64 * 1024);
        let events = collect(&mut adapter, &[&a, &a, &b]);
        assert!(matches!(&events[0], ResponseEvent::Call(call) if call.id == "toolu_1"));
        assert_eq!(events[1], ResponseEvent::Text(b));
        assert_eq!(adapter.stats().parsed, 3);
        assert_eq!(adapter.stats().deduped, 1);
        assert_eq!(adapter.stats().call_limit_exceeded, 1);
    }

    #[test]
    fn finite_repair_telemetry_is_retained() {
        let repaired = "<|tool_call|>call:lookup{query:<|\"|>hi<|\"|>}<tool_call|>";
        let mut adapter = ToolResponseAdapter::new(registry(), Dialect::OpenAi);
        let events = collect(&mut adapter, &[repaired]);
        assert!(matches!(&events[0], ResponseEvent::Call(call) if call.name == "lookup"));
        assert_eq!(adapter.stats().parsed, 1);
        assert_eq!(adapter.stats().wellformed, 0);
        assert_eq!(adapter.stats().repaired, 1);
    }

    #[test]
    fn explicit_none_is_byte_exact_text_passthrough() {
        let no_tools = Arc::new(
            ToolRegistry::from_openai(OpenAiToolsInput {
                tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
            })
            .unwrap(),
        );
        let mut adapter = ToolResponseAdapter::new(no_tools, Dialect::OpenAi);
        let events = collect(&mut adapter, &["a", "β", "c"]);
        assert_eq!(
            events,
            vec![
                ResponseEvent::Text("a".into()),
                ResponseEvent::Text("β".into()),
                ResponseEvent::Text("c".into())
            ]
        );
        assert_eq!(adapter.stats(), ToolCallStats::default());
    }
}
