//! Request-local adaptation from native Gemma 4 output into validated response
//! events. Parsing, dedupe, repair, and call bounds remain owned by
//! [`ToolCallParser`]; this module adds registry validation and server-minted
//! provider call IDs without knowing about HTTP framing.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::dialect::Dialect;
use crate::tool_call::{ToolCall, ToolCallEvent, ToolCallParser, ToolCallStats};
use crate::tool_schema::{ToolMode, ToolRegistry};

/// Process-wide response nonce. Combined with the per-response call index so
/// IDs remain valid when clients feed multiple assistant turns back as
/// history. Exhaustion requires 2^64 prepared responses in one process.
static NEXT_RESPONSE_NONCE: AtomicU64 = AtomicU64::new(1);
static PROCESS_NAMESPACE: OnceLock<ProcessNamespace> = OnceLock::new();

fn next_response_nonce() -> u64 {
    NEXT_RESPONSE_NONCE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("process-wide tool-call response nonce exhausted")
}

/// A randomized process namespace mixed with non-secret lifecycle material.
/// Two words retain collision resistance when independent processes start
/// their response counters from the same value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProcessNamespace([u64; 2]);

impl ProcessNamespace {
    fn initialize() -> Self {
        let state = RandomState::new();
        let process_id = process::id();
        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let word = |domain: u8| {
            let mut hasher = state.build_hasher();
            hasher.write_u8(domain);
            hasher.write_u32(process_id);
            hasher.write_u128(started_at);
            hasher.finish()
        };
        Self([word(0), word(1)])
    }
}

fn process_namespace() -> ProcessNamespace {
    *PROCESS_NAMESPACE.get_or_init(ProcessNamespace::initialize)
}

/// Pure response-local call-ID generator with an injectable process namespace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CallIdGenerator {
    namespace: ProcessNamespace,
    response_nonce: u64,
}

impl CallIdGenerator {
    fn new() -> Self {
        Self {
            namespace: process_namespace(),
            response_nonce: next_response_nonce(),
        }
    }

    fn mint(self, dialect: Dialect, call_index: usize) -> String {
        let prefix = match dialect {
            Dialect::OpenAi => "call",
            Dialect::Anthropic => "toolu",
        };
        format!(
            "{prefix}_{:016x}{:016x}_{}_{}",
            self.namespace.0[0], self.namespace.0[1], self.response_nonce, call_index
        )
    }
}

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
    id_generator: CallIdGenerator,
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
            id_generator: CallIdGenerator::new(),
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
        let registry = &self.registry;
        let mut admission = |call: &ToolCall| registry.validate_generated(call).is_ok();
        let events = parser.push_with_admission(fragment, &mut admission);
        adapt_events(
            self.dialect,
            self.id_generator,
            &mut self.next_call_index,
            &mut self.accepted_calls,
            events,
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
        let registry = &self.registry;
        let mut admission = |call: &ToolCall| registry.validate_generated(call).is_ok();
        let events = parser.finish_with_admission(&mut admission);
        adapt_events(
            self.dialect,
            self.id_generator,
            &mut self.next_call_index,
            &mut self.accepted_calls,
            events,
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
            id_generator: CallIdGenerator::new(),
            next_call_index: 0,
            accepted_calls: 0,
            finished: false,
        }
    }

    #[cfg(test)]
    fn with_identity(
        registry: Arc<ToolRegistry>,
        dialect: Dialect,
        namespace: ProcessNamespace,
        response_nonce: u64,
    ) -> Self {
        let parser = (registry.mode() == ToolMode::Auto).then(ToolCallParser::new);
        Self {
            registry,
            parser,
            dialect,
            id_generator: CallIdGenerator {
                namespace,
                response_nonce,
            },
            next_call_index: 0,
            accepted_calls: 0,
            finished: false,
        }
    }
}

fn adapt_events(
    dialect: Dialect,
    id_generator: CallIdGenerator,
    next_call_index: &mut usize,
    accepted_calls: &mut usize,
    events: Vec<ToolCallEvent>,
) -> Vec<ResponseEvent> {
    events
        .into_iter()
        .map(|event| match event {
            ToolCallEvent::Text(text) => ResponseEvent::Text(text),
            ToolCallEvent::Call(call) => {
                let id = id_generator.mint(dialect, *next_call_index + 1);
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
    use crate::history::{AnthropicHistoryInput, OpenAiHistoryInput};
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

    fn call_ids(events: &[ResponseEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                ResponseEvent::Call(call) => Some(call.id.as_str()),
                ResponseEvent::Text(_) => None,
            })
            .collect()
    }

    fn minted_id(
        dialect: Dialect,
        namespace: ProcessNamespace,
        response_nonce: u64,
        query: &str,
    ) -> String {
        let source = format!("{TOOL_CALL_OPENER}lookup{{\"query\":\"{query}\"}}{TOOL_CALL_CLOSER}");
        let mut adapter =
            ToolResponseAdapter::with_identity(registry(), dialect, namespace, response_nonce);
        let events = collect(&mut adapter, &[&source]);
        call_ids(&events)[0].to_owned()
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
            let id = call_ids(&events)[0].to_owned();
            assert!(id.starts_with("call_"), "split {split}: {id}");
            assert!(id.ends_with("_1"), "split {split}: {id}");
            assert_eq!(
                events,
                vec![
                    ResponseEvent::Text("before ".into()),
                    ResponseEvent::Call(AcceptedToolCall {
                        id,
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
        for split in 0..=raw.len() {
            let mut adapter = ToolResponseAdapter::new(registry(), Dialect::Anthropic);
            assert_eq!(
                coalesce_text(collect(&mut adapter, &[&raw[..split], &raw[split..]])),
                vec![ResponseEvent::Text(raw.clone())],
                "split {split}"
            );
            assert!(!adapter.has_calls());
            assert_eq!(adapter.stats().parsed, 1);
            assert_eq!(adapter.stats().repaired, 1);
            assert_eq!(adapter.stats().deduped, 0);
            assert_eq!(adapter.stats().call_limit_exceeded, 0);
        }

        let incomplete = raw.strip_suffix(TOOL_CALL_CLOSER).unwrap();
        let mut adapter = ToolResponseAdapter::new(registry(), Dialect::Anthropic);
        assert_eq!(
            collect(&mut adapter, &[incomplete]),
            vec![ResponseEvent::Text(incomplete.to_owned())]
        );
        assert_eq!(adapter.stats(), ToolCallStats::default());
    }

    #[test]
    fn invalid_calls_neither_dedupe_nor_consume_the_call_limit() {
        let invalid = format!("{TOOL_CALL_OPENER}lookup{{query:7}}{TOOL_CALL_CLOSER}");
        let valid = format!("{TOOL_CALL_OPENER}lookup{{query:<|\"|>ok<|\"|>}}{TOOL_CALL_CLOSER}");
        let mut adapter =
            ToolResponseAdapter::with_limits(registry(), Dialect::OpenAi, 1, 64 * 1024);
        let invalids = std::iter::repeat_n(invalid.as_str(), 12).collect::<Vec<_>>();
        let mut fragments = invalids;
        fragments.push(valid.as_str());

        let events = collect(&mut adapter, &fragments);
        assert_eq!(events.len(), 13);
        assert!(
            events[..12]
                .iter()
                .all(|event| { matches!(event, ResponseEvent::Text(text) if text == &invalid) })
        );
        assert!(
            matches!(&events[12], ResponseEvent::Call(call) if call.arguments == json!({"query": "ok"}))
        );
        assert_eq!(adapter.stats().parsed, 13);
        assert_eq!(adapter.stats().wellformed, 13);
        assert_eq!(adapter.stats().repaired, 0);
        assert_eq!(adapter.stats().deduped, 0);
        assert_eq!(adapter.stats().call_limit_exceeded, 0);
    }

    #[test]
    fn duplicates_and_call_limit_remain_parser_owned() {
        let a = format!("{TOOL_CALL_OPENER}lookup{{\"query\":\"a\"}}{TOOL_CALL_CLOSER}");
        let b = format!("{TOOL_CALL_OPENER}lookup{{\"query\":\"b\"}}{TOOL_CALL_CLOSER}");
        let mut adapter =
            ToolResponseAdapter::with_limits(registry(), Dialect::Anthropic, 1, 64 * 1024);
        let events = collect(&mut adapter, &[&a, &a, &b]);
        assert!(
            matches!(&events[0], ResponseEvent::Call(call) if call.id.starts_with("toolu_") && call.id.ends_with("_1"))
        );
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
    fn separate_responses_never_reuse_ids_and_indices_stay_contiguous() {
        let first_call = format!("{TOOL_CALL_OPENER}lookup{{\"query\":\"a\"}}{TOOL_CALL_CLOSER}");
        let second_call = format!("{TOOL_CALL_OPENER}lookup{{\"query\":\"b\"}}{TOOL_CALL_CLOSER}");
        let mut first = ToolResponseAdapter::new(registry(), Dialect::OpenAi);
        let first_events = collect(&mut first, &[&first_call, &second_call]);
        let first_ids = call_ids(&first_events);
        assert_eq!(first_ids.len(), 2);
        let response_prefix = first_ids[0].strip_suffix("_1").unwrap();
        assert_eq!(first_ids[1], format!("{response_prefix}_2"));

        let mut second = ToolResponseAdapter::new(registry(), Dialect::OpenAi);
        let second_events = collect(&mut second, &[&first_call]);
        let second_ids = call_ids(&second_events);
        assert_eq!(second_ids.len(), 1);
        assert!(second_ids[0].ends_with("_1"));
        assert_ne!(first_ids[0], second_ids[0]);
    }

    #[test]
    fn independent_namespaces_do_not_collide_and_both_replay_in_history() {
        let namespace_a = ProcessNamespace([0x1111, 0xaaaa]);
        let namespace_b = ProcessNamespace([0x2222, 0xbbbb]);
        let first_call = format!("{TOOL_CALL_OPENER}lookup{{\"query\":\"a\"}}{TOOL_CALL_CLOSER}");
        let second_call = format!("{TOOL_CALL_OPENER}lookup{{\"query\":\"b\"}}{TOOL_CALL_CLOSER}");

        for (dialect, prefix) in [(Dialect::OpenAi, "call_"), (Dialect::Anthropic, "toolu_")] {
            let mut first = ToolResponseAdapter::with_identity(registry(), dialect, namespace_a, 1);
            let first_events = collect(&mut first, &[&first_call, &second_call]);
            let first_ids = call_ids(&first_events);
            assert_eq!(first_ids.len(), 2);
            assert!(first_ids[0].starts_with(prefix));
            assert!(first_ids[0].len() < 256);
            let response_prefix = first_ids[0].strip_suffix("_1").unwrap();
            assert_eq!(first_ids[1], format!("{response_prefix}_2"));

            let mut restarted =
                ToolResponseAdapter::with_identity(registry(), dialect, namespace_b, 1);
            let restarted_events = collect(&mut restarted, &[&first_call]);
            let restarted_ids = call_ids(&restarted_events);
            assert_eq!(restarted_ids.len(), 1);
            assert!(restarted_ids[0].starts_with(prefix));
            assert!(restarted_ids[0].ends_with("_1"));
            assert_ne!(first_ids[0], restarted_ids[0]);
        }

        let openai_a = minted_id(Dialect::OpenAi, namespace_a, 1, "first");
        let openai_b = minted_id(Dialect::OpenAi, namespace_b, 1, "second");
        let openai_history = json!([
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": openai_a,
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"query\":\"first\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": openai_a, "content": "ok"},
            {"role": "user", "content": "second"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": openai_b,
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"query\":\"second\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": openai_b, "content": "ok"},
        ]);
        let registry = registry();
        assert!(
            registry
                .normalize_openai_history(OpenAiHistoryInput {
                    messages: &openai_history,
                })
                .is_ok()
        );

        let anthropic_a = minted_id(Dialect::Anthropic, namespace_a, 1, "first");
        let anthropic_b = minted_id(Dialect::Anthropic, namespace_b, 1, "second");
        let anthropic_history = json!([
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": anthropic_a,
                    "name": "lookup",
                    "input": {"query": "first"},
                }],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": anthropic_a, "content": "ok"}],
            },
            {"role": "user", "content": "second"},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": anthropic_b,
                    "name": "lookup",
                    "input": {"query": "second"},
                }],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": anthropic_b, "content": "ok"}],
            },
        ]);
        assert!(
            registry
                .normalize_anthropic_history(AnthropicHistoryInput {
                    system: None,
                    messages: &anthropic_history,
                })
                .is_ok()
        );
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
