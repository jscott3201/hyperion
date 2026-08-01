//! Bounded, transport-free provider tool-history normalization.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::io::{self, Write};

use hyperion_tokenizer::renderer::ChatMessage;
use serde_json::{Map, Value};

use crate::tool_call::decode_unique_json;
use crate::tool_schema::{
    MAX_ARGUMENT_BYTES, MAX_ARGUMENT_NODES, MAX_DEPTH, ToolRegistry, reserved_control,
    validate_tool_name,
};

const MAX_SOURCE_MESSAGES: usize = 4_096;
const MAX_CALLS_PER_TURN: usize = 8;
const MAX_EXTERNAL_ID_BYTES: usize = 256;
const MAX_RAW_ARGUMENT_BYTES: usize = 64 * 1024;
const MAX_AGGREGATE_ARGUMENT_BYTES: usize = 512 * 1024;
const MAX_AGGREGATE_TEXT_BYTES: usize = 32 * 1024 * 1024;
const MAX_AGGREGATE_TEXT_BLOCKS: usize = 4_096;

/// Borrowed OpenAI-compatible history input.
pub struct OpenAiHistoryInput<'a> {
    /// Provider `messages` value.
    pub messages: &'a Value,
}

/// Borrowed Anthropic-compatible history input.
pub struct AnthropicHistoryInput<'a> {
    /// Provider top-level `system` value, when supplied.
    pub system: Option<&'a Value>,
    /// Provider `messages` value.
    pub messages: &'a Value,
}

/// A deterministic provider-history error that never includes client payloads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolHistoryError(String);

impl fmt::Display for ToolHistoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ToolHistoryError {}

#[derive(Clone, Copy)]
struct HistoryLimits {
    source_messages: usize,
    calls_per_turn: usize,
    external_id_bytes: usize,
    raw_argument_bytes: usize,
    compact_argument_bytes: usize,
    aggregate_argument_bytes: usize,
    aggregate_text_bytes: usize,
    aggregate_text_blocks: usize,
}

const PRODUCTION_LIMITS: HistoryLimits = HistoryLimits {
    source_messages: MAX_SOURCE_MESSAGES,
    calls_per_turn: MAX_CALLS_PER_TURN,
    external_id_bytes: MAX_EXTERNAL_ID_BYTES,
    raw_argument_bytes: MAX_RAW_ARGUMENT_BYTES,
    compact_argument_bytes: MAX_ARGUMENT_BYTES,
    aggregate_argument_bytes: MAX_AGGREGATE_ARGUMENT_BYTES,
    aggregate_text_bytes: MAX_AGGREGATE_TEXT_BYTES,
    aggregate_text_blocks: MAX_AGGREGATE_TEXT_BLOCKS,
};

#[derive(Clone, Copy)]
enum Provider {
    OpenAi,
    Anthropic,
}

impl Provider {
    const fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "OpenAI",
            Self::Anthropic => "Anthropic",
        }
    }
}

struct HistoricalCall {
    id: String,
    name: String,
    arguments: Value,
}

struct Normalizer<'a> {
    registry: &'a ToolRegistry,
    provider: Provider,
    limits: HistoryLimits,
    text_bytes: usize,
    text_blocks: usize,
    argument_bytes: usize,
    seen_call_ids: HashSet<String>,
}

impl ToolRegistry {
    /// Normalize a strict, text-only OpenAI history into renderer messages.
    pub fn normalize_openai_history(
        &self,
        input: OpenAiHistoryInput<'_>,
    ) -> Result<Vec<ChatMessage>, ToolHistoryError> {
        Normalizer::new(self, Provider::OpenAi, PRODUCTION_LIMITS).normalize_openai(input)
    }

    /// Normalize a strict, text-only Anthropic history into renderer messages.
    pub fn normalize_anthropic_history(
        &self,
        input: AnthropicHistoryInput<'_>,
    ) -> Result<Vec<ChatMessage>, ToolHistoryError> {
        Normalizer::new(self, Provider::Anthropic, PRODUCTION_LIMITS).normalize_anthropic(input)
    }

    #[cfg(test)]
    fn normalize_openai_history_with_limits(
        &self,
        input: OpenAiHistoryInput<'_>,
        limits: HistoryLimits,
    ) -> Result<Vec<ChatMessage>, ToolHistoryError> {
        Normalizer::new(self, Provider::OpenAi, limits).normalize_openai(input)
    }

    #[cfg(test)]
    fn normalize_anthropic_history_with_limits(
        &self,
        input: AnthropicHistoryInput<'_>,
        limits: HistoryLimits,
    ) -> Result<Vec<ChatMessage>, ToolHistoryError> {
        Normalizer::new(self, Provider::Anthropic, limits).normalize_anthropic(input)
    }
}

impl<'a> Normalizer<'a> {
    fn new(registry: &'a ToolRegistry, provider: Provider, limits: HistoryLimits) -> Self {
        Self {
            registry,
            provider,
            limits,
            text_bytes: 0,
            text_blocks: 0,
            argument_bytes: 0,
            seen_call_ids: HashSet::new(),
        }
    }

    fn normalize_openai(
        mut self,
        input: OpenAiHistoryInput<'_>,
    ) -> Result<Vec<ChatMessage>, ToolHistoryError> {
        let messages = self.message_array(input.messages, "$.messages")?;
        let mut output = Vec::with_capacity(messages.len());
        let mut index = 0;

        while index < messages.len() {
            let path = format!("$.messages[{index}]");
            let message = self.object(&messages[index], &path)?;
            let role = self.required_string(message, "role", &path)?;
            match role {
                "system" | "developer" => {
                    if index != 0 {
                        return Err(self.error(
                            &format!("{path}.role"),
                            "system/developer is only allowed as the first message",
                        ));
                    }
                    self.ensure_fields(message, &["role", "content"], &path)?;
                    let content = self.required_text(message, "content", &path, false)?;
                    output.push(canonical_message("system", Some(content)));
                    index += 1;
                }
                "user" => {
                    self.ensure_fields(message, &["role", "content"], &path)?;
                    let content = self.required_text(message, "content", &path, false)?;
                    output.push(canonical_message("user", Some(content)));
                    index += 1;
                }
                "assistant" if message.contains_key("tool_calls") => {
                    self.ensure_fields(message, &["role", "content", "tool_calls"], &path)?;
                    let content = match message.get("content") {
                        None | Some(Value::Null) => None,
                        Some(value) => {
                            Some(self.text_content(value, &format!("{path}.content"), false)?)
                        }
                    };
                    let calls = self.openai_calls(
                        message.get("tool_calls").expect("contains_key was checked"),
                        &format!("{path}.tool_calls"),
                    )?;
                    let result_start = index + 1;
                    let results = self.openai_results(messages, result_start, &calls)?;
                    let consumed_results = calls.len();
                    let (assistant, tool_rows) = canonical_batch(content, calls, results);
                    output.push(assistant);
                    output.extend(tool_rows);
                    index = result_start + consumed_results;
                }
                "assistant" => {
                    self.ensure_fields(message, &["role", "content"], &path)?;
                    let content = self.required_text(message, "content", &path, false)?;
                    output.push(canonical_message("assistant", Some(content)));
                    index += 1;
                }
                "tool" => {
                    return Err(self.error(
                        &format!("{path}.role"),
                        "tool result has no pending assistant tool-call batch",
                    ));
                }
                _ => {
                    return Err(self.error(&format!("{path}.role"), "unsupported message role"));
                }
            }
        }
        Ok(output)
    }

    fn normalize_anthropic(
        mut self,
        input: AnthropicHistoryInput<'_>,
    ) -> Result<Vec<ChatMessage>, ToolHistoryError> {
        let messages = self.message_array(input.messages, "$.messages")?;
        let mut output = Vec::with_capacity(messages.len() + usize::from(input.system.is_some()));
        if let Some(system) = input.system
            && !system.is_null()
        {
            let content = self.text_content(system, "$.system", false)?;
            output.push(canonical_message("system", Some(content)));
        }

        let mut index = 0;
        while index < messages.len() {
            let path = format!("$.messages[{index}]");
            let message = self.object(&messages[index], &path)?;
            self.ensure_fields(message, &["role", "content"], &path)?;
            let role = self.required_string(message, "role", &path)?;
            let content = message
                .get("content")
                .ok_or_else(|| self.error(&format!("{path}.content"), "is required"))?;

            match role {
                "assistant"
                    if self.is_anthropic_tool_use(content, &format!("{path}.content"))? =>
                {
                    let calls = self.anthropic_calls(content, &format!("{path}.content"))?;
                    let result_index = index + 1;
                    let results = self.anthropic_results(messages, result_index, &calls)?;
                    let (assistant, tool_rows) = canonical_batch(None, calls, results);
                    output.push(assistant);
                    output.extend(tool_rows);
                    index = result_index + 1;
                }
                "assistant" => {
                    let content = self.text_content(content, &format!("{path}.content"), true)?;
                    output.push(canonical_message("assistant", Some(content)));
                    index += 1;
                }
                "user" => {
                    if self.anthropic_has_block_type(
                        content,
                        "tool_result",
                        &format!("{path}.content"),
                    )? {
                        return Err(self.error(
                            &format!("{path}.content"),
                            "tool_result has no pending assistant tool-use batch",
                        ));
                    }
                    let content = self.text_content(content, &format!("{path}.content"), true)?;
                    output.push(canonical_message("user", Some(content)));
                    index += 1;
                }
                _ => {
                    return Err(self.error(&format!("{path}.role"), "unsupported message role"));
                }
            }
        }
        Ok(output)
    }

    fn message_array<'value>(
        &self,
        value: &'value Value,
        path: &str,
    ) -> Result<&'value [Value], ToolHistoryError> {
        let Value::Array(messages) = value else {
            return Err(self.error(path, "must be an array"));
        };
        if messages.len() > self.limits.source_messages {
            return Err(self.error(
                path,
                format_args!(
                    "exceeds the maximum of {} source messages",
                    self.limits.source_messages
                ),
            ));
        }
        Ok(messages)
    }

    fn openai_calls(
        &mut self,
        value: &Value,
        path: &str,
    ) -> Result<Vec<HistoricalCall>, ToolHistoryError> {
        let Value::Array(values) = value else {
            return Err(self.error(path, "must be a nonempty array"));
        };
        if values.is_empty() {
            return Err(self.error(path, "must be a nonempty array"));
        }
        if values.len() > self.limits.calls_per_turn {
            return Err(self.error(
                path,
                format_args!(
                    "exceeds the maximum of {} calls per turn",
                    self.limits.calls_per_turn
                ),
            ));
        }

        let mut calls = Vec::with_capacity(values.len());
        for (index, value) in values.iter().enumerate() {
            let call_path = format!("{path}[{index}]");
            let call = self.object(value, &call_path)?;
            self.ensure_fields(call, &["id", "type", "function", "index"], &call_path)?;
            let id = self.required_string(call, "id", &call_path)?;
            self.register_call_id(id, &format!("{call_path}.id"))?;
            self.require_exact_string(call, "type", "function", &call_path)?;
            if let Some(stream_index) = call.get("index")
                && stream_index.as_u64().is_none()
            {
                return Err(self.error(
                    &format!("{call_path}.index"),
                    "must be a nonnegative integer",
                ));
            }

            let function_path = format!("{call_path}.function");
            let function = self.object(
                call.get("function")
                    .ok_or_else(|| self.error(&function_path, "is required"))?,
                &function_path,
            )?;
            self.ensure_fields(function, &["name", "arguments"], &function_path)?;
            let name = self.required_string(function, "name", &function_path)?;
            let name_path = format!("{function_path}.name");
            validate_tool_name(name, &name_path)
                .map_err(|_| self.error(&name_path, "must match the declared tool-name grammar"))?;
            let arguments_path = format!("{function_path}.arguments");
            let raw_arguments = self.required_string(function, "arguments", &function_path)?;
            if raw_arguments.len() > self.limits.raw_argument_bytes {
                return Err(self.error(
                    &arguments_path,
                    format_args!(
                        "exceeds the maximum of {} raw bytes",
                        self.limits.raw_argument_bytes
                    ),
                ));
            }
            let arguments = decode_unique_json(raw_arguments).map_err(|_| {
                self.error(
                    &arguments_path,
                    "must be valid JSON without duplicate object keys",
                )
            })?;
            if !arguments.is_object() {
                return Err(self.error(&arguments_path, "must decode to an object"));
            }
            self.validate_arguments(name, &arguments, &arguments_path)?;
            calls.push(HistoricalCall {
                id: id.to_owned(),
                name: name.to_owned(),
                arguments,
            });
        }
        Ok(calls)
    }

    fn openai_results(
        &mut self,
        messages: &[Value],
        start: usize,
        calls: &[HistoricalCall],
    ) -> Result<Vec<String>, ToolHistoryError> {
        let Some(end) = start.checked_add(calls.len()) else {
            return Err(self.error("$.messages", "tool-result position overflow"));
        };
        if end > messages.len() {
            return Err(self.error(
                "$.messages",
                "ends with an unresolved assistant tool-call batch",
            ));
        }

        let mut ordered = vec![None; calls.len()];
        for (offset, value) in messages[start..end].iter().enumerate() {
            let message_index = start + offset;
            let path = format!("$.messages[{message_index}]");
            let message = self.object(value, &path)?;
            let role = self.required_string(message, "role", &path)?;
            if role != "tool" {
                return Err(self.error(
                    &format!("{path}.role"),
                    "interleaves a pending assistant tool-call batch",
                ));
            }
            self.ensure_fields(message, &["role", "content", "tool_call_id", "name"], &path)?;
            let id = self.required_string(message, "tool_call_id", &path)?;
            self.validate_external_id(id, &format!("{path}.tool_call_id"))?;
            let Some(position) = calls.iter().position(|call| call.id == id) else {
                return Err(self.error(
                    &format!("{path}.tool_call_id"),
                    "does not match the pending assistant tool-call batch",
                ));
            };
            if ordered[position].is_some() {
                return Err(self.error(
                    &format!("{path}.tool_call_id"),
                    "duplicates a result in the pending assistant tool-call batch",
                ));
            }
            if let Some(name) = message.get("name") {
                let Value::String(name) = name else {
                    return Err(self.error(&format!("{path}.name"), "must be a string"));
                };
                if name != &calls[position].name {
                    return Err(self.error(
                        &format!("{path}.name"),
                        "does not match the declared call name",
                    ));
                }
            }
            let content = message
                .get("content")
                .ok_or_else(|| self.error(&format!("{path}.content"), "is required"))?;
            ordered[position] =
                Some(self.text_content(content, &format!("{path}.content"), false)?);
        }
        self.complete_results(ordered, "$.messages")
    }

    fn is_anthropic_tool_use(&self, content: &Value, path: &str) -> Result<bool, ToolHistoryError> {
        let Value::Array(blocks) = content else {
            return Ok(false);
        };
        self.precheck_anthropic_content_array(blocks, path)?;
        let tool_uses = blocks
            .iter()
            .filter(|block| {
                block
                    .as_object()
                    .and_then(|object| object.get("type"))
                    .and_then(Value::as_str)
                    == Some("tool_use")
            })
            .count();
        if tool_uses > 0 && tool_uses != blocks.len() {
            return Err(self.error(path, "must not mix tool_use and non-tool_use blocks"));
        }
        Ok(tool_uses > 0)
    }

    fn anthropic_has_block_type(
        &self,
        content: &Value,
        expected: &str,
        path: &str,
    ) -> Result<bool, ToolHistoryError> {
        let Value::Array(blocks) = content else {
            return Ok(false);
        };
        self.precheck_anthropic_content_array(blocks, path)?;
        Ok(blocks.iter().any(|block| {
            block
                .as_object()
                .and_then(|object| object.get("type"))
                .and_then(Value::as_str)
                == Some(expected)
        }))
    }

    fn precheck_anthropic_content_array(
        &self,
        blocks: &[Value],
        path: &str,
    ) -> Result<(), ToolHistoryError> {
        let remaining_text_blocks = self
            .limits
            .aggregate_text_blocks
            .checked_sub(self.text_blocks)
            .ok_or_else(|| self.error(path, "text-block accounting underflow"))?;
        let first_is_tool_block = blocks
            .first()
            .and_then(Value::as_object)
            .and_then(|block| block.get("type"))
            .and_then(Value::as_str)
            .is_some_and(|kind| matches!(kind, "tool_use" | "tool_result"));
        let scan_limit = if first_is_tool_block {
            self.limits.calls_per_turn
        } else {
            remaining_text_blocks
        };
        if blocks.len() > scan_limit {
            return Err(self.error(
                path,
                "exceeds the remaining text-block or tool-use scan budget",
            ));
        }
        Ok(())
    }

    fn anthropic_calls(
        &mut self,
        content: &Value,
        path: &str,
    ) -> Result<Vec<HistoricalCall>, ToolHistoryError> {
        let Value::Array(blocks) = content else {
            return Err(self.error(path, "must be a nonempty tool_use array"));
        };
        if blocks.is_empty() {
            return Err(self.error(path, "must be a nonempty tool_use array"));
        }
        if blocks.len() > self.limits.calls_per_turn {
            return Err(self.error(
                path,
                format_args!(
                    "exceeds the maximum of {} calls per turn",
                    self.limits.calls_per_turn
                ),
            ));
        }

        let mut calls = Vec::with_capacity(blocks.len());
        for (index, value) in blocks.iter().enumerate() {
            let block_path = format!("{path}[{index}]");
            let block = self.object(value, &block_path)?;
            self.ensure_fields(block, &["type", "id", "name", "input"], &block_path)?;
            self.require_exact_string(block, "type", "tool_use", &block_path)?;
            let id = self.required_string(block, "id", &block_path)?;
            self.register_call_id(id, &format!("{block_path}.id"))?;
            let name = self.required_string(block, "name", &block_path)?;
            let name_path = format!("{block_path}.name");
            validate_tool_name(name, &name_path)
                .map_err(|_| self.error(&name_path, "must match the declared tool-name grammar"))?;
            let input_path = format!("{block_path}.input");
            let arguments = block
                .get("input")
                .ok_or_else(|| self.error(&input_path, "is required"))?;
            if !arguments.is_object() {
                return Err(self.error(&input_path, "must be an object"));
            }
            self.validate_arguments(name, arguments, &input_path)?;
            calls.push(HistoricalCall {
                id: id.to_owned(),
                name: name.to_owned(),
                arguments: arguments.clone(),
            });
        }
        Ok(calls)
    }

    fn anthropic_results(
        &mut self,
        messages: &[Value],
        index: usize,
        calls: &[HistoricalCall],
    ) -> Result<Vec<String>, ToolHistoryError> {
        let Some(value) = messages.get(index) else {
            return Err(self.error(
                "$.messages",
                "ends with an unresolved assistant tool-use batch",
            ));
        };
        let path = format!("$.messages[{index}]");
        let message = self.object(value, &path)?;
        let role = self.required_string(message, "role", &path)?;
        if role != "user" {
            return Err(self.error(
                &format!("{path}.role"),
                "interleaves a pending assistant tool-use batch",
            ));
        }
        self.ensure_fields(message, &["role", "content"], &path)?;
        let content_path = format!("{path}.content");
        let content = message
            .get("content")
            .ok_or_else(|| self.error(&content_path, "is required"))?;
        let Value::Array(blocks) = content else {
            return Err(self.error(&content_path, "must be a nonempty tool_result array"));
        };
        if blocks.is_empty() || blocks.len() != calls.len() {
            return Err(self.error(
                &content_path,
                "must contain exactly one tool_result per pending call",
            ));
        }

        let mut ordered = vec![None; calls.len()];
        for (block_index, value) in blocks.iter().enumerate() {
            let block_path = format!("{content_path}[{block_index}]");
            let block = self.object(value, &block_path)?;
            self.ensure_fields(
                block,
                &["type", "tool_use_id", "content", "is_error"],
                &block_path,
            )?;
            self.require_exact_string(block, "type", "tool_result", &block_path)?;
            let id = self.required_string(block, "tool_use_id", &block_path)?;
            self.validate_external_id(id, &format!("{block_path}.tool_use_id"))?;
            let Some(position) = calls.iter().position(|call| call.id == id) else {
                return Err(self.error(
                    &format!("{block_path}.tool_use_id"),
                    "does not match the pending assistant tool-use batch",
                ));
            };
            if ordered[position].is_some() {
                return Err(self.error(
                    &format!("{block_path}.tool_use_id"),
                    "duplicates a result in the pending assistant tool-use batch",
                ));
            }
            match block.get("is_error") {
                None | Some(Value::Bool(false)) => {}
                Some(Value::Bool(true)) => {
                    return Err(
                        self.error(&format!("{block_path}.is_error"), "true is unsupported")
                    );
                }
                Some(_) => {
                    return Err(self.error(&format!("{block_path}.is_error"), "must be a boolean"));
                }
            }
            let result = match block.get("content") {
                None => String::new(),
                Some(value) => self.text_content(value, &format!("{block_path}.content"), false)?,
            };
            ordered[position] = Some(result);
        }
        self.complete_results(ordered, &content_path)
    }

    fn complete_results(
        &self,
        values: Vec<Option<String>>,
        path: &str,
    ) -> Result<Vec<String>, ToolHistoryError> {
        values
            .into_iter()
            .map(|value| {
                value.ok_or_else(|| self.error(path, "is missing a result for a pending tool call"))
            })
            .collect()
    }

    fn required_text(
        &mut self,
        object: &Map<String, Value>,
        field: &str,
        path: &str,
        require_nonempty_blocks: bool,
    ) -> Result<String, ToolHistoryError> {
        let field_path = format!("{path}.{field}");
        let value = object
            .get(field)
            .ok_or_else(|| self.error(&field_path, "is required"))?;
        self.text_content(value, &field_path, require_nonempty_blocks)
    }

    fn text_content(
        &mut self,
        value: &Value,
        path: &str,
        require_nonempty_blocks: bool,
    ) -> Result<String, ToolHistoryError> {
        match value {
            Value::String(text) => {
                let next_text_bytes = self.checked_text_bytes(text.len(), path)?;
                self.inspect_text(text, path)?;
                self.text_bytes = next_text_bytes;
                Ok(text.clone())
            }
            Value::Array(blocks) => {
                let next_text_blocks = self.checked_text_blocks(blocks.len(), path)?;
                if require_nonempty_blocks && blocks.is_empty() {
                    return Err(self.error(path, "must be a string or nonempty text-block array"));
                }
                let mut total = 0usize;
                for (index, value) in blocks.iter().enumerate() {
                    let block_path = format!("{path}[{index}]");
                    let block = self.object(value, &block_path)?;
                    self.ensure_fields(block, &["type", "text"], &block_path)?;
                    self.require_exact_string(block, "type", "text", &block_path)?;
                    let text = self.required_string(block, "text", &block_path)?;
                    total = total
                        .checked_add(text.len())
                        .ok_or_else(|| self.error(path, "text byte accounting overflow"))?;
                    self.checked_text_bytes(total, path)?;
                }
                let next_text_bytes = self.checked_text_bytes(total, path)?;
                let mut normalized = String::with_capacity(total);
                for (index, value) in blocks.iter().enumerate() {
                    let text = value
                        .as_object()
                        .and_then(|block| block.get("text"))
                        .and_then(Value::as_str)
                        .expect("the first pass validated every canonical text block");
                    self.inspect_text(text, &format!("{path}[{index}].text"))?;
                    normalized.push_str(text);
                }
                self.text_bytes = next_text_bytes;
                self.text_blocks = next_text_blocks;
                Ok(normalized)
            }
            _ => Err(self.error(path, "must be a string or canonical text-block array")),
        }
    }

    fn inspect_text(&self, text: &str, path: &str) -> Result<(), ToolHistoryError> {
        if reserved_control(text).is_some() {
            return Err(self.error(path, "contains a reserved renderer control"));
        }
        Ok(())
    }

    fn checked_text_bytes(&self, amount: usize, path: &str) -> Result<usize, ToolHistoryError> {
        let total = self
            .text_bytes
            .checked_add(amount)
            .ok_or_else(|| self.error(path, "aggregate text byte accounting overflow"))?;
        if total > self.limits.aggregate_text_bytes {
            return Err(self.error(
                path,
                format_args!(
                    "exceeds the aggregate maximum of {} text bytes",
                    self.limits.aggregate_text_bytes
                ),
            ));
        }
        Ok(total)
    }

    fn checked_text_blocks(&self, amount: usize, path: &str) -> Result<usize, ToolHistoryError> {
        let total = self
            .text_blocks
            .checked_add(amount)
            .ok_or_else(|| self.error(path, "aggregate text-block accounting overflow"))?;
        if total > self.limits.aggregate_text_blocks {
            return Err(self.error(
                path,
                format_args!(
                    "exceeds the aggregate maximum of {} text blocks",
                    self.limits.aggregate_text_blocks
                ),
            ));
        }
        Ok(total)
    }

    fn register_call_id(&mut self, id: &str, path: &str) -> Result<(), ToolHistoryError> {
        self.validate_external_id(id, path)?;
        if !self.seen_call_ids.insert(id.to_owned()) {
            return Err(self.error(path, "duplicates an earlier historical call ID"));
        }
        Ok(())
    }

    fn validate_external_id(&self, id: &str, path: &str) -> Result<(), ToolHistoryError> {
        if id.is_empty() {
            return Err(self.error(path, "must be nonempty"));
        }
        if id.len() > self.limits.external_id_bytes {
            return Err(self.error(
                path,
                format_args!(
                    "exceeds the maximum of {} UTF-8 bytes",
                    self.limits.external_id_bytes
                ),
            ));
        }
        if id.chars().any(char::is_control) {
            return Err(self.error(path, "must not contain control characters"));
        }
        if reserved_control(id).is_some() {
            return Err(self.error(path, "contains a reserved renderer control"));
        }
        Ok(())
    }

    fn preflight_arguments(
        &self,
        arguments: &Value,
        key_string_maximum: usize,
        path: &str,
    ) -> Result<(), ToolHistoryError> {
        if !arguments.is_object() {
            return Err(self.error(path, "must be an object"));
        }

        let initial_capacity = (MAX_DEPTH + 1).min(MAX_ARGUMENT_NODES);
        let mut stack = Vec::with_capacity(initial_capacity);
        stack.push((arguments, 0usize));
        let mut nodes = 0usize;
        let mut key_string_bytes = 0usize;

        while let Some((value, depth)) = stack.pop() {
            if depth > MAX_DEPTH {
                return Err(self.error(
                    path,
                    format_args!("exceeds the maximum argument depth of {MAX_DEPTH}"),
                ));
            }
            nodes = nodes
                .checked_add(1)
                .ok_or_else(|| self.error(path, "argument node accounting overflow"))?;
            if nodes > MAX_ARGUMENT_NODES {
                return Err(self.error(
                    path,
                    format_args!("exceeds the maximum of {MAX_ARGUMENT_NODES} argument nodes"),
                ));
            }

            match value {
                Value::Object(object) => {
                    let child_depth = self.reserve_argument_children(
                        &mut stack,
                        object.len(),
                        depth,
                        nodes,
                        path,
                    )?;
                    for key in object.keys() {
                        key_string_bytes = self.checked_argument_leaf_bytes(
                            key_string_bytes,
                            key.len(),
                            key_string_maximum,
                            path,
                        )?;
                    }
                    for child in object.values() {
                        stack.push((child, child_depth));
                    }
                }
                Value::Array(values) => {
                    let child_depth = self.reserve_argument_children(
                        &mut stack,
                        values.len(),
                        depth,
                        nodes,
                        path,
                    )?;
                    for child in values {
                        stack.push((child, child_depth));
                    }
                }
                Value::String(value) => {
                    key_string_bytes = self.checked_argument_leaf_bytes(
                        key_string_bytes,
                        value.len(),
                        key_string_maximum,
                        path,
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn reserve_argument_children(
        &self,
        stack: &mut Vec<(&Value, usize)>,
        child_count: usize,
        depth: usize,
        processed_nodes: usize,
        path: &str,
    ) -> Result<usize, ToolHistoryError> {
        if child_count == 0 {
            return Ok(depth);
        }
        if depth >= MAX_DEPTH {
            return Err(self.error(
                path,
                format_args!("exceeds the maximum argument depth of {MAX_DEPTH}"),
            ));
        }
        let pending_nodes = stack
            .len()
            .checked_add(child_count)
            .ok_or_else(|| self.error(path, "argument pending-node accounting overflow"))?;
        let minimum_nodes = processed_nodes
            .checked_add(pending_nodes)
            .ok_or_else(|| self.error(path, "argument node accounting overflow"))?;
        if minimum_nodes > MAX_ARGUMENT_NODES {
            return Err(self.error(
                path,
                format_args!("exceeds the maximum of {MAX_ARGUMENT_NODES} argument nodes"),
            ));
        }
        stack.try_reserve_exact(child_count).map_err(|_| {
            self.error(
                path,
                "could not reserve the bounded argument traversal stack",
            )
        })?;
        depth
            .checked_add(1)
            .ok_or_else(|| self.error(path, "argument depth accounting overflow"))
    }

    fn checked_argument_leaf_bytes(
        &self,
        current: usize,
        amount: usize,
        maximum: usize,
        path: &str,
    ) -> Result<usize, ToolHistoryError> {
        let total = current
            .checked_add(amount)
            .ok_or_else(|| self.error(path, "argument key/string byte accounting overflow"))?;
        if total > maximum {
            return Err(self.error(
                path,
                format_args!(
                    "exceeds the preflight maximum of {maximum} argument key/string bytes"
                ),
            ));
        }
        Ok(total)
    }

    fn validate_arguments(
        &mut self,
        name: &str,
        arguments: &Value,
        path: &str,
    ) -> Result<(), ToolHistoryError> {
        let remaining_aggregate = self
            .limits
            .aggregate_argument_bytes
            .checked_sub(self.argument_bytes)
            .ok_or_else(|| self.error(path, "aggregate argument byte accounting underflow"))?;
        let sizing_limit = self.limits.compact_argument_bytes.min(remaining_aggregate);
        self.preflight_arguments(arguments, sizing_limit, path)?;
        let compact_bytes = match bounded_compact_json_bytes(arguments, sizing_limit) {
            Ok(bytes) => bytes,
            Err(CompactJsonSizeError::LimitExceeded)
                if remaining_aggregate < self.limits.compact_argument_bytes =>
            {
                return Err(self.error(
                    path,
                    format_args!(
                        "exceeds the aggregate maximum of {} compact argument bytes",
                        self.limits.aggregate_argument_bytes
                    ),
                ));
            }
            Err(CompactJsonSizeError::LimitExceeded) => {
                return Err(self.error(
                    path,
                    format_args!(
                        "compact JSON exceeds the maximum of {} bytes",
                        self.limits.compact_argument_bytes
                    ),
                ));
            }
            Err(CompactJsonSizeError::Encoding) => {
                return Err(self.error(path, "compact JSON byte accounting failed"));
            }
        };
        let aggregate = self
            .argument_bytes
            .checked_add(compact_bytes)
            .ok_or_else(|| self.error(path, "aggregate argument byte accounting overflow"))?;
        if aggregate > self.limits.aggregate_argument_bytes {
            return Err(self.error(
                path,
                format_args!(
                    "exceeds the aggregate maximum of {} compact argument bytes",
                    self.limits.aggregate_argument_bytes
                ),
            ));
        }
        self.registry
            .validate_call_arguments(name, arguments)
            .map_err(|_| {
                self.error(
                    path,
                    "must satisfy the schema of a declared tool without disallowed content",
                )
            })?;
        self.argument_bytes = aggregate;
        Ok(())
    }

    fn object<'value>(
        &self,
        value: &'value Value,
        path: &str,
    ) -> Result<&'value Map<String, Value>, ToolHistoryError> {
        value
            .as_object()
            .ok_or_else(|| self.error(path, "must be an object"))
    }

    fn ensure_fields(
        &self,
        object: &Map<String, Value>,
        allowed: &[&str],
        path: &str,
    ) -> Result<(), ToolHistoryError> {
        if object
            .keys()
            .any(|field| !allowed.contains(&field.as_str()))
        {
            return Err(self.error(path, "contains an unsupported field"));
        }
        Ok(())
    }

    fn required_string<'value>(
        &self,
        object: &'value Map<String, Value>,
        field: &str,
        path: &str,
    ) -> Result<&'value str, ToolHistoryError> {
        match object.get(field) {
            Some(Value::String(value)) => Ok(value),
            Some(_) => Err(self.error(&format!("{path}.{field}"), "must be a string")),
            None => Err(self.error(&format!("{path}.{field}"), "is required")),
        }
    }

    fn require_exact_string(
        &self,
        object: &Map<String, Value>,
        field: &str,
        expected: &str,
        path: &str,
    ) -> Result<(), ToolHistoryError> {
        match object.get(field) {
            Some(Value::String(value)) if value == expected => Ok(()),
            Some(_) => Err(self.error(&format!("{path}.{field}"), "has an unsupported value")),
            None => Err(self.error(&format!("{path}.{field}"), "is required")),
        }
    }

    fn error(&self, path: &str, reason: impl fmt::Display) -> ToolHistoryError {
        ToolHistoryError(format!("{} history {path}: {reason}", self.provider.name()))
    }
}

fn canonical_message(role: &str, content: Option<String>) -> ChatMessage {
    ChatMessage {
        role: role.to_owned(),
        content: content.map(Value::String),
        tool_calls: None,
        tool_responses: None,
        reasoning: None,
        reasoning_content: None,
        tool_call_id: None,
        name: None,
    }
}

fn canonical_batch(
    content: Option<String>,
    calls: Vec<HistoricalCall>,
    results: Vec<String>,
) -> (ChatMessage, Vec<ChatMessage>) {
    debug_assert_eq!(calls.len(), results.len());
    let mut canonical_calls = Vec::with_capacity(calls.len());
    let mut tool_rows = Vec::with_capacity(calls.len());

    for (call, result) in calls.into_iter().zip(results) {
        tool_rows.push(ChatMessage {
            role: "tool".to_owned(),
            content: Some(Value::String(result)),
            tool_calls: None,
            tool_responses: None,
            reasoning: None,
            reasoning_content: None,
            tool_call_id: Some(call.id.clone()),
            name: Some(call.name.clone()),
        });

        let mut function = Map::new();
        function.insert("name".to_owned(), Value::String(call.name));
        function.insert("arguments".to_owned(), call.arguments);
        let mut canonical_call = Map::new();
        canonical_call.insert("id".to_owned(), Value::String(call.id));
        canonical_call.insert("type".to_owned(), Value::String("function".to_owned()));
        canonical_call.insert("function".to_owned(), Value::Object(function));
        canonical_calls.push(Value::Object(canonical_call));
    }

    let mut assistant = canonical_message("assistant", content);
    assistant.tool_calls = Some(Value::Array(canonical_calls));
    (assistant, tool_rows)
}

struct BoundedJsonByteCounter {
    bytes: usize,
    maximum: usize,
    exceeded: bool,
}

impl Write for BoundedJsonByteCounter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(total) = self.bytes.checked_add(buffer.len()) else {
            self.exceeded = true;
            return Err(io::Error::other("compact JSON size overflow"));
        };
        if total > self.maximum {
            self.exceeded = true;
            return Err(io::Error::other("compact JSON exceeds byte budget"));
        }
        self.bytes = total;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
enum CompactJsonSizeError {
    LimitExceeded,
    Encoding,
}

fn bounded_compact_json_bytes(
    value: &Value,
    maximum: usize,
) -> Result<usize, CompactJsonSizeError> {
    let mut counter = BoundedJsonByteCounter {
        bytes: 0,
        maximum,
        exceeded: false,
    };
    match serde_json::to_writer(&mut counter, value) {
        Ok(()) => Ok(counter.bytes),
        Err(_) if counter.exceeded => Err(CompactJsonSizeError::LimitExceeded),
        Err(_) => Err(CompactJsonSizeError::Encoding),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool_schema::{OpenAiToolsInput, ToolSchemaError};

    fn registry_with_schema(names: &[&str], schema: &Value) -> ToolRegistry {
        let tools = Value::Array(
            names
                .iter()
                .map(|name| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": "history test tool",
                            "parameters": schema,
                        }
                    })
                })
                .collect(),
        );
        ToolRegistry::from_openai(OpenAiToolsInput {
            tools: Some(&tools),
            tool_choice: None,
            parallel_tool_calls: None,
        })
        .unwrap()
    }

    fn permissive_registry(names: &[&str]) -> ToolRegistry {
        registry_with_schema(
            names,
            &json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": true,
            }),
        )
    }

    fn strict_registry() -> ToolRegistry {
        registry_with_schema(
            &["echo"],
            &json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": false,
            }),
        )
    }

    fn openai_call(id: &str, name: &str, arguments: &str) -> Value {
        json!({
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
    }

    fn openai_result(id: &str, content: Value) -> Value {
        json!({"role": "tool", "tool_call_id": id, "content": content})
    }

    fn openai_batch(calls: Vec<Value>, results: Vec<Value>) -> Value {
        let mut messages = vec![json!({
            "role": "assistant",
            "content": null,
            "tool_calls": calls,
        })];
        messages.extend(results);
        Value::Array(messages)
    }

    fn anthropic_call(id: &str, name: &str, arguments: Value) -> Value {
        json!({"type": "tool_use", "id": id, "name": name, "input": arguments})
    }

    fn anthropic_result(id: &str, content: Option<Value>) -> Value {
        let mut result = Map::new();
        result.insert("type".to_owned(), Value::String("tool_result".to_owned()));
        result.insert("tool_use_id".to_owned(), Value::String(id.to_owned()));
        if let Some(content) = content {
            result.insert("content".to_owned(), content);
        }
        Value::Object(result)
    }

    fn anthropic_batch(calls: Vec<Value>, results: Vec<Value>) -> Value {
        json!([
            {"role": "assistant", "content": calls},
            {"role": "user", "content": results},
        ])
    }

    fn normalize_openai(
        registry: &ToolRegistry,
        messages: &Value,
    ) -> Result<Vec<ChatMessage>, ToolHistoryError> {
        registry.normalize_openai_history(OpenAiHistoryInput { messages })
    }

    fn normalize_anthropic(
        registry: &ToolRegistry,
        system: Option<&Value>,
        messages: &Value,
    ) -> Result<Vec<ChatMessage>, ToolHistoryError> {
        registry.normalize_anthropic_history(AnthropicHistoryInput { system, messages })
    }

    #[test]
    fn provider_parallel_histories_converge_and_reorder_results() {
        let registry = permissive_registry(&["lookup"]);
        let openai = json!([
            {
                "role": "developer",
                "content": [
                    {"type": "text", "text": "system"},
                    {"type": "text", "text": " rules"},
                ],
            },
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call-a",
                        "type": "function",
                        "index": 0,
                        "function": {"name": "lookup", "arguments": "{\"value\":\"a\"}"},
                    },
                    {
                        "id": "call-b",
                        "type": "function",
                        "index": 1,
                        "function": {"name": "lookup", "arguments": "{\"value\":\"b\"}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-b",
                "name": "lookup",
                "content": [{"type": "text", "text": "result-b"}],
            },
            {"role": "tool", "tool_call_id": "call-a", "content": "result-a"},
        ]);
        let system = json!([
            {"type": "text", "text": "system"},
            {"type": "text", "text": " rules"},
        ]);
        let anthropic = json!([
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call-a", "name": "lookup", "input": {"value": "a"}},
                    {"type": "tool_use", "id": "call-b", "name": "lookup", "input": {"value": "b"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-b",
                        "content": [{"type": "text", "text": "result-b"}],
                    },
                    {"type": "tool_result", "tool_use_id": "call-a", "content": "result-a"},
                ],
            },
        ]);

        let openai_output = normalize_openai(&registry, &openai).unwrap();
        let anthropic_output = normalize_anthropic(&registry, Some(&system), &anthropic).unwrap();
        let openai_json = serde_json::to_value(&openai_output).unwrap();
        assert_eq!(
            openai_json,
            serde_json::to_value(&anthropic_output).unwrap()
        );
        assert_eq!(
            openai_json,
            json!([
                {"role": "system", "content": "system rules"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"value": "a"}},
                        },
                        {
                            "id": "call-b",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"value": "b"}},
                        },
                    ],
                },
                {"role": "tool", "content": "result-a", "tool_call_id": "call-a", "name": "lookup"},
                {"role": "tool", "content": "result-b", "tool_call_id": "call-b", "name": "lookup"},
            ])
        );
    }

    #[test]
    fn declaration_schema_and_unique_json_are_enforced_without_payload_leaks() {
        let registry = strict_registry();
        for (name, arguments) in [
            ("missing", "{\"secret-value\":true}"),
            ("echo", "{\"value\":1}"),
            ("echo", "{"),
            ("echo", "[]"),
            ("echo", "{\"value\":\"first\",\"value\":\"second\"}"),
        ] {
            let history = openai_batch(
                vec![openai_call("call-1", name, arguments)],
                vec![openai_result("call-1", json!("ok"))],
            );
            let error = normalize_openai(&registry, &history).unwrap_err();
            assert!(!error.to_string().contains("secret-value"));
            assert!(
                error
                    .to_string()
                    .starts_with("OpenAI history $.messages[0]")
            );
        }

        let anthropic = anthropic_batch(
            vec![anthropic_call("call-1", "echo", json!({"value": 1}))],
            vec![anthropic_result("call-1", Some(json!("ok")))],
        );
        assert!(normalize_anthropic(&registry, None, &anthropic).is_err());
    }

    #[test]
    fn openai_id_and_correlation_invariants_fail_closed() {
        let registry = permissive_registry(&["echo"]);
        let missing_id = json!([
            {
                "role": "assistant",
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ]);
        assert!(normalize_openai(&registry, &missing_id).is_err());
        for id in ["", "bad\nid", "x<|turn>y"] {
            let history = openai_batch(
                vec![openai_call(id, "echo", "{}")],
                vec![openai_result(id, json!("ok"))],
            );
            assert!(normalize_openai(&registry, &history).is_err());
        }
        let oversized = "x".repeat(MAX_EXTERNAL_ID_BYTES + 1);
        let history = openai_batch(
            vec![openai_call(&oversized, "echo", "{}")],
            vec![openai_result(&oversized, json!("ok"))],
        );
        assert!(normalize_openai(&registry, &history).is_err());

        let duplicate_calls = openai_batch(
            vec![
                openai_call("same", "echo", "{}"),
                openai_call("same", "echo", "{}"),
            ],
            vec![
                openai_result("same", json!("a")),
                openai_result("same", json!("b")),
            ],
        );
        assert!(normalize_openai(&registry, &duplicate_calls).is_err());

        let duplicate_later = json!([
            {
                "role": "assistant",
                "tool_calls": [openai_call("same", "echo", "{}")],
            },
            openai_result("same", json!("first")),
            {
                "role": "assistant",
                "tool_calls": [openai_call("same", "echo", "{}")],
            },
            openai_result("same", json!("second")),
        ]);
        assert!(normalize_openai(&registry, &duplicate_later).is_err());

        let orphan = json!([openai_result("orphan", json!("no call"))]);
        assert!(normalize_openai(&registry, &orphan).is_err());

        let unknown_result = openai_batch(
            vec![openai_call("call-1", "echo", "{}")],
            vec![openai_result("unknown", json!("ok"))],
        );
        assert!(normalize_openai(&registry, &unknown_result).is_err());

        let duplicate_results = openai_batch(
            vec![
                openai_call("call-1", "echo", "{}"),
                openai_call("call-2", "echo", "{}"),
            ],
            vec![
                openai_result("call-1", json!("first")),
                openai_result("call-1", json!("again")),
            ],
        );
        assert!(normalize_openai(&registry, &duplicate_results).is_err());

        let pending_eof = json!([{
            "role": "assistant",
            "tool_calls": [openai_call("call-1", "echo", "{}")],
        }]);
        assert!(normalize_openai(&registry, &pending_eof).is_err());

        let interleaved = json!([
            {
                "role": "assistant",
                "tool_calls": [openai_call("call-1", "echo", "{}")],
            },
            {"role": "user", "content": "interrupt"},
        ]);
        assert!(normalize_openai(&registry, &interleaved).is_err());

        let extra = openai_batch(
            vec![openai_call("call-1", "echo", "{}")],
            vec![
                openai_result("call-1", json!("ok")),
                openai_result("call-1", json!("extra")),
            ],
        );
        assert!(normalize_openai(&registry, &extra).is_err());

        let result_without_id = json!([
            {
                "role": "assistant",
                "tool_calls": [openai_call("call-1", "echo", "{}")],
            },
            {"role": "tool", "content": "missing id"},
        ]);
        assert!(normalize_openai(&registry, &result_without_id).is_err());
    }

    #[test]
    fn anthropic_id_and_correlation_invariants_fail_closed() {
        let registry = permissive_registry(&["echo"]);
        let orphan = json!([{
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "orphan"}],
        }]);
        assert!(normalize_anthropic(&registry, None, &orphan).is_err());

        let duplicate_calls = anthropic_batch(
            vec![
                anthropic_call("same", "echo", json!({})),
                anthropic_call("same", "echo", json!({})),
            ],
            vec![
                anthropic_result("same", None),
                anthropic_result("same", None),
            ],
        );
        assert!(normalize_anthropic(&registry, None, &duplicate_calls).is_err());

        let unknown = anthropic_batch(
            vec![anthropic_call("call-1", "echo", json!({}))],
            vec![anthropic_result("unknown", None)],
        );
        assert!(normalize_anthropic(&registry, None, &unknown).is_err());

        let duplicate_results = anthropic_batch(
            vec![
                anthropic_call("call-1", "echo", json!({})),
                anthropic_call("call-2", "echo", json!({})),
            ],
            vec![
                anthropic_result("call-1", None),
                anthropic_result("call-1", None),
            ],
        );
        assert!(normalize_anthropic(&registry, None, &duplicate_results).is_err());

        let pending = json!([{
            "role": "assistant",
            "content": [anthropic_call("call-1", "echo", json!({}))],
        }]);
        assert!(normalize_anthropic(&registry, None, &pending).is_err());

        let interleaved = json!([
            {
                "role": "assistant",
                "content": [anthropic_call("call-1", "echo", json!({}))],
            },
            {"role": "assistant", "content": "interrupt"},
        ]);
        assert!(normalize_anthropic(&registry, None, &interleaved).is_err());

        let missing_and_extra = [
            anthropic_batch(
                vec![anthropic_call("call-1", "echo", json!({}))],
                Vec::new(),
            ),
            anthropic_batch(
                vec![anthropic_call("call-1", "echo", json!({}))],
                vec![
                    anthropic_result("call-1", None),
                    anthropic_result("call-1", None),
                ],
            ),
        ];
        for history in missing_and_extra {
            assert!(normalize_anthropic(&registry, None, &history).is_err());
        }
    }

    #[test]
    fn system_rules_and_ordinary_content_are_canonical() {
        let registry = permissive_registry(&[]);
        let system = json!("instructions");
        let anthropic_messages = json!([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "assistant", "content": ""},
        ]);
        let expected = json!([
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ]);
        let anthropic = normalize_anthropic(&registry, Some(&system), &anthropic_messages).unwrap();
        assert_eq!(serde_json::to_value(anthropic).unwrap(), expected);

        for role in ["system", "developer"] {
            let openai = json!([
                {"role": role, "content": "instructions"},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": ""},
            ]);
            let output = normalize_openai(&registry, &openai).unwrap();
            assert_eq!(serde_json::to_value(output).unwrap(), expected);
        }

        for bad in [
            json!([
                {"role": "user", "content": "first"},
                {"role": "system", "content": "late"},
            ]),
            json!([
                {"role": "system", "content": "first"},
                {"role": "developer", "content": "second"},
            ]),
            json!([{"role": "user"}]),
            json!([{"role": "assistant", "content": null}]),
        ] {
            assert!(normalize_openai(&registry, &bad).is_err());
        }

        let no_system = normalize_anthropic(&registry, Some(&Value::Null), &json!([])).unwrap();
        assert!(no_system.is_empty());
    }

    #[test]
    fn unsupported_semantics_and_rich_blocks_are_rejected() {
        let registry = permissive_registry(&["echo"]);
        let mixed = json!([
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "preface"},
                    anthropic_call("call-1", "echo", json!({})),
                ],
            },
            {"role": "user", "content": [anthropic_result("call-1", None)]},
        ]);
        assert!(normalize_anthropic(&registry, None, &mixed).is_err());

        for block in [
            json!({"type": "image", "source": {}}),
            json!({"type": "thinking", "thinking": "secret"}),
            json!({"type": "server_tool_use", "id": "x", "name": "search", "input": {}}),
            json!({"type": "document", "source": {}}),
        ] {
            let history = json!([{"role": "user", "content": [block]}]);
            assert!(normalize_anthropic(&registry, None, &history).is_err());
        }

        let mut is_error = anthropic_result("call-1", Some(json!("failed")));
        is_error
            .as_object_mut()
            .unwrap()
            .insert("is_error".to_owned(), Value::Bool(true));
        let history = anthropic_batch(
            vec![anthropic_call("call-1", "echo", json!({}))],
            vec![is_error],
        );
        assert!(normalize_anthropic(&registry, None, &history).is_err());

        let history = json!([{"role": "system", "content": "not a message role"}]);
        assert!(normalize_anthropic(&registry, None, &history).is_err());
        let history = json!([{"role": "assistant", "content": "answer", "reasoning": "x"}]);
        assert!(normalize_anthropic(&registry, None, &history).is_err());

        for history in [
            json!([{"role": "function", "name": "echo", "content": "legacy"}]),
            json!([{"role": "assistant", "content": null, "function_call": {"name": "echo", "arguments": "{}"}}]),
            json!([{"role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}]),
            json!([
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "custom",
                        "function": {"name": "echo", "arguments": "{}"},
                    }],
                },
                openai_result("call-1", json!("ok")),
            ]),
            json!([
                {
                    "role": "assistant",
                    "tool_calls": [openai_call("call-1", "echo", "{}")],
                },
                openai_result("call-1", json!([{"type": "image", "image_url": "x"}])),
            ]),
        ] {
            assert!(normalize_openai(&registry, &history).is_err());
        }
    }

    #[test]
    fn reserved_controls_and_unsafe_arguments_fail_every_history_surface() {
        let registry = permissive_registry(&["echo"]);
        for history in [
            json!([{"role": "user", "content": "bad <|turn> text"}]),
            openai_batch(
                vec![openai_call("bad<|tool>id", "echo", "{}")],
                vec![openai_result("bad<|tool>id", json!("ok"))],
            ),
            openai_batch(
                vec![openai_call("call-1", "echo", "{\"value\":\"<|think|>\"}")],
                vec![openai_result("call-1", json!("ok"))],
            ),
            openai_batch(
                vec![openai_call("call-1", "echo", "{\"bad key\":true}")],
                vec![openai_result("call-1", json!("ok"))],
            ),
            openai_batch(
                vec![openai_call("call-1", "echo", "{}")],
                vec![openai_result("call-1", json!("bad <|tool_response>"))],
            ),
        ] {
            assert!(normalize_openai(&registry, &history).is_err());
        }

        let system = json!("bad <bos> system");
        assert!(normalize_anthropic(&registry, Some(&system), &json!([])).is_err());
        let history = anthropic_batch(
            vec![anthropic_call("call-1", "echo", json!({}))],
            vec![anthropic_result(
                "call-1",
                Some(json!("bad <|tool_call> result")),
            )],
        );
        assert!(normalize_anthropic(&registry, None, &history).is_err());
    }

    #[test]
    fn tiny_limits_cover_exact_and_one_over_transport_boundaries() {
        let no_tools = permissive_registry(&[]);
        let one_message = json!([{"role": "user", "content": ""}]);
        assert!(
            no_tools
                .normalize_openai_history_with_limits(
                    OpenAiHistoryInput {
                        messages: &one_message,
                    },
                    HistoryLimits {
                        source_messages: 1,
                        ..PRODUCTION_LIMITS
                    },
                )
                .is_ok()
        );
        assert!(
            no_tools
                .normalize_openai_history_with_limits(
                    OpenAiHistoryInput {
                        messages: &one_message,
                    },
                    HistoryLimits {
                        source_messages: 0,
                        ..PRODUCTION_LIMITS
                    },
                )
                .is_err()
        );

        let registry = permissive_registry(&["echo"]);
        let two_calls = openai_batch(
            vec![
                openai_call("a", "echo", "{}"),
                openai_call("b", "echo", "{}"),
            ],
            vec![openai_result("b", json!("")), openai_result("a", json!(""))],
        );
        for (maximum, succeeds) in [(2, true), (1, false)] {
            let result = registry.normalize_openai_history_with_limits(
                OpenAiHistoryInput {
                    messages: &two_calls,
                },
                HistoryLimits {
                    calls_per_turn: maximum,
                    ..PRODUCTION_LIMITS
                },
            );
            assert_eq!(result.is_ok(), succeeds);
        }

        let unicode_id = openai_batch(
            vec![openai_call("é", "echo", "{}")],
            vec![openai_result("é", json!(""))],
        );
        for (maximum, succeeds) in [(2, true), (1, false)] {
            let result = registry.normalize_openai_history_with_limits(
                OpenAiHistoryInput {
                    messages: &unicode_id,
                },
                HistoryLimits {
                    external_id_bytes: maximum,
                    ..PRODUCTION_LIMITS
                },
            );
            assert_eq!(result.is_ok(), succeeds);
        }

        let empty_arguments = openai_batch(
            vec![openai_call("call-1", "echo", "{}")],
            vec![openai_result("call-1", json!(""))],
        );
        assert_eq!(bounded_compact_json_bytes(&json!({}), 2).unwrap(), 2);
        assert_eq!(
            bounded_compact_json_bytes(&json!({}), 1),
            Err(CompactJsonSizeError::LimitExceeded)
        );
        for (field, maximum, succeeds) in [
            ("raw", 2, true),
            ("raw", 1, false),
            ("compact", 2, true),
            ("compact", 1, false),
        ] {
            let mut limits = PRODUCTION_LIMITS;
            if field == "raw" {
                limits.raw_argument_bytes = maximum;
            } else {
                limits.compact_argument_bytes = maximum;
            }
            let result = registry.normalize_openai_history_with_limits(
                OpenAiHistoryInput {
                    messages: &empty_arguments,
                },
                limits,
            );
            assert_eq!(result.is_ok(), succeeds, "{field} maximum {maximum}");
        }

        let two_empty_arguments = openai_batch(
            vec![
                openai_call("call-1", "echo", "{}"),
                openai_call("call-2", "echo", "{}"),
            ],
            vec![
                openai_result("call-1", json!("")),
                openai_result("call-2", json!("")),
            ],
        );
        for (maximum, succeeds) in [(4, true), (3, false)] {
            let result = registry.normalize_openai_history_with_limits(
                OpenAiHistoryInput {
                    messages: &two_empty_arguments,
                },
                HistoryLimits {
                    aggregate_argument_bytes: maximum,
                    ..PRODUCTION_LIMITS
                },
            );
            assert_eq!(result.is_ok(), succeeds);
        }

        let unicode_text = json!([{"role": "user", "content": "é"}]);
        for (maximum, succeeds) in [(2, true), (1, false)] {
            let result = no_tools.normalize_openai_history_with_limits(
                OpenAiHistoryInput {
                    messages: &unicode_text,
                },
                HistoryLimits {
                    aggregate_text_bytes: maximum,
                    ..PRODUCTION_LIMITS
                },
            );
            assert_eq!(result.is_ok(), succeeds);
        }

        let two_text_blocks = json!([
            {"role": "user", "content": [{"type": "text", "text": ""}]},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]);
        for (maximum, succeeds) in [(2, true), (1, false)] {
            let result = no_tools.normalize_openai_history_with_limits(
                OpenAiHistoryInput {
                    messages: &two_text_blocks,
                },
                HistoryLimits {
                    aggregate_text_blocks: maximum,
                    ..PRODUCTION_LIMITS
                },
            );
            assert_eq!(result.is_ok(), succeeds);
        }

        let anthropic_two_calls = anthropic_batch(
            vec![
                anthropic_call("a", "echo", json!({})),
                anthropic_call("b", "echo", json!({})),
            ],
            vec![anthropic_result("b", None), anthropic_result("a", None)],
        );
        for (maximum, succeeds) in [(2, true), (1, false)] {
            let result = registry.normalize_anthropic_history_with_limits(
                AnthropicHistoryInput {
                    system: None,
                    messages: &anthropic_two_calls,
                },
                HistoryLimits {
                    calls_per_turn: maximum,
                    ..PRODUCTION_LIMITS
                },
            );
            assert_eq!(result.is_ok(), succeeds);
        }
    }

    #[test]
    fn text_budgets_precede_scans_and_failed_validation_is_transactional() {
        let registry = permissive_registry(&[]);
        let mut normalizer = Normalizer::new(
            &registry,
            Provider::OpenAi,
            HistoryLimits {
                aggregate_text_bytes: 1,
                ..PRODUCTION_LIMITS
            },
        );
        let error = normalizer
            .text_content(&json!("oversized <|turn>"), "$.content", false)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("aggregate maximum of 1 text bytes")
        );
        assert!(!error.to_string().contains("reserved"));
        assert_eq!(normalizer.text_bytes, 0);
        assert_eq!(normalizer.text_blocks, 0);

        let mut normalizer = Normalizer::new(
            &registry,
            Provider::OpenAi,
            HistoryLimits {
                aggregate_text_bytes: 64,
                aggregate_text_blocks: 2,
                ..PRODUCTION_LIMITS
            },
        );
        let content = json!([
            {"type": "text", "text": "accepted first"},
            {"type": "text", "text": "rejected <|turn> second"},
        ]);
        let error = normalizer
            .text_content(&content, "$.content", false)
            .unwrap_err();
        assert!(error.to_string().contains("reserved"));
        assert_eq!(normalizer.text_bytes, 0);
        assert_eq!(normalizer.text_blocks, 0);

        let many_empty_blocks = Value::Array(
            (0..=MAX_AGGREGATE_TEXT_BLOCKS)
                .map(|_| json!({"type": "text", "text": ""}))
                .collect(),
        );
        let history = json!([{"role": "user", "content": many_empty_blocks}]);
        let error = normalize_anthropic(&registry, None, &history).unwrap_err();
        assert!(error.to_string().contains("scan budget"));
    }

    #[test]
    fn argument_leaf_preflight_precedes_serialization_and_schema_traversal() {
        let registry = permissive_registry(&["echo"]);
        let mut keyed = Map::new();
        keyed.insert("x".repeat(2 * 1024 * 1024), Value::Bool(true));
        let oversized_value = "x".repeat(2 * 1024 * 1024);

        for arguments in [Value::Object(keyed), json!({"value": oversized_value})] {
            let history = anthropic_batch(
                vec![anthropic_call("call-1", "echo", arguments)],
                vec![anthropic_result("call-1", None)],
            );
            let error = normalize_anthropic(&registry, None, &history).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("preflight maximum of 65536 argument key/string bytes")
            );
        }
    }

    #[test]
    fn deeply_nested_arguments_fail_without_recursive_serialization_or_teardown() {
        let registry = permissive_registry(&["echo"]);
        let mut nested = Value::Null;
        for _ in 0..10_000 {
            nested = Value::Array(vec![nested]);
        }
        let mut arguments = Map::new();
        arguments.insert("value".to_owned(), nested);
        let mut block = Map::new();
        block.insert("type".to_owned(), Value::String("tool_use".to_owned()));
        block.insert("id".to_owned(), Value::String("call-1".to_owned()));
        block.insert("name".to_owned(), Value::String("echo".to_owned()));
        block.insert("input".to_owned(), Value::Object(arguments));
        let mut message = Map::new();
        message.insert("role".to_owned(), Value::String("assistant".to_owned()));
        message.insert(
            "content".to_owned(),
            Value::Array(vec![Value::Object(block)]),
        );
        let history = Value::Array(vec![Value::Object(message)]);

        let result = normalize_anthropic(&registry, None, &history);
        std::mem::forget(history);
        match result {
            Err(error) => assert!(error.to_string().contains("maximum argument depth of 48")),
            Ok(output) => {
                std::mem::forget(output);
                panic!("deep arguments unexpectedly normalized");
            }
        }
    }

    #[test]
    fn call_name_grammar_precedes_all_argument_sizing() {
        let registry = permissive_registry(&["echo"]);
        let raw_arguments = "x".repeat(MAX_RAW_ARGUMENT_BYTES + 1);
        let name_65 = "n".repeat(65);
        let openai = openai_batch(
            vec![openai_call("call-1", &name_65, &raw_arguments)],
            vec![openai_result("call-1", json!(""))],
        );
        let error = normalize_openai(&registry, &openai).unwrap_err();
        assert!(error.to_string().contains("tool-name grammar"));
        assert!(!error.to_string().contains(name_65.as_str()));

        let huge_name = "n".repeat(2 * 1024 * 1024);
        let anthropic = anthropic_batch(
            vec![anthropic_call(
                "call-1",
                &huge_name,
                json!({"value": "x".repeat(2 * 1024 * 1024)}),
            )],
            vec![anthropic_result("call-1", None)],
        );
        let error = normalize_anthropic(&registry, None, &anthropic).unwrap_err();
        assert!(error.to_string().contains("tool-name grammar"));
        assert!(!error.to_string().contains(huge_name.as_str()));
    }

    #[test]
    fn argument_node_and_depth_boundaries_are_exact() {
        let registry = permissive_registry(&["echo"]);
        let exact_nodes = json!({"values": vec![Value::Null; 4_094]});
        let over_nodes = json!({"values": vec![Value::Null; 4_095]});
        for (arguments, succeeds) in [(exact_nodes, true), (over_nodes, false)] {
            let history = anthropic_batch(
                vec![anthropic_call("call-1", "echo", arguments)],
                vec![anthropic_result("call-1", None)],
            );
            assert_eq!(
                normalize_anthropic(&registry, None, &history).is_ok(),
                succeeds
            );
        }

        fn nested_arguments(array_count: usize) -> Value {
            let mut value = Value::Null;
            for _ in 0..array_count {
                value = Value::Array(vec![value]);
            }
            json!({"value": value})
        }
        for (arguments, succeeds) in [(nested_arguments(47), true), (nested_arguments(48), false)] {
            let history = anthropic_batch(
                vec![anthropic_call("call-1", "echo", arguments)],
                vec![anthropic_result("call-1", None)],
            );
            assert_eq!(
                normalize_anthropic(&registry, None, &history).is_ok(),
                succeeds
            );
        }
    }

    #[test]
    fn empty_results_names_and_input_immutability_are_preserved() {
        let registry = permissive_registry(&["echo"]);
        let openai = json!([
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [openai_call("call-1", "echo", "{}")],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "echo", "content": ""},
        ]);
        let openai_before = openai.clone();
        let output = normalize_openai(&registry, &openai).unwrap();
        assert_eq!(openai, openai_before);
        let output = serde_json::to_value(output).unwrap();
        assert_eq!(output[1]["content"], "");
        assert_eq!(output[1]["name"], "echo");

        let anthropic = anthropic_batch(
            vec![anthropic_call("call-1", "echo", json!({}))],
            vec![anthropic_result("call-1", None)],
        );
        let anthropic_before = anthropic.clone();
        let output = normalize_anthropic(&registry, None, &anthropic).unwrap();
        assert_eq!(anthropic, anthropic_before);
        assert_eq!(serde_json::to_value(output).unwrap()[1]["content"], "");

        let wrong_name = json!([
            {
                "role": "assistant",
                "tool_calls": [openai_call("call-1", "echo", "{}")],
            },
            {"role": "tool", "tool_call_id": "call-1", "name": "other", "content": ""},
        ]);
        let before = wrong_name.clone();
        assert!(normalize_openai(&registry, &wrong_name).is_err());
        assert_eq!(wrong_name, before);

        let malformed = json!([
            {
                "role": "assistant",
                "tool_calls": [openai_call("call-1", "echo", "{\"x\":1,\"x\":2}")],
            },
            openai_result("call-1", json!("")),
        ]);
        let before = malformed.clone();
        assert!(normalize_openai(&registry, &malformed).is_err());
        assert_eq!(malformed, before);
    }

    #[test]
    fn registry_constructor_error_type_remains_independent() {
        fn declaration_error() -> Result<ToolRegistry, ToolSchemaError> {
            let tools = json!([{
                "type": "function",
                "function": {"name": "bad name", "parameters": {"type": "object"}},
            }]);
            ToolRegistry::from_openai(OpenAiToolsInput {
                tools: Some(&tools),
                tool_choice: None,
                parallel_tool_calls: None,
            })
        }

        assert!(declaration_error().is_err());
    }
}
