//! Transport-neutral incremental parsing for Gemma 4 native tool calls.
//!
//! The scanner owns no HTTP- or dialect-specific policy. It recognizes native
//! model output, preserves bytes that cannot safely become a call as text, and
//! exposes the raw call block and repair telemetry for later adapters.

use std::collections::HashSet;
use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};

/// Gemma 4's canonical native tool-call opener.
pub const TOOL_CALL_OPENER: &str = "<|tool_call>call:";
/// The native tool-call closer.
pub const TOOL_CALL_CLOSER: &str = "<tool_call|>";
const STALE_TOOL_CALL_OPENER: &str = "<|tool_call|>call:";
const NATIVE_QUOTE: &str = "<|\"|>";
const MAX_ARGUMENT_NESTING: usize = 64;
const SCANNER_LOOKBEHIND: usize = STALE_TOOL_CALL_OPENER.len() - 1;

/// Default maximum number of unique calls surfaced from one parser.
pub const DEFAULT_MAX_TOOL_CALLS: usize = 8;
/// Default maximum retained byte length of an in-progress candidate.
pub const DEFAULT_MAX_CANDIDATE_BYTES: usize = 64 * 1024;

/// A parsed tool call, independent of any public protocol's call IDs.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    /// Function name emitted by the model.
    pub name: String,
    /// Parsed argument object.
    pub arguments: Value,
    /// Exact source block, including opener and closer.
    pub raw: String,
    /// Whether the finite repair path was required.
    pub repaired: bool,
}

/// Ordered output from [`ToolCallParser`].
#[derive(Clone, Debug, PartialEq)]
pub enum ToolCallEvent {
    /// Exact model text that was not surfaced as a call.
    Text(String),
    /// A unique call within the configured per-turn limit.
    Call(ToolCall),
}

/// Raw-fidelity counters for one parser lifetime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ToolCallStats {
    /// Candidates successfully parsed, including duplicates and over-limit calls.
    pub parsed: u64,
    /// Successfully parsed candidates using canonical native syntax.
    pub wellformed: u64,
    /// Successfully parsed candidates that used one or both finite repairs.
    pub repaired: u64,
    /// Successfully parsed exact duplicates that were suppressed.
    pub deduped: u64,
    /// Candidates abandoned as text after exceeding the retained-byte cap.
    pub candidate_overflows: u64,
    /// Unique parsed calls surfaced as text because the per-turn cap was reached.
    pub call_limit_exceeded: u64,
}

#[derive(Debug)]
struct Candidate {
    raw: String,
    stale_opener: bool,
    scan_position: usize,
    lexical_state: LexicalState,
}

#[derive(Debug)]
struct DiscardCandidate {
    lexical_state: LexicalState,
    lookbehind: String,
}

#[derive(Clone, Copy, Debug)]
enum LexicalState {
    Normal,
    JsonString { escaped: bool },
    NativeString,
}

impl Candidate {
    fn scan_for_closer(&mut self) -> Option<usize> {
        scan_for_closer(&self.raw, &mut self.scan_position, &mut self.lexical_state)
    }
}

fn scan_for_closer(
    source: &str,
    position: &mut usize,
    lexical_state: &mut LexicalState,
) -> Option<usize> {
    while *position < source.len() {
        let remaining = &source[*position..];
        match *lexical_state {
            LexicalState::Normal => {
                if remaining.starts_with(TOOL_CALL_CLOSER) {
                    *position += TOOL_CALL_CLOSER.len();
                    return Some(*position);
                }
                if remaining.starts_with(NATIVE_QUOTE) {
                    *position += NATIVE_QUOTE.len();
                    *lexical_state = LexicalState::NativeString;
                    continue;
                }
                if TOOL_CALL_CLOSER.starts_with(remaining) || NATIVE_QUOTE.starts_with(remaining) {
                    break;
                }

                let character = remaining
                    .chars()
                    .next()
                    .expect("scan position is before source end");
                *position += character.len_utf8();
                if character == '"' {
                    *lexical_state = LexicalState::JsonString { escaped: false };
                }
            }
            LexicalState::JsonString { escaped } => {
                let character = remaining
                    .chars()
                    .next()
                    .expect("scan position is before source end");
                *position += character.len_utf8();
                *lexical_state = if escaped {
                    LexicalState::JsonString { escaped: false }
                } else if character == '\\' {
                    LexicalState::JsonString { escaped: true }
                } else if character == '"' {
                    LexicalState::Normal
                } else {
                    LexicalState::JsonString { escaped: false }
                };
            }
            LexicalState::NativeString => {
                if remaining.starts_with(NATIVE_QUOTE) {
                    *position += NATIVE_QUOTE.len();
                    *lexical_state = LexicalState::Normal;
                    continue;
                }
                if NATIVE_QUOTE.starts_with(remaining) {
                    break;
                }
                let character = remaining
                    .chars()
                    .next()
                    .expect("scan position is before source end");
                *position += character.len_utf8();
            }
        }
    }
    None
}

/// Incremental, transport-neutral Gemma 4 tool-call parser.
///
/// Feed sequential UTF-8 fragments with [`Self::push`], then call
/// [`Self::finish`] at end of stream. Exact duplicates are suppressed by
/// function name plus recursively key-sorted arguments. Unrecoverable input is
/// always returned as [`ToolCallEvent::Text`].
#[derive(Debug)]
pub struct ToolCallParser {
    pending: String,
    candidate: Option<Candidate>,
    discard: Option<DiscardCandidate>,
    seen: HashSet<(String, String)>,
    surfaced_calls: usize,
    max_tool_calls: usize,
    max_candidate_bytes: usize,
    stats: ToolCallStats,
    #[cfg(test)]
    peak_retained_scanner_bytes: usize,
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self::with_limits(DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_CANDIDATE_BYTES)
    }
}

impl ToolCallParser {
    /// Create a parser with the safe defaults of eight unique calls and a
    /// 64-KiB retained candidate cap.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a parser with explicit per-turn call and retained-candidate caps.
    ///
    /// Zero is valid for either cap: calls or candidates then round-trip as
    /// text and increment the corresponding limit counter.
    #[must_use]
    pub fn with_limits(max_tool_calls: usize, max_candidate_bytes: usize) -> Self {
        Self {
            pending: String::new(),
            candidate: None,
            discard: None,
            seen: HashSet::new(),
            surfaced_calls: 0,
            max_tool_calls,
            max_candidate_bytes,
            stats: ToolCallStats::default(),
            #[cfg(test)]
            peak_retained_scanner_bytes: 0,
        }
    }

    /// Consume the next sequential UTF-8 fragment and return newly available
    /// ordered text/call events.
    pub fn push(&mut self, fragment: &str) -> Vec<ToolCallEvent> {
        let mut events = self.process();
        self.record_retained_peak();
        let mut consumed = 0;

        while consumed < fragment.len() {
            let retained = self.retained_scanner_bytes();
            let bound = self.max_candidate_bytes.saturating_add(SCANNER_LOOKBEHIND);
            let available = bound.saturating_sub(retained);
            let remaining = &fragment[consumed..];
            let take = char_boundary_at_or_before(remaining, available);

            if take == 0 {
                debug_assert!(self.candidate.is_none() && self.discard.is_none());
                let character_len = remaining
                    .chars()
                    .next()
                    .expect("fragment has unconsumed text")
                    .len_utf8();
                let mut possible_opener = std::mem::take(&mut self.pending);
                possible_opener.push_str(&remaining[..character_len]);
                if possible_opener == TOOL_CALL_OPENER || possible_opener == STALE_TOOL_CALL_OPENER
                {
                    consumed += character_len;
                    self.discard = Some(DiscardCandidate {
                        lexical_state: LexicalState::Normal,
                        lookbehind: String::new(),
                    });
                    self.stats.candidate_overflows =
                        self.stats.candidate_overflows.saturating_add(1);
                    emit_text(&mut events, possible_opener);
                } else {
                    possible_opener.truncate(possible_opener.len() - character_len);
                    emit_text(&mut events, possible_opener);
                }
                continue;
            }

            self.pending.push_str(&remaining[..take]);
            consumed += take;
            self.record_retained_peak();
            events.extend(self.process());
            self.record_retained_peak();
        }

        events
    }

    /// Finish the stream, returning any incomplete candidate or delimiter
    /// prefix exactly as text.
    pub fn finish(&mut self) -> Vec<ToolCallEvent> {
        let mut events = self.process();
        if let Some(candidate) = self.candidate.take() {
            emit_text(&mut events, candidate.raw);
        }
        self.discard = None;
        if !self.pending.is_empty() {
            emit_text(&mut events, std::mem::take(&mut self.pending));
        }
        events
    }

    /// Return a snapshot of the parser's saturating telemetry counters.
    #[must_use]
    pub const fn stats(&self) -> ToolCallStats {
        self.stats
    }

    fn process(&mut self) -> Vec<ToolCallEvent> {
        let mut events = Vec::new();

        loop {
            if self.discard.is_some() {
                if self.process_discard(&mut events) {
                    continue;
                }
                break;
            }

            if self.candidate.is_some() {
                if self.process_candidate(&mut events) {
                    continue;
                }
                break;
            }

            let Some((opener_index, stale_opener)) = next_opener(&self.pending) else {
                let keep = opener_lookbehind_len(&self.pending);
                let split = self.pending.len() - keep;
                if split > 0 {
                    let suffix = self.pending.split_off(split);
                    let text = std::mem::replace(&mut self.pending, suffix);
                    emit_text(&mut events, text);
                }
                break;
            };

            if opener_index > 0 {
                let suffix = self.pending.split_off(opener_index);
                let text = std::mem::replace(&mut self.pending, suffix);
                emit_text(&mut events, text);
            }

            let opener = if stale_opener {
                STALE_TOOL_CALL_OPENER
            } else {
                TOOL_CALL_OPENER
            };
            self.pending.drain(..opener.len());
            self.candidate = Some(Candidate {
                raw: opener.to_owned(),
                stale_opener,
                scan_position: 0,
                lexical_state: LexicalState::Normal,
            });
        }

        events
    }

    /// Returns whether scanning can immediately continue.
    fn process_candidate(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let candidate_len = self.candidate.as_ref().map_or(0, |value| value.raw.len());
        if candidate_len > self.max_candidate_bytes {
            let candidate = self
                .candidate
                .take()
                .expect("candidate state checked above");
            self.start_discard(candidate, events);
            return true;
        }

        let available = self.max_candidate_bytes - candidate_len;
        if !self.pending.is_empty() {
            let take = if self.pending.len() <= available {
                self.pending.len()
            } else {
                char_boundary_at_or_before(&self.pending, available)
            };
            let suffix = self.pending.split_off(take);
            let segment = std::mem::replace(&mut self.pending, suffix);
            self.candidate
                .as_mut()
                .expect("candidate state checked above")
                .raw
                .push_str(&segment);
        }

        let closer_end = self
            .candidate
            .as_mut()
            .expect("candidate state checked above")
            .scan_for_closer();
        if let Some(end) = closer_end {
            let mut candidate = self
                .candidate
                .take()
                .expect("candidate state checked above");
            let trailing = candidate.raw.split_off(end);
            if !trailing.is_empty() {
                let mut pending = trailing;
                pending.push_str(&self.pending);
                self.pending = pending;
            }
            self.complete_candidate(candidate, events);
            return true;
        }

        if self.pending.is_empty() {
            false
        } else {
            let candidate = self
                .candidate
                .take()
                .expect("candidate state checked above");
            self.start_discard(candidate, events);
            true
        }
    }

    fn start_discard(&mut self, mut candidate: Candidate, events: &mut Vec<ToolCallEvent>) {
        let _ = candidate.scan_for_closer();
        let lookbehind = candidate.raw[candidate.scan_position..].to_owned();
        self.discard = Some(DiscardCandidate {
            lexical_state: candidate.lexical_state,
            lookbehind,
        });
        self.stats.candidate_overflows = self.stats.candidate_overflows.saturating_add(1);
        emit_text(events, candidate.raw);
    }

    /// Returns whether normal scanning can immediately continue.
    fn process_discard(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        if self.pending.is_empty() {
            return false;
        }

        let mut discard = self.discard.take().expect("discard state checked above");
        let already_emitted = discard.lookbehind.len();
        let mut source = discard.lookbehind;
        source.push_str(&std::mem::take(&mut self.pending));
        let mut position = 0;
        let closer_end = scan_for_closer(&source, &mut position, &mut discard.lexical_state);

        if let Some(end) = closer_end {
            emit_text(events, source[already_emitted..end].to_owned());
            self.pending.push_str(&source[end..]);
            true
        } else {
            emit_text(events, source[already_emitted..].to_owned());
            discard.lookbehind = source[position..].to_owned();
            self.discard = Some(discard);
            false
        }
    }

    fn retained_scanner_bytes(&self) -> usize {
        self.pending
            .len()
            .saturating_add(
                self.candidate
                    .as_ref()
                    .map_or(0, |candidate| candidate.raw.len()),
            )
            .saturating_add(
                self.discard
                    .as_ref()
                    .map_or(0, |discard| discard.lookbehind.len()),
            )
    }

    fn record_retained_peak(&mut self) {
        #[cfg(test)]
        {
            self.peak_retained_scanner_bytes = self
                .peak_retained_scanner_bytes
                .max(self.retained_scanner_bytes());
        }
    }

    fn complete_candidate(&mut self, candidate: Candidate, events: &mut Vec<ToolCallEvent>) {
        let Some(parsed) = parse_candidate(&candidate.raw, candidate.stale_opener) else {
            emit_text(events, candidate.raw);
            return;
        };

        self.stats.parsed = self.stats.parsed.saturating_add(1);
        if parsed.repaired {
            self.stats.repaired = self.stats.repaired.saturating_add(1);
        } else {
            self.stats.wellformed = self.stats.wellformed.saturating_add(1);
        }

        let canonical = canonical_json(&parsed.arguments);
        let dedupe_key = (parsed.name.clone(), canonical);
        if self.seen.contains(&dedupe_key) {
            self.stats.deduped = self.stats.deduped.saturating_add(1);
            return;
        }

        if self.surfaced_calls >= self.max_tool_calls {
            self.stats.call_limit_exceeded = self.stats.call_limit_exceeded.saturating_add(1);
            emit_text(events, candidate.raw);
            return;
        }

        self.seen.insert(dedupe_key);
        self.surfaced_calls = self.surfaced_calls.saturating_add(1);
        events.push(ToolCallEvent::Call(ToolCall {
            name: parsed.name,
            arguments: parsed.arguments,
            raw: candidate.raw,
            repaired: parsed.repaired,
        }));
    }
}

#[derive(Debug)]
struct ParsedCandidate {
    name: String,
    arguments: Value,
    repaired: bool,
}

fn parse_candidate(raw: &str, stale_opener: bool) -> Option<ParsedCandidate> {
    let opener = if stale_opener {
        STALE_TOOL_CALL_OPENER
    } else {
        TOOL_CALL_OPENER
    };
    let body = raw.strip_prefix(opener)?.strip_suffix(TOOL_CALL_CLOSER)?;
    let object_start = body.find('{')?;
    let name = &body[..object_start];
    if name.is_empty() || name.chars().any(char::is_whitespace) {
        return None;
    }
    let argument_source = &body[object_start..];

    if !native_syntax_has_external_whitespace(argument_source)
        && let Some(arguments) = NativeParser::parse_arguments(argument_source)
    {
        return Some(ParsedCandidate {
            name: name.to_owned(),
            arguments: Value::Object(arguments),
            repaired: stale_opener,
        });
    }

    let arguments = serde_json::from_str::<UniqueJsonValue>(argument_source)
        .ok()?
        .0;
    if !arguments.is_object() {
        return None;
    }
    Some(ParsedCandidate {
        name: name.to_owned(),
        arguments,
        repaired: true,
    })
}

fn native_syntax_has_external_whitespace(source: &str) -> bool {
    let mut position = 0;
    let mut in_native_string = false;
    while position < source.len() {
        let remaining = &source[position..];
        if remaining.starts_with(NATIVE_QUOTE) {
            position += NATIVE_QUOTE.len();
            in_native_string = !in_native_string;
            continue;
        }
        let character = remaining
            .chars()
            .next()
            .expect("position is before source end");
        if !in_native_string && character.is_whitespace() {
            return true;
        }
        position += character.len_utf8();
    }
    false
}

struct UniqueJsonValue(Value);

impl<'de> Deserialize<'de> for UniqueJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

struct UniqueJsonVisitor;

impl<'de> Visitor<'de> for UniqueJsonVisitor {
    type Value = UniqueJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .map(UniqueJsonValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<UniqueJsonValue>()? {
            values.push(value.0);
        }
        Ok(UniqueJsonValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some((key, value)) = object.next_entry::<String, UniqueJsonValue>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom("duplicate JSON object key"));
            }
            values.insert(key, value.0);
        }
        Ok(UniqueJsonValue(Value::Object(values)))
    }
}

struct NativeParser<'a> {
    source: &'a str,
    position: usize,
}

impl<'a> NativeParser<'a> {
    fn parse_arguments(source: &'a str) -> Option<Map<String, Value>> {
        let mut parser = Self {
            source,
            position: 0,
        };
        let Value::Object(arguments) = parser.parse_value(0)? else {
            return None;
        };
        parser.skip_whitespace();
        (parser.position == source.len()).then_some(arguments)
    }

    fn parse_value(&mut self, depth: usize) -> Option<Value> {
        self.skip_whitespace();
        let remaining = self.remaining();
        if remaining.starts_with('{') {
            (depth < MAX_ARGUMENT_NESTING)
                .then(|| self.parse_object(depth + 1))?
                .map(Value::Object)
        } else if remaining.starts_with('[') {
            (depth < MAX_ARGUMENT_NESTING)
                .then(|| self.parse_array(depth + 1))?
                .map(Value::Array)
        } else if remaining.starts_with(NATIVE_QUOTE) {
            self.parse_native_string().map(Value::String)
        } else if self.consume_keyword("true") {
            Some(Value::Bool(true))
        } else if self.consume_keyword("false") {
            Some(Value::Bool(false))
        } else if self.consume_keyword("null") {
            Some(Value::Null)
        } else {
            self.parse_number().map(Value::Number)
        }
    }

    fn parse_object(&mut self, depth: usize) -> Option<Map<String, Value>> {
        self.consume_char('{')?;
        self.skip_whitespace();
        let mut object = Map::new();
        if self.consume_char('}').is_some() {
            return Some(object);
        }

        loop {
            self.skip_whitespace();
            let key_start = self.position;
            let colon_offset = self.remaining().find(':')?;
            let key_end = self.position + colon_offset;
            let key = &self.source[key_start..key_end];
            if key.is_empty()
                || key.chars().any(|character| {
                    character.is_whitespace()
                        || matches!(character, '"' | '<' | '>' | '{' | '}' | '[' | ']' | ',')
                })
            {
                return None;
            }
            self.position = key_end + 1;
            let value = self.parse_value(depth)?;
            if object.contains_key(key) {
                return None;
            }
            object.insert(key.to_owned(), value);
            self.skip_whitespace();
            if self.consume_char('}').is_some() {
                return Some(object);
            }
            self.consume_char(',')?;
        }
    }

    fn parse_array(&mut self, depth: usize) -> Option<Vec<Value>> {
        self.consume_char('[')?;
        self.skip_whitespace();
        let mut array = Vec::new();
        if self.consume_char(']').is_some() {
            return Some(array);
        }

        loop {
            array.push(self.parse_value(depth)?);
            self.skip_whitespace();
            if self.consume_char(']').is_some() {
                return Some(array);
            }
            self.consume_char(',')?;
        }
    }

    fn parse_native_string(&mut self) -> Option<String> {
        self.position += NATIVE_QUOTE.len();
        let end_offset = self.remaining().find(NATIVE_QUOTE)?;
        let end = self.position + end_offset;
        let value = self.source[self.position..end].to_owned();
        self.position = end + NATIVE_QUOTE.len();
        Some(value)
    }

    fn parse_number(&mut self) -> Option<serde_json::Number> {
        let end_offset = self
            .remaining()
            .find(|character: char| {
                character.is_whitespace() || matches!(character, ',' | ']' | '}')
            })
            .unwrap_or_else(|| self.remaining().len());
        if end_offset == 0 {
            return None;
        }
        let end = self.position + end_offset;
        let token = &self.source[self.position..end];
        let Value::Number(number) = serde_json::from_str(token).ok()? else {
            return None;
        };
        self.position = end;
        Some(number)
    }

    fn consume_keyword(&mut self, keyword: &str) -> bool {
        if !self.remaining().starts_with(keyword) {
            return false;
        }
        let boundary = self.position + keyword.len();
        if self
            .source
            .get(boundary..)
            .and_then(|remaining| remaining.chars().next())
            .is_some_and(|character| {
                !character.is_whitespace() && !matches!(character, ',' | ']' | '}')
            })
        {
            return false;
        }
        self.position = boundary;
        true
    }

    fn consume_char(&mut self, expected: char) -> Option<()> {
        let actual = self.remaining().chars().next()?;
        if actual != expected {
            return None;
        }
        self.position += actual.len_utf8();
        Some(())
    }

    fn skip_whitespace(&mut self) {
        while let Some(character) = self.remaining().chars().next() {
            if !character.is_whitespace() {
                break;
            }
            self.position += character.len_utf8();
        }
    }

    fn remaining(&self) -> &'a str {
        &self.source[self.position..]
    }
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("JSON string serialization"),
        Value::Array(values) => {
            let values = values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{values}]")
        }
        Value::Object(object) => {
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| *key);
            let entries = entries
                .into_iter()
                .map(|(key, value)| {
                    let key = serde_json::to_string(key).expect("JSON key serialization");
                    format!("{key}:{}", canonical_json(value))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{entries}}}")
        }
    }
}

fn next_opener(source: &str) -> Option<(usize, bool)> {
    let canonical = source.find(TOOL_CALL_OPENER).map(|index| (index, false));
    let stale = source
        .find(STALE_TOOL_CALL_OPENER)
        .map(|index| (index, true));
    match (canonical, stale) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(found), None) | (None, Some(found)) => Some(found),
        (None, None) => None,
    }
}

fn opener_lookbehind_len(source: &str) -> usize {
    let max = source
        .len()
        .min(TOOL_CALL_OPENER.len().max(STALE_TOOL_CALL_OPENER.len()) - 1);
    (1..=max)
        .rev()
        .find(|&length| {
            source.ends_with(&TOOL_CALL_OPENER[..length])
                || source.ends_with(&STALE_TOOL_CALL_OPENER[..length])
        })
        .unwrap_or(0)
}

fn char_boundary_at_or_before(source: &str, byte_limit: usize) -> usize {
    let mut boundary = byte_limit.min(source.len());
    while !source.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

fn emit_text(events: &mut Vec<ToolCallEvent>, text: String) {
    if text.is_empty() {
        return;
    }
    if let Some(ToolCallEvent::Text(previous)) = events.last_mut() {
        previous.push_str(&text);
    } else {
        events.push(ToolCallEvent::Text(text));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn collect(parser: &mut ToolCallParser, fragments: &[&str]) -> Vec<ToolCallEvent> {
        let mut events = Vec::new();
        for fragment in fragments {
            events.extend(parser.push(fragment));
        }
        events.extend(parser.finish());
        events
    }

    fn calls(events: &[ToolCallEvent]) -> Vec<&ToolCall> {
        events
            .iter()
            .filter_map(|event| match event {
                ToolCallEvent::Text(_) => None,
                ToolCallEvent::Call(call) => Some(call),
            })
            .collect()
    }

    fn text(events: &[ToolCallEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                ToolCallEvent::Text(text) => Some(text.as_str()),
                ToolCallEvent::Call(_) => None,
            })
            .collect()
    }

    #[test]
    fn canonical_call_parses_at_every_ascii_split_boundary() {
        let raw = "<|tool_call>call:weather{city:<|\"|>Boston<|\"|>}<tool_call|>";
        for split in 0..=raw.len() {
            let mut parser = ToolCallParser::new();
            let events = collect(&mut parser, &[&raw[..split], &raw[split..]]);
            let parsed = calls(&events);
            assert_eq!(parsed.len(), 1, "split {split}");
            assert_eq!(parsed[0].name, "weather", "split {split}");
            assert_eq!(parsed[0].arguments, json!({"city": "Boston"}));
            assert_eq!(parsed[0].raw, raw, "split {split}");
            assert!(!parsed[0].repaired, "split {split}");
            assert_eq!(parser.stats().parsed, 1, "split {split}");
            assert_eq!(parser.stats().wellformed, 1, "split {split}");
        }

        let fragments = raw
            .as_bytes()
            .iter()
            .map(|byte| std::str::from_utf8(std::slice::from_ref(byte)).unwrap())
            .collect::<Vec<_>>();
        let mut parser = ToolCallParser::new();
        assert_eq!(calls(&collect(&mut parser, &fragments)).len(), 1);
    }

    #[test]
    fn native_values_cover_recursive_types_and_edge_values() {
        let raw = concat!(
            "<|tool_call>call:mix{outer:{z:<|\"|>雪<|\"|>,",
            "a:[<|\"|><|\"|>,-12,1.25,6.02e23,true,false,null]},empty:{}}",
            "<tool_call|>"
        );
        let mut parser = ToolCallParser::new();
        let events = collect(&mut parser, &[raw]);
        let parsed = calls(&events);
        assert_eq!(parsed.len(), 1);
        assert_eq!(
            parsed[0].arguments,
            json!({
                "outer": {"z": "雪", "a": ["", -12, 1.25, 6.02e23, true, false, null]},
                "empty": {}
            })
        );
        assert_eq!(parser.stats().wellformed, 1);
    }

    #[test]
    fn text_and_partial_opener_prefixes_preserve_order() {
        let first = "<|tool_call>call:first{}<tool_call|>";
        let second = "<|tool_call>call:second{}<tool_call|>";
        let mut parser = ToolCallParser::new();
        let events = collect(
            &mut parser,
            &[
                "before<|tool_",
                "call>call:first{}<tool_call|>between<",
                "|tool_call>call:second{}<tool_call|>after",
            ],
        );
        assert_eq!(
            events,
            vec![
                ToolCallEvent::Text("before".to_owned()),
                ToolCallEvent::Call(ToolCall {
                    name: "first".to_owned(),
                    arguments: json!({}),
                    raw: first.to_owned(),
                    repaired: false,
                }),
                ToolCallEvent::Text("between".to_owned()),
                ToolCallEvent::Call(ToolCall {
                    name: "second".to_owned(),
                    arguments: json!({}),
                    raw: second.to_owned(),
                    repaired: false,
                }),
                ToolCallEvent::Text("after".to_owned()),
            ]
        );
    }

    #[test]
    fn adjacent_parallel_calls_preserve_order() {
        let raw = concat!(
            "<|tool_call>call:a{x:1}<tool_call|>",
            "<|tool_call>call:b{x:2}<tool_call|>"
        );
        let mut parser = ToolCallParser::new();
        let events = collect(&mut parser, &[raw]);
        assert_eq!(
            calls(&events)
                .iter()
                .map(|call| call.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(text(&events), "");
    }

    #[test]
    fn canonical_arguments_dedupe_recursively_but_preserve_distinct_calls() {
        let input = concat!(
            "<|tool_call>call:f{a:1,nested:{x:2,y:[3,{z:4}]}}<tool_call|>",
            "<|tool_call>call:f{nested:{y:[3,{z:4}],x:2},a:1}<tool_call|>",
            "<|tool_call>call:f{a:2,nested:{x:2,y:[3,{z:4}]}}<tool_call|>"
        );
        let mut parser = ToolCallParser::new();
        let events = collect(&mut parser, &[input]);
        assert_eq!(calls(&events).len(), 2);
        assert_eq!(calls(&events)[0].arguments["a"], 1);
        assert_eq!(calls(&events)[1].arguments["a"], 2);
        assert_eq!(parser.stats().parsed, 3);
        assert_eq!(parser.stats().wellformed, 3);
        assert_eq!(parser.stats().deduped, 1);
    }

    #[test]
    fn stale_opener_and_ordinary_json_are_finite_repairs() {
        let input = concat!(
            "<|tool_call|>call:stale{value:<|\"|>ok<|\"|>}<tool_call|>",
            "<|tool_call>call:json{\"value\":\"ok\",\"nested\":{\"n\":-1.5e2}}<tool_call|>"
        );
        let mut parser = ToolCallParser::new();
        let events = collect(&mut parser, &[input]);
        let parsed = calls(&events);
        assert_eq!(parsed.len(), 2);
        assert!(parsed.iter().all(|call| call.repaired));
        assert_eq!(parsed[1].arguments["nested"]["n"], -150.0);
        assert_eq!(parser.stats().parsed, 2);
        assert_eq!(parser.stats().wellformed, 0);
        assert_eq!(parser.stats().repaired, 2);
    }

    #[test]
    fn malformed_complete_and_incomplete_eof_round_trip_as_text() {
        let malformed = "<|tool_call>call:x{a:wat}<tool_call|>";
        let incomplete = "<|tool_call>call:y{a:<|\"|>open";
        let input = format!("pre{malformed}mid{incomplete}");
        let mut parser = ToolCallParser::new();
        let events = collect(&mut parser, &[&input]);
        assert!(calls(&events).is_empty());
        assert_eq!(text(&events), input);
        assert_eq!(parser.stats().parsed, 0);
    }

    #[test]
    fn ninth_unique_call_is_text_and_sets_limit_telemetry() {
        let input = (0..9)
            .map(|index| format!("<|tool_call>call:f{{n:{index}}}<tool_call|>"))
            .collect::<String>();
        let ninth = "<|tool_call>call:f{n:8}<tool_call|>";
        let mut parser = ToolCallParser::new();
        let events = collect(&mut parser, &[&input]);
        assert_eq!(calls(&events).len(), 8);
        assert_eq!(text(&events), ninth);
        assert_eq!(parser.stats().parsed, 9);
        assert_eq!(parser.stats().wellformed, 9);
        assert_eq!(parser.stats().call_limit_exceeded, 1);
    }

    #[test]
    fn candidate_overflow_round_trips_and_scanner_recovers() {
        let oversized = "<|tool_call>call:huge{value:<|\"|>abcdefghij<|\"|>}<tool_call|>";
        let later = "<|tool_call>call:ok{}<tool_call|>";
        let input = format!("{oversized}tail{later}");
        let mut parser = ToolCallParser::with_limits(8, oversized.len() - 1);
        let events = collect(&mut parser, &[&input]);
        let parsed = calls(&events);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "ok");
        assert_eq!(text(&events), format!("{oversized}tail"));
        assert_eq!(parser.stats().candidate_overflows, 1);
    }

    #[test]
    fn overflow_at_exact_cap_preserves_a_split_later_opener() {
        let incomplete = "<|tool_call>call:large{value:123456}";
        let outer_closer = "<tool_call|>";
        let later = "<|tool_call>call:ok{}<tool_call|>";
        let mut parser = ToolCallParser::with_limits(8, incomplete.len());
        let mut events = parser.push(incomplete);
        events.extend(parser.push("<tool_call|><"));
        events.extend(parser.push("|tool_call>call:ok{}<tool_call|>"));
        events.extend(parser.finish());

        assert_eq!(calls(&events).len(), 1);
        assert_eq!(calls(&events)[0].name, "ok");
        assert_eq!(text(&events), format!("{incomplete}{outer_closer}"));
        assert_eq!(calls(&events)[0].raw, later);
        assert_eq!(parser.stats().candidate_overflows, 1);
    }

    #[test]
    fn oversized_complete_fragment_is_bounded_and_later_call_survives() {
        let oversized = format!(
            "<|tool_call>call:large{{value:{}}}<tool_call|>",
            "1".repeat(10_000)
        );
        let later = "<|tool_call>call:ok{}<tool_call|>";
        let input = format!("{oversized}{later}");
        let mut parser = ToolCallParser::with_limits(8, 128);
        let events = collect(&mut parser, &[&input]);

        assert_eq!(calls(&events).len(), 1);
        assert_eq!(calls(&events)[0].name, "ok");
        assert_eq!(text(&events), oversized);
        assert_eq!(parser.stats().candidate_overflows, 1);
    }

    #[test]
    fn closer_text_inside_native_and_json_strings_is_not_structural() {
        let native = "<|tool_call>call:native{x:<|\"|>before<tool_call|>after<|\"|>}<tool_call|>";
        let json = "<|tool_call>call:json{\"x\":\"before<tool_call|>after\"}<tool_call|>";
        let mut parser = ToolCallParser::new();
        let events = collect(&mut parser, &[native, json]);
        let parsed = calls(&events);

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].arguments["x"], "before<tool_call|>after");
        assert!(!parsed[0].repaired);
        assert_eq!(parsed[1].arguments["x"], "before<tool_call|>after");
        assert!(parsed[1].repaired);
    }

    #[test]
    fn excessive_native_nesting_degrades_to_text_without_panicking() {
        let nested = format!(
            "{}0{}",
            "[".repeat(MAX_ARGUMENT_NESTING + 1),
            "]".repeat(MAX_ARGUMENT_NESTING + 1)
        );
        let raw = format!("<|tool_call>call:deep{{value:{nested}}}<tool_call|>");
        let mut parser = ToolCallParser::new();
        let events = collect(&mut parser, &[&raw]);

        assert!(calls(&events).is_empty());
        assert_eq!(text(&events), raw);
        assert_eq!(parser.stats().parsed, 0);
    }

    #[test]
    fn multi_megabyte_single_push_keeps_parser_staging_bounded() {
        let cap = 32;
        let raw = format!(
            "<|tool_call>call:large{{value:<|\"|>{}<|\"|>}}<tool_call|>",
            "x".repeat(2 * 1024 * 1024)
        );
        let mut parser = ToolCallParser::with_limits(8, cap);
        let events = parser.push(&raw);

        assert!(calls(&events).is_empty());
        assert_eq!(text(&events), raw);
        assert!(parser.candidate.is_none());
        assert!(parser.discard.is_none());
        assert!(parser.pending.len() <= SCANNER_LOOKBEHIND);
        assert!(parser.peak_retained_scanner_bytes <= cap + SCANNER_LOOKBEHIND);
        assert_eq!(parser.stats().candidate_overflows, 1);
    }

    #[test]
    fn oversized_native_string_ignores_nested_call_until_outer_closer() {
        let nested = "<|tool_call>call:evil{}<tool_call|>";
        let outer = format!(
            "<|tool_call>call:outer{{value:<|\"|>{nested}{}<|\"|>}}<tool_call|>",
            "x".repeat(128)
        );
        let later = "<|tool_call>call:ok{}<tool_call|>";
        let input = format!("{outer}{later}");
        let mut parser = ToolCallParser::with_limits(8, 48);
        let events = collect(&mut parser, &[&input]);

        assert_eq!(calls(&events).len(), 1);
        assert_eq!(calls(&events)[0].name, "ok");
        assert_eq!(text(&events), outer);
        assert_eq!(parser.stats().candidate_overflows, 1);
    }

    #[test]
    fn oversized_json_string_preserves_lexical_state_across_one_byte_fragments() {
        let nested = "<|tool_call>call:evil{}<tool_call|>";
        let outer = format!(
            "<|tool_call>call:outer{{\"value\":\"prefix \\\"quoted\\\" <|\\\"|> {nested} {}\"}}<tool_call|>",
            "x".repeat(128)
        );
        let later = "<|tool_call>call:ok{}<tool_call|>";
        let input = format!("{outer}{later}");
        let mut parser = ToolCallParser::with_limits(8, 56);
        let fragments = input
            .as_bytes()
            .iter()
            .map(|byte| std::str::from_utf8(std::slice::from_ref(byte)).unwrap())
            .collect::<Vec<_>>();
        let events = collect(&mut parser, &fragments);

        assert_eq!(calls(&events).len(), 1);
        assert_eq!(calls(&events)[0].name, "ok");
        assert_eq!(text(&events), outer);
        assert_eq!(parser.stats().candidate_overflows, 1);
    }

    #[test]
    fn duplicate_keys_in_native_or_json_objects_round_trip_as_text() {
        let native_top = "<|tool_call>call:native{a:1,a:2}<tool_call|>";
        let native_nested = "<|tool_call>call:native_nested{outer:{x:1,x:2}}<tool_call|>";
        let json_top = "<|tool_call>call:json{\"a\":1,\"a\":2}<tool_call|>";
        let json_nested = concat!(
            "<|tool_call>call:json_nested{\"outer\":{\"x\":1,\"x\":2}}",
            "<tool_call|>"
        );
        let input = format!("{native_top}{native_nested}{json_top}{json_nested}");
        let mut parser = ToolCallParser::new();
        let events = collect(&mut parser, &[&input]);

        assert!(calls(&events).is_empty());
        assert_eq!(text(&events), input);
        assert_eq!(parser.stats().parsed, 0);
        assert_eq!(parser.stats().wellformed, 0);
        assert_eq!(parser.stats().repaired, 0);
    }

    #[test]
    fn whitespace_bearing_unquoted_native_syntax_is_not_wellformed() {
        let unquoted = "<|tool_call>call:native{a: 1}<tool_call|>";
        let json = "<|tool_call>call:json{ \"a\": 1 }<tool_call|>";
        let input = format!("{unquoted}{json}");
        let mut parser = ToolCallParser::new();
        let events = collect(&mut parser, &[&input]);

        assert_eq!(calls(&events).len(), 1);
        assert_eq!(calls(&events)[0].name, "json");
        assert!(calls(&events)[0].repaired);
        assert_eq!(text(&events), unquoted);
        assert_eq!(parser.stats().parsed, 1);
        assert_eq!(parser.stats().wellformed, 0);
        assert_eq!(parser.stats().repaired, 1);
    }

    #[test]
    fn tiny_cap_unicode_after_partial_stale_opener_round_trips_once() {
        let input = "<|tool_call|>call雪text";
        let mut parser = ToolCallParser::with_limits(8, 1);
        let events = collect(&mut parser, &[input]);

        assert!(calls(&events).is_empty());
        assert_eq!(text(&events), input);
        assert_eq!(parser.stats().candidate_overflows, 0);
        assert!(parser.peak_retained_scanner_bytes <= 1 + SCANNER_LOOKBEHIND);
    }

    #[test]
    fn zero_cap_stale_opener_enters_bounded_discard() {
        let raw = "<|tool_call|>call:x{}<tool_call|>";
        let mut parser = ToolCallParser::with_limits(8, 0);
        let events = collect(&mut parser, &[raw]);

        assert!(calls(&events).is_empty());
        assert_eq!(text(&events), raw);
        assert_eq!(parser.stats().candidate_overflows, 1);
        assert!(parser.peak_retained_scanner_bytes <= SCANNER_LOOKBEHIND);
    }

    #[test]
    fn unmatched_and_malformed_inputs_do_not_panic() {
        let cases = [
            "<",
            "<|",
            "<|tool_call>",
            "<|tool_call>call:",
            "<|tool_call>call:x{{{{",
            "<|tool_call>call:x{a:[1,}<tool_call|>",
            "<|tool_call>call:x{a:<|\"|>unterminated}<tool_call|>",
            "<|tool_call>call:x{a:truex}<tool_call|>",
        ];
        for input in cases {
            let mut parser = ToolCallParser::new();
            let events = collect(&mut parser, &[input]);
            assert!(calls(&events).is_empty(), "{input}");
            assert_eq!(text(&events), input, "{input}");
        }
    }

    #[test]
    fn zero_caps_are_safe_and_saturating_counters_are_observable() {
        let raw = "<|tool_call>call:x{}<tool_call|>";
        let mut parser = ToolCallParser::with_limits(0, 0);
        parser.stats.candidate_overflows = u64::MAX;
        let events = collect(&mut parser, &[raw]);
        assert_eq!(text(&events), raw);
        assert_eq!(parser.stats().candidate_overflows, u64::MAX);
    }
}
