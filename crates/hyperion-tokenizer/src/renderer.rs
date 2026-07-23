//! The chat-template renderer (M3) — a real Jinja2 evaluation via minijinja.
//!
//! This is the deferred half of the M3 "in-process tokenizer + template golden
//! fixtures" deliverable (`10:40`). The tokenizer half (PR #20) sealed
//! encode/decode parity; this half renders the gemma4 `chat_template.jinja` so
//! the request path never shells out to Python (`01:49`, `04:80-82`).
//!
//! ## The minijinja `.get()` gap (and the fix)
//!
//! The gemma4 `chat_template.jinja` calls dict `.get()` extensively
//! (`message.get('reasoning')`, `follow.get('name')`, `tc.get('id')`, …) and
//! chains it with `or`: `message.get('reasoning') or message.get('reasoning_content')`
//! (lines 239–240, 287, 289, 294, 300–301, 324, 330, 332, 338…). minijinja's
//! built-in map value exposes attribute/`[]` lookup but **not** `.get()` as a
//! method, and serde conversion cannot add methods. The fix chosen (per the M3
//! handoff) is a **custom minijinja `Object`** that wraps a `serde_json::Value`
//! map and dispatches `.get(key[, default])` via `call_method`, while exposing
//! every field through `get_value` for `message['role']`/`message.role` and
//! `enumerate` for `| dictsort`. The wrapping is **recursive**: the template
//! reaches nested maps (`tool_call['function']`, `value['items']['required']`),
//! and *those* must also support `.get()`, so every `serde_json::Value::Object`
//! in the tree becomes a [`JsonMapObject`]; arrays/scalars/strings fall through
//! to minijinja's default serde conversion (which is correct — `is sequence`
//! must stay true for arrays, `is string` for strings, etc.).
//!
//! `raise_exception(msg)` (template line 258) is not a minijinja builtin, so it
//! is registered as a function that errors with the message — matching the
//! transformers behavior the template was authored against.

use std::path::Path;
use std::sync::Arc;

use minijinja::value::{Enumerator, Object, ObjectRepr, Value};
use minijinja::{Environment, Error, ErrorKind};

use crate::TokenizerHandle;

/// Read `bos_token`/`eos_token` from a model artifact's `tokenizer_config.json`.
/// These are JSON values (transformers stores them as either bare strings or
/// `AddedToken` objects); we coerce to the bare string the template emits. The
/// 12B stores them as the bare strings `"<bos>"`/`"<eos>"`.
fn read_special_tokens(artifact_dir: &Path) -> Result<(String, String), RenderError> {
    let cfg_path = artifact_dir.join("tokenizer_config.json");
    let raw = std::fs::read_to_string(&cfg_path)
        .map_err(|e| RenderError::Load(format!("{cfg_path:?}: {e}")))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| RenderError::Load(format!("{cfg_path:?}: {e}")))?;
    let bos = special_token_string(&cfg, "bos_token").unwrap_or_else(|| "<bos>".to_string());
    let eos = special_token_string(&cfg, "eos_token").unwrap_or_else(|| "<eos>".to_string());
    Ok((bos, eos))
}

/// Extract a special-token string from the config: a bare string
/// (`"<bos>"`) or an `AddedToken` object (`{"content": "<bos>", ...}`).
fn special_token_string(cfg: &serde_json::Value, key: &str) -> Option<String> {
    match cfg.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(map) => map
            .get("content")
            .and_then(|v| v.as_str())
            .map(String::from),
        _ => None,
    }
}

/// A chat-template load error.
#[derive(Debug)]
pub enum RenderError {
    /// The template file could not be read.
    Load(String),
    /// minijinja failed to parse the template.
    Parse(String),
    /// minijinja failed to render (a runtime error, e.g. `raise_exception`).
    Render(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(s) => write!(f, "template load error: {s}"),
            Self::Parse(s) => write!(f, "template parse error: {s}"),
            Self::Render(s) => write!(f, "template render error: {s}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// A compiled chat template bound to a tokenizer (for `bos_token`/`eos_token`
/// substitution and the `has_thinking` default). Cheap to clone — the
/// `Environment` is shared behind an `Arc`.
#[derive(Clone)]
pub struct ChatTemplate {
    env: Arc<Environment<'static>>,
    /// The template source name (used for auto-escape selection; minijinja
    /// defaults to no escaping for unknown extensions, which is what we want —
    /// the gemma4 template emits raw `<|...|>` special tokens and must NOT be
    /// HTML-escaped).
    name: Arc<str>,
    /// The model's `has_thinking` default (mlx-lm: true iff the vocab has
    /// `<|channel>` AND `<channel|>`). Used to resolve
    /// `RenderOptions::enable_thinking == None`, mirroring the oracle's
    /// `apply_chat_template` default.
    has_thinking: bool,
}

/// The conversation message shape the renderer accepts. A mirror of the
/// transformers `apply_chat_template` message dict: `role` + `content`
/// (string or content-parts array) + the optional `tool_calls`/
/// `tool_responses`/`reasoning` keys the gemma4 template reads via `.get()`.
/// Serialized to `serde_json::Value` and recursively wrapped so every nested
/// map — including caller-supplied `tool_calls[].function.arguments` —
/// supports `.get()`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_responses: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<serde_json::Value>,
    /// OpenAI Chat-Completions tool-result fields (role="tool").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Render parameters mirroring the transformers `apply_chat_template` kwargs
/// the gemma4 template consumes.
#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    /// Whether to append the generation prompt (`<|turn>model\n…`).
    pub add_generation_prompt: bool,
    /// Enable the thinking channel (the `<|think|>` injection at the first
    /// system turn). `None` → the model default: mlx-lm's `has_thinking`
    /// (true iff the vocab has `<|channel>` + `<channel|>`, the gemma4 multi-
    /// token thinking mode — true for both the 12B and E4B). The gemma4
    /// template's own `enable_thinking | default(false)` is overridden by the
    /// model wrapper to `has_thinking`, so `None` mirrors the oracle's actual
    /// default behavior (verified: `apply_chat_template` with no
    /// `enable_thinking` emits the `<|think|>` block). Pass an explicit
    /// `Some(false)` to suppress thinking (the `<|channel>thought` generation
    /// prompt).
    pub enable_thinking: Option<bool>,
    /// Preserve thinking blocks across tool-call chains within the current
    /// turn. `None` → `false` (the template's `preserve_thinking | default(false)`;
    /// mlx-lm does not override this one).
    pub preserve_thinking: Option<bool>,
    /// Tool declarations to render into the `<|tool|>` block (the OpenAI
    /// `tools` array; each entry `{function: {name, description, parameters,
    /// response?}}`).
    pub tools: Vec<serde_json::Value>,
}

impl ChatTemplate {
    /// Load and compile a chat template from a `chat_template.jinja` file in a
    /// model artifact directory. `artifact_dir` must also contain
    /// `tokenizer_config.json` (for `bos_token`/`eos_token`, which the template
    /// references via `{{ bos_token }}`); the `tokenizer.json` there provides
    /// the vocab for `has_token`. The `tokenizer` supplies the model's
    /// `has_thinking` default (true iff the vocab has `<|channel>` +
    /// `<channel|>`); pass `None` only for tests that don't exercise the
    /// thinking branch (defaults to `false`).
    pub fn from_artifact(
        artifact_dir: &Path,
        tokenizer: Option<&TokenizerHandle>,
    ) -> Result<Self, RenderError> {
        let template_path = artifact_dir.join("chat_template.jinja");
        let source = std::fs::read_to_string(&template_path)
            .map_err(|e| RenderError::Load(format!("{template_path:?}: {e}")))?;
        let (bos, eos) = read_special_tokens(artifact_dir)?;
        let has_thinking = tokenizer.is_some_and(model_has_thinking);
        Self::from_source_with_thinking(&source, &bos, &eos, has_thinking)
    }

    /// Compile a chat template from an inline source string with explicit
    /// bos/eos token strings. The `has_thinking` default is `false` (use
    /// [`ChatTemplate::from_source_with_thinking`] to set it).
    pub fn from_source(
        source: &str,
        bos_token: &str,
        eos_token: &str,
    ) -> Result<Self, RenderError> {
        Self::from_source_with_thinking(source, bos_token, eos_token, false)
    }

    /// Compile a chat template from an inline source string with the model's
    /// `has_thinking` default (used to resolve `RenderOptions::enable_thinking
    /// == None`, mirroring mlx-lm's `apply_chat_template`).
    pub fn from_source_with_thinking(
        source: &str,
        bos_token: &str,
        eos_token: &str,
        has_thinking: bool,
    ) -> Result<Self, RenderError> {
        let mut env: Environment<'static> = Environment::new();
        // The gemma4 template emits raw special-token bytes (`<|turn>`,
        // `<|channel>`, `<|"|>`); auto-escaping would corrupt them. minijinja's
        // default auto-escape callback only escapes known HTML extensions, and
        // our template name carries none, so this is a no-op — but explicit is
        // safer (the M2 `near_tie_events` discipline: don't rely on a default
        // staying a default).
        env.set_auto_escape_callback(|_name| minijinja::AutoEscape::None);
        // `raise_exception(msg)` — the gemma4 template calls it (line 258) on a
        // malformed `tool_calls[].function.arguments`. transformers raises a
        // Python exception with the message; we surface the same message as a
        // render error.
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        let name: Arc<str> = Arc::from("chat_template.jinja");
        env.add_template_owned::<_, _>(name.to_string(), source.to_string())
            .map_err(|e| RenderError::Parse(e.to_string()))?;
        env.add_global("bos_token", Value::from(bos_token));
        env.add_global("eos_token", Value::from(eos_token));
        Ok(Self {
            env: Arc::new(env),
            name,
            has_thinking,
        })
    }

    /// Render a conversation. Returns the templated prompt string (the bytes
    /// the encoder then tokenizes — BOS + the templated body).
    pub fn render(
        &self,
        messages: &[ChatMessage],
        options: &RenderOptions,
    ) -> Result<String, RenderError> {
        // Build the context as a serde_json::Value tree, then recursively wrap
        // every Object so `.get()` works on all nested maps. `render` passes a
        // `Value` context through WITHOUT re-serializing (per the minijinja
        // docs), so the custom Objects survive into evaluation.
        let messages_val = serde_json::to_value(messages)
            .map_err(|e| RenderError::Render(format!("message serialization: {e}")))?;
        let tools_val = serde_json::to_value(&options.tools)
            .map_err(|e| RenderError::Render(format!("tools serialization: {e}")))?;
        // Resolve the Option<bool> defaults against the model's has_thinking,
        // mirroring mlx-lm's apply_chat_template (enable_thinking defaults to
        // has_thinking; preserve_thinking defaults to false — the template's own
        // default, which mlx-lm does NOT override).
        let enable_thinking = options.enable_thinking.unwrap_or(self.has_thinking);
        let preserve_thinking = options.preserve_thinking.unwrap_or(false);
        let ctx = serde_json::json!({
            "messages": messages_val,
            "tools": tools_val,
            "add_generation_prompt": options.add_generation_prompt,
            "enable_thinking": enable_thinking,
            "preserve_thinking": preserve_thinking,
        });
        let wrapped = json_to_minijinja(&ctx, 0)?;
        self.env
            .get_template(self.name.as_ref())
            .map_err(|e| RenderError::Parse(e.to_string()))?
            .render(wrapped)
            .map_err(|e| RenderError::Render(e.to_string()))
    }
}

/// The model's `has_thinking` flag (mlx-lm `tokenizer_utils.py`: true iff the
/// vocab contains `<|channel>` AND `<channel|>`, the gemma4 multi-token
/// thinking mode). The single-token `THINK_TOKENS` mode (chatml-style
/// start/end tags) is not exercised by the gemma4 family, so it's omitted
/// here; add it if a model that needs it joins the test path.
fn model_has_thinking(tokenizer: &TokenizerHandle) -> bool {
    tokenizer.has_token("<|channel>") && tokenizer.has_token("<channel|>")
}

/// The maximum JSON nesting [`json_to_minijinja`] will walk before failing
/// closed. This is the guard against a deep caller-/network-supplied
/// `tool_calls.arguments` overflowing the stack (which `panic = "abort"`
/// turns into a hard process crash, not a catchable error). It is far above
/// any depth the gemma4 template reaches (~6 levels) and far below
/// `serde_json`'s 128-deep `from_str` parser bound.
const MAX_NESTING_DEPTH: u32 = 64;

/// The recursive core: turn a `serde_json::Value` into a minijinja `Value`,
/// wrapping every Object in [`JsonMapObject`] and recursing into arrays/maps.
///
/// Recursion is **depth-bounded**: a `serde_json::Value` can carry an
/// arbitrarily-deep tree, and the `Value` that reaches here may bypass
/// `serde_json::from_str`'s 128-deep guard (the `serde_json::to_value` path
/// [`ChatTemplate::render`] uses does no depth checking). An unbounded walk
/// would overflow the stack, and the release profile's `panic = "abort"`
/// turns that overflow into a hard process crash. Past [`MAX_NESTING_DEPTH`]
/// we fail closed with a [`RenderError::Render`] instead.
fn json_to_minijinja(v: &serde_json::Value, depth: u32) -> Result<Value, RenderError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(RenderError::Render(format!(
            "chat-template context nesting exceeds {MAX_NESTING_DEPTH} levels"
        )));
    }
    match v {
        serde_json::Value::Object(map) => {
            // Recursively wrap each value so nested maps also get `.get()`.
            let entries: Vec<(String, Value)> = map
                .iter()
                .map(|(k, child)| Ok((k.clone(), json_to_minijinja(child, depth + 1)?)))
                .collect::<Result<_, _>>()?;
            Ok(Value::from_object(JsonMapObject { entries }))
        }
        serde_json::Value::Array(arr) => {
            // A sequence of (possibly map) values. Build a minijinja seq.
            let items: Vec<Value> = arr
                .iter()
                .map(|child| json_to_minijinja(child, depth + 1))
                .collect::<Result<_, _>>()?;
            Ok(Value::from(items))
        }
        // Scalars/None fall through to minijinja's native conversions so the
        // `is string`/`is none`/`is boolean`/`is number` tests stay exact.
        serde_json::Value::String(s) => Ok(Value::from(s.as_str())),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::from(i))
            } else if let Some(u) = n.as_u64() {
                Ok(Value::from(u))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::from(f))
            } else {
                Ok(Value::from(n.to_string()))
            }
        }
        serde_json::Value::Bool(b) => Ok(Value::from(*b)),
        serde_json::Value::Null => Ok(Value::from(())),
    }
}

/// A minijinja `Object` wrapping a JSON object map. Supports:
/// - `message['role']` / `message.role` via `get_value`
/// - `message.get('reasoning')` / `message.get('reasoning', default)` via
///   `call_method` (returns `Value::UNDEFINED` for a missing key with no
///   default, which is falsy → `a.get('x') or b.get('y')` works)
/// - `| dictsort` / iteration via `enumerate` (yields the keys)
/// - `is mapping` via the default `ObjectRepr::Map`
///
/// This is the load-bearing fix for the gemma4 `.get()` gap.
#[derive(Debug)]
struct JsonMapObject {
    /// The (key, value) pairs in insertion order (dictsort is stable).
    entries: Vec<(String, Value)>,
}

impl Object for JsonMapObject {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        let key = key.as_str()?;
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    }

    fn get_value_by_str(self: &Arc<Self>, key: &str) -> Option<Value> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        // `| dictsort` needs the keys; yield them as values. dictsort then
        // re-fetches each via get_value. We use the Iter enumerator because the
        // keys are dynamic (caller-supplied), not a static slice.
        let keys: Vec<Value> = self
            .entries
            .iter()
            .map(|(k, _)| Value::from(k.as_str()))
            .collect();
        Enumerator::Iter(Box::new(keys.into_iter()))
    }

    fn call_method(
        self: &Arc<Self>,
        _state: &minijinja::State<'_, '_>,
        method: &str,
        args: &[Value],
    ) -> Result<Value, Error> {
        match method {
            "get" => {
                // Python dict.get(key[, default]): missing key → default, or
                // None if no default given. The gemma4 template relies on the
                // `or`-chainable falsiness of a missing key, so we return
                // `Value::from(())` (None, not UNDEFINED) when no default is
                // supplied — matching `dict.get` exactly (Python returns None,
                // not a sentinel, and `None or x` → x).
                let key = args.first().and_then(|v| v.as_str()).ok_or_else(|| {
                    Error::new(ErrorKind::InvalidOperation, "get() requires a key")
                })?;
                let default = args.get(1).cloned().unwrap_or(Value::from(()));
                Ok(self.get_value(&Value::from(key)).unwrap_or(default))
            }
            "items" => {
                // dict.items() → list of [key, value] pairs (the gemma4 template
                // uses `| dictsort` instead, but `items()` is the natural
                // complement and transformers exposes it).
                let pairs: Vec<Value> = self
                    .entries
                    .iter()
                    .map(|(k, v)| Value::from(vec![Value::from(k.as_str()), v.clone()]))
                    .collect();
                Ok(Value::from(pairs))
            }
            "keys" => {
                let keys: Vec<Value> = self
                    .entries
                    .iter()
                    .map(|(k, _)| Value::from(k.as_str()))
                    .collect();
                Ok(Value::from(keys))
            }
            "values" => {
                let vals: Vec<Value> = self.entries.iter().map(|(_, v)| v.clone()).collect();
                Ok(Value::from(vals))
            }
            _ => Err(Error::new(
                ErrorKind::UnknownMethod,
                format!("unknown method: {method}"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gemma4 template's `.get()`-or-`.get()` pattern (line 239) must
    /// resolve: a present `reasoning` wins; a missing key falls through to the
    /// next `.get()`; an all-missing chain yields None (falsy). Rendered through
    /// a real minijinja template — the exact code path the gemma4 template
    /// exercises. This is the load-bearing test for the `.get()` gap fix.
    #[test]
    fn json_map_get_or_chain_resolves_like_python_dict() {
        let tpl = ChatTemplate::from_source(
            "{{ message.get('reasoning') or message.get('reasoning_content') }}",
            "<bos>",
            "<eos>",
        )
        .expect("template compiles");
        // Present reasoning wins.
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: None,
            tool_responses: None,
            reasoning: Some(serde_json::Value::String("because I said so".to_string())),
            reasoning_content: None,
            tool_call_id: None,
            name: None,
        };
        // Wrap a single message the way `render` does.
        let ctx = serde_json::json!({ "message": serde_json::to_value(&msg).unwrap() });
        let wrapped = json_to_minijinja(&ctx, 0).unwrap();
        let out = tpl
            .env
            .get_template("chat_template.jinja")
            .unwrap()
            .render(wrapped)
            .unwrap();
        assert_eq!(out, "because I said so");

        // Missing reasoning → falls through to reasoning_content.
        let msg2 = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: None,
            tool_responses: None,
            reasoning: None,
            reasoning_content: Some(serde_json::Value::String("fallback".to_string())),
            tool_call_id: None,
            name: None,
        };
        let ctx2 = serde_json::json!({ "message": serde_json::to_value(&msg2).unwrap() });
        let out2 = tpl
            .env
            .get_template("chat_template.jinja")
            .unwrap()
            .render(json_to_minijinja(&ctx2, 0).unwrap())
            .unwrap();
        assert_eq!(out2, "fallback");

        // Both missing → `None or None` evaluates to None. The gemma4 template
        // never renders this bare (line 239 `set`s it; line 241 guards with `if
        // thinking_text`), so the None→"none" render divergence vs Python's
        // "None" is NOT exercised by the real template. What matters is that the
        // `if`-guard sees it as falsy — verified by the next assertion.
        let msg3 = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: None,
            tool_responses: None,
            reasoning: None,
            reasoning_content: None,
            tool_call_id: None,
            name: None,
        };
        let ctx3 = serde_json::json!({ "message": serde_json::to_value(&msg3).unwrap() });
        let out3 = tpl
            .env
            .get_template("chat_template.jinja")
            .unwrap()
            .render(json_to_minijinja(&ctx3, 0).unwrap())
            .unwrap();
        // minijinja renders None as "none" (lowercase); Python Jinja renders
        // "None". The gemma4 template guards this value with `if`, so the raw
        // render output is never emitted — we assert the value itself is None
        // (falsy) via the `if` test below instead of the rendered string.
        assert_eq!(out3, "none");

        // The critical property: a None result is FALSY in `if`, so the gemma4
        // `if thinking_text` guard correctly skips. This is the behavior the
        // template actually depends on.
        let guard_tpl = ChatTemplate::from_source(
            "{% if message.get('reasoning') or message.get('reasoning_content') %}YES{% else %}NO{% endif %}",
            "<bos>",
            "<eos>",
        )
        .expect("guard template compiles");
        let guard_ctx = serde_json::json!({ "message": serde_json::to_value(&msg3).unwrap() });
        let guard_out = guard_tpl
            .env
            .get_template("chat_template.jinja")
            .unwrap()
            .render(json_to_minijinja(&guard_ctx, 0).unwrap())
            .unwrap();
        assert_eq!(
            guard_out, "NO",
            "a missing key chain is falsy → the if-guard skips"
        );
    }

    /// `is mapping` must be true for our wrapped object and false for a string
    /// or array — the gemma4 template branches on `value['items'] is mapping`
    /// and `message['content'] is sequence`. Verified through real `is` tests.
    #[test]
    fn wrapped_value_reports_correct_kinds() {
        let tpl = ChatTemplate::from_source(
            "{{ m is mapping }}|{{ a is sequence }}|{{ s is string }}|{{ n is none }}",
            "<bos>",
            "<eos>",
        )
        .expect("template compiles");
        let ctx = serde_json::json!({
            "m": {"a": 1},
            "a": [1, 2, 3],
            "s": "hi",
            "n": null,
        });
        let out = tpl
            .env
            .get_template("chat_template.jinja")
            .unwrap()
            .render(json_to_minijinja(&ctx, 0).unwrap())
            .unwrap();
        assert_eq!(out, "true|true|true|true");
    }

    /// `dictsort` over a wrapped map must yield key-value pairs in insertion
    /// order (the gemma4 `format_parameters` macro iterates `properties |
    /// dictsort`). And `items()` must return [k, v] pairs.
    #[test]
    fn dictsort_and_items_work_on_wrapped_maps() {
        let tpl = ChatTemplate::from_source(
            "{% for k, v in m | dictsort %}{{ k }}={{ v }};{% endfor %}|{{ m.items() | length }}",
            "<bos>",
            "<eos>",
        )
        .expect("template compiles");
        let ctx = serde_json::json!({ "m": {"b": 2, "a": 1} });
        let out = tpl
            .env
            .get_template("chat_template.jinja")
            .unwrap()
            .render(json_to_minijinja(&ctx, 0).unwrap())
            .unwrap();
        // dictsort sorts by key → a=1;b=2; then items() length = 2.
        assert_eq!(out, "a=1;b=2;|2");
    }

    /// `json_to_minijinja` must fail closed (not overflow the stack) on a
    /// deeply-nested context. This is the adversarial-review fix: the
    /// `serde_json::to_value` path `render()` uses bypasses `from_str`'s
    /// 128-deep guard, so without an explicit bound a caller/network-supplied
    /// `tool_calls.arguments` tree could overflow the stack — which
    /// `panic = "abort"` turns into a hard process crash. We build a Value
    /// nest deeper than `MAX_NESTING_DEPTH` *without* going through `from_str`
    /// (constructing it in-memory, exactly as a deserialized-then-rebuilt
    /// request body would), and assert `json_to_minijinja` returns a
    /// `RenderError::Render` rather than recursuring to a crash.
    #[test]
    fn json_to_minijinja_fails_closed_on_excessive_nesting() {
        // Build a value nested far deeper than MAX_NESTING_DEPTH (64) in memory,
        // bypassing serde_json::from_str's 128-deep parser guard — the path a
        // rebuilt request body takes. `{"a": {"a": {...}}}`.
        let mut deep = serde_json::Value::String("bottom".to_string());
        for _ in 0..200 {
            deep = serde_json::Value::Object(std::iter::once(("a".to_string(), deep)).collect());
        }
        let err = json_to_minijinja(&deep, 0).expect_err("over-depth nest must fail closed");
        let msg = err.to_string();
        assert!(
            msg.contains("nesting exceeds"),
            "error should name the depth bound: {msg}"
        );

        // A shallow nest well under the bound still succeeds (sanity).
        let shallow = serde_json::json!({"a": {"b": {"c": 1}}});
        json_to_minijinja(&shallow, 0).expect("shallow nest wraps fine");
    }

    /// A minimal end-to-end render: a one-turn user message with the generation
    /// prompt appended. Exercises bos_token, the turn markers, and
    /// add_generation_prompt — the simplest path through the real gemma4
    /// template. Uses the committed canonical template (M5-gated on the
    /// artifact being present; self-skips on CI).
    #[test]
    fn renders_simple_user_turn_against_canonical_template() {
        let dir = match std::env::var("HYPERION_12B_ARTIFACT") {
            Ok(d) if !d.is_empty() => d,
            _ => {
                eprintln!("renderer test: HYPERION_12B_ARTIFACT unset; skipping (M5-gated)");
                return;
            }
        };
        let artifact = std::path::Path::new(&dir);
        let tok = TokenizerHandle::from_file(&artifact.join("tokenizer.json"))
            .expect("12B tokenizer.json loads");
        let tpl =
            ChatTemplate::from_artifact(artifact, Some(&tok)).expect("canonical template compiles");
        let prompt = tpl
            .render(
                &[ChatMessage {
                    role: "user".to_string(),
                    content: Some(serde_json::Value::String("Hello".to_string())),
                    tool_calls: None,
                    tool_responses: None,
                    reasoning: None,
                    reasoning_content: None,
                    tool_call_id: None,
                    name: None,
                }],
                &RenderOptions {
                    add_generation_prompt: true,
                    ..Default::default()
                },
            )
            .expect("render succeeds");
        // BOS + the user turn + the model generation prompt. The exact bytes
        // are the 21-id golden target's prefix; a full parity check belongs in
        // the M5 golden test, but the shape must be right.
        assert!(
            prompt.starts_with("<bos>"),
            "starts with bos_token: {prompt:?}"
        );
        assert!(
            prompt.contains("<|turn>user\n"),
            "opens a user turn: {prompt:?}"
        );
        assert!(prompt.contains("Hello"), "carries the content: {prompt:?}");
        assert!(
            prompt.ends_with("<|turn>model\n"),
            "ends with the generation prompt: {prompt:?}"
        );
    }
}
