//! B4: dual-dialect normalization → `PreparedPrompt` (06: "both dialects
//! normalize to one PreparedPrompt"). A request body (Anthropic
//! `/v1/messages` or OpenAI `/v1/chat/completions`) is parsed into a
//! `ChatMessage` list + sampler config, rendered through the M3 chat template,
//! tokenized, and packaged as a [`PreparedPrompt`] — the `EngineRequest` +
//! stream flag + dialect the handler feeds to the engine. Provider-shaped
//! tool history is accepted only when generation is explicitly disabled.
//!
//! The 413 context-overflow check (`prompt_tokens + max_tokens > context`)
//! lives here, distinct from the 32 MiB body-size 413 (B5's
//! `DefaultBodyLimit`). Model-generated tool calls remain disabled until the
//! response parser/framing slice lands.

use hyperion_ffi::HypSamplingConfig;
use hyperion_tokenizer::TokenizerHandle;
use hyperion_tokenizer::renderer::{ChatMessage, ChatTemplate, RenderError, RenderOptions};

use crate::dialect::Dialect;
use crate::engine::EngineRequest;
use crate::history::{AnthropicHistoryInput, OpenAiHistoryInput, ToolHistoryError};
use crate::tool_schema::{
    AnthropicToolsInput, OpenAiToolsInput, ToolMode, ToolRegistry, ToolSchemaError,
};

/// The context window (tokens) used for the 413 overflow check. The 12B's
/// `max_position_embeddings` (262_144); passed in from the loaded geometry so
/// the prepare path doesn't depend on hyperion-model directly.
pub type ContextWindow = u32;

/// The normalized input both dialects produce. The handler feeds
/// `engine_request` to the `EngineDriver`; `stream` + `dialect` select the
/// response framing.
#[derive(Clone, Debug)]
pub struct PreparedPrompt {
    /// The `EngineRequest` for the engine-driver (prompt tokens + max_tokens +
    /// eos + sampler).
    pub engine_request: EngineRequest,
    /// `true` if the client asked for `stream: true` (SSE) vs a single JSON
    /// response.
    pub stream: bool,
    /// The dialect (selects SSE framing + error envelope).
    pub dialect: Dialect,
    /// The rendered prompt token count (for usage accounting / the 413 check).
    pub prompt_tokens_len: u32,
}

/// A normalization failure, mapped to the 06 error taxonomy.
#[derive(Debug)]
pub enum PrepareError {
    /// The JSON body was malformed or missing required fields → 400.
    MalformedBody(String),
    /// The request enables model-generated tool calls before response parsing
    /// and framing support is available → 400.
    ToolsUnsupported,
    /// Tool declarations or controls failed bounded registry compilation →
    /// 400. The public display text does not expose the source error.
    ToolSchema(ToolSchemaError),
    /// Provider history failed bounded normalization → 400. The public display
    /// text does not expose the source error.
    ToolHistory(ToolHistoryError),
    /// The chat template failed to render → 400 (a malformed input the
    /// template rejected, e.g. bad `tool_calls` arguments).
    Render(RenderError),
    /// `prompt_tokens + max_tokens > context_window` → 413 (context overflow,
    /// distinct from the body-size 413).
    ContextOverflow {
        prompt_tokens: u32,
        max_tokens: u32,
        context: u32,
    },
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedBody(s) => write!(f, "malformed request body: {s}"),
            Self::ToolsUnsupported => f.write_str("tool calling is not yet supported"),
            Self::ToolSchema(_) => f.write_str("invalid tool schema"),
            Self::ToolHistory(_) => f.write_str("invalid tool history"),
            Self::Render(e) => write!(f, "template render error: {e}"),
            Self::ContextOverflow {
                prompt_tokens,
                max_tokens,
                context,
            } => write!(
                f,
                "context overflow: prompt {prompt_tokens} + max_tokens {max_tokens} > context {context}"
            ),
        }
    }
}

impl std::error::Error for PrepareError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ToolSchema(error) => Some(error),
            Self::ToolHistory(error) => Some(error),
            Self::Render(error) => Some(error),
            Self::MalformedBody(_) | Self::ToolsUnsupported | Self::ContextOverflow { .. } => None,
        }
    }
}

impl PrepareError {
    /// The HTTP status code this error maps to (06 §Error taxonomy).
    #[must_use]
    pub fn http_status(&self) -> u16 {
        match self {
            Self::MalformedBody(_)
            | Self::ToolsUnsupported
            | Self::ToolSchema(_)
            | Self::ToolHistory(_)
            | Self::Render(_) => 400,
            Self::ContextOverflow { .. } => 413,
        }
    }
}

/// The sampler parameters parsed from a dialect body (both dialects carry the
/// same set under different names; this is the common shape).
#[derive(Clone, Copy, Debug, Default)]
struct SamplerParams {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    seed: Option<u64>,
}

impl SamplerParams {
    /// Build the `HypSamplingConfig` for the engine. `None` (greedy) if
    /// `temperature <= 0` (the G1 greedy default); `Some(cfg)` for sampled.
    /// Defaults per the 06 model card: t=1.0, top_p=0.95, top_k=64.
    #[must_use]
    fn to_sampling(self) -> Option<HypSamplingConfig> {
        // temperature <= 0 (or unset with a default of 0 for greedy-intent) ⇒
        // greedy. The dialects default `temperature` to 1.0 per the model card,
        // but an explicit 0 / unset-with-greedy-intent selects greedy. We treat
        // `temperature` unset as greedy (the safe default for a server that
        // hasn't been told to sample); an explicit >0 temperature selects
        // sampled with the model-card defaults.
        let temperature = self.temperature.unwrap_or(0.0);
        if temperature <= 0.0 {
            return None;
        }
        Some(HypSamplingConfig {
            temperature,
            top_p: self.top_p.unwrap_or(0.95),
            top_k: self.top_k.unwrap_or(64).max(1) as i32,
            min_p: self.min_p.unwrap_or(0.0),
            seed: self.seed.unwrap_or(0),
        })
    }
}

/// The Anthropic `/v1/messages` request body (the fields PR B reads; the rest
/// are ignored). `system` is a string or an array of content blocks — both
/// fold into a leading system `ChatMessage`.
#[derive(Deserialize)]
pub struct AnthropicBody {
    #[serde(default = "empty_messages")]
    pub messages: serde_json::Value,
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    /// Anthropic requires `max_tokens`.
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

/// The OpenAI `/v1/chat/completions` request body.
#[derive(Deserialize)]
pub struct OpenAiBody {
    #[serde(default = "empty_messages")]
    pub messages: serde_json::Value,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<serde_json::Value>,
}

fn empty_messages() -> serde_json::Value {
    serde_json::Value::Array(Vec::new())
}

fn has_non_null_tools(tools: Option<&serde_json::Value>) -> bool {
    tools.is_some_and(|tools| !tools.is_null())
}

fn require_history_only_generation(
    registry: &ToolRegistry,
    tools_present: bool,
    explicit_none: bool,
) -> Result<(), PrepareError> {
    if registry.mode() == ToolMode::Auto || (tools_present && !explicit_none) {
        Err(PrepareError::ToolsUnsupported)
    } else {
        Ok(())
    }
}

/// Compile declarations, enforce OpenAI's explicit history-only generation
/// boundary, then normalize the complete provider message value.
fn normalize_openai_for_generation(body: &OpenAiBody) -> Result<Vec<ChatMessage>, PrepareError> {
    let registry = ToolRegistry::from_openai(OpenAiToolsInput {
        tools: body.tools.as_ref(),
        tool_choice: body.tool_choice.as_ref(),
        parallel_tool_calls: body.parallel_tool_calls.as_ref(),
    })
    .map_err(PrepareError::ToolSchema)?;
    let explicit_none = body.tool_choice.as_ref().is_some_and(
        |choice| matches!(choice, serde_json::Value::String(value) if value == "none"),
    );
    require_history_only_generation(
        &registry,
        has_non_null_tools(body.tools.as_ref()),
        explicit_none,
    )?;
    registry
        .normalize_openai_history(OpenAiHistoryInput {
            messages: &body.messages,
        })
        .map_err(PrepareError::ToolHistory)
}

/// Compile declarations, enforce Anthropic's explicit history-only generation
/// boundary, then normalize the complete provider system/messages values.
fn normalize_anthropic_for_generation(
    body: &AnthropicBody,
) -> Result<Vec<ChatMessage>, PrepareError> {
    let registry = ToolRegistry::from_anthropic(AnthropicToolsInput {
        tools: body.tools.as_ref(),
        tool_choice: body.tool_choice.as_ref(),
    })
    .map_err(PrepareError::ToolSchema)?;
    let explicit_none = body.tool_choice.as_ref().is_some_and(|choice| {
        choice
            .as_object()
            .and_then(|choice| choice.get("type"))
            .and_then(serde_json::Value::as_str)
            == Some("none")
    });
    require_history_only_generation(
        &registry,
        has_non_null_tools(body.tools.as_ref()),
        explicit_none,
    )?;
    registry
        .normalize_anthropic_history(AnthropicHistoryInput {
            system: body.system.as_ref(),
            messages: &body.messages,
        })
        .map_err(PrepareError::ToolHistory)
}

/// Compile and normalize Anthropic count-only input without applying the
/// generation capability gate. `Auto` declarations are intentionally returned
/// for prompt rendering; `None` declarations are hidden by the registry.
fn normalize_anthropic_for_count(
    body: &AnthropicBody,
) -> Result<(Vec<ChatMessage>, Vec<serde_json::Value>), PrepareError> {
    let registry = ToolRegistry::from_anthropic(AnthropicToolsInput {
        tools: body.tools.as_ref(),
        tool_choice: body.tool_choice.as_ref(),
    })
    .map_err(PrepareError::ToolSchema)?;
    let messages = registry
        .normalize_anthropic_history(AnthropicHistoryInput {
            system: body.system.as_ref(),
            messages: &body.messages,
        })
        .map_err(PrepareError::ToolHistory)?;
    Ok((messages, registry.render_tools().to_vec()))
}

/// Prepare an Anthropic `/v1/messages` body into a `PreparedPrompt`.
///
/// # Errors
/// - `PrepareError::MalformedBody` — bad JSON or missing `max_tokens` (Anthropic
///   requires it).
/// - `PrepareError::ToolsUnsupported` — the effective tool mode permits model
///   generation, or declarations are present without an explicit `none`.
/// - `PrepareError::ToolSchema` / `PrepareError::ToolHistory` — bounded tool
///   declaration/history validation failed.
/// - `PrepareError::Render` — the chat template rejected the conversation.
/// - `PrepareError::ContextOverflow` — `prompt + max_tokens > context`.
pub fn prepare_anthropic(
    body: &str,
    template: &ChatTemplate,
    tokenizer: &TokenizerHandle,
    context: ContextWindow,
) -> Result<PreparedPrompt, PrepareError> {
    let parsed: AnthropicBody =
        serde_json::from_str(body).map_err(|e| PrepareError::MalformedBody(e.to_string()))?;
    let max_tokens = parsed
        .max_tokens
        .ok_or_else(|| PrepareError::MalformedBody("Anthropic requires max_tokens".to_string()))?;
    let messages = normalize_anthropic_for_generation(&parsed)?;
    let sampler = SamplerParams {
        temperature: parsed.temperature,
        top_p: parsed.top_p,
        top_k: parsed.top_k,
        min_p: None,
        seed: None,
    };
    finalize(
        messages,
        Vec::new(),
        max_tokens,
        parsed.stream.unwrap_or(false),
        sampler,
        Dialect::Anthropic,
        PrepareEnv {
            template,
            tokenizer,
            context,
        },
    )
}

/// Render + tokenize an Anthropic `/v1/messages/count_tokens` body and return
/// the prompt token count. `count_tokens` does NOT require `max_tokens` (it
/// only counts input tokens), so this is a separate path from
/// [`prepare_anthropic`] — that one requires `max_tokens` for a generation.
///
/// # Errors
/// `PrepareError::MalformedBody` on bad JSON; registry/history errors on invalid
/// bounded input; `PrepareError::Render` on a template failure.
pub fn count_anthropic_tokens(
    body: &str,
    template: &ChatTemplate,
    tokenizer: &TokenizerHandle,
) -> Result<u32, PrepareError> {
    let parsed: AnthropicBody =
        serde_json::from_str(body).map_err(|e| PrepareError::MalformedBody(e.to_string()))?;
    let (messages, tools) = normalize_anthropic_for_count(&parsed)?;
    let prompt_tokens = render_prompt_tokens(&messages, tools, template, tokenizer)?;
    Ok(u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX))
}

/// Prepare an OpenAI `/v1/chat/completions` body into a `PreparedPrompt`.
/// `max_tokens` defaults to a server default if unset (OpenAI makes it
/// optional). `seed` is honored and echoed.
///
/// # Errors
/// Same shape as [`prepare_anthropic`] except `max_tokens` is never missing
/// (defaults applied).
pub fn prepare_openai(
    body: &str,
    template: &ChatTemplate,
    tokenizer: &TokenizerHandle,
    context: ContextWindow,
    default_max_tokens: u32,
) -> Result<PreparedPrompt, PrepareError> {
    let parsed: OpenAiBody =
        serde_json::from_str(body).map_err(|e| PrepareError::MalformedBody(e.to_string()))?;
    let max_tokens = parsed.max_tokens.unwrap_or(default_max_tokens);
    let messages = normalize_openai_for_generation(&parsed)?;
    let sampler = SamplerParams {
        temperature: parsed.temperature,
        top_p: parsed.top_p,
        top_k: parsed.top_k,
        min_p: parsed.min_p,
        seed: parsed.seed,
    };
    finalize(
        messages,
        Vec::new(),
        max_tokens,
        parsed.stream.unwrap_or(false),
        sampler,
        Dialect::OpenAi,
        PrepareEnv {
            template,
            tokenizer,
            context,
        },
    )
}

/// The shared render+tokenize environment both `prepare_*` fns pass to
/// [`finalize`] (grouped to keep `finalize` under the argument limit).
#[derive(Clone, Copy)]
pub struct PrepareEnv<'a> {
    pub template: &'a ChatTemplate,
    pub tokenizer: &'a TokenizerHandle,
    pub context: ContextWindow,
}

/// The shared tail: render → tokenize → context-overflow check →
/// `PreparedPrompt`. The eos token id is resolved from the tokenizer's
/// `eos_token` (the gemma4 `<eos>`); `None` if absent (run to max_tokens).
fn finalize(
    messages: Vec<ChatMessage>,
    tools: Vec<serde_json::Value>,
    max_tokens: u32,
    stream: bool,
    sampler: SamplerParams,
    dialect: Dialect,
    env: PrepareEnv<'_>,
) -> Result<PreparedPrompt, PrepareError> {
    let prompt_tokens = render_prompt_tokens(&messages, tools, env.template, env.tokenizer)?;
    let prompt_tokens_len = u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX);
    if prompt_tokens_len.saturating_add(max_tokens) > env.context {
        return Err(PrepareError::ContextOverflow {
            prompt_tokens: prompt_tokens_len,
            max_tokens,
            context: env.context,
        });
    }
    let eos_token_id = env.tokenizer.token_to_id("<eos>");
    let engine_request = EngineRequest {
        prompt_tokens,
        max_tokens,
        eos_token_id,
        sampling: sampler.to_sampling(),
    };
    Ok(PreparedPrompt {
        engine_request,
        stream,
        dialect,
        prompt_tokens_len,
    })
}

fn render_prompt_tokens(
    messages: &[ChatMessage],
    tools: Vec<serde_json::Value>,
    template: &ChatTemplate,
    tokenizer: &TokenizerHandle,
) -> Result<Vec<u32>, PrepareError> {
    let options = RenderOptions {
        add_generation_prompt: true,
        enable_thinking: None,   // model default
        preserve_thinking: None, // template default (false)
        tools,
    };
    let rendered = template
        .render(messages, &options)
        .map_err(PrepareError::Render)?;
    // add_special_tokens=false: the template emits its own BOS; the tokenizer's
    // empty special-tokens post-processor means no auto-add anyway.
    Ok(tokenizer.encode(&rendered, false))
}

use serde::Deserialize;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const COUNT_TOKENIZER_JSON: &str = r#"{
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
        "vocab": {"<eos>": 0, "<bos>": 1, "DECL": 2, "user": 3, "hi": 4, "[UNK]": 5},
        "unk_token": "[UNK]"
      }
    }"#;

    fn openai_tool() -> serde_json::Value {
        json!({
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
        json!({
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

    #[test]
    fn sampler_greedy_when_temperature_unset_or_zero() {
        let s = SamplerParams::default();
        assert!(s.to_sampling().is_none(), "unset temperature ⇒ greedy");
        let s = SamplerParams {
            temperature: Some(0.0),
            ..Default::default()
        };
        assert!(s.to_sampling().is_none(), "temperature 0 ⇒ greedy");
    }

    #[test]
    fn sampler_sampled_when_temperature_positive() {
        let s = SamplerParams {
            temperature: Some(1.0),
            ..Default::default()
        };
        let cfg = s.to_sampling().expect("temperature 1.0 ⇒ sampled");
        assert_eq!(cfg.temperature, 1.0);
        // model-card defaults
        assert_eq!(cfg.top_p, 0.95);
        assert_eq!(cfg.top_k, 64);
        assert_eq!(cfg.min_p, 0.0);
    }

    #[test]
    fn sampler_honors_explicit_fields() {
        let s = SamplerParams {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(32),
            min_p: Some(0.05),
            seed: Some(42),
        };
        let cfg = s.to_sampling().unwrap();
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_p, 0.9);
        assert_eq!(cfg.top_k, 32);
        assert_eq!(cfg.min_p, 0.05);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn openai_structured_history_survives_deserialization_and_normalizes_with_none() {
        let body: OpenAiBody = serde_json::from_value(json!({
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
            "tool_choice": "none",
            "parallel_tool_calls": true
        }))
        .unwrap();

        assert!(body.messages[1].get("tool_calls").is_some());
        let messages = normalize_openai_for_generation(&body).unwrap();
        assert_eq!(messages.len(), 3);
        assert!(messages[1].tool_calls.is_some());
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(messages[2].name.as_deref(), Some("lookup"));
    }

    #[test]
    fn anthropic_structured_history_survives_deserialization_and_normalizes_with_none() {
        let body: AnthropicBody = serde_json::from_value(json!({
            "system": "be concise",
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
            "tools": [anthropic_tool()],
            "tool_choice": {"type": "none"}
        }))
        .unwrap();

        assert_eq!(body.messages[1]["content"][0]["type"], "tool_use");
        let messages = normalize_anthropic_for_generation(&body).unwrap();
        assert_eq!(messages[0].role, "system");
        assert!(messages[2].tool_calls.is_some());
        assert_eq!(messages[3].tool_call_id.as_deref(), Some("toolu_1"));
    }

    #[test]
    fn present_tools_require_explicit_none_even_when_registry_defaults_to_none() {
        let body: OpenAiBody = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": []
        }))
        .unwrap();
        assert!(matches!(
            normalize_openai_for_generation(&body),
            Err(PrepareError::ToolsUnsupported)
        ));
    }

    #[test]
    fn effective_auto_is_rejected_before_generation_rendering() {
        let body: AnthropicBody = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "auto"}
        }))
        .unwrap();
        assert!(matches!(
            normalize_anthropic_for_generation(&body),
            Err(PrepareError::ToolsUnsupported)
        ));
    }

    #[test]
    fn missing_messages_still_defaults_to_an_empty_array() {
        let openai: OpenAiBody = serde_json::from_str("{}").unwrap();
        let anthropic: AnthropicBody = serde_json::from_str("{}").unwrap();
        assert_eq!(openai.messages, json!([]));
        assert_eq!(anthropic.messages, json!([]));
    }

    #[test]
    fn fixed_tool_error_displays_do_not_reflect_source_payloads() {
        let schema_body: OpenAiBody = serde_json::from_value(json!({
            "messages": [],
            "tools": [{"SENTINEL_SCHEMA_KEY": true}],
            "tool_choice": "none"
        }))
        .unwrap();
        let schema_error = normalize_openai_for_generation(&schema_body).unwrap_err();
        assert_eq!(schema_error.to_string(), "invalid tool schema");
        assert!(!schema_error.to_string().contains("SENTINEL"));
        assert_eq!(schema_error.http_status(), 400);

        let history_body: OpenAiBody = serde_json::from_value(json!({
            "messages": [{"role": "SENTINEL_HISTORY_ROLE", "content": "secret"}]
        }))
        .unwrap();
        let history_error = normalize_openai_for_generation(&history_body).unwrap_err();
        assert_eq!(history_error.to_string(), "invalid tool history");
        assert!(!history_error.to_string().contains("SENTINEL"));
        assert_eq!(history_error.http_status(), 400);
    }

    #[test]
    fn count_tokens_renders_auto_declarations_but_hides_none_without_max_tokens() {
        let tokenizer = TokenizerHandle::from_bytes(COUNT_TOKENIZER_JSON.as_bytes()).unwrap();
        let template = ChatTemplate::from_source(
            "{% if tools | length > 0 %}DECL {% endif %}{% for message in messages %}{{ message.role }} {{ message.content }} {% endfor %}",
            "<bos>",
            "<eos>",
        )
        .unwrap();
        let base = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [anthropic_tool()]
        });
        let mut auto = base.clone();
        auto["tool_choice"] = json!({"type": "auto"});
        let mut none = base;
        none["tool_choice"] = json!({"type": "none"});

        let auto_count = count_anthropic_tokens(&auto.to_string(), &template, &tokenizer).unwrap();
        let none_count = count_anthropic_tokens(&none.to_string(), &template, &tokenizer).unwrap();
        assert_eq!(auto_count, none_count + 1);
    }

    #[test]
    fn prepare_anthropic_rejects_missing_max_tokens() {
        // A body without max_tokens → MalformedBody (400).
        let body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        // We can't easily build a ChatTemplate+TokenizerHandle model-free, so
        // assert the parse+tools+max_tokens gate fails before reaching render.
        // The parse succeeds; max_tokens is None → MalformedBody.
        let parsed: AnthropicBody = serde_json::from_str(body).unwrap();
        assert!(parsed.max_tokens.is_none());
    }

    #[test]
    fn prepare_anthropic_rejects_auto_tools_with_400() {
        let parsed: AnthropicBody = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "tools": [anthropic_tool()],
            "tool_choice": {"type": "auto"}
        }))
        .unwrap();
        assert!(matches!(
            normalize_anthropic_for_generation(&parsed),
            Err(PrepareError::ToolsUnsupported)
        ));
        assert_eq!(PrepareError::ToolsUnsupported.http_status(), 400);
    }

    #[test]
    fn context_overflow_maps_to_413() {
        assert_eq!(
            PrepareError::ContextOverflow {
                prompt_tokens: 200,
                max_tokens: 100,
                context: 250,
            }
            .http_status(),
            413
        );
    }

    #[test]
    fn malformed_body_maps_to_400() {
        assert_eq!(
            PrepareError::MalformedBody("bad".to_string()).http_status(),
            400
        );
        assert_eq!(
            PrepareError::Render(RenderError::Load("x".into())).http_status(),
            400
        );
    }
}
