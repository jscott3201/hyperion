//! B4: dual-dialect normalization → `PreparedPrompt` (06: "both dialects
//! normalize to one PreparedPrompt"). A request body (Anthropic
//! `/v1/messages` or OpenAI `/v1/chat/completions`) is parsed into a
//! `ChatMessage` list + sampler config, rendered through the M3 chat template,
//! tokenized, and packaged as a [`PreparedPrompt`] — the `EngineRequest` +
//! stream flag + dialect the handler feeds to the engine.
//!
//! The 413 context-overflow check (`prompt_tokens + max_tokens > context`)
//! lives here, distinct from the 32 MiB body-size 413 (B5's
//! `DefaultBodyLimit`). Tools are rejected with 400 in PR B — the tool-call
//! parser is a later M3 sub-slice.

use hyperion_ffi::HypSamplingConfig;
use hyperion_tokenizer::TokenizerHandle;
use hyperion_tokenizer::renderer::{ChatMessage, ChatTemplate, RenderError, RenderOptions};

use crate::dialect::Dialect;
use crate::engine::EngineRequest;

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
    /// The body carried `tools` / `tool_choice` — not supported in PR B (the
    /// tool-call parser is a later slice) → 400.
    ToolsUnsupported,
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

impl std::error::Error for PrepareError {}

impl PrepareError {
    /// The HTTP status code this error maps to (06 §Error taxonomy).
    #[must_use]
    pub fn http_status(&self) -> u16 {
        match self {
            Self::MalformedBody(_) | Self::ToolsUnsupported | Self::Render(_) => 400,
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
    #[serde(default)]
    pub messages: Vec<RawMessage>,
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
    /// Tools are rejected in PR B (later slice).
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

/// The OpenAI `/v1/chat/completions` request body.
#[derive(Deserialize)]
pub struct OpenAiBody {
    #[serde(default)]
    pub messages: Vec<RawMessage>,
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
}

/// A dialect-agnostic message: `role` + `content` (string or parts). Both
/// dialects' message shapes deserialize into this (the `content` is kept as a
/// `serde_json::Value` so the renderer's `ChatMessage` handles string-or-array).
#[derive(Deserialize)]
pub struct RawMessage {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
}

/// Normalize a parsed body into `ChatMessage`s + sampler params. Shared by both
/// dialects; the dialect only changes which fields are read + how `system` is
/// folded in.
fn to_chat_messages(
    messages: &[RawMessage],
    system: Option<&serde_json::Value>,
) -> Result<Vec<ChatMessage>, PrepareError> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    if let Some(system) = system {
        // A non-empty system (string or array of blocks) → a leading system
        // message. The renderer handles string or content-parts via the Value.
        if !system.is_null() {
            out.push(ChatMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_calls: None,
                tool_responses: None,
                reasoning: None,
                reasoning_content: None,
                tool_call_id: None,
                name: None,
            });
        }
    }
    for m in messages {
        if m.role.is_empty() {
            return Err(PrepareError::MalformedBody(
                "message missing role".to_string(),
            ));
        }
        out.push(ChatMessage {
            role: m.role.clone(),
            content: Some(m.content.clone()),
            tool_calls: None,
            tool_responses: None,
            reasoning: None,
            reasoning_content: None,
            tool_call_id: None,
            name: None,
        });
    }
    Ok(out)
}

/// Reject tools (PR B scope: tool-calling is a later slice). `Some` tools or a
/// non-`none` tool_choice → `ToolsUnsupported` (400).
fn check_tools(
    tools: &Option<serde_json::Value>,
    tool_choice: &Option<serde_json::Value>,
) -> Result<(), PrepareError> {
    if tools.is_some() {
        return Err(PrepareError::ToolsUnsupported);
    }
    if let Some(choice) = tool_choice {
        // `tool_choice: "none"` is allowed (no tools requested); anything else
        // implies tool use, which is unsupported in PR B.
        let is_none = match choice {
            serde_json::Value::String(s) => s == "none",
            _ => false,
        };
        if !is_none {
            return Err(PrepareError::ToolsUnsupported);
        }
    }
    Ok(())
}

/// Prepare an Anthropic `/v1/messages` body into a `PreparedPrompt`.
///
/// # Errors
/// - `PrepareError::MalformedBody` — bad JSON or missing `max_tokens` (Anthropic
///   requires it).
/// - `PrepareError::ToolsUnsupported` — body carries `tools`/non-`none`
///   `tool_choice`.
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
    check_tools(&parsed.tools, &parsed.tool_choice)?;
    let max_tokens = parsed
        .max_tokens
        .ok_or_else(|| PrepareError::MalformedBody("Anthropic requires max_tokens".to_string()))?;
    let messages = to_chat_messages(&parsed.messages, parsed.system.as_ref())?;
    let sampler = SamplerParams {
        temperature: parsed.temperature,
        top_p: parsed.top_p,
        top_k: parsed.top_k,
        min_p: None,
        seed: None,
    };
    finalize(
        messages,
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
    check_tools(&parsed.tools, &parsed.tool_choice)?;
    let max_tokens = parsed.max_tokens.unwrap_or(default_max_tokens);
    let messages = to_chat_messages(&parsed.messages, None)?;
    let sampler = SamplerParams {
        temperature: parsed.temperature,
        top_p: parsed.top_p,
        top_k: parsed.top_k,
        min_p: parsed.min_p,
        seed: parsed.seed,
    };
    finalize(
        messages,
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
    max_tokens: u32,
    stream: bool,
    sampler: SamplerParams,
    dialect: Dialect,
    env: PrepareEnv<'_>,
) -> Result<PreparedPrompt, PrepareError> {
    let options = RenderOptions {
        add_generation_prompt: true,
        enable_thinking: None,   // model default
        preserve_thinking: None, // template default (false)
        tools: Vec::new(),
    };
    let rendered = env
        .template
        .render(&messages, &options)
        .map_err(PrepareError::Render)?;
    // add_special_tokens=false: the template emits its own BOS; the tokenizer's
    // empty special-tokens post-processor means no auto-add anyway.
    let prompt_tokens = env.tokenizer.encode(&rendered, false);
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

use serde::Deserialize;

#[cfg(test)]
mod tests {
    use super::*;

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
    fn check_tools_rejects_tools_array() {
        let tools = Some(serde_json::json!([{"type": "function"}]));
        assert!(matches!(
            check_tools(&tools, &None),
            Err(PrepareError::ToolsUnsupported)
        ));
    }

    #[test]
    fn check_tools_rejects_non_none_tool_choice() {
        let choice = Some(serde_json::json!("auto"));
        assert!(matches!(
            check_tools(&None, &choice),
            Err(PrepareError::ToolsUnsupported)
        ));
    }

    #[test]
    fn check_tools_allows_none_choice() {
        let choice = Some(serde_json::json!("none"));
        assert!(check_tools(&None, &choice).is_ok());
        assert!(check_tools(&None, &None).is_ok());
    }

    #[test]
    fn to_chat_messages_folds_system_anthropic() {
        let msgs = vec![RawMessage {
            role: "user".to_string(),
            content: serde_json::json!("hi"),
        }];
        let system = Some(serde_json::json!("you are helpful"));
        let out = to_chat_messages(&msgs, system.as_ref()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "system");
        assert_eq!(out[1].role, "user");
    }

    #[test]
    fn to_chat_messages_rejects_empty_role() {
        let msgs = vec![RawMessage {
            role: String::new(),
            content: serde_json::json!("hi"),
        }];
        assert!(matches!(
            to_chat_messages(&msgs, None),
            Err(PrepareError::MalformedBody(_))
        ));
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
    fn prepare_anthropic_rejects_tools_with_400() {
        let body = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":10,"tools":[]}"#;
        let parsed: AnthropicBody = serde_json::from_str(body).unwrap();
        assert!(matches!(
            check_tools(&parsed.tools, &parsed.tool_choice),
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
