//! Validated Gemma 4 geometry parsed from a Hugging Face ``config.json``.
//!
//! From M2 onward this is the single source the native graph, heterogeneous KV
//! layout, per-layer-kind masks, and the governor read. Two disciplines are
//! load-bearing (02-architecture, Helios lesson):
//!
//! * No silent default-through — the raw ``text_config`` is parsed with
//!   ``#[serde(deny_unknown_fields)]`` and every field declared required, so a
//!   new upstream field surfaces as a parse error for review instead of being
//!   silently dropped, and a missing field never defaults.
//! * Masks and KV are built per layer *kind*, sourced from the array — the
//!   ``layer_types`` order is the truth, never a hard-coded 5:1 rule (E2B is 4:1,
//!   the rest 5:1). The first layer of each kind is what the mask-construction
//!   code must later source from, never layer 0 unconditionally.

use std::fmt;

use serde::Deserialize;

/// A per-token attention layer kind in the Gemma 4 hybrid layout.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LayerType {
    /// Sliding-window (local) attention; head_dim 256.
    Sliding,
    /// Full causal (global) attention; head_dim 512, K=V when attention_k_eq_v.
    Full,
}

impl LayerType {
    /// The HF ``layer_types`` string this variant is parsed from.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sliding => "sliding_attention",
            Self::Full => "full_attention",
        }
    }
}

/// The RoPE scheme for one attention kind.
#[derive(Copy, Clone, Debug)]
pub struct RopeSpec {
    /// ``rope_theta`` from config.
    pub theta: f64,
    /// Fraction of head_dim rotated; ``None`` means full rotary (local default).
    pub partial_rotary_factor: Option<f32>,
    /// Whether this is the proportional (global) or default (local) scheme.
    pub proportional: bool,
}

/// Mixture-of-experts block geometry (26B-A4B; absent on the dense sizes).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MoeConfig {
    pub num_experts: u32,
    pub top_k: u32,
    pub moe_intermediate_size: u32,
}

/// The Gemma 4 text backbone model_type accepted by Hyperion v1.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TextModelType {
    /// Dense unified text (12B).
    Gemma4UnifiedText,
    /// E-series / 26B / 31B text.
    Gemma4Text,
}

/// Validated, engine-ready geometry derived from a HF ``config.json``.
///
/// Construct via [`Geometry::from_text_config_str`]; every other crate consumes
/// the validated struct, never the raw file.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub model_type: TextModelType,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub layer_types: Vec<LayerType>,
    pub num_attention_heads: u32,
    pub head_dim_local: u32,
    pub head_dim_global: u32,
    pub num_kv_heads_local: u32,
    pub num_kv_heads_global: u32,
    /// Global-only flag; true ⇒ the key tensor IS the value tensor (12B/26B/31B).
    pub attention_k_eq_v_global: bool,
    pub num_kv_shared_layers: u32,
    pub sliding_window: u32,
    pub rope_local: RopeSpec,
    pub rope_global: RopeSpec,
    pub final_logit_softcapping: f32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub tie_word_embeddings: bool,
    /// PLE ``hidden_size_per_layer_input``; 0 on the dense sizes.
    pub ple_hidden_per_layer_input: u32,
    /// PLE ``vocab_size_per_layer_input``; 0 on the dense sizes.
    pub ple_vocab_per_layer_input: u32,
    pub use_double_wide_mlp: bool,
    pub moe: Option<MoeConfig>,
}

/// A geometry parse or validation failure.
#[derive(Debug)]
pub struct GeometryError {
    message: String,
}

impl GeometryError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for GeometryError {}

// --- Raw serde mirrors (strict; deny_unknown_fields) ------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RopeSpecRaw {
    rope_theta: f64,
    rope_type: String,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RopeParametersRaw {
    full_attention: RopeSpecRaw,
    sliding_attention: RopeSpecRaw,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // every field is declared so a new upstream field surfaces as a
// parse error (no silent default-through); not all are consumed
// by the engine yet — e.g. bos/eos/pad tokens, dtype, use_cache.
struct Gemma4TextConfigRaw {
    attention_bias: bool,
    attention_dropout: f32,
    attention_k_eq_v: bool,
    bos_token_id: u32,
    dtype: String,
    enable_moe_block: bool,
    eos_token_id: u32,
    final_logit_softcapping: f32,
    global_head_dim: u32,
    head_dim: u32,
    hidden_activation: String,
    hidden_size: u32,
    hidden_size_per_layer_input: u32,
    initializer_range: f32,
    intermediate_size: u32,
    layer_types: Vec<String>,
    max_position_embeddings: u32,
    model_type: String,
    moe_intermediate_size: Option<u32>,
    num_attention_heads: u32,
    num_experts: Option<u32>,
    num_global_key_value_heads: Option<u32>,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    num_kv_shared_layers: u32,
    pad_token_id: u32,
    rms_norm_eps: f32,
    rope_parameters: RopeParametersRaw,
    sliding_window: u32,
    tie_word_embeddings: bool,
    top_k_experts: Option<u32>,
    use_bidirectional_attention: Option<String>,
    use_cache: bool,
    use_double_wide_mlp: bool,
    vocab_size: u32,
    vocab_size_per_layer_input: u32,
}

impl Geometry {
    /// Parse and validate a Gemma 4 ``text_config`` JSON object.
    ///
    /// Accepts the ``text_config`` value (not the full multimodal wrapper); the
    /// audio/vision configs are sanitized out (hyperion v1 is text-only).
    pub fn from_text_config_str(json: &str) -> Result<Self, GeometryError> {
        let raw: Gemma4TextConfigRaw = serde_json::from_str(json).map_err(|error| {
            GeometryError::new(format!("gemma4 text_config parse failed: {error}"))
        })?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: Gemma4TextConfigRaw) -> Result<Self, GeometryError> {
        let model_type = match raw.model_type.as_str() {
            "gemma4_unified_text" => TextModelType::Gemma4UnifiedText,
            "gemma4_text" => TextModelType::Gemma4Text,
            other => {
                return Err(GeometryError::new(format!(
                    "unsupported text model_type {other:?}; expected gemma4_text or gemma4_unified_text"
                )));
            }
        };

        if raw.hidden_activation != "gelu_pytorch_tanh" {
            return Err(GeometryError::new(format!(
                "unsupported hidden_activation {:?}; only gelu_pytorch_tanh is implemented",
                raw.hidden_activation
            )));
        }
        if !raw.tie_word_embeddings {
            return Err(GeometryError::new(
                "tie_word_embeddings must be true (lm_head is tied to the embedding table)",
            ));
        }

        let layer_types = raw
            .layer_types
            .iter()
            .map(|name| match name.as_str() {
                "sliding_attention" => Ok(LayerType::Sliding),
                "full_attention" => Ok(LayerType::Full),
                other => Err(GeometryError::new(format!(
                    "unsupported layer_type {other:?}"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if u32::try_from(layer_types.len())
            .map_err(|_| GeometryError::new("layer_types length overflows u32"))?
            != raw.num_hidden_layers
        {
            return Err(GeometryError::new(format!(
                "layer_types length {} != num_hidden_layers {}",
                layer_types.len(),
                raw.num_hidden_layers
            )));
        }
        if layer_types.last() != Some(&LayerType::Full) {
            return Err(GeometryError::new(
                "the last layer must be full_attention (Gemma 4 invariant)",
            ));
        }
        if layer_types.is_empty() {
            return Err(GeometryError::new("layer_types is empty"));
        }

        let rope_local = Self::rope_spec(&raw.rope_parameters.sliding_attention, false)?;
        let rope_global = Self::rope_spec(&raw.rope_parameters.full_attention, true)?;
        if rope_local.partial_rotary_factor.is_some() {
            return Err(GeometryError::new(
                "sliding_attention rope must be full rotary (no partial_rotary_factor)",
            ));
        }
        if rope_global.partial_rotary_factor.is_none() {
            return Err(GeometryError::new(
                "full_attention rope must set partial_rotary_factor",
            ));
        }

        if raw.sliding_window == 0 {
            return Err(GeometryError::new("sliding_window must be non-zero"));
        }
        if raw.num_kv_shared_layers > raw.num_hidden_layers {
            return Err(GeometryError::new(format!(
                "num_kv_shared_layers {} exceeds num_hidden_layers {}",
                raw.num_kv_shared_layers, raw.num_hidden_layers
            )));
        }

        let num_kv_heads_global = raw
            .num_global_key_value_heads
            .unwrap_or(raw.num_key_value_heads);

        let moe = if raw.enable_moe_block {
            let num_experts = raw
                .num_experts
                .ok_or_else(|| GeometryError::new("enable_moe_block=true requires num_experts"))?;
            let top_k = raw.top_k_experts.ok_or_else(|| {
                GeometryError::new("enable_moe_block=true requires top_k_experts")
            })?;
            let moe_intermediate_size = raw.moe_intermediate_size.ok_or_else(|| {
                GeometryError::new("enable_moe_block=true requires moe_intermediate_size")
            })?;
            Some(MoeConfig {
                num_experts,
                top_k,
                moe_intermediate_size,
            })
        } else {
            None
        };

        Ok(Self {
            model_type,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            layer_types,
            num_attention_heads: raw.num_attention_heads,
            head_dim_local: raw.head_dim,
            head_dim_global: raw.global_head_dim,
            num_kv_heads_local: raw.num_key_value_heads,
            num_kv_heads_global,
            attention_k_eq_v_global: raw.attention_k_eq_v,
            num_kv_shared_layers: raw.num_kv_shared_layers,
            sliding_window: raw.sliding_window,
            rope_local,
            rope_global,
            final_logit_softcapping: raw.final_logit_softcapping,
            rms_norm_eps: raw.rms_norm_eps,
            attention_bias: raw.attention_bias,
            vocab_size: raw.vocab_size,
            max_position_embeddings: raw.max_position_embeddings,
            tie_word_embeddings: raw.tie_word_embeddings,
            ple_hidden_per_layer_input: raw.hidden_size_per_layer_input,
            ple_vocab_per_layer_input: raw.vocab_size_per_layer_input,
            use_double_wide_mlp: raw.use_double_wide_mlp,
            moe,
        })
    }

    fn rope_spec(raw: &RopeSpecRaw, proportional: bool) -> Result<RopeSpec, GeometryError> {
        let want = if proportional {
            "proportional"
        } else {
            "default"
        };
        if raw.rope_type != want {
            return Err(GeometryError::new(format!(
                "rope_type {:?} for {want} rope; expected {want:?}",
                raw.rope_type
            )));
        }
        if raw.rope_theta <= 0.0 {
            return Err(GeometryError::new("rope_theta must be positive"));
        }
        Ok(RopeSpec {
            theta: raw.rope_theta,
            partial_rotary_factor: raw.partial_rotary_factor,
            proportional,
        })
    }

    /// Indices of the full (global) layers, in ascending order.
    #[must_use]
    pub fn global_layer_indices(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter_map(|(index, kind)| (*kind == LayerType::Full).then_some(index))
            .collect()
    }

    /// Indices of the sliding (local) layers, in ascending order.
    #[must_use]
    pub fn local_layer_indices(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter_map(|(index, kind)| (*kind == LayerType::Sliding).then_some(index))
            .collect()
    }

    /// The first global layer index — the source for the global attention mask.
    #[must_use]
    pub fn first_global_layer(&self) -> usize {
        self.global_layer_indices()[0]
    }

    /// The first sliding layer index — the source for the local attention mask.
    #[must_use]
    pub fn first_sliding_layer(&self) -> usize {
        self.local_layer_indices()[0]
    }

    /// Whether this is the dense unified 12B-class geometry.
    #[must_use]
    pub const fn is_dense_unified(&self) -> bool {
        matches!(self.model_type, TextModelType::Gemma4UnifiedText)
    }

    /// Whether this is a Mixture-of-Experts geometry (26B-A4B).
    #[must_use]
    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// A 5:1 ``layer_types`` array of length ``n`` (last entry always full).
    fn five_to_one(n: usize) -> Vec<String> {
        let mut out = Vec::with_capacity(n);
        for index in 0..n {
            out.push(
                if (index + 1) % 6 == 0 || index + 1 == n {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
                .to_string(),
            );
        }
        out
    }

    /// A 4:1 ``layer_types`` array of length ``n`` (E2B shape; last always full).
    fn four_to_one(n: usize) -> Vec<String> {
        let mut out = Vec::with_capacity(n);
        for index in 0..n {
            out.push(
                if (index + 1) % 5 == 0 || index + 1 == n {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
                .to_string(),
            );
        }
        out
    }

    /// Base 12B text_config as a JSON Value; tests mutate fields then serialize.
    fn base_12b() -> Value {
        json!({
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": true,
            "bos_token_id": 2,
            "dtype": "bfloat16",
            "enable_moe_block": false,
            "eos_token_id": 1,
            "final_logit_softcapping": 30.0,
            "global_head_dim": 512,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 3840,
            "hidden_size_per_layer_input": 0,
            "initializer_range": 0.02,
            "intermediate_size": 15360,
            "layer_types": five_to_one(48),
            "max_position_embeddings": 262144,
            "model_type": "gemma4_unified_text",
            "moe_intermediate_size": null,
            "num_attention_heads": 16,
            "num_experts": null,
            "num_global_key_value_heads": 1,
            "num_hidden_layers": 48,
            "num_key_value_heads": 8,
            "num_kv_shared_layers": 0,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            },
            "sliding_window": 1024,
            "tie_word_embeddings": true,
            "top_k_experts": null,
            "use_bidirectional_attention": "vision",
            "use_cache": true,
            "use_double_wide_mlp": false,
            "vocab_size": 262144,
            "vocab_size_per_layer_input": 0
        })
    }

    fn parse(value: &Value) -> Result<Geometry, GeometryError> {
        Geometry::from_text_config_str(&value.to_string())
    }

    #[test]
    fn parses_dense_12b_geometry() {
        let geometry = parse(&base_12b()).unwrap();
        assert!(geometry.is_dense_unified());
        assert!(!geometry.is_moe());
        assert_eq!(geometry.num_hidden_layers, 48);
        assert_eq!(geometry.hidden_size, 3840);
        assert_eq!(geometry.intermediate_size, 15360);
        assert_eq!(geometry.head_dim_local, 256);
        assert_eq!(geometry.head_dim_global, 512);
        assert_eq!(geometry.num_attention_heads, 16);
        assert_eq!(geometry.num_kv_heads_local, 8);
        assert_eq!(geometry.num_kv_heads_global, 1);
        assert!(geometry.attention_k_eq_v_global);
        assert_eq!(geometry.sliding_window, 1024);
        assert_eq!(geometry.vocab_size, 262_144);
        assert_eq!(geometry.max_position_embeddings, 262_144);
        assert_eq!(geometry.final_logit_softcapping, 30.0);
        assert_eq!(geometry.rope_local.theta, 10_000.0);
        assert!(geometry.rope_local.partial_rotary_factor.is_none());
        assert_eq!(geometry.rope_global.theta, 1_000_000.0);
        assert_eq!(geometry.rope_global.partial_rotary_factor, Some(0.25));
        assert_eq!(
            geometry.global_layer_indices(),
            vec![5, 11, 17, 23, 29, 35, 41, 47]
        );
        assert_eq!(geometry.first_global_layer(), 5);
        assert_eq!(geometry.first_sliding_layer(), 0);
        assert_eq!(geometry.global_layer_indices().len(), 8);
        assert_eq!(geometry.local_layer_indices().len(), 40);
        assert_eq!(geometry.layer_types.last(), Some(&LayerType::Full));
    }

    #[test]
    fn parses_e4b_shared_kv_and_ple() {
        let mut value = base_12b();
        value["model_type"] = json!("gemma4_text");
        value["attention_k_eq_v"] = json!(false);
        value["hidden_size"] = json!(2560);
        value["intermediate_size"] = json!(10240);
        value["num_attention_heads"] = json!(8);
        value["num_key_value_heads"] = json!(2);
        value["num_global_key_value_heads"] = json!(null);
        value["num_kv_shared_layers"] = json!(18);
        value["num_hidden_layers"] = json!(42);
        value["layer_types"] = json!(five_to_one(42));
        value["sliding_window"] = json!(512);
        value["max_position_embeddings"] = json!(131072);
        value["hidden_size_per_layer_input"] = json!(256);
        value["vocab_size_per_layer_input"] = json!(262144);
        value["use_bidirectional_attention"] = json!(null);

        let geometry = parse(&value).unwrap();
        assert!(!geometry.is_dense_unified());
        assert!(!geometry.attention_k_eq_v_global);
        assert_eq!(geometry.num_hidden_layers, 42);
        assert_eq!(geometry.num_kv_heads_local, 2);
        // null num_global_key_value_heads falls back to num_key_value_heads.
        assert_eq!(geometry.num_kv_heads_global, 2);
        assert_eq!(geometry.sliding_window, 512);
        assert_eq!(geometry.ple_hidden_per_layer_input, 256);
        assert_eq!(geometry.num_kv_shared_layers, 18);
        assert_eq!(geometry.global_layer_indices().len(), 7);
        assert_eq!(geometry.local_layer_indices().len(), 35);
        assert_eq!(geometry.layer_types.last(), Some(&LayerType::Full));
    }

    #[test]
    fn parses_e2b_four_to_one_pattern() {
        // E2B is the 4:1 exception (28 local / 7 global = 35 layers); proves the
        // parser derives indices from the array rather than a hard-coded 5:1 rule.
        let mut value = base_12b();
        value["model_type"] = json!("gemma4_text");
        value["hidden_size"] = json!(1536);
        value["intermediate_size"] = json!(6144);
        value["num_attention_heads"] = json!(8);
        value["num_key_value_heads"] = json!(1);
        value["num_global_key_value_heads"] = json!(null);
        value["num_hidden_layers"] = json!(35);
        value["layer_types"] = json!(four_to_one(35));
        value["sliding_window"] = json!(512);
        value["max_position_embeddings"] = json!(131072);
        value["hidden_size_per_layer_input"] = json!(256);
        value["num_kv_shared_layers"] = json!(20);
        value["use_double_wide_mlp"] = json!(true);
        value["vocab_size_per_layer_input"] = json!(262144);
        value["use_bidirectional_attention"] = json!(null);

        let geometry = parse(&value).unwrap();
        assert_eq!(geometry.num_hidden_layers, 35);
        assert_eq!(geometry.global_layer_indices().len(), 7);
        assert_eq!(geometry.local_layer_indices().len(), 28);
        assert!(geometry.use_double_wide_mlp);
    }

    #[test]
    fn parses_26b_moe_geometry() {
        let mut value = base_12b();
        value["model_type"] = json!("gemma4_text");
        value["hidden_size"] = json!(2816);
        value["intermediate_size"] = json!(2112);
        value["enable_moe_block"] = json!(true);
        value["moe_intermediate_size"] = json!(704);
        value["num_experts"] = json!(128);
        value["top_k_experts"] = json!(8);
        value["num_attention_heads"] = json!(16);
        value["num_key_value_heads"] = json!(8);
        value["num_global_key_value_heads"] = json!(2);
        value["num_hidden_layers"] = json!(30);
        value["layer_types"] = json!(five_to_one(30));
        value["use_bidirectional_attention"] = json!(null);

        let geometry = parse(&value).unwrap();
        assert!(geometry.is_moe());
        let moe = geometry.moe.unwrap();
        assert_eq!(moe.num_experts, 128);
        assert_eq!(moe.top_k, 8);
        assert_eq!(moe.moe_intermediate_size, 704);
        assert_eq!(geometry.num_kv_heads_global, 2);
        assert!(geometry.attention_k_eq_v_global);
    }

    #[test]
    fn rejects_unknown_text_config_field() {
        let mut value = base_12b();
        value["fabricated_future_field"] = json!(true);
        let error = parse(&value).unwrap_err();
        assert!(
            error.to_string().contains("fabricated_future_field"),
            "expected the unknown field to surface, got: {error}"
        );
    }

    #[test]
    fn rejects_missing_required_field() {
        let mut value = base_12b();
        value.as_object_mut().unwrap().remove("sliding_window");
        assert!(parse(&value).is_err());
    }

    #[test]
    fn rejects_last_layer_not_full() {
        let mut value = base_12b();
        let mut layers = five_to_one(48);
        layers[47] = "sliding_attention".to_string();
        value["layer_types"] = json!(layers);
        let error = parse(&value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("last layer must be full_attention")
        );
    }

    #[test]
    fn rejects_layer_types_length_mismatch() {
        let mut value = base_12b();
        value["layer_types"] = json!(five_to_one(42));
        // num_hidden_layers stays 48 → mismatch.
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("layer_types length"));
    }

    #[test]
    fn rejects_unsupported_activation() {
        let mut value = base_12b();
        value["hidden_activation"] = json!("relu");
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("gelu_pytorch_tanh"));
    }

    #[test]
    fn rejects_untied_embeddings() {
        let mut value = base_12b();
        value["tie_word_embeddings"] = json!(false);
        assert!(parse(&value).is_err());
    }

    #[test]
    fn rejects_local_rope_with_partial_factor() {
        let mut value = base_12b();
        value["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = json!(0.5);
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("full rotary"));
    }

    #[test]
    fn rejects_moe_enabled_without_expert_fields() {
        let mut value = base_12b();
        value["enable_moe_block"] = json!(true);
        // num_experts/top_k/moe_intermediate stay null → missing.
        let error = parse(&value).unwrap_err();
        assert!(error.to_string().contains("num_experts"));
    }
}
