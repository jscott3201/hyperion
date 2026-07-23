//! Bridge a validated [`Geometry`] to the C-ABI [`HypGeometryParams`] the native
//! loader expects.
//!
//! `hyperion-model` stays FFI-free, so this seam lives in `hyperion-core` (the
//! crate that already depends on both `hyperion-model` and `hyperion-ffi`).
//!
//! The one lifetime hazard is `HypGeometryParams.layer_types: *const c_int`,
//! which borrows a buffer. [`AbiGeometry`] owns that buffer (`Vec<c_int>`)
//! alongside the params struct, so a single value keeps the pointer valid for
//! as long as the `&HypGeometryParams` handed to [`Model::load`] is live.

use std::os::raw::c_int;

use hyperion_ffi::{
    HYP_GEMMA4_TEXT, HYP_GEMMA4_UNIFIED_TEXT, HYP_LAYER_FULL, HYP_LAYER_SLIDING, HypGeometryParams,
    HypMoeConfig, HypRopeSpec,
};
use hyperion_model::geometry::{Geometry, LayerType, RopeSpec, TextModelType};

/// An owned `HypGeometryParams` plus the `layer_types` buffer it borrows.
///
/// Build with [`AbiGeometry::from_geometry`]; borrow the params with
/// [`AbiGeometry::params`]. The buffer lives as long as this value, so the
/// `layer_types` pointer inside `params` stays valid — never move the params
/// out and outlive this owner (the API makes that hard: `params` returns a
/// `&HypGeometryParams` tied to `&self`).
pub struct AbiGeometry {
    /// The C-ABI params. `layer_types` points into `layer_types_buf` below.
    params: HypGeometryParams,
    /// Owns the buffer `params.layer_types` points at. Kept as a field so the
    /// pointer is valid for the lifetime of this `AbiGeometry`.
    layer_types_buf: Vec<c_int>,
}

impl AbiGeometry {
    /// Translate a validated `Geometry` into the ABI params the native loader
    /// consumes. Every `Geometry` field is mapped; no field is silently dropped.
    ///
    /// # Panics
    /// Never in practice — `Geometry` validation guarantees `layer_types` is
    /// non-empty and the rope specs are well-formed, so the conversions are
    /// total. The `f64` rope theta → `f32` ABI field is a narrowing that the
    /// native side accepts (the header declares `theta` as `f64`, so this is
    /// in fact a straight copy; see `HypRopeSpec`).
    #[must_use]
    pub fn from_geometry(geometry: &Geometry) -> Self {
        // layer_types: Vec<LayerType> -> Vec<c_int> (the buffer the pointer borrows).
        let layer_types_buf: Vec<c_int> = geometry
            .layer_types
            .iter()
            .map(|kind| match kind {
                LayerType::Sliding => HYP_LAYER_SLIDING,
                LayerType::Full => HYP_LAYER_FULL,
            })
            .collect();

        let model_type = match geometry.model_type {
            TextModelType::Gemma4UnifiedText => HYP_GEMMA4_UNIFIED_TEXT,
            TextModelType::Gemma4Text => HYP_GEMMA4_TEXT,
        };

        let (moe, has_moe) = match &geometry.moe {
            Some(m) => (
                HypMoeConfig {
                    num_experts: m.num_experts,
                    top_k: m.top_k,
                    moe_intermediate_size: m.moe_intermediate_size,
                },
                1,
            ),
            None => (
                HypMoeConfig {
                    num_experts: 0,
                    top_k: 0,
                    moe_intermediate_size: 0,
                },
                0,
            ),
        };

        let mut params = HypGeometryParams {
            model_type,
            hidden_size: geometry.hidden_size,
            intermediate_size: geometry.intermediate_size,
            num_hidden_layers: geometry.num_hidden_layers,
            layer_types: layer_types_buf.as_ptr(),
            num_attention_heads: geometry.num_attention_heads,
            head_dim_local: geometry.head_dim_local,
            head_dim_global: geometry.head_dim_global,
            num_kv_heads_local: geometry.num_kv_heads_local,
            num_kv_heads_global: geometry.num_kv_heads_global,
            attention_k_eq_v_global: c_int::from(geometry.attention_k_eq_v_global),
            num_kv_shared_layers: geometry.num_kv_shared_layers,
            sliding_window: geometry.sliding_window,
            rope_local: rope_to_abi(&geometry.rope_local),
            rope_global: rope_to_abi(&geometry.rope_global),
            final_logit_softcapping: geometry.final_logit_softcapping,
            rms_norm_eps: geometry.rms_norm_eps,
            attention_bias: c_int::from(geometry.attention_bias),
            vocab_size: geometry.vocab_size,
            max_position_embeddings: geometry.max_position_embeddings,
            tie_word_embeddings: c_int::from(geometry.tie_word_embeddings),
            ple_hidden_per_layer_input: geometry.ple_hidden_per_layer_input,
            ple_vocab_per_layer_input: geometry.ple_vocab_per_layer_input,
            use_double_wide_mlp: c_int::from(geometry.use_double_wide_mlp),
            has_moe,
            moe,
        };

        // Pin the pointer to the owned buffer (already set above, but make the
        // intent explicit: the buffer is a field, the pointer borrows it).
        params.layer_types = layer_types_buf.as_ptr();

        Self {
            params,
            layer_types_buf,
        }
    }

    /// Borrow the ABI params. The `layer_types` pointer inside is valid for
    /// as long as this `AbiGeometry` is live.
    #[must_use]
    pub const fn params(&self) -> &HypGeometryParams {
        &self.params
    }

    /// The `layer_types` values as a safe slice. This is the same buffer the
    /// `params().layer_types` pointer borrows, exposed without `unsafe` (which
    /// is confined to `hyperion-ffi`).
    #[must_use]
    pub fn layer_types_slice(&self) -> &[c_int] {
        &self.layer_types_buf
    }
}

/// Map one RoPE spec to its ABI mirror. The `partial_rotary_factor` `Option`
/// becomes an explicit `has_partial_rotary_factor` flag + value (the C struct
/// has no nullable floats); `Geometry` validation already guarantees the local
/// rope is full-rotary (no factor) and the global rope sets one.
fn rope_to_abi(spec: &RopeSpec) -> HypRopeSpec {
    match spec.partial_rotary_factor {
        Some(factor) => HypRopeSpec {
            theta: spec.theta,
            has_partial_rotary_factor: 1,
            partial_rotary_factor: factor,
            proportional: c_int::from(spec.proportional),
        },
        None => HypRopeSpec {
            theta: spec.theta,
            has_partial_rotary_factor: 0,
            partial_rotary_factor: 0.0,
            proportional: c_int::from(spec.proportional),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// Reuses the hyperion-model 12B base config shape; kept local so this test
    /// does not depend on a private helper in another crate.
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
            "layer_types": (0..48).map(|i| if (i + 1) % 6 == 0 || i + 1 == 48 { "full_attention" } else { "sliding_attention" }).collect::<Vec<_>>(),
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

    #[test]
    fn maps_every_dense_12b_field_into_abi_params() {
        let geometry = Geometry::from_text_config_str(&base_12b().to_string()).unwrap();
        let abi = AbiGeometry::from_geometry(&geometry);
        let p = abi.params();

        assert_eq!(p.model_type, HYP_GEMMA4_UNIFIED_TEXT);
        assert_eq!(p.hidden_size, 3840);
        assert_eq!(p.intermediate_size, 15360);
        assert_eq!(p.num_hidden_layers, 48);
        assert_eq!(p.num_attention_heads, 16);
        assert_eq!(p.head_dim_local, 256);
        assert_eq!(p.head_dim_global, 512);
        assert_eq!(p.num_kv_heads_local, 8);
        assert_eq!(p.num_kv_heads_global, 1);
        assert_eq!(p.attention_k_eq_v_global, 1);
        assert_eq!(p.num_kv_shared_layers, 0);
        assert_eq!(p.sliding_window, 1024);
        assert_eq!(p.final_logit_softcapping, 30.0);
        assert_eq!(p.rms_norm_eps, 1e-6);
        assert_eq!(p.attention_bias, 0);
        assert_eq!(p.vocab_size, 262_144);
        assert_eq!(p.max_position_embeddings, 262_144);
        assert_eq!(p.tie_word_embeddings, 1);
        assert_eq!(p.ple_hidden_per_layer_input, 0);
        assert_eq!(p.ple_vocab_per_layer_input, 0);
        assert_eq!(p.use_double_wide_mlp, 0);
        assert_eq!(p.has_moe, 0);
        assert_eq!(p.moe.num_experts, 0);
        // Rope: local full-rotary (no factor), global partial 0.25 proportional.
        assert_eq!(p.rope_local.theta, 10_000.0);
        assert_eq!(p.rope_local.has_partial_rotary_factor, 0);
        assert_eq!(p.rope_local.proportional, 0);
        assert_eq!(p.rope_global.theta, 1_000_000.0);
        assert_eq!(p.rope_global.has_partial_rotary_factor, 1);
        assert_eq!(p.rope_global.partial_rotary_factor, 0.25);
        assert_eq!(p.rope_global.proportional, 1);
    }

    #[test]
    fn layer_types_buffer_is_the_pointer_target_and_matches_kinds() {
        // The pointer inside params must point at the owned buffer, and the
        // kinds must map 5:1 (full at indices 5,11,...,47; sliding elsewhere).
        let geometry = Geometry::from_text_config_str(&base_12b().to_string()).unwrap();
        let abi = AbiGeometry::from_geometry(&geometry);
        let p = abi.params();
        assert!(!p.layer_types.is_null());
        // The safe slice is the same buffer the pointer borrows.
        let slice = abi.layer_types_slice();
        assert_eq!(slice.len(), 48);
        for (i, kind) in slice.iter().enumerate() {
            let expected = if (i + 1) % 6 == 0 || i + 1 == 48 {
                HYP_LAYER_FULL
            } else {
                HYP_LAYER_SLIDING
            };
            assert_eq!(*kind, expected, "layer {i} kind mismatch");
        }
        // The pointer must equal the buffer's base (proof of ownership). This
        // raw-pointer comparison is not an `unsafe` operation (no dereference).
        assert_eq!(p.layer_types, abi.layer_types_buf.as_ptr());
    }

    #[test]
    fn maps_moe_geometry_with_has_moe_flag() {
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
        value["layer_types"] = json!(
            (0..30)
                .map(|i| if (i + 1) % 6 == 0 || i + 1 == 30 {
                    "full_attention"
                } else {
                    "sliding_attention"
                })
                .collect::<Vec<_>>()
        );
        value["use_bidirectional_attention"] = json!(null);

        let geometry = Geometry::from_text_config_str(&value.to_string()).unwrap();
        let abi = AbiGeometry::from_geometry(&geometry);
        let p = abi.params();
        assert_eq!(p.model_type, HYP_GEMMA4_TEXT);
        assert_eq!(p.has_moe, 1);
        assert_eq!(p.moe.num_experts, 128);
        assert_eq!(p.moe.top_k, 8);
        assert_eq!(p.moe.moe_intermediate_size, 704);
    }

    #[test]
    fn e4b_attention_k_eq_v_false_maps_to_zero_flag() {
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
        value["layer_types"] = json!(
            (0..42)
                .map(|i| if (i + 1) % 6 == 0 || i + 1 == 42 {
                    "full_attention"
                } else {
                    "sliding_attention"
                })
                .collect::<Vec<_>>()
        );
        value["sliding_window"] = json!(512);
        value["max_position_embeddings"] = json!(131072);
        value["hidden_size_per_layer_input"] = json!(256);
        value["vocab_size_per_layer_input"] = json!(262144);
        value["use_bidirectional_attention"] = json!(null);

        let geometry = Geometry::from_text_config_str(&value.to_string()).unwrap();
        let abi = AbiGeometry::from_geometry(&geometry);
        let p = abi.params();
        assert_eq!(p.attention_k_eq_v_global, 0);
        assert_eq!(p.num_kv_heads_global, 2); // null fell back to num_key_value_heads
        assert_eq!(p.ple_hidden_per_layer_input, 256);
        assert_eq!(p.num_kv_shared_layers, 18);
    }
}
