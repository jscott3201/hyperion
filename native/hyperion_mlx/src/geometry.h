#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hyperion::model {

/// A per-token attention layer kind in the Gemma 4 hybrid layout.
enum class LayerType : std::uint8_t {
    /// Sliding-window (local) attention; head_dim 256.
    Sliding,
    /// Full causal (global) attention; head_dim 512, K=V when attention_k_eq_v.
    Full,
};

/// The Gemma 4 text backbone model_type accepted by Hyperion v1.
enum class TextModelType : std::uint8_t {
    /// E-series / 26B / 31B text.
    Gemma4Text,
    /// Dense unified text (12B).
    Gemma4UnifiedText,
};

/// The RoPE scheme for one attention kind.
struct RopeSpec {
    /// ``rope_theta`` from config.
    double theta;
    /// Fraction of head_dim rotated; ``std::nullopt`` means full rotary (local).
    std::optional<float> partial_rotary_factor;
    /// Whether this is the proportional (global) or default (local) scheme.
    bool proportional;
};

/// Mixture-of-experts block geometry (26B-A4B; absent on the dense sizes).
struct MoeConfig {
    std::uint32_t num_experts;
    std::uint32_t top_k;
    std::uint32_t moe_intermediate_size;
};

/// Validated, engine-ready geometry mirrored from ``hyperion-model::geometry``.
///
/// Rust owns config.json parsing and validation; the C ABI passes the validated
/// fields here (1.3). This struct re-checks only the invariants the native graph
/// needs as a second line of defense against ABI misuse. The native graph, KV
/// layout, per-layer-kind masks, and governor read from this struct.
struct Geometry {
    TextModelType model_type;
    std::uint32_t hidden_size;
    std::uint32_t intermediate_size;
    std::uint32_t num_hidden_layers;
    std::vector<LayerType> layer_types;
    std::uint32_t num_attention_heads;
    std::uint32_t head_dim_local;
    std::uint32_t head_dim_global;
    std::uint32_t num_kv_heads_local;
    std::uint32_t num_kv_heads_global;
    /// Global-only flag; true => the key tensor IS the value tensor.
    bool attention_k_eq_v_global;
    std::uint32_t num_kv_shared_layers;
    std::uint32_t sliding_window;
    RopeSpec rope_local;
    RopeSpec rope_global;
    float final_logit_softcapping;
    float rms_norm_eps;
    bool attention_bias;
    std::uint32_t vocab_size;
    std::uint32_t max_position_embeddings;
    bool tie_word_embeddings;
    std::uint32_t ple_hidden_per_layer_input;
    std::uint32_t ple_vocab_per_layer_input;
    bool use_double_wide_mlp;
    std::optional<MoeConfig> moe;

    /// Re-check the graph-critical invariants. Returns ``std::nullopt`` on
    /// success or an explanatory message on failure (no exceptions across ABI).
    [[nodiscard]] std::optional<std::string> validate() const;

    /// Indices of the full (global) layers, ascending.
    [[nodiscard]] std::vector<std::size_t> global_layer_indices() const;
    /// Indices of the sliding (local) layers, ascending.
    [[nodiscard]] std::vector<std::size_t> local_layer_indices() const;
    /// First global layer index — the source for the global attention mask.
    [[nodiscard]] std::size_t first_global_layer() const;
    /// First sliding layer index — the source for the local attention mask.
    [[nodiscard]] std::size_t first_sliding_layer() const;
    /// Whether this is the dense unified 12B-class geometry.
    [[nodiscard]] bool is_dense_unified() const;
    /// Whether this is a Mixture-of-Experts geometry (26B-A4B).
    [[nodiscard]] bool is_moe() const;
};

} // namespace hyperion::model
