#include "geometry.h"

#include <algorithm>
#include <stdexcept>

namespace hyperion::model {

namespace {

std::optional<std::string> rope_from_abi(const HypRopeSpec& src, RopeSpec& out) {
    out.theta = src.theta;
    if (src.has_partial_rotary_factor != 0) {
        out.partial_rotary_factor = src.partial_rotary_factor;
    } else {
        out.partial_rotary_factor = std::nullopt;
    }
    out.proportional = (src.proportional != 0);
    return std::nullopt;
}

} // namespace

std::optional<std::string> Geometry::from_abi(const HypGeometryParams& p, Geometry& out) {
    if (p.layer_types == nullptr && p.num_hidden_layers > 0) {
        return "layer_types pointer is null";
    }
    switch (p.model_type) {
        case HYP_GEMMA4_TEXT: out.model_type = TextModelType::Gemma4Text; break;
        case HYP_GEMMA4_UNIFIED_TEXT: out.model_type = TextModelType::Gemma4UnifiedText; break;
        default: return "unknown HypTextModelType";
    }
    out.hidden_size = p.hidden_size;
    out.intermediate_size = p.intermediate_size;
    out.num_hidden_layers = p.num_hidden_layers;
    out.layer_types.clear();
    out.layer_types.reserve(p.num_hidden_layers);
    for (std::uint32_t i = 0; i < p.num_hidden_layers; ++i) {
        switch (p.layer_types[i]) {
            case HYP_LAYER_SLIDING: out.layer_types.push_back(LayerType::Sliding); break;
            case HYP_LAYER_FULL: out.layer_types.push_back(LayerType::Full); break;
            default: return "unknown HypLayerType at index " + std::to_string(i);
        }
    }
    out.num_attention_heads = p.num_attention_heads;
    out.head_dim_local = p.head_dim_local;
    out.head_dim_global = p.head_dim_global;
    out.num_kv_heads_local = p.num_kv_heads_local;
    out.num_kv_heads_global = p.num_kv_heads_global;
    out.attention_k_eq_v_global = (p.attention_k_eq_v_global != 0);
    out.num_kv_shared_layers = p.num_kv_shared_layers;
    out.sliding_window = p.sliding_window;
    if (auto err = rope_from_abi(p.rope_local, out.rope_local); err.has_value()) {
        return "rope_local: " + *err;
    }
    if (auto err = rope_from_abi(p.rope_global, out.rope_global); err.has_value()) {
        return "rope_global: " + *err;
    }
    out.final_logit_softcapping = p.final_logit_softcapping;
    out.rms_norm_eps = p.rms_norm_eps;
    out.attention_bias = (p.attention_bias != 0);
    out.vocab_size = p.vocab_size;
    out.max_position_embeddings = p.max_position_embeddings;
    out.tie_word_embeddings = (p.tie_word_embeddings != 0);
    out.ple_hidden_per_layer_input = p.ple_hidden_per_layer_input;
    out.ple_vocab_per_layer_input = p.ple_vocab_per_layer_input;
    out.use_double_wide_mlp = (p.use_double_wide_mlp != 0);
    if (p.has_moe != 0) {
        // Positional aggregate init (designated initializers are a C++20 extension;
        // the project is C++17 with -Wpedantic -Werror).
        out.moe = MoeConfig{p.moe.num_experts, p.moe.top_k, p.moe.moe_intermediate_size};
    } else {
        out.moe = std::nullopt;
    }
    return std::nullopt;
}


std::optional<std::string> Geometry::validate() const {
    if (layer_types.empty()) {
        return "layer_types is empty";
    }
    if (layer_types.size() !=
        static_cast<std::vector<LayerType>::size_type>(num_hidden_layers)) {
        return "layer_types length does not match num_hidden_layers";
    }
    if (layer_types.back() != LayerType::Full) {
        return "the last layer must be full_attention (Gemma 4 invariant)";
    }
    if (num_hidden_layers == 0) {
        return "num_hidden_layers must be non-zero";
    }
    if (sliding_window == 0) {
        return "sliding_window must be non-zero";
    }
    if (num_kv_shared_layers > num_hidden_layers) {
        return "num_kv_shared_layers exceeds num_hidden_layers";
    }
    if (head_dim_local == 0 || head_dim_global == 0) {
        return "head_dim_local and head_dim_global must be non-zero";
    }
    if (num_attention_heads == 0 || num_kv_heads_local == 0 || num_kv_heads_global == 0) {
        return "attention/KV head counts must be non-zero";
    }
    if (vocab_size == 0) {
        return "vocab_size must be non-zero";
    }
    if (rope_local.proportional) {
        return "sliding rope must be the default (full rotary) scheme";
    }
    if (rope_local.partial_rotary_factor.has_value()) {
        return "sliding rope must be full rotary (no partial_rotary_factor)";
    }
    if (!rope_global.proportional) {
        return "global rope must be the proportional scheme";
    }
    if (!rope_global.partial_rotary_factor.has_value()) {
        return "global rope must set partial_rotary_factor";
    }
    if (rope_local.theta <= 0.0 || rope_global.theta <= 0.0) {
        return "rope_theta must be positive";
    }
    if (!tie_word_embeddings) {
        return "tie_word_embeddings must be true (lm_head is tied to the embedding table)";
    }
    if (moe.has_value()) {
        const auto& moe_config = *moe;
        if (moe_config.num_experts == 0 || moe_config.top_k == 0 ||
            moe_config.moe_intermediate_size == 0) {
            return "moe block requires non-zero num_experts/top_k/moe_intermediate_size";
        }
    }
    return std::nullopt;
}

std::vector<std::size_t> Geometry::global_layer_indices() const {
    std::vector<std::size_t> indices;
    for (std::size_t index = 0; index < layer_types.size(); ++index) {
        if (layer_types[index] == LayerType::Full) {
            indices.push_back(index);
        }
    }
    return indices;
}

std::vector<std::size_t> Geometry::local_layer_indices() const {
    std::vector<std::size_t> indices;
    for (std::size_t index = 0; index < layer_types.size(); ++index) {
        if (layer_types[index] == LayerType::Sliding) {
            indices.push_back(index);
        }
    }
    return indices;
}

std::size_t Geometry::first_global_layer() const {
    const auto indices = global_layer_indices();
    return indices.front();
}

std::size_t Geometry::first_sliding_layer() const {
    const auto indices = local_layer_indices();
    return indices.front();
}

bool Geometry::is_dense_unified() const {
    return model_type == TextModelType::Gemma4UnifiedText;
}

bool Geometry::is_moe() const {
    return moe.has_value();
}

} // namespace hyperion::model
