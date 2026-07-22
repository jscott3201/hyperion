#include "geometry.h"

#include <algorithm>
#include <stdexcept>

namespace hyperion::model {

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
