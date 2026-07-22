#include "weights_loader.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace mx = mlx::core;

namespace hyperion::model {

namespace {

constexpr const char* kLayerPrefix = "language_model.model.layers.";
constexpr const char* kEmbedWeight = "language_model.model.embed_tokens.weight";
constexpr const char* kEmbedScales = "language_model.model.embed_tokens.scales";
constexpr const char* kEmbedBiases = "language_model.model.embed_tokens.biases";
constexpr const char* kFinalNorm = "language_model.model.norm.weight";

const mx::array& require_tensor(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& name) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) {
        throw std::runtime_error("weight manifest is missing tensor: " + name);
    }
    return it->second;
}

std::string layer_tensor(std::size_t layer, const std::string& suffix) {
    return std::string(kLayerPrefix) + std::to_string(layer) + "." + suffix;
}

QuantizedLinear make_proj(
    const std::unordered_map<std::string, mx::array>& tensors,
    const std::string& base,
    int group_size,
    int bits) {
    return QuantizedLinear{
        require_tensor(tensors, base + ".weight"),
        require_tensor(tensors, base + ".scales"),
        std::optional<mx::array>(require_tensor(tensors, base + ".biases")),
        group_size,
        bits,
    };
}

bool has_v_proj(const std::unordered_map<std::string, mx::array>& tensors, std::size_t layer) {
    return tensors.find(layer_tensor(layer, "self_attn.v_proj.weight")) != tensors.end();
}

} // namespace

mx::array ModelWeights::embed(const mx::array& ids, mx::Stream stream) const {
    // Gather rows of the packed table + their scales/biases, then dequantize ONLY the
    // gathered rows ([L, in]) — never the full [vocab, in] table (the 12B embed is
    // 262144×3840 ≈ 1 GB packed; dequantizing it residently is the OOM path).
    mx::array w = mx::take(embed_weight, ids, /*axis=*/0, stream);   // [L, in_packed]
    mx::array s = mx::take(embed_scales, ids, /*axis=*/0, stream);  // [L, groups]
    mx::array b = mx::take(embed_biases, ids, /*axis=*/0, stream);   // [L]
    return mx::dequantize(
        w,
        s,
        std::optional<mx::array>(b),
        std::optional<int>(embed_group_size),
        std::optional<int>(embed_bits),
        /*mode=*/"affine",
        /*global_scale=*/std::nullopt,
        /*dtype=*/std::optional<mx::Dtype>(mx::bfloat16),
        stream);
}

ModelWeights load_model_weights(
    const std::filesystem::path& artifact_dir,
    const Geometry& geometry,
    int group_size,
    int bits,
    mx::Stream cpu_stream) {
    if (!std::filesystem::is_directory(artifact_dir)) {
        throw std::runtime_error("weights artifact directory does not exist: " + artifact_dir.string());
    }

    // Load every model-*.safetensors shard in the directory and merge into one tensor
    // map. MLX's safetensors Load has no GPU kernel (the production CPU-load path); the
    // arrays stay lazy (mmap'd) until the forward pass evals them on the GPU stream,
    // which triggers the CPU→GPU transfer.
    std::unordered_map<std::string, mx::array> tensors;
    std::vector<std::filesystem::path> shards;
    for (const auto& entry : std::filesystem::directory_iterator(artifact_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            shards.push_back(entry.path());
        }
    }
    std::sort(shards.begin(), shards.end());
    if (shards.empty()) {
        throw std::runtime_error("no .safetensors shards found in " + artifact_dir.string());
    }
    for (const auto& shard : shards) {
        auto loaded = mx::load_safetensors(shard.string(), cpu_stream);
        for (auto& [key, value] : loaded.first) {
            tensors.emplace(std::move(key), std::move(value));
        }
    }

    // Positional aggregate init in struct-member order (designated initializers are
    // a C++20 extension; the project is C++17 with -Wpedantic -Werror).
    ModelWeights weights{
        require_tensor(tensors, kEmbedWeight),
        require_tensor(tensors, kEmbedScales),
        require_tensor(tensors, kEmbedBiases),
        group_size,
        bits,
        require_tensor(tensors, kFinalNorm),
        {},
    };
    weights.layers.reserve(geometry.num_hidden_layers);

    for (std::size_t i = 0; i < geometry.num_hidden_layers; ++i) {
        const bool schema_sliding = has_v_proj(tensors, i);
        const LayerType schema_kind = schema_sliding ? LayerType::Sliding : LayerType::Full;
        // Parity check: the weight schema's derived kind must match geometry. A mismatch
        // is a bug (never a coercion), mirroring WeightManifest::layer_kind vs geometry.
        if (schema_kind != geometry.layer_types[i]) {
            throw std::runtime_error(
                "layer " + std::to_string(i) +
                " weight schema (" + (schema_sliding ? "sliding" : "global") +
                ") disagrees with geometry layer_types");
        }

        const std::string attn = layer_tensor(i, "self_attn.");
        // Positional aggregate init in struct-member order (weights_loader.h).
        LayerWeights lw{
            make_proj(tensors, attn + "q_proj", group_size, bits),
            make_proj(tensors, attn + "k_proj", group_size, bits),
            schema_sliding
                ? std::optional<QuantizedLinear>(
                      make_proj(tensors, attn + "v_proj", group_size, bits))
                : std::nullopt,
            make_proj(tensors, attn + "o_proj", group_size, bits),
            make_proj(tensors, layer_tensor(i, "mlp.gate_proj"), group_size, bits),
            make_proj(tensors, layer_tensor(i, "mlp.up_proj"), group_size, bits),
            make_proj(tensors, layer_tensor(i, "mlp.down_proj"), group_size, bits),
            require_tensor(tensors, attn + "q_norm.weight"),
            require_tensor(tensors, attn + "k_norm.weight"),
            require_tensor(tensors, layer_tensor(i, "input_layernorm.weight")),
            require_tensor(tensors, layer_tensor(i, "post_attention_layernorm.weight")),
            require_tensor(tensors, layer_tensor(i, "pre_feedforward_layernorm.weight")),
            require_tensor(tensors, layer_tensor(i, "post_feedforward_layernorm.weight")),
            require_tensor(tensors, layer_tensor(i, "layer_scalar")),
            schema_kind,
        };
        weights.layers.push_back(std::move(lw));
    }
    return weights;
}

} // namespace hyperion::model
