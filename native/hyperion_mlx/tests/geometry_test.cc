#include "geometry.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using hyperion::model::Geometry;
using hyperion::model::LayerType;
using hyperion::model::MoeConfig;
using hyperion::model::RopeSpec;
using hyperion::model::TextModelType;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "geometry_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::vector<LayerType> five_to_one(std::size_t n) {
    std::vector<LayerType> out;
    out.reserve(n);
    for (std::size_t index = 0; index < n; ++index) {
        out.push_back((index + 1) % 6 == 0 || index + 1 == n ? LayerType::Full
                                                              : LayerType::Sliding);
    }
    return out;
}

Geometry make_12b() {
    Geometry geometry{};
    geometry.model_type = TextModelType::Gemma4UnifiedText;
    geometry.hidden_size = 3840;
    geometry.intermediate_size = 15360;
    geometry.num_hidden_layers = 48;
    geometry.layer_types = five_to_one(48);
    geometry.num_attention_heads = 16;
    geometry.head_dim_local = 256;
    geometry.head_dim_global = 512;
    geometry.num_kv_heads_local = 8;
    geometry.num_kv_heads_global = 1;
    geometry.attention_k_eq_v_global = true;
    geometry.num_kv_shared_layers = 0;
    geometry.sliding_window = 1024;
    geometry.rope_local = RopeSpec{10000.0, std::optional<float>(), false};
    geometry.rope_global =
        RopeSpec{1000000.0, std::optional<float>(0.25F), true};
    geometry.final_logit_softcapping = 30.0F;
    geometry.rms_norm_eps = 1e-6F;
    geometry.attention_bias = false;
    geometry.vocab_size = 262144;
    geometry.max_position_embeddings = 262144;
    geometry.tie_word_embeddings = true;
    geometry.ple_hidden_per_layer_input = 0;
    geometry.ple_vocab_per_layer_input = 0;
    geometry.use_double_wide_mlp = false;
    geometry.moe = std::optional<MoeConfig>();
    return geometry;
}

} // namespace

int main() {
    const auto geometry = make_12b();
    require(!geometry.validate().has_value(), "12B geometry should validate");
    require(geometry.is_dense_unified(), "12B is dense unified");
    require(!geometry.is_moe(), "12B is not MoE");
    require(geometry.num_hidden_layers == 48, "num_hidden_layers");
    require(geometry.head_dim_local == 256, "head_dim_local");
    require(geometry.head_dim_global == 512, "head_dim_global");
    require(geometry.num_kv_heads_local == 8, "num_kv_heads_local");
    require(geometry.num_kv_heads_global == 1, "num_kv_heads_global");
    require(geometry.attention_k_eq_v_global, "attention_k_eq_v_global");
    require(geometry.sliding_window == 1024, "sliding_window");
    require(geometry.global_layer_indices() == std::vector<std::size_t>{
                                                    5, 11, 17, 23, 29, 35, 41, 47},
            "global indices");
    require(geometry.first_global_layer() == 5, "first global layer");
    require(geometry.first_sliding_layer() == 0, "first sliding layer");
    require(geometry.global_layer_indices().size() == 8, "global count");
    require(geometry.local_layer_indices().size() == 40, "local count");

    // E4B: shared-KV + PLE + K!=V + null global kv heads -> local count.
    auto e4b = make_12b();
    e4b.model_type = TextModelType::Gemma4Text;
    e4b.attention_k_eq_v_global = false;
    e4b.hidden_size = 2560;
    e4b.intermediate_size = 10240;
    e4b.num_attention_heads = 8;
    e4b.num_kv_heads_local = 2;
    e4b.num_kv_heads_global = 2; // falls back to num_key_value_heads on the Rust side
    e4b.num_kv_shared_layers = 18;
    e4b.num_hidden_layers = 42;
    e4b.layer_types = five_to_one(42);
    e4b.sliding_window = 512;
    e4b.max_position_embeddings = 131072;
    e4b.ple_hidden_per_layer_input = 256;
    e4b.ple_vocab_per_layer_input = 262144;
    require(!e4b.validate().has_value(), "E4B geometry should validate");
    require(!e4b.attention_k_eq_v_global, "E4B global K!=V");
    require(e4b.global_layer_indices().size() == 7, "E4B global count");
    require(e4b.local_layer_indices().size() == 35, "E4B local count");

    // 26B: MoE.
    auto twentysix = make_12b();
    twentysix.model_type = TextModelType::Gemma4Text;
    twentysix.hidden_size = 2816;
    twentysix.intermediate_size = 2112;
    twentysix.num_hidden_layers = 30;
    twentysix.layer_types = five_to_one(30);
    twentysix.num_kv_heads_global = 2;
    twentysix.moe = std::optional<MoeConfig>(MoeConfig{128, 8, 704});
    require(!twentysix.validate().has_value(), "26B MoE geometry should validate");
    require(twentysix.is_moe(), "26B is MoE");

    // Rejections.
    auto bad = make_12b();
    bad.layer_types.back() = LayerType::Sliding;
    require(bad.validate().has_value() &&
                 bad.validate()->find("last layer must be full_attention") != std::string::npos,
             "last-layer-not-full must reject");

    auto mismatch = make_12b();
    mismatch.layer_types = five_to_one(42); // num_hidden_layers stays 48
    require(mismatch.validate().has_value() &&
                 mismatch.validate()->find("does not match") != std::string::npos,
             "length mismatch must reject");

    auto window0 = make_12b();
    window0.sliding_window = 0;
    require(window0.validate().has_value() &&
                 window0.validate()->find("sliding_window") != std::string::npos,
             "zero sliding_window must reject");

    auto local_prop = make_12b();
    local_prop.rope_local.proportional = true;
    require(local_prop.validate().has_value() &&
                 local_prop.validate()->find("sliding rope must be the default") != std::string::npos,
             "proportional local rope must reject");

    auto global_no_partial = make_12b();
    global_no_partial.rope_global.partial_rotary_factor = std::optional<float>();
    require(global_no_partial.validate().has_value() &&
                 global_no_partial.validate()->find("global rope must set partial") !=
                     std::string::npos,
             "global rope without partial must reject");

    auto untied = make_12b();
    untied.tie_word_embeddings = false;
    require(untied.validate().has_value(), "untied embeddings must reject");

    auto bad_moe = make_12b();
    bad_moe.moe = std::optional<MoeConfig>(MoeConfig{0, 8, 704});
    require(bad_moe.validate().has_value() &&
                 bad_moe.validate()->find("moe block requires non-zero") != std::string::npos,
             "MoE with zero experts must reject");

    return 0;
}
