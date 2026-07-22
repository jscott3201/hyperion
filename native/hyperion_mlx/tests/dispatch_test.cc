#include "dispatch.h"
#include "geometry.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using hyperion::model::build_dispatch;
using hyperion::model::build_kv_stash_policy;
using hyperion::model::CacheKind;
using hyperion::model::Geometry;
using hyperion::model::LayerType;
using hyperion::model::RopeSpec;
using hyperion::model::TextModelType;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "dispatch_test: " << message << '\n';
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
    geometry.moe = std::optional<hyperion::model::MoeConfig>();
    return geometry;
}

} // namespace

int main() {
    const auto geometry = make_12b();
    const auto table = build_dispatch(geometry);

    // Sliding spec.
    require(table.sliding.kind == LayerType::Sliding, "sliding kind");
    require(table.sliding.cache_kind == CacheKind::LocalRing, "sliding cache kind");
    require(table.sliding.head_dim == 256, "sliding head_dim");
    require(table.sliding.num_kv_heads == 8, "sliding kv heads");
    require(!table.sliding.k_eq_v, "sliding K!=V");
    require(!table.sliding.rope.proportional, "sliding rope default");
    require(!table.sliding.rope.partial_rotary_factor.has_value(),
            "sliding rope full rotary");
    require(table.sliding.rope.theta == 10000.0, "sliding rope theta");
    require(table.sliding.window == 1024, "sliding window");

    // Global spec.
    require(table.global.kind == LayerType::Full, "global kind");
    require(table.global.cache_kind == CacheKind::GlobalSteppedKV,
            "global cache kind");
    require(table.global.head_dim == 512, "global head_dim");
    require(table.global.num_kv_heads == 1, "global kv heads");
    require(table.global.k_eq_v, "global K=V");
    require(table.global.rope.proportional, "global rope proportional");
    require(table.global.rope.partial_rotary_factor.has_value(),
            "global rope partial");
    require(table.global.rope.theta == 1000000.0, "global rope theta");
    require(table.global.window == 0, "global window (full causal)");

    // Per-layer dispatch resolved once; indices match the layer_types array.
    require(table.per_layer.size() == 48, "per_layer length");
    require(table.per_layer[0] == LayerType::Sliding, "layer 0 sliding");
    require(table.per_layer[5] == LayerType::Full, "layer 5 full");
    require(table.per_layer[47] == LayerType::Full, "last layer full");
    require(&table.for_layer(0) == &table.sliding, "layer 0 -> sliding spec");
    require(&table.for_layer(5) == &table.global, "layer 5 -> global spec");
    require(&table.for_layer(47) == &table.global, "last -> global spec");

    // A3: masks are sourced from the FIRST layer of each kind, never layer 0
    // unconditionally. For the 12B layout first_sliding_layer == 0 (so the rule
    // and layer-0 coincide) — but the global mask source MUST be 5, not 0.
    require(table.sliding.mask_source_layer == 0,
            "sliding mask source = first sliding layer");
    require(table.global.mask_source_layer == 5,
            "global mask source = first global layer (A3), not layer 0");

    // A3 defensive case: a geometry whose layer 0 is Full must source the
    // sliding mask from the first Sliding layer (not 0), proving the rule holds
    // when "first of kind" != 0.
    auto leading_global = make_12b();
    leading_global.layer_types.assign(48, LayerType::Sliding);
    leading_global.layer_types[0] = LayerType::Full;
    leading_global.layer_types[47] = LayerType::Full; // last must be full
    const auto table_b = build_dispatch(leading_global);
    require(table_b.sliding.mask_source_layer == 1,
            "sliding mask source follows first sliding layer, not layer 0");
    require(table_b.global.mask_source_layer == 0,
            "global mask source = first global layer (0 here)");

    // The returned table is self-contained (no dangling Geometry pointer): the
    // rope specs are copies, so mutating the source must not change the table.
    auto mutated = make_12b();
    const auto table_c = build_dispatch(mutated);
    mutated.rope_local.theta = 999.0;
    require(table_c.sliding.rope.theta == 10000.0,
            "dispatch table must copy rope specs (no source aliasing)");

    // Stash policy: 12B has no cross-layer KV sharing.
    const auto stash = build_kv_stash_policy(geometry);
    require(!stash.shares_kv(), "12B does not share cross-layer KV");
    require(stash.num_shared_layers == 0, "12B shared layers = 0");

    // E4B records its shared-KV tail (full wiring lands at M8).
    auto e4b = make_12b();
    e4b.num_kv_shared_layers = 18;
    const auto stash_e4b = build_kv_stash_policy(e4b);
    require(stash_e4b.shares_kv(), "E4B shares cross-layer KV");
    require(stash_e4b.num_shared_layers == 18, "E4B shared layers = 18");

    return 0;
}
