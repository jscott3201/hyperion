#include "dispatch.h"

namespace hyperion::model {

const ArchSpec& DispatchTable::for_layer(std::size_t index) const {
    return per_layer[index] == LayerType::Full ? global : sliding;
}

DispatchTable build_dispatch(const Geometry& geometry) {
    // Source masks from the FIRST layer of each kind (A3) — never layer 0
    // unconditionally. For the 12B layout first_sliding_layer() == 0, but the
    // rule is "first of kind", so a geometry whose layer 0 is global would
    // correctly source the sliding mask from the first sliding layer instead.
    const std::size_t sliding_source = geometry.first_sliding_layer();
    const std::size_t global_source = geometry.first_global_layer();

    DispatchTable table{};

    table.sliding.kind = LayerType::Sliding;
    table.sliding.cache_kind = CacheKind::LocalRing;
    table.sliding.head_dim = geometry.head_dim_local;
    table.sliding.num_kv_heads = geometry.num_kv_heads_local;
    table.sliding.k_eq_v = false;
    table.sliding.rope = geometry.rope_local;
    table.sliding.mask_source_layer = sliding_source;
    table.sliding.window = geometry.sliding_window;

    table.global.kind = LayerType::Full;
    table.global.cache_kind = CacheKind::GlobalSteppedKV;
    table.global.head_dim = geometry.head_dim_global;
    table.global.num_kv_heads = geometry.num_kv_heads_global;
    table.global.k_eq_v = geometry.attention_k_eq_v_global;
    table.global.rope = geometry.rope_global;
    table.global.mask_source_layer = global_source;
    table.global.window = 0; // full causal attention

    table.per_layer.reserve(geometry.layer_types.size());
    for (const auto layer_type : geometry.layer_types) {
        table.per_layer.push_back(layer_type);
    }
    return table;
}

KvStashPolicy build_kv_stash_policy(const Geometry& geometry) {
    return KvStashPolicy{geometry.num_kv_shared_layers};
}

} // namespace hyperion::model
