#include "kv_cache.h"

#include <algorithm>
#include <stdexcept>

namespace mx = mlx::core;

namespace hyperion::model {

namespace {

/// Convert a (head, dim, capacity) triple to an MLX shape.
mx::Shape kv_shape(std::uint32_t capacity, std::uint32_t heads, std::uint32_t dim) {
    return {static_cast<int>(capacity), static_cast<int>(heads), static_cast<int>(dim)};
}

} // namespace

// ---------------------------------------------------------------------------
// LocalKvCache
// ---------------------------------------------------------------------------

mx::array LocalKvCache::allocate(
    std::uint32_t capacity,
    std::uint32_t num_kv_heads,
    std::uint32_t head_dim,
    mx::Dtype dtype,
    mx::Stream stream) {
    return mx::zeros(kv_shape(capacity, num_kv_heads, head_dim), dtype, stream);
}

LocalKvCache::LocalKvCache(
    std::uint32_t window,
    std::uint32_t gamma_max,
    std::uint32_t num_kv_heads,
    std::uint32_t head_dim,
    mx::Dtype dtype,
    mx::Stream stream)
    : index_(window, gamma_max),
      k_(allocate(window + gamma_max, num_kv_heads, head_dim, dtype, stream)),
      v_(allocate(window + gamma_max, num_kv_heads, head_dim, dtype, stream)),
      stream_(stream),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim) {
    if (num_kv_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("LocalKvCache num_kv_heads/head_dim must be non-zero");
    }
}

mx::array LocalKvCache::write_ring(
    mx::array buf,
    const mx::array& update,
    std::uint32_t start_slot,
    std::uint32_t n,
    std::uint32_t capacity,
    std::uint32_t num_kv_heads,
    std::uint32_t head_dim,
    mx::Stream stream) {
    // A contiguous logical append of n tokens at physical slot start_slot may
    // straddle the capacity boundary. MLX slice_update is linear (no wrap), so
    // split at the boundary: [start_slot : capacity] then [0 : remainder].
    const std::uint32_t first = std::min(n, capacity - start_slot);
    const auto h = static_cast<int>(num_kv_heads);
    const auto d = static_cast<int>(head_dim);

    const mx::array first_update = mx::slice(
        update, {0, 0, 0}, {static_cast<int>(first), h, d}, {1, 1, 1}, stream);
    buf = mx::slice_update(
        buf,
        first_update,
        {static_cast<int>(start_slot), 0, 0},
        {static_cast<int>(start_slot + first), h, d},
        {1, 1, 1},
        stream);

    const std::uint32_t remaining = n - first;
    if (remaining == 0) {
        return buf;
    }
    const mx::array second_update = mx::slice(
        update,
        {static_cast<int>(first), 0, 0},
        {static_cast<int>(n), h, d},
        {1, 1, 1},
        stream);
    buf = mx::slice_update(
        buf,
        second_update,
        {0, 0, 0},
        {static_cast<int>(remaining), h, d},
        {1, 1, 1},
        stream);
    return buf;
}

bool LocalKvCache::append(
    const mx::array& k_update,
    const mx::array& v_update,
    std::uint32_t n) {
    if (n == 0) {
        return true;
    }
    const std::uint32_t start_pos = index_.next_write_pos();
    if (!index_.append_speculative(n)) {
        return false; // gamma slack overflow; caller must commit/discard first
    }
    // next_write_pos did not move past the reserved region — write at start_pos.
    k_ = write_ring(k_, k_update, index_.slot_for(start_pos), n, capacity(), num_kv_heads_, head_dim_, stream_);
    v_ = write_ring(v_, v_update, index_.slot_for(start_pos), n, capacity(), num_kv_heads_, head_dim_, stream_);
    return true;
}

// ---------------------------------------------------------------------------
// GlobalKvCache
// ---------------------------------------------------------------------------

GlobalKvCache::GlobalKvCache(
    std::uint32_t step,
    std::uint32_t num_kv_heads,
    std::uint32_t head_dim,
    bool k_eq_v,
    mx::Dtype dtype,
    mx::Stream stream)
    : index_(step),
      allocated_capacity_(0),
      k_(mx::zeros(kv_shape(0, num_kv_heads, head_dim), dtype, stream)),
      v_(mx::zeros(kv_shape(0, num_kv_heads, head_dim), dtype, stream)),
      stream_(stream),
      step_(step),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      k_eq_v_(k_eq_v),
      dtype_(dtype) {
    if (num_kv_heads == 0 || head_dim == 0) {
        throw std::invalid_argument("GlobalKvCache num_kv_heads/head_dim must be non-zero");
    }
}

void GlobalKvCache::append(
    const mx::array& k_update,
    const mx::array& v_update,
    std::uint32_t n) {
    if (n == 0) {
        return;
    }
    const std::uint32_t start = index_.committed_len();
    index_.append(n); // grows step_count_ (and thus capacity) at bucket boundaries

    const std::uint32_t new_capacity = index_.capacity();
    if (new_capacity > allocated_capacity_) {
        // The only allocation event: cross a bucket boundary (or the first append).
        // Allocate the grown buffer and copy the committed prefix so existing KV
        // survives the grow. BOTH K and V are grown + written — V is NOT aliased to
        // K even when ``k_eq_v_`` (M2-2.7 fix): gemma4's k_eq_v means V comes from
        // the SAME k_proj as K (no separate v_proj weight), but V = v_norm(k_proj)
        // is a DISTINCT tensor from K = rope(k_norm(k_proj)). Aliasing V=K (as the
        // 2.1 design did) served K as V at the offset>0 read → wrong logits. The
        // forward computes the correct distinct V; the cache must store it.
        const auto h = static_cast<int>(num_kv_heads_);
        const auto d = static_cast<int>(head_dim_);
        auto grow_buf = [&](const mx::array& old) {
            mx::array g = mx::zeros(kv_shape(new_capacity, num_kv_heads_, head_dim_), dtype_, stream_);
            if (allocated_capacity_ > 0) {
                g = mx::slice_update(
                    g, old, {0, 0, 0},
                    {static_cast<int>(allocated_capacity_), h, d}, {1, 1, 1}, stream_);
            }
            return g;
        };
        k_ = grow_buf(k_);
        v_ = grow_buf(v_);
        allocated_capacity_ = new_capacity;
    }

    // Write the appended tokens contiguously at [start : start+n) (full causal —
    // no ring, no straddle). Both K and V are written (see the aliasing note above).
    const auto h = static_cast<int>(num_kv_heads_);
    const auto d = static_cast<int>(head_dim_);
    k_ = mx::slice_update(
        k_,
        k_update,
        {static_cast<int>(start), 0, 0},
        {static_cast<int>(start + n), h, d},
        {1, 1, 1},
        stream_);
    v_ = mx::slice_update(
        v_,
        v_update,
        {static_cast<int>(start), 0, 0},
        {static_cast<int>(start + n), h, d},
        {1, 1, 1},
        stream_);
}

// ---------------------------------------------------------------------------
// build_kv_state
// ---------------------------------------------------------------------------

KvState build_kv_state(
    const DispatchTable& dispatch,
    std::uint32_t gamma_max,
    mx::Dtype dtype,
    mx::Stream stream) {
    KvState state;
    const std::size_t num_layers = dispatch.per_layer.size();
    state.local.reserve(dispatch.per_layer.size());
    state.global.reserve(dispatch.per_layer.size());
    // One cache per layer, pushed into the per-kind vector in layer order so
    // local[j] is the j-th sliding layer and global[j] the j-th global layer.
    for (std::size_t i = 0; i < num_layers; ++i) {
        const ArchSpec& spec = dispatch.for_layer(i);
        if (spec.kind == LayerType::Sliding) {
            state.local.emplace_back(
                spec.window, gamma_max, spec.num_kv_heads, spec.head_dim, dtype, stream);
        } else {
            // The global step is the 256-token capacity bucket (the prefill
            // chunk / snapshot boundary), not a per-layer geometry field.
            state.global.emplace_back(
                256u,
                spec.num_kv_heads,
                spec.head_dim,
                spec.k_eq_v,
                dtype,
                stream);
        }
    }
    return state;
}

} // namespace hyperion::model
