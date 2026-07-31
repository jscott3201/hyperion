#include "kv_cache.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

namespace mx = mlx::core;

namespace hyperion::model {

namespace {

/// Convert a (head, dim, capacity) triple to an MLX shape.
mx::Shape kv_shape(std::uint32_t capacity, std::uint32_t heads, std::uint32_t dim) {
    return {static_cast<int>(capacity), static_cast<int>(heads), static_cast<int>(dim)};
}

} // namespace

KvGrowthPlan plan_kv_growth(const KvState& state, std::uint32_t n_tokens) {
    KvGrowthPlan plan;
    plan.n_tokens = n_tokens;

    for (const auto& cache : state.local) {
        const std::uint64_t proposed =
            static_cast<std::uint64_t>(cache.committed_len()) + n_tokens;
        if (proposed > std::numeric_limits<std::uint32_t>::max()) {
            plan.representable = false;
            return plan;
        }
    }

    for (const auto& cache : state.global) {
        const std::uint64_t proposed =
            static_cast<std::uint64_t>(cache.committed_len()) + n_tokens;
        if (proposed > std::numeric_limits<std::uint32_t>::max()) {
            plan.representable = false;
            return plan;
        }
        const std::uint64_t step = cache.step();
        const std::uint64_t steps = proposed / step + (proposed % step != 0);
        const std::uint64_t required = steps * step;
        if (required > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
            required > std::numeric_limits<std::uint32_t>::max()) {
            plan.representable = false;
            return plan;
        }
        const auto projected = static_cast<std::uint32_t>(
            std::max<std::uint64_t>(cache.capacity(), required));
        plan.requires_transaction =
            plan.requires_transaction || projected > cache.capacity();
    }
    return plan;
}

std::uint32_t projected_global_capacity(
    const GlobalKvCache& cache,
    std::uint32_t n_tokens) {
    const std::uint64_t proposed =
        static_cast<std::uint64_t>(cache.committed_len()) + n_tokens;
    const std::uint64_t step = cache.step();
    const std::uint64_t steps = proposed / step + (proposed % step != 0);
    return static_cast<std::uint32_t>(
        std::max<std::uint64_t>(cache.capacity(), steps * step));
}

KvGrowthTransaction::KvGrowthTransaction(
    std::unique_ptr<KvState>& live,
    const KvGrowthPlan& plan)
    : live_(live) {
    if (live_ == nullptr) {
        throw std::invalid_argument("KV growth transaction requires live state");
    }
    if (!plan.representable) {
        throw std::overflow_error("KV operation exceeds representable cache dimensions");
    }
    if (plan.requires_transaction) {
        staged_ = std::make_unique<KvState>(*live_);
    }
}

std::vector<mx::array> KvGrowthTransaction::staged_outputs() const {
    std::vector<mx::array> outputs;
    if (!active()) {
        return outputs;
    }
    outputs.reserve(2 * (staged_->local.size() + staged_->global.size()));
    for (const auto& cache : staged_->local) {
        outputs.push_back(cache.keys());
        outputs.push_back(cache.values());
    }
    for (const auto& cache : staged_->global) {
        outputs.push_back(cache.keys());
        outputs.push_back(cache.values());
    }
    return outputs;
}

mx::array KvGrowthTransaction::root_forward_result(const mx::array& output) const {
    if (!active()) {
        return output;
    }
    return mx::depends({output}, staged_outputs()).front();
}

void KvGrowthTransaction::materialize() {
    if (!active()) {
        return;
    }
    mx::eval(staged_outputs());
    materialized_ = true;
}

void KvGrowthTransaction::publish() {
    if (published_) {
        throw std::logic_error("KV growth transaction published more than once");
    }
    if (!active()) {
        return;
    }
    if (!materialized_) {
        throw std::logic_error("KV growth transaction published before materialization");
    }
    live_.swap(staged_);
    staged_.reset();
    published_ = true;
}

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

void LocalKvCache::append_committed(const mx::array& k_update, const mx::array& v_update, std::uint32_t n) {
    if (n == 0) {
        return;
    }
    const std::uint32_t cap = capacity();
    const auto h = static_cast<int>(num_kv_heads_);
    const auto d = static_cast<int>(head_dim_);
    // Chunk by cap so each write_ring writes ≤ cap tokens (write_ring's 2-slice split
    // assumes n ≤ cap). The ring rotates naturally: once committed > cap, older
    // out-of-window tokens are overwritten (the rotation read reconstructs logical order).
    // The gamma speculative slack is bypassed — prefill commits directly (commit(take)
    // with speculative_len==0 promotes `take` as fresh committed).
    for (std::uint32_t written = 0; written < n;) {
        const std::uint32_t start_pos = index_.committed_len(); // logical start of this sub-chunk
        const std::uint32_t take = std::min(cap, n - written);
        index_.commit(take); // advance committed_len by `take` (speculative stays 0)
        const auto w0 = static_cast<int>(written);
        const auto w1 = static_cast<int>(written + take);
        auto slice_update = [&](const mx::array& u) {
            return mx::slice(u, {w0, 0, 0}, {w1, h, d}, {1, 1, 1}, stream_);
        };
        k_ = write_ring(k_, slice_update(k_update), index_.slot_for(start_pos), take, cap, num_kv_heads_, head_dim_, stream_);
        v_ = write_ring(v_, slice_update(v_update), index_.slot_for(start_pos), take, cap, num_kv_heads_, head_dim_, stream_);
        written += take;
    }
}

mx::array LocalKvCache::read_window(const mx::array& buf, std::uint32_t window, int h, int d, mx::Stream s) const {
    const std::uint32_t committed = index_.attention_len(); // == committed_len
    const std::uint32_t cap = capacity();
    const std::uint32_t n = std::min(window, committed);
    // n ≥ 1 for offset>0 (committed ≥ 1); the general path handles any n.
    const auto start = static_cast<double>(committed - n);
    const auto stop = static_cast<double>(committed);
    mx::array idx = mx::remainder(
        mx::arange(start, stop, mx::int32, s),
        mx::array(static_cast<int>(cap), mx::int32),
        s); // [n] physical slots in logical order
    mx::array gathered = mx::take(buf, idx, /*axis=*/0, s); // [n, h, d] logical order
    mx::array r = mx::reshape(gathered, {1, static_cast<int>(n), h, d}, s);
    return mx::transpose(r, {0, 2, 1, 3}, s); // [1, h, n, d]
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
