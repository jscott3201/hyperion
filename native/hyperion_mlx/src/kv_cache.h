#pragma once

#include "kv.h"
#include "dispatch.h"

#include <cstdint>
#include <vector>

#include <mlx/mlx.h>

namespace hyperion::model {

/// Default speculative slack for a local ring (MTP draft/verify, M7).
/// An engine constant — not a model-config field; the builder may override.
inline constexpr std::uint32_t kDefaultGammaMax = 8;

/// One sliding (local) layer's KV ring, backed by ``mlx::core::array`` tensors.
///
/// Shape: ``[capacity, num_kv_heads, head_dim]`` in the cache dtype.
/// ``capacity = window + gamma_max`` (flat forever, regardless of context).
///
/// Writes are functional (MLX arrays are immutable graph nodes): an append
/// reassigns the buffer via ``slice_update``. The ring is physically linear, so
/// an append of ``n`` tokens at sequence position ``next_write_pos`` may straddle
/// the capacity boundary; ``append`` splits it into one or two ``slice_update``s
/// so tokens land in the correct physical slots (``slot_for(pos) = pos % cap``).
///
/// Speculative appends (MTP draft) live in the gamma slack and are discarded by
/// *not committing* — never by trimming committed state (A2). Attention reads
/// committed tokens only (``attention_len``), excluding the speculative region
/// (A3): a verify pass reads committed KV, never uncommitted draft KV.
class LocalKvCache {
  public:
    LocalKvCache(
        std::uint32_t window,
        std::uint32_t gamma_max,
        std::uint32_t num_kv_heads,
        std::uint32_t head_dim,
        mlx::core::Dtype dtype,
        mlx::core::Stream stream);

    /// Capacity (``window + gamma_max``); flat forever.
    [[nodiscard]] std::uint32_t capacity() const { return index_.capacity(); }
    [[nodiscard]] std::uint32_t num_kv_heads() const { return num_kv_heads_; }
    [[nodiscard]] std::uint32_t head_dim() const { return head_dim_; }

    /// The ring index (arithmetic + A2/A3 bookkeeping).
    [[nodiscard]] const LocalRingIndex& index() const { return index_; }
    [[nodiscard]] std::uint32_t committed_len() const { return index_.committed_len(); }
    [[nodiscard]] std::uint32_t speculative_len() const { return index_.speculative_len(); }
    [[nodiscard]] std::uint32_t attention_len() const { return index_.attention_len(); }
    [[nodiscard]] std::uint32_t slot_for(std::uint32_t sequence_pos) const {
        return index_.slot_for(sequence_pos);
    }

    /// Write ``n`` tokens of K and V at the next write position (speculative —
    /// not visible to ``attention_len`` until ``commit``). ``k_update`` /
    /// ``v_update`` must be shape ``[n, num_kv_heads, head_dim]`` in the cache
    /// dtype. Returns false if ``n`` would overflow the gamma slack (caller
    /// must commit or discard first). Splits the write across the capacity
    /// boundary so wrapped tokens land in the correct physical slots.
    [[nodiscard]] bool append(const mlx::core::array& k_update, const mlx::core::array& v_update, std::uint32_t n);

    /// Promote ``n`` speculative tokens to committed (A2: append-only, no trim).
    void commit(std::uint32_t n) { index_.commit(n); }

    /// Discard all speculative tokens (MTP reject); committed is untouched (A2).
    void discard_speculative() { index_.discard_speculative(); }

    /// The full key/value tensors (shape ``[capacity, h, d]``). Attention (2.3)
    /// reads the committed prefix via ``attention_len`` and the slot map.
    [[nodiscard]] const mlx::core::array& keys() const { return k_; }
    [[nodiscard]] const mlx::core::array& values() const { return v_; }

  private:
    static mlx::core::array allocate(
        std::uint32_t capacity,
        std::uint32_t num_kv_heads,
        std::uint32_t head_dim,
        mlx::core::Dtype dtype,
        mlx::core::Stream stream);

    /// Write ``update`` (shape ``[n, h, d]``) into ``buf`` starting at physical
    /// slot ``start_slot``, splitting across the capacity boundary. Returns the
    /// updated buffer (functional ``slice_update`` reassigns).
    static mlx::core::array write_ring(
        mlx::core::array buf,
        const mlx::core::array& update,
        std::uint32_t start_slot,
        std::uint32_t n,
        std::uint32_t capacity,
        std::uint32_t num_kv_heads,
        std::uint32_t head_dim,
        mlx::core::Stream stream);

    LocalRingIndex index_;
    mlx::core::array k_;
    mlx::core::array v_;
    mlx::core::Stream stream_;
    std::uint32_t num_kv_heads_;
    std::uint32_t head_dim_;
};

/// One global (full) layer's capacity-stepped KV cache. K=V when ``k_eq_v`` —
/// one tensor, and ``keys()``/``values()`` alias it.
///
/// Shape: ``[capacity, num_kv_heads, head_dim]``; ``capacity`` grows in
/// ``step``-token increments (default 256, the prefill chunk / bucket size) at
/// bucket boundaries only — never mid-token. The only allocation event is a
/// capacity step crossing: allocate ``[new_cap, h, d]``, copy the old prefix
/// via ``slice_update``, then write the appended tokens. The governor checks KV
/// growth at bucket transitions, not every token, because this is the only
/// place the global cache allocates.
class GlobalKvCache {
  public:
    GlobalKvCache(
        std::uint32_t step,
        std::uint32_t num_kv_heads,
        std::uint32_t head_dim,
        bool k_eq_v,
        mlx::core::Dtype dtype,
        mlx::core::Stream stream);

    [[nodiscard]] std::uint32_t capacity() const { return index_.capacity(); }
    [[nodiscard]] std::uint32_t committed_len() const { return index_.committed_len(); }
    [[nodiscard]] std::uint32_t step_count() const { return index_.step_count(); }
    [[nodiscard]] std::uint32_t step() const { return step_; }
    [[nodiscard]] std::uint32_t num_kv_heads() const { return num_kv_heads_; }
    [[nodiscard]] std::uint32_t head_dim() const { return head_dim_; }
    [[nodiscard]] bool k_eq_v() const { return k_eq_v_; }

    /// The index (capacity-stepped arithmetic).
    [[nodiscard]] const GlobalCacheIndex& index() const { return index_; }

    /// Append ``n`` committed tokens, growing capacity by whole steps at bucket
    /// boundaries. ``k_update``/``v_update`` shape ``[n, h, d]`` in the cache
    /// dtype; ``v_update`` is ignored when ``k_eq_v`` (K IS V).
    void append(const mlx::core::array& k_update, const mlx::core::array& v_update, std::uint32_t n);

    [[nodiscard]] const mlx::core::array& keys() const { return k_; }
    /// V is stored separately from K (M2-2.7 fix: gemma4's k_eq_v means V comes from
    /// the same k_proj as K, but V = v_norm(k_proj) ≠ K = rope(k_norm(k_proj));
    /// aliasing V=K served K as V at the offset>0 read). ``k_eq_v_`` is now a
    /// forward-computation fact (no v_proj weight), not a cache-storage flag.
    [[nodiscard]] const mlx::core::array& values() const { return v_; }

  private:
    GlobalCacheIndex index_;
    /// Tracked alongside the index so we detect a step crossing (the tensor's
    /// own dim 0 is the allocated capacity; the index owns the arithmetic).
    std::uint32_t allocated_capacity_;
    mlx::core::array k_;
    mlx::core::array v_;
    mlx::core::Stream stream_;
    std::uint32_t step_;
    std::uint32_t num_kv_heads_;
    std::uint32_t head_dim_;
    bool k_eq_v_;
    mlx::core::Dtype dtype_;
};

/// Heterogeneous KV state for one model instance: one ``LocalKvCache`` per
/// sliding layer and one ``GlobalKvCache`` per global layer, in layer order.
/// ``local[j]`` is the j-th sliding layer's cache; ``global[j]`` the j-th global
/// layer's. The forward pass (2.3) resolves a layer index to its cache via the
/// DispatchTable's per-layer kind + the running local/global counts.
struct KvState {
    std::vector<LocalKvCache> local;
    std::vector<GlobalKvCache> global;
};

/// Allocate the heterogeneous KV state from a resolved dispatch table.
/// ``gamma_max`` defaults to ``kDefaultGammaMax``; ``dtype`` defaults to bf16
/// (the production KV dtype; tests may pass float32 for exact-value readback).
[[nodiscard]] KvState build_kv_state(
    const DispatchTable& dispatch,
    std::uint32_t gamma_max = kDefaultGammaMax,
    mlx::core::Dtype dtype = mlx::core::bfloat16,
    mlx::core::Stream stream = mlx::core::default_stream(mlx::core::Device::gpu));

} // namespace hyperion::model
