#pragma once

#include "geometry.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hyperion::model {

/// How a layer kind stores its KV cache.
enum class CacheKind : std::uint8_t {
    /// Sliding layers: an O(1) ring of window + gamma_max speculative slack.
    LocalRing,
    /// Global layers: a capacity-stepped single K=V tensor (8 KiB/token).
    GlobalSteppedKV,
};

/// Per-layer-kind architecture spec, resolved once per kind (A3/B2). A future
/// concrete ``LayerRunner`` (2.x, owns ``mx::array`` ops) reads this; the
/// dispatch table maps every layer index to its kind so dispatch is not
/// recomputed per token.
struct ArchSpec {
    LayerType kind;
    CacheKind cache_kind;
    std::uint32_t head_dim;
    std::uint32_t num_kv_heads;
    /// Global K=V (attention_k_eq_v); the key tensor IS the value tensor.
    bool k_eq_v;
    /// Copied from Geometry so the table is self-contained (no dangling).
    RopeSpec rope;
    /// First layer of this kind — the mask-construction source. NEVER layer 0
    /// unconditionally (A3): a rotating sliding cache clamps its offset to
    /// window-1, so a global mask built from a sliding layer's cache truncates
    /// and crashes (broadcast_shapes) once the sequence exceeds the window.
    std::size_t mask_source_layer;
    /// sliding_window for Sliding; 0 for Full (full causal attention).
    std::uint32_t window;
};

/// Per-layer dispatch table resolved once from Geometry. ``per_layer[i]`` is the
/// kind of layer ``i``; ``for_layer(i)`` returns its ArchSpec (a reference into
/// ``sliding`` or ``global``, which live with the table — no dangling pointers).
struct DispatchTable {
    ArchSpec sliding;
    ArchSpec global;
    /// The kind for each layer index; resolved once, not per token.
    std::vector<LayerType> per_layer;

    /// The ArchSpec for layer ``index``.
    [[nodiscard]] const ArchSpec& for_layer(std::size_t index) const;
};

/// Cross-layer KV stash policy ("stash only when consumed", B2). For the dense
/// 12B (num_kv_shared_layers == 0) nothing is stashed; E-series shared-KV wiring
/// (which owner layers feed which shared layers) is config-derived at M8.
struct KvStashPolicy {
    std::uint32_t num_shared_layers;
    [[nodiscard]] bool shares_kv() const { return num_shared_layers > 0; }
};

/// Resolve the per-kind dispatch table from a validated Geometry.
[[nodiscard]] DispatchTable build_dispatch(const Geometry& geometry);

/// Resolve the cross-layer KV stash policy from a validated Geometry.
[[nodiscard]] KvStashPolicy build_kv_stash_policy(const Geometry& geometry);

} // namespace hyperion::model
