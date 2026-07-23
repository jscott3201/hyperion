#pragma once

#include "dispatch.h"
#include "geometry.h"
#include "kv_cache.h"
#include "rope.h"
#include "weights_loader.h"

#include <cstdint>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace hyperion::model {

/// The native forward pass for the dense Gemma 4 12B text stack (04). One instance per
/// loaded model; owns the per-kind RoPE + the per-layer math. Define-by-run, uncompiled
/// (``mx::compile`` shape bucketing is a later sub-task — correctness-first).
///
/// Scope (M2-2.3a): the per-layer math — embed → ``num_hidden_layers``× DecoderLayer →
/// final RMSNorm — proven model-free against an inline reference. The attention reads the
/// chunk's own K/V (offset 0 prefill); the cache is APPENDED to advance state (the cached
///-prefix attention read + chunked prefill land at 2.6, decode + greedy sampler + lm_head
/// softcap epilogue at 2.7). ``resolve_layer`` is the per-layer-kind cache dispatch the
/// ``dispatch.h`` note flags as "the forward pass (2.3) resolves a layer index via
/// per-layer kind + running counts."
class ForwardPass {
  public:
    ForwardPass(
        const Geometry& geometry,
        const DispatchTable& dispatch,
        ModelWeights& weights,
        mlx::core::Stream stream);

    /// Embed token ids ``[L]`` → ``[1, L, hidden]`` scaled by ``sqrt(hidden)`` (the Gemma
    /// embed scale). The lm_head is TIED to this table but its application + softcap is 2.7.
    [[nodiscard]] mlx::core::array embed(const mlx::core::array& ids) const;

    /// Run all decoder layers over ``h`` ``[B, L, hidden]`` at ``offset`` (current committed
    /// KV length; 0 for a fresh prefill). Appends each layer's K/V to ``kvstate``. Returns
    /// the final-RMSNorm hidden state ``[B, L, hidden]``. Asserts ``offset == 0`` for 2.3a
    /// (cached-prefix attention read is 2.6).
    [[nodiscard]] mlx::core::array forward(
        const mlx::core::array& h,
        KvState& kvstate,
        std::uint32_t offset);

    /// Final RMSNorm (``model.norm``). Public so the 2.7 epilogue can reuse it.
    [[nodiscard]] mlx::core::array final_norm(const mlx::core::array& h) const;

    /// The attention mask for one kind, shape ``[q_len, kv_len]`` boolean (True = attend).
    /// ``offset`` is the committed prefix length; ``kv_len`` the attention read length.
    /// Sourced per-kind — A3: the caller never builds a global mask from a sliding cache.
    [[nodiscard]] mlx::core::array build_mask(
        LayerType kind,
        std::uint32_t q_len,
        std::uint32_t kv_len,
        std::uint32_t offset) const;

    /// Per-layer cache dispatch: ``local[j]`` is the j-th sliding layer's ``LocalKvCache``,
    /// ``global[j]`` the j-th global. Resolved once at construction from
    /// ``dispatch.per_layer`` (running local/global counts), not per token.
    struct LayerCacheRef {
        LayerType kind;
        std::size_t per_kind_index;
    };
    [[nodiscard]] const std::vector<LayerCacheRef>& cache_refs() const { return cache_refs_; }

    /// Drive one decoder layer (the parity test calls this per-layer to localize the
    /// first divergence vs the oracle). Equivalent to one iteration of ``forward``'s loop.
    [[nodiscard]] mlx::core::array run_layer(
        const mlx::core::array& x,
        std::size_t layer,
        const mlx::core::array& mask,
        KvState& kvstate,
        std::uint32_t offset) {
        return decoder_layer(x, layer, mask, kvstate, offset);
    }

    /// Attention internals for parity debugging: the post-o_proj output, the post-rope K,
    /// and the post-v_norm V (both ``[B, n_kv_heads, L, head_dim]``), before the cache
    /// append. The 2.3b test compares these to the oracle per-layer to localize a divergence.
    struct AttentionInternals {
        mlx::core::array out;
        mlx::core::array k;
        mlx::core::array v;
    };
    [[nodiscard]] AttentionInternals inspect_attention(
        const mlx::core::array& x,
        std::size_t layer,
        const mlx::core::array& mask,
        KvState& kvstate,
        std::uint32_t offset) {
        return attention(x, layer, mask, kvstate, offset);
    }

    /// Raw (pre-norm) projections for parity debugging — isolates the quantized matmul
    /// from the norms/RoPE. ``which`` ∈ {"q", "k"}.
    [[nodiscard]] mlx::core::array raw_proj(const mlx::core::array& x, std::size_t layer, char which) const {
        const auto& lw = weights_.layers[layer];
        return (which == 'k') ? lw.k_proj.apply(x, stream_) : lw.q_proj.apply(x, stream_);
    }

  private:
    /// One decoder layer (the 4-norm sandwich + attention + MLP + layer_scalar).
    [[nodiscard]] mlx::core::array decoder_layer(
        const mlx::core::array& x,
        std::size_t layer,
        const mlx::core::array& mask,
        KvState& kvstate,
        std::uint32_t offset);

    /// One attention block: Q/K/V proj → QK-norm (q/k scaled, v unscaled) → RoPE (q,k) →
    /// append to cache → SDPA(scale=1.0) → o_proj. Reads the chunk K/V (offset 0). Returns
    /// the post-o_proj output + post-rope K / post-v_norm V (for parity debugging).
    [[nodiscard]] AttentionInternals attention(
        const mx::array& x,
        std::size_t layer,
        const mx::array& mask,
        KvState& kvstate,
        std::uint32_t offset);

    /// GeGLU MLP: ``down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))``.
    [[nodiscard]] mx::array mlp(const mx::array& x, std::size_t layer) const;

    /// gelu_pytorch_tanh: ``0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))``.
    [[nodiscard]] mx::array gelu_tanh(const mx::array& x) const;

    const Geometry& geometry_;
    const DispatchTable& dispatch_;
    ModelWeights& weights_;
    mx::Stream stream_;
    Rope rope_local_;
    Rope rope_global_;
    float eps_;
    float embed_scale_;
    std::vector<LayerCacheRef> cache_refs_;
};

} // namespace hyperion::model
