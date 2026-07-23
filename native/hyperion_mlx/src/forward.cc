#include "forward.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace mx = mlx::core;

namespace hyperion::model {

namespace {

// RMSNorm over the last axis. ``weight`` = the learnable scale (layernorms, q_norm,
// k_norm); ``nullopt`` = the parameterless RMSNormNoScale used for v_norm (no tensor
// exists for it — 04/03). Mirrors mlx-lm's ``nn.RMSNorm`` / ``RMSNormNoScale``.
mx::array rms_norm(const mx::array& x, const std::optional<mx::array>& weight, float eps, mx::Stream s) {
    return mx::fast::rms_norm(x, weight, eps, s);
}

// Prefill append to a local ring: write L committed tokens, chunked by gamma_max (the
// ring's speculative slack — append reserves speculative slots ≤ gamma_max; prefill
// commits directly, so chunk). The ring rotates old out-of-window tokens. A single
// commit-past-speculative append (no chunking) is a 2.6 refinement.
void prefill_append_local(
    LocalKvCache& cache,
    const mx::array& k,
    const mx::array& v,
    int n,
    int n_kv_heads,
    int head_dim,
    mx::Stream s) {
    const auto h = static_cast<int>(n_kv_heads);
    const auto d = static_cast<int>(head_dim);
    const std::uint32_t gamma = kDefaultGammaMax;
    for (std::uint32_t written = 0; written < static_cast<std::uint32_t>(n);) {
        const std::uint32_t take = std::min(gamma, static_cast<std::uint32_t>(n) - written);
        const int start = static_cast<int>(written);
        const int stop = static_cast<int>(written + take);
        auto slice = [&](const mx::array& buf) {
            return mx::slice(buf, {start, 0, 0}, {stop, h, d}, {1, 1, 1}, s);
        };
        if (!cache.append(slice(k), slice(v), take)) {
            throw std::runtime_error("local KV ring gamma slack overflow (prefill chunk)");
        }
        cache.commit(take);
        written += take;
    }
}


} // namespace

ForwardPass::ForwardPass(
    const Geometry& geometry,
    const DispatchTable& dispatch,
    ModelWeights& weights,
    mx::Stream stream)
    : geometry_(geometry),
      dispatch_(dispatch),
      weights_(weights),
      stream_(stream),
      rope_local_(geometry.rope_local, geometry.head_dim_local, stream),
      rope_global_(geometry.rope_global, geometry.head_dim_global, stream),
      eps_(geometry.rms_norm_eps),
      embed_scale_(std::sqrt(static_cast<float>(geometry.hidden_size))) {
    // Precompute the per-layer cache dispatch (running local/global counts). This is the
    // "forward pass (2.3) resolves a layer index via per-layer kind + running counts"
    // from dispatch.h — resolved once at load, not per token. The A3 mask-by-kind rule
    // (source each mask from the first layer of its kind) is handled in build_mask; this
    // is the cache dispatch, which uses the same per-layer kind.
    cache_refs_.reserve(dispatch.per_layer.size());
    std::size_t local_seen = 0;
    std::size_t global_seen = 0;
    for (std::size_t i = 0; i < dispatch.per_layer.size(); ++i) {
        const LayerType kind = dispatch.per_layer[i];
        cache_refs_.push_back(LayerCacheRef{
            kind,
            (kind == LayerType::Sliding) ? local_seen : global_seen,
        });
        if (kind == LayerType::Sliding) {
            ++local_seen;
        } else {
            ++global_seen;
        }
    }
}

mx::array ForwardPass::embed(const mx::array& ids) const {
    // ids: [L] int32 → dequantized rows [L, hidden] → scaled by sqrt(hidden) → [1, L, hidden].
    mx::array rows = weights_.embed(ids, stream_);            // [L, hidden] bf16
    mx::array scaled = mx::multiply(rows, mx::array(embed_scale_, mx::bfloat16), stream_);
    return mx::reshape(scaled, {1, static_cast<int>(rows.shape(0)), static_cast<int>(rows.shape(1))}, stream_);
}

mx::array ForwardPass::final_norm(const mx::array& h) const {
    return rms_norm(h, std::optional<mx::array>(weights_.final_norm), eps_, stream_);
}

mx::array ForwardPass::gelu_tanh(const mx::array& x) const {
    // gelu_pytorch_tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    constexpr float kCoeff = 0.044715F;
    constexpr float kInner = 0.7978845608028654F; // sqrt(2/pi)
    mx::array x3 = mx::multiply(x, mx::square(x, stream_), stream_);          // x * x^2 = x^3
    mx::array inner = mx::multiply(mx::array(kCoeff, x.dtype()), x3, stream_); // 0.044715 * x^3
    inner = mx::add(inner, x, stream_);                                          // x + 0.044715*x^3
    inner = mx::multiply(mx::array(kInner, x.dtype()), inner, stream_);        // sqrt(2/pi) * (...)
    mx::array one = mx::array(1.0F, x.dtype());
    mx::array t = mx::tanh(inner, stream_);
    mx::array half = mx::array(0.5F, x.dtype());
    return mx::multiply(mx::multiply(half, x, stream_), mx::add(one, t, stream_), stream_);
}

mx::array ForwardPass::mlp(const mx::array& x, std::size_t layer) const {
    const LayerWeights& lw = weights_.layers[layer];
    mx::array gate = lw.gate_proj.apply(x, stream_);
    mx::array up = lw.up_proj.apply(x, stream_);
    return lw.down_proj.apply(mx::multiply(gelu_tanh(gate), up, stream_), stream_);
}

mx::array ForwardPass::build_mask(
    LayerType kind,
    std::uint32_t q_len,
    std::uint32_t kv_len,
    std::uint32_t offset) const {
    // Absolute positions. q covers [offset, offset+q_len); k covers the last kv_len
    // committed tokens [committed-kv_len, committed), committed = offset+q_len.
    const std::int64_t committed = static_cast<std::int64_t>(offset) + q_len;
    const std::int64_t base_k = committed - kv_len;
    mx::array abs_q = mx::arange(
        static_cast<double>(offset),
        static_cast<double>(offset + q_len),
        mx::int32,
        stream_); // [q_len]
    mx::array abs_k = mx::arange(
        static_cast<double>(base_k),
        static_cast<double>(committed),
        mx::int32,
        stream_); // [kv_len]
    abs_q = mx::reshape(abs_q, {static_cast<int>(q_len), 1}, stream_);
    abs_k = mx::reshape(abs_k, {1, static_cast<int>(kv_len)}, stream_);
    // causal: k <= q
    mx::array mask = mx::less_equal(abs_k, abs_q, stream_);
    if (kind == LayerType::Sliding) {
        // window: k > q - window  (trailing-window constraint)
        mx::array floor = mx::subtract(abs_q, mx::array(static_cast<int>(geometry_.sliding_window), mx::int32), stream_);
        mask = mx::logical_and(mask, mx::greater(abs_k, floor, stream_), stream_);
    }
    return mask; // [q_len, kv_len] bool
}

ForwardPass::AttentionInternals ForwardPass::attention(
    const mx::array& x,
    std::size_t layer,
    const mx::array& mask,
    KvState& kvstate,
    std::uint32_t offset) {
    const LayerWeights& lw = weights_.layers[layer];
    const ArchSpec& spec = dispatch_.for_layer(layer);
    const bool k_eq_v = spec.k_eq_v;
    const int n_heads = static_cast<int>(geometry_.num_attention_heads);
    const int n_kv_heads = static_cast<int>(spec.num_kv_heads);
    const int head_dim = static_cast<int>(spec.head_dim);
    const Rope& rope = (spec.kind == LayerType::Sliding) ? rope_local_ : rope_global_;

    // B = 1 for v1 (single-sequence prefill/decode; batched prefill is post-v1).
    const int B = static_cast<int>(x.shape(0));
    const int L = static_cast<int>(x.shape(1));

    // Projections → reshape to [B, L, h, head_dim]. QK-norm happens BEFORE RoPE.
    mx::array q = lw.q_proj.apply(x, stream_);                 // [B, L, n_heads*head_dim]
    q = mx::reshape(q, {B, L, n_heads, head_dim}, stream_);
    q = rms_norm(q, std::optional<mx::array>(lw.q_norm), eps_, stream_);

    mx::array k_raw = lw.k_proj.apply(x, stream_);             // [B, L, n_kv_heads*head_dim]
    mx::array k = mx::reshape(k_raw, {B, L, n_kv_heads, head_dim}, stream_);
    k = rms_norm(k, std::optional<mx::array>(lw.k_norm), eps_, stream_); // k_norm (scaled)

    // V: for k_eq_v (global) the value comes from the SAME k_proj but a DIFFERENT norm —
    // v_norm (parameterless RMSNorm, no weight tensor), NOT k_norm, and NO RoPE. So
    // K = rope(k_norm(raw)) and V = v_norm(raw) are distinct tensors even though no
    // v_proj weight exists (the gemma4_text.Attention applies v_norm unconditionally).
    // For sliding, V = v_norm(v_proj(x)). (Gemma 4: v_norm is RMSNormNoScale.)
    mx::array v = [&] {
        mx::array vv = k_eq_v ? k_raw : lw.v_proj->apply(x, stream_);
        vv = mx::reshape(vv, {B, L, n_kv_heads, head_dim}, stream_);
        return rms_norm(vv, std::nullopt, eps_, stream_);
    }();

    // Transpose to [B, h, L, head_dim] then RoPE on q,k (NOT v).
    q = mx::transpose(q, {0, 2, 1, 3}, stream_);
    k = mx::transpose(k, {0, 2, 1, 3}, stream_);
    v = mx::transpose(v, {0, 2, 1, 3}, stream_);
    q = rope.apply(q, offset);
    k = rope.apply(k, offset);

    // Append this chunk's post-rope K/V to the layer's cache (advances state for 2.6's
    // cached-prefix read). The cache is [cap, n_kv_heads, head_dim] (no batch): squeeze B.
    const LayerCacheRef ref = cache_refs_[layer];
    // Capture the post-rope K / post-v_norm V ([B, n_kv_heads, L, head_dim]) before the
    // reshape-for-append, for parity debugging.
    mx::array k_post = k;
    mx::array v_post = v;
    mx::array k_append = mx::reshape(k, {L, n_kv_heads, head_dim}, stream_);
    mx::array v_append = mx::reshape(v, {L, n_kv_heads, head_dim}, stream_);
    if (ref.kind == LayerType::Sliding) {
        LocalKvCache& cache = kvstate.local[ref.per_kind_index];
        prefill_append_local(cache, k_append, v_append, L, n_kv_heads, head_dim, stream_);
    } else {
        GlobalKvCache& cache = kvstate.global[ref.per_kind_index];
        cache.append(k_append, v_append, static_cast<std::uint32_t>(L));
    }

    // 2.3a: attend over the chunk's own K/V (offset 0 prefill). The cached-prefix read
    // (kv_len = offset+L) is 2.6; here kv_len == L. scale=1.0 — Gemma 4 drops the 1/sqrt(d)
    // scaling (QK-norm compensates); follow the reference exactly.
    const std::string mask_mode = ""; // explicit boolean mask via mask_arr
    mx::array out = mx::fast::scaled_dot_product_attention(
        q, k, v,
        /*scale=*/1.0F,
        mask_mode,
        std::optional<mx::array>(mask),
        /*sinks=*/std::nullopt,
        stream_); // [B, n_heads, L, head_dim]
    out = mx::transpose(out, {0, 2, 1, 3}, stream_);                       // [B, L, n_heads, head_dim]
    out = mx::reshape(out, {B, L, n_heads * head_dim}, stream_);
    return AttentionInternals{lw.o_proj.apply(out, stream_), k_post, v_post};
}

mx::array ForwardPass::decoder_layer(
    const mx::array& x,
    std::size_t layer,
    const mx::array& mask,
    KvState& kvstate,
    std::uint32_t offset) {
    const LayerWeights& lw = weights_.layers[layer];

    // Gemma 2 pre/post sandwich: norm the BRANCH OUTPUT before the residual add.
    mx::array residual = x;
    mx::array h = rms_norm(x, std::optional<mx::array>(lw.input_layernorm), eps_, stream_);
    h = attention(h, layer, mask, kvstate, offset).out;
    h = rms_norm(h, std::optional<mx::array>(lw.post_attention_layernorm), eps_, stream_);
    h = mx::add(residual, h, stream_);

    residual = h;
    h = rms_norm(h, std::optional<mx::array>(lw.pre_feedforward_layernorm), eps_, stream_);
    h = mlp(h, layer);
    h = rms_norm(h, std::optional<mx::array>(lw.post_feedforward_layernorm), eps_, stream_);
    h = mx::add(residual, h, stream_);

    // Per-layer residual scale (04). layer_scalar is [] or [1]; broadcast over hidden.
    h = mx::multiply(h, lw.layer_scalar, stream_);
    return h;
}

mx::array ForwardPass::forward(const mx::array& h, KvState& kvstate, std::uint32_t offset) {
    // 2.3a: offset 0 prefill only (the cached-prefix attention read is 2.6).
    assert(offset == 0 && "M2-2.3a forward supports offset 0 (cached-prefix read is 2.6)");
    const int L = static_cast<int>(h.shape(1));
    mx::array state = h;
    for (std::size_t layer = 0; layer < weights_.layers.size(); ++layer) {
        const LayerType kind = dispatch_.per_layer[layer];
        const std::uint32_t kv_len = static_cast<std::uint32_t>(L);
        mx::array mask = build_mask(kind, static_cast<std::uint32_t>(L), kv_len, offset);
        state = decoder_layer(state, layer, mask, kvstate, offset);
    }
    return final_norm(state);
}

} // namespace hyperion::model
