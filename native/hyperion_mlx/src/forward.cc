#include "forward.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
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

/// Read ``n`` cached tokens from a cache buffer ``[cap, h, d]`` (no batch dim) as
/// ``[B=1, h, n, d]`` for the SDPA: slice ``[0:n]`` → reshape ``[1, n, h, d]`` →
/// transpose ``{0,2,1,3}``. The reverse of the append path's squeeze. ``n`` must
/// be ≤ capacity and, for a local ring, in the linear (un-rotated) region
/// (``slot_for(pos) == pos`` for ``pos < capacity``) — the rotation read past the
/// window is 2.6.
mx::array read_kv_view(const mx::array& buf, std::uint32_t n, int h, int d, mx::Stream s) {
    const auto nn = static_cast<int>(n);
    mx::array sl = mx::slice(buf, {0, 0, 0}, {nn, h, d}, {1, 1, 1}, s); // [n, h, d]
    mx::array r = mx::reshape(sl, {1, nn, h, d}, s);                      // [1, n, h, d]
    return mx::transpose(r, {0, 2, 1, 3}, s);                             // [1, h, n, d]
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

mx::array ForwardPass::lm_head(const mx::array& h) const {
    // Tied lm_head: the quantized embed table IS the lm_head weight, stored
    // [vocab, hidden] = [out, in] (the QuantizedLinear::w layout). transpose=true
    // fuses dequant + the transposed matmul → [B, L, vocab] logits. No manual
    // transpose, no full-table dequant (the q/k/v/o path, weights.h).
    return mx::quantized_matmul(
        h,
        weights_.embed_weight,
        weights_.embed_scales,
        std::optional<mx::array>(weights_.embed_biases),
        /*transpose=*/true,
        std::optional<int>(weights_.embed_group_size),
        std::optional<int>(weights_.embed_bits),
        /*mode=*/"affine",
        stream_);
}

mx::array ForwardPass::softcap(const mx::array& logits) const {
    // mlx-lm logit_softcap: tanh(x / softcap) * softcap, no fp32 cast (operates
    // in the lm_head output dtype). final_logit_softcapping = 30.0 for Gemma 4.
    const float sc = geometry_.final_logit_softcapping;
    mx::array scaled = mx::divide(logits, mx::array(sc, logits.dtype()), stream_);
    return mx::multiply(mx::tanh(scaled, stream_), mx::array(sc, logits.dtype()), stream_);
}

ForwardPass::GreedySample ForwardPass::sample_greedy(const mx::array& logits_last) const {
    // THE 2.3b LESSON: MLX lazy-graph mx::argmax/mx::max scalars go STALE across
    // repeated evals in a decode loop (graph-cache aliasing). Read raw contiguous
    // data<float>() pointers and host-scan for top-1 AND top-2 in one pass. This
    // is the load-bearing choice in the slice (the seam): a GPU mx::argmax+topk
    // port would risk the stale-scalar bug; the 262144-vocab CPU scan is a G2
    // perf flag, not a block.
    const mx::Stream cs = mx::default_stream(mx::Device::cpu);
    mx::array lf = mx::astype(mx::contiguous(logits_last, false, cs), mx::float32, cs);
    mx::eval(lf); // MLX lazy-graph materialization (NOT JS/Python eval): force-eval before data<float>().
    const float* p = lf.data<float>();
    const std::size_t n = lf.size();
    float top1 = -std::numeric_limits<float>::infinity();
    float top2 = -std::numeric_limits<float>::infinity();
    std::uint32_t arg = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const float v = p[i];
        if (v > top1) {
            top2 = top1;
            top1 = v;
            arg = static_cast<std::uint32_t>(i);
        } else if (v > top2) {
            top2 = v;
        }
    }
    // near_tie = top-2 logit gap < 0.5 (A1, 08:49 — the near_tie_events counter).
    return GreedySample{arg, top1, (top1 - top2) < 0.5F};
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
    // k/v are [B=1, n_kv_heads, L, head_dim] (post-transpose). The cache slot layout is
    // [n_kv_heads, head_dim] per position → append needs [L, n_kv_heads, head_dim].
    // reshape([1,nkv,L,hd] → [L,nkv,hd]) would SCRAMBLE (reinterprets the flat buffer
    // with nkv and L swapped); transpose to [1,L,nkv,hd] FIRST so the flat order matches
    // [L,nkv,hd]. For nkv==1 (global) this is a no-op (size-1 axis) — which is why the
    // 2.3b global path was fine and only the sliding (nkv>1) cache was scrambled. The
    // offset==0 SDPA read chunk-local k (not the cache) so 2.3b never exercised this.
    mx::array k_append = mx::reshape(mx::transpose(k, {0, 2, 1, 3}, stream_), {L, n_kv_heads, head_dim}, stream_);
    mx::array v_append = mx::reshape(mx::transpose(v, {0, 2, 1, 3}, stream_), {L, n_kv_heads, head_dim}, stream_);
    if (ref.kind == LayerType::Sliding) {
        LocalKvCache& cache = kvstate.local[ref.per_kind_index];
        // Direct committed append (chunked by cap internally) — replaces the gamma-
        // chunked prefill_append_local (which looped L/gamma times; impractical for a
        // 2048-token prefill chunk). The gamma-limited speculative append stays for MTP.
        cache.append_committed(k_append, v_append, static_cast<std::uint32_t>(L));
    } else {
        GlobalKvCache& cache = kvstate.global[ref.per_kind_index];
        cache.append(k_append, v_append, static_cast<std::uint32_t>(L));
    }

    // SDPA K/V: offset==0 attends over the chunk's own post-rope K/V (the 2.3b
    // bit-exact path, UNCHANGED — do not risk the proven seal). offset>0 (decode)
    // reads the full cached prefix INCL. this chunk's just-appended K/V from the
    // cache buffers — the cached-prefix attention read (2.7). The sliding ring is
    // in its linear region for pos < window (slot_for(pos) == pos); the rotation
    // read past the window is 2.6. attn_len == committed == offset + L.
    mx::array k_attn = k;
    mx::array v_attn = v;
    if (offset > 0) {
        if (ref.kind == LayerType::Sliding) {
            LocalKvCache& cache = kvstate.local[ref.per_kind_index];
            const std::uint32_t attn = cache.attention_len(); // == committed
            const std::uint32_t win = geometry_.sliding_window;
            // Sliding attention reads only the last min(window, committed) tokens.
            // For attn <= window all committed tokens are in-window → the linear slice
            // [0:attn] is exactly the last attn tokens (the 2.7 decode fast-path, bit-
            // identical to read_kv_rotated for this case — keeps the 2.7 seal on the same
            // code path, zero risk). Past the window the ring has rotated (or will, once
            // attn > cap) → read_kv_rotated gathers the last min(window, attn) in logical
            // order via slot_for (the 2.6 rotation read). The mask (kv_len =
            // min(window, offset+L)) matches the read length.
            if (attn <= win) {
                k_attn = read_kv_view(cache.keys(), attn, n_kv_heads, head_dim, stream_);
                v_attn = read_kv_view(cache.values(), attn, n_kv_heads, head_dim, stream_);
            } else {
                k_attn = cache.read_window(cache.keys(), win, n_kv_heads, head_dim, stream_);
                v_attn = cache.read_window(cache.values(), win, n_kv_heads, head_dim, stream_);
            }
        } else {
            GlobalKvCache& cache = kvstate.global[ref.per_kind_index];
            // Global cache never rotates (capacity grows with committed); linear read.
            k_attn = read_kv_view(cache.keys(), cache.committed_len(), n_kv_heads, head_dim, stream_);
            v_attn = read_kv_view(cache.values(), cache.committed_len(), n_kv_heads, head_dim, stream_);
        }
    }
    // scale=1.0 — Gemma 4 drops the 1/sqrt(d) scaling (QK-norm compensates).
    const std::string mask_mode = ""; // explicit boolean mask via mask_arr
    mx::array out = mx::fast::scaled_dot_product_attention(
        q, k_attn, v_attn,
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
    // The mask's kv_len must match the SDPA's K/V read length:
    //  - offset==0: the SDPA reads the chunk-local K/V (kv_len = committed = offset+L = L),
    //    for BOTH kinds (the 2.3b bit-exact path; the cache is appended but not read).
    //  - offset>0 sliding: the cache read returns only min(window, committed) tokens.
    //  - offset>0 global: the cache read returns all committed (full causal, no window).
    // For offset==0 with L > window (a 2048-token prefill chunk) the sliding mask is
    // [L, L] with the trailing-window constraint (zeroes out-of-window) — matching the
    // chunk-local K/V of length L (NOT a window-sized slice).
    const int L = static_cast<int>(h.shape(1));
    const std::uint32_t q_len = static_cast<std::uint32_t>(L);
    const std::uint32_t committed = offset + q_len;
    const std::uint32_t win = geometry_.sliding_window;
    mx::array state = h;
    for (std::size_t layer = 0; layer < weights_.layers.size(); ++layer) {
        const LayerType kind = dispatch_.per_layer[layer];
        const std::uint32_t kv_len =
            (offset > 0 && kind == LayerType::Sliding) ? std::min(win, committed) : committed;
        mx::array mask = build_mask(kind, q_len, kv_len, offset);
        state = decoder_layer(state, layer, mask, kvstate, offset);
    }
    return final_norm(state);
}

} // namespace hyperion::model
