#include "forward.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
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

/// The exact K/V-axis length consumed by attention. A multi-token continuation on a
/// sliding layer needs the prior trailing window plus every current token so each query
/// row can select its own causal window. Single-token decode keeps its sealed final-window
/// axis, while offset-0 prefill and global attention keep the full committed axis.
std::uint32_t attention_kv_len(
    LayerType kind,
    std::uint32_t q_len,
    std::uint32_t offset,
    std::uint32_t window) {
    const std::uint32_t committed = offset + q_len;
    if (kind != LayerType::Sliding || offset == 0) {
        return committed;
    }
    if (q_len == 1) {
        return std::min(window, committed);
    }
    return std::min(window, offset) + q_len;
}

/// Assemble a bounded transient ``[prior trailing prefix] + [current chunk]`` view for
/// continuation prefill. The persistent ring remains ``window + gamma_max``; this scratch
/// tensor is populated with slice_update so no grow-and-copy cache is introduced.
mx::array assemble_continuation_kv(
    const mx::array& prefix,
    const mx::array& current,
    int prefix_len,
    int q_len,
    int h,
    int d,
    mx::Stream s) {
    mx::array assembled = mx::zeros({1, h, prefix_len + q_len, d}, current.dtype(), s);
    assembled = mx::slice_update(
        assembled,
        prefix,
        {0, 0, 0, 0},
        {1, h, prefix_len, d},
        {1, 1, 1, 1},
        s);
    return mx::slice_update(
        assembled,
        current,
        {0, 0, prefix_len, 0},
        {1, h, prefix_len + q_len, d},
        {1, 1, 1, 1},
        s);
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

ForwardPass::StochasticSample ForwardPass::sample_stochastic(
    const mx::array& logits_last,
    const HypSamplingConfig& cfg,
    std::uint64_t rng_state) const {
    // REUSES the 2.3b host-scan seam (sample_greedy): contiguous f32 on the CPU stream,
    // eval, then raw data<float>(). NO mx::argmax/mx::random graph scalars (the stale-
    // scalar bug). All filter/sort/softmax/draw work is over the host float* vector.
    const mx::Stream cs = mx::default_stream(mx::Device::cpu);
    mx::array lf = mx::astype(mx::contiguous(logits_last, false, cs), mx::float32, cs);
    mx::eval(lf); // MLX lazy-graph materialization (NOT JS/Python eval).
    const std::size_t n = lf.size();
    const float* src = lf.data<float>();

    // Greedy default: temperature == 0 → the argmax (byte-identical to sample_greedy's
    // top-1). This also guards against a direct call with temp==0 (1/0 = inf would NaN
    // the logits); run_epilogue_sampled routes temp==0 here too. Fill the top-k sidecar.
    if (cfg.temperature == 0.0F) {
        StochasticSample out{};
        float top1 = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < n; ++i) if (src[i] > top1) { top1 = src[i]; out.token_id = static_cast<std::uint32_t>(i); }
        out.logit = top1;
        std::vector<std::uint32_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + HYP_TOP_K_LOGPROBS, idx.end(),
                          [&](std::uint32_t a, std::uint32_t b) { return src[a] > src[b]; });
        out.top_k_count = std::min<std::size_t>(HYP_TOP_K_LOGPROBS, n);
        for (std::size_t i = 0; i < out.top_k_count; ++i) {
            out.top_k_ids[i] = idx[i]; out.top_k_logprobs[i] = src[idx[i]];
        }
        return out;
    }

    std::vector<float> logits(src, src + n);

    // Temperature: logits *= (1/temperature). temp > 0 is validated by the caller.
    const float inv_temp = 1.0F / cfg.temperature;
    for (float& v : logits) v *= inv_temp;

    // ── mlx-lm filter order (sample_utils.py): top-p, min-p, top-k — each sets the
    //    losers to -inf (a mask), NOT a removal. All over the host vector. ──
    // First compute the softmax probabilities (post-temp, pre-filter) for the
    // mass-based filters (top-p, min-p need probs).
    float max_l = -std::numeric_limits<float>::infinity();
    for (float v : logits) max_l = std::max(max_l, v);
    std::vector<float> probs(n);
    float z = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        probs[i] = std::exp((logits[i] - max_l));
        z += probs[i];
    }
    if (z > 0.0F) for (float& p : probs) p /= z;

    // top-p (nucleus): sort ascending by prob, cumsum, keep the smallest set whose
    // cumulative mass >= top_p (the nucleus). Set the rest to -inf. mlx-lm keeps the
    // tokens where cumsum <= 1 - top_p removed; equivalently the nucleus is the top
    // tokens summing to >= top_p. We mark non-nucleus logits -inf.
    if (cfg.top_p > 0.0F && cfg.top_p <= 1.0F) {
        // indices sorted by prob descending
        std::vector<std::uint32_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](std::uint32_t a, std::uint32_t b) { return probs[a] > probs[b]; });
        float cum = 0.0F;
        std::vector<char> in_nucleus(n, 0);
        for (std::uint32_t i : idx) {
            in_nucleus[i] = 1;
            cum += probs[i];
            if (cum >= cfg.top_p) break;
        }
        for (std::size_t i = 0; i < n; ++i) if (!in_nucleus[i]) logits[i] = -std::numeric_limits<float>::infinity();
    }
    // min-p: keep tokens with prob >= min_p * max_prob. Set the rest to -inf.
    if (cfg.min_p > 0.0F && cfg.min_p < 1.0F) {
        float max_prob = 0.0F;
        for (float p : probs) max_prob = std::max(max_prob, p);
        const float threshold = cfg.min_p * max_prob;
        for (std::size_t i = 0; i < n; ++i) if (probs[i] < threshold) logits[i] = -std::numeric_limits<float>::infinity();
    }
    // top-k: keep the top-k logits; set the rest to -inf. partial_sort by logit desc.
    if (cfg.top_k > 0 && static_cast<std::uint32_t>(cfg.top_k) < n) {
        std::vector<std::uint32_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + cfg.top_k, idx.end(),
                          [&](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b]; });
        std::vector<char> keep(n, 0);
        for (int32_t i = 0; i < cfg.top_k; ++i) keep[idx[i]] = 1;
        for (std::size_t i = 0; i < n; ++i) if (!keep[i]) logits[i] = -std::numeric_limits<float>::infinity();
    }

    // ── Softmax over the (filtered, temp-scaled) logits, then a seeded categorical
    //    draw via inverse-CDF on a host PRNG (std::mt19937_64). ──
    float fmax = -std::numeric_limits<float>::infinity();
    for (float v : logits) fmax = std::max(fmax, v);
    std::vector<float> sp(n);
    float sz = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        sp[i] = (logits[i] == -std::numeric_limits<float>::infinity()) ? 0.0F : std::exp(logits[i] - fmax);
        sz += sp[i];
    }
    // Draw a uniform in [0, sz) from the seeded PRNG. std::mt19937_64 — host-side,
    // per-request reproducible (same seed → same stream).
    std::mt19937_64 rng(rng_state);
    std::uniform_real_distribution<float> uni(0.0F, sz);
    const float u = uni(rng);
    // Inverse-CDF search: the first index where the cumulative sum >= u.
    float c = 0.0F;
    std::uint32_t arg = 0;
    for (std::size_t i = 0; i < n; ++i) {
        c += sp[i];
        if (u <= c) { arg = static_cast<std::uint32_t>(i); break; }
        arg = static_cast<std::uint32_t>(i); // fallback (FP roundoff at the tail)
    }

    // ── Top-k logprob sidecar (the ≤HYP_TOP_K_LOGPROBS highest logits, sorted desc).
    //    Computed from the PRE-filter logits (the raw post-softcap logprobs, not the
    //    filtered -inf ones) — the caller asked for the top-k logprobs, not the
    //    nucleus. Reuse the pre-filter `lf` (the original post-softcap vector).
    StochasticSample out{};
    out.token_id = arg;
    out.logit = src[arg];
    {
        std::vector<std::uint32_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + HYP_TOP_K_LOGPROBS, idx.end(),
                          [&](std::uint32_t a, std::uint32_t b) { return src[a] > src[b]; });
        out.top_k_count = std::min<std::size_t>(HYP_TOP_K_LOGPROBS, n);
        for (std::size_t i = 0; i < out.top_k_count; ++i) {
            out.top_k_ids[i] = idx[i];
            out.top_k_logprobs[i] = src[idx[i]];
        }
    }
    return out;
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

    // SDPA defaults to the current chunk. For a multi-token sliding continuation, gather
    // the prior committed trailing window BEFORE the append can overwrite ring slots,
    // then assemble the bounded transient prefix+chunk axis. The existing absolute-position
    // mask bands and causal-masks that axis per query row.
    mx::array k_attn = k;
    mx::array v_attn = v;
    const bool sliding_continuation_prefill =
        ref.kind == LayerType::Sliding && offset > 0 && L > 1;
    if (sliding_continuation_prefill) {
        LocalKvCache& cache = kvstate.local[ref.per_kind_index];
        const std::uint32_t prefix_len = std::min(geometry_.sliding_window, offset);
        mx::array k_prefix = cache.read_window(
            cache.keys(), prefix_len, n_kv_heads, head_dim, stream_);
        mx::array v_prefix = cache.read_window(
            cache.values(), prefix_len, n_kv_heads, head_dim, stream_);
        k_attn = assemble_continuation_kv(
            k_prefix, k, static_cast<int>(prefix_len), L, n_kv_heads, head_dim, stream_);
        v_attn = assemble_continuation_kv(
            v_prefix, v, static_cast<int>(prefix_len), L, n_kv_heads, head_dim, stream_);
    }

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

    // offset==0 attends over the current chunk (the sealed 2.3b path). Multi-token
    // sliding continuation already holds prior-window+chunk above. Single-token decode
    // and global continuation retain their existing post-append cache reads.
    if (offset > 0) {
        if (ref.kind == LayerType::Sliding) {
            if (!sliding_continuation_prefill) {
                LocalKvCache& cache = kvstate.local[ref.per_kind_index];
                const std::uint32_t attn = cache.attention_len(); // == committed
                const std::uint32_t win = geometry_.sliding_window;
                // Sealed single-token decode: for attn <= window, the linear prefix is
                // already the final window; past it, gather the rotated final window.
                if (attn <= win) {
                    k_attn = read_kv_view(cache.keys(), attn, n_kv_heads, head_dim, stream_);
                    v_attn = read_kv_view(cache.values(), attn, n_kv_heads, head_dim, stream_);
                } else {
                    k_attn = cache.read_window(cache.keys(), win, n_kv_heads, head_dim, stream_);
                    v_attn = cache.read_window(cache.values(), win, n_kv_heads, head_dim, stream_);
                }
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
    std::uint32_t offset,
    std::optional<float> layer_scalar_factor) {
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
    // The G1 fault-boundary test (forward_faulted) multiplies this layer's scalar by
    // ``layer_scalar_factor`` — the single-layer structural fault. The production path
    // (factor == nullopt) is bit-identical (multiply by the raw layer_scalar).
    mx::array scalar = lw.layer_scalar;
    if (layer_scalar_factor.has_value()) {
        scalar = mx::multiply(scalar, mx::array(*layer_scalar_factor, scalar.dtype()), stream_);
    }
    h = mx::multiply(h, scalar, stream_);
    return h;
}

mx::array ForwardPass::forward(const mx::array& h, KvState& kvstate, std::uint32_t offset) {
    // The mask's kv_len must match the SDPA's K/V read length:
    //  - offset==0: the SDPA reads the chunk-local K/V (kv_len = committed = offset+L = L),
    //    for BOTH kinds (the 2.3b bit-exact path; the cache is appended but not read).
    //  - multi-token sliding continuation: prior min(window, offset) + all current rows.
    //  - single-token sliding decode: the cache read returns min(window, committed).
    //  - offset>0 global: the cache read returns all committed (full causal, no window).
    // For offset==0 with L > window (a 2048-token prefill chunk) the sliding mask is
    // [L, L] with the trailing-window constraint (zeroes out-of-window) — matching the
    // chunk-local K/V of length L (NOT a window-sized slice).
    const int L = static_cast<int>(h.shape(1));
    const std::uint32_t q_len = static_cast<std::uint32_t>(L);
    const std::uint32_t win = geometry_.sliding_window;
    mx::array state = h;
    for (std::size_t layer = 0; layer < weights_.layers.size(); ++layer) {
        const LayerType kind = dispatch_.per_layer[layer];
        const std::uint32_t kv_len = attention_kv_len(kind, q_len, offset, win);
        mx::array mask = build_mask(kind, q_len, kv_len, offset);
        state = decoder_layer(state, layer, mask, kvstate, offset);
    }
    return final_norm(state);
}

mx::array ForwardPass::forward_faulted(
    const mx::array& h,
    KvState& kvstate,
    std::uint32_t offset,
    std::size_t fault_layer,
    float layer_scalar_factor) {
    // Duplicate of forward()'s loop body (kept in sync by reading the same geometry
    // + dispatch). The ONLY difference: at fault_layer, decoder_layer multiplies that
    // layer's per-layer residual scale (layer_scalar) by layer_scalar_factor — the
    // single-layer structural fault. The mask is built with the REAL offset at every
    // layer (the attention pattern is unchanged), so the kv_len computation below is
    // identical to forward().
    const int L = static_cast<int>(h.shape(1));
    const std::uint32_t q_len = static_cast<std::uint32_t>(L);
    const std::uint32_t win = geometry_.sliding_window;
    mx::array state = h;
    for (std::size_t layer = 0; layer < weights_.layers.size(); ++layer) {
        const LayerType kind = dispatch_.per_layer[layer];
        const std::uint32_t kv_len = attention_kv_len(kind, q_len, offset, win);
        mx::array mask = build_mask(kind, q_len, kv_len, offset);
        // The faulted layer's per-layer residual scale is multiplied by
        // layer_scalar_factor (the single-layer structural fault); every other layer
        // passes nullopt → decoder_layer uses the raw layer_scalar (bit-identical to
        // the production path).
        const std::optional<float> factor =
            (layer == fault_layer)
                ? std::optional<float>(layer_scalar_factor)
                : std::nullopt;
        state = decoder_layer(state, layer, mask, kvstate, offset, factor);
    }
    return final_norm(state);
}

} // namespace hyperion::model
