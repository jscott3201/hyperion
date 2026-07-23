#pragma once

#include "dispatch.h"
#include "geometry.h"
#include "kv_cache.h"
#include "rope.h"
#include "weights_loader.h"

#include <array>
#include <cstdint>
#include <optional>
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
    /// KV length; 0 for a fresh prefill, =prompt_len for decode). Appends each layer's K/V
    /// to ``kvstate``. Returns the final-RMSNorm hidden state ``[B, L, hidden]``. The
    /// offset>0 cached-prefix attention read (``attention()``) is M2-2.7; the offset==0
    /// prefill path is the 2.3b bit-exact path (unchanged).
    [[nodiscard]] mlx::core::array forward(
        const mlx::core::array& h,
        KvState& kvstate,
        std::uint32_t offset);

    /// TEST-ONLY fault-injection forward pass for the G1 logit fault-boundary
    /// calibration (08-correctness-and-verification.md §18-22). Identical to
    /// ``forward`` EXCEPT at ``fault_layer``: that one layer's per-layer residual
    /// scale (``layer_scalar``) is multiplied by ``layer_scalar_factor`` — a
    /// single-layer structural perturbation (amplify/drop/negate the layer's
    /// contribution to the residual stream). The attention pattern, RoPE, cache,
    /// and every other layer are unchanged.
    ///
    /// WHY NOT RoPE-offset (the 08 "e.g." fault): a single-layer RoPE offset was
    /// measured on the Gemma-4 12B and produces a logit delta AT OR BELOW the
    /// engine noise floor (sig_rel ~0.019 vs clean ~0.021 — no separation; the
    /// residual stream + QK-norm absorb a one-position rotation). 08:21 says
    /// "Gemma numbers will differ; derive, don't port" — so the SEPARATING fault
    /// is derived empirically. A ``layer_scalar × 5`` at layer 0 gives sig_rel
    /// ~0.41 (~20× the noise floor, token-stable) — the derived fault this gate
    /// calibrates against. This is a real single-layer structural fault, NOT a
    /// quantization-noise proxy (08:23-25).
    ///
    /// ``forward`` (the production path) is UNCHANGED — this is a separate method the
    /// fault-boundary test calls; the token-exact 2.3b/2.7 seal is untouched. NOT an
    /// ABI function (no ``extern "C"``, not in the ratchet count).
    [[nodiscard]] mlx::core::array forward_faulted(
        const mlx::core::array& h,
        KvState& kvstate,
        std::uint32_t offset,
        std::size_t fault_layer,
        float layer_scalar_factor);

    /// Final RMSNorm (``model.norm``). Public so the 2.7 epilogue can reuse it.
    [[nodiscard]] mlx::core::array final_norm(const mlx::core::array& h) const;

    /// ── Generation epilogue (M2-2.7) ──────────────────────────────────────────
    /// The lm_head + softcap + greedy-sampler path that turns the final-norm hidden
    /// state into a sampled token. ``lm_head`` is TIED to the quantized embedding
    /// table (``ModelWeights::embed_*``); ``geometry.tie_word_embeddings`` is
    /// validated true at load (``geometry.cc`` rejects false), so there is no
    /// untied path. ``softcap`` follows the mlx-lm ``logit_softcap`` exactly:
    /// ``softcap * tanh(logits / softcap)`` with NO fp32 cast (operates in the
    /// lm_head output dtype). Greedy argmax is host-side (see ``sample_greedy``).
    ///@{

    /// Tied lm_head: ``[B, L, hidden] @ embed_weightᵀ → [B, L, vocab]``. The embed
    /// table is stored ``[vocab, hidden] = [out, in]`` — the same layout
    /// ``QuantizedLinear::w`` uses — so ``quantized_matmul`` with ``transpose=true``
    /// fuses dequant + the transposed matmul (the q/k/v/o path, ``weights.h``).
    [[nodiscard]] mlx::core::array lm_head(const mlx::core::array& h) const;

    /// ``final_logit_softcapping * tanh(logits / final_logit_softcapping)``.
    [[nodiscard]] mlx::core::array softcap(const mlx::core::array& logits) const;

    /// A greedy argmax + top-2 near-tie over one position's logits.
    struct GreedySample {
        std::uint32_t token_id; ///< argmax token.
        float logit;            ///< the winning post-softcap logit.
        bool near_tie;          ///< top-2 gap < 0.5 (A1, the near_tie_events counter).
    };
    /// ``logits_last`` is ``[vocab]`` (one position, post-softcap). Returns the
    /// argmax token + its logit + whether the top-2 gap is < 0.5.
    ///
    /// THE 2.3b LESSON: MLX lazy-graph ``mx::argmax``/``mx::max`` scalars go STALE
    /// across repeated evals in a decode loop (graph-cache aliasing) — so this
    /// reads raw contiguous ``data<float>()`` pointers and host-scans for top-1
    /// AND top-2 in one pass. (A GPU ``mx::argmax``+``topk`` port would risk the
    /// stale-scalar bug; the 262144-vocab CPU scan is a G2 perf flag, not a block.)
    /// This is the load-bearing choice in the slice — the seam.
    [[nodiscard]] GreedySample sample_greedy(const mlx::core::array& logits_last) const;

    /// ── M3 sampler surface (sampled mode, 03 §Sampling) ────────────────────────
    /// A sampled token + its top-k logprobs. ``logit`` is the WINNING post-softcap
    /// logit (the sampled token's, for telemetry); ``top_k_logprobs`` is the
    /// ≤``HYP_TOP_K_LOGPROBS`` highest-logit (ids, logprobs), sorted descending —
    /// the small sidecar that crosses the ABI (03:105-106: NOT the full frame).
    struct StochasticSample {
        std::uint32_t token_id;
        float logit;
        std::uint32_t top_k_count;
        std::array<std::uint32_t, HYP_TOP_K_LOGPROBS> top_k_ids;
        std::array<float, HYP_TOP_K_LOGPROBS> top_k_logprobs;
    };
    /// ``logits_last`` is ``[vocab]`` (one position, post-softcap), same input as
    /// ``sample_greedy``. ``cfg`` drives the sampled epilogue: temperature scaling,
    /// then the mlx-lm filter order (top-p, min-p, top-k) as ``-inf`` masks, then a
    /// host-side softmax + seeded categorical draw (``std::mt19937_64``).
    ///
    /// REUSES the 2.3b host-scan seam: ``astype(contiguous, float32, CPU)`` →
    /// ``eval`` → ``data<float>()`` — NO ``mx::argmax``/``mx::random`` graph scalars
    /// (the stale-scalar bug). The RNG is a per-request seeded host PRNG
    /// (statistical-faithfulness bar per 08 §35-37 + the M2-2.6c fault-boundary
    /// threshold; NOT bit-exact vs mlx-lm — C++/Metal vs MLX FP reduction order
    /// differs, so seeded-token-exact draws against the oracle are out of scope).
    [[nodiscard]] StochasticSample sample_stochastic(
        const mlx::core::array& logits_last,
        const HypSamplingConfig& cfg,
        std::uint64_t rng_state) const;
    ///@}

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
    /// ``layer_scalar_factor`` (test-only, G1 fault-boundary): when set, multiplies
    /// this layer's per-layer residual scale (``layer_scalar``) by ``*factor`` —
    /// the single-layer structural fault. Unset → production path (bit-identical).
    [[nodiscard]] mlx::core::array decoder_layer(
        const mlx::core::array& x,
        std::size_t layer,
        const mx::array& mask,
        KvState& kvstate,
        std::uint32_t offset,
        std::optional<float> layer_scalar_factor = std::nullopt);

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
