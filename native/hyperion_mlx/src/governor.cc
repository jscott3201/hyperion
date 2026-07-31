#include "governor.h"

#include <algorithm>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace hyperion::governor {

using hyperion::model::LayerType;

namespace {

/// BF16 element size (bytes). The KV cache dtype is always bfloat16 in production.
constexpr std::size_t kBf16Bytes = 2;

/// Safety factor for the attention transient (the SDPA peak buffer may be larger
/// than the steady-state attribution due to MLX graph buffering).
constexpr float kTransientSafety = 1.25F;

/// Extra materialized K+V buffers used by sliding continuation prefill. Forward
/// evaluates layers sequentially, so this is one local-layer peak rather than a sum
/// across every sliding layer. Decode and fresh-prefill paths do not assemble it.
std::uint64_t continuation_local_kv_scratch(
    const Geometry& geometry,
    std::uint32_t n_tokens,
    std::uint32_t offset,
    StepKind step_kind) {
    if (step_kind != StepKind::Prefill || offset == 0 || n_tokens <= 1 ||
        std::find(geometry.layer_types.begin(), geometry.layer_types.end(),
            LayerType::Sliding) == geometry.layer_types.end()) {
        return 0;
    }

    const std::uint64_t prefix = std::min(geometry.sliding_window, offset);
    const std::uint64_t q = n_tokens;
    const std::uint64_t kv_heads = geometry.num_kv_heads_local;
    const std::uint64_t head_dim = geometry.head_dim_local;
    return 2ULL * (prefix + q) * kv_heads * head_dim * kBf16Bytes; // K + V
}

} // namespace

Governor::Governor(
    const Geometry& geometry,
    std::uint64_t budget_ceiling_bytes,
    std::uint64_t soft_watermark_bytes)
    : geometry_(geometry),
      budget_ceiling_bytes_(budget_ceiling_bytes),
      soft_watermark_bytes_(soft_watermark_bytes) {}

std::uint64_t Governor::local_kv_bytes(const KvState& kvstate) const {
    std::uint64_t total = 0;
    for (const auto& cache : kvstate.local) {
        // Each LocalKvCache stores K and V, each [capacity, n_kv_heads, head_dim] bf16.
        const std::uint64_t per_tensor =
            static_cast<std::uint64_t>(cache.capacity()) *
            static_cast<std::uint64_t>(cache.num_kv_heads()) *
            static_cast<std::uint64_t>(cache.head_dim()) *
            kBf16Bytes;
        total += 2 * per_tensor; // K + V
    }
    return total;
}

std::uint64_t Governor::global_kv_bytes(const KvState& kvstate) const {
    std::uint64_t total = 0;
    for (const auto& cache : kvstate.global) {
        // Each GlobalKvCache stores K and V separately (M2-2.7 fix), each
        // [capacity, n_kv_heads, head_dim] bf16.
        const std::uint64_t per_tensor =
            static_cast<std::uint64_t>(cache.capacity()) *
            static_cast<std::uint64_t>(cache.num_kv_heads()) *
            static_cast<std::uint64_t>(cache.head_dim()) *
            kBf16Bytes;
        total += 2 * per_tensor; // K + V (never aliased, even when k_eq_v)
    }
    return total;
}

std::uint64_t Governor::attention_transient(
    std::uint32_t n_tokens,
    std::uint32_t offset,
    StepKind step_kind) const {
    // The attention transient is the peak SDPA buffer allocation during the step.
    // MLX's mx::fast::scaled_dot_product_attention is a FUSED kernel that does NOT
    // materialize the [B, n_heads, q, ctx] attention-SCORES matrix — it streams the
    // computation in tiles. The buffer that IS materialized is the attention OUTPUT
    // [B, n_heads, q, head_dim] per layer. Modeling the full [q, ctx] scores (as the
    // M2-2.6b governor did) over-predicts by the scores-vs-output ratio, worsening with
    // context — the M3 calibration (benchmarks/m3/governor-calibration.json) measured
    // +129% at 8K and +259% at 32K (the governor rejected prefills that fit). So the
    // transient is the OUTPUT term:
    //   output = q * head_dim * n_heads * dtype
    // scaled by the per-layer head_dim (local for sliding, global for full). n_heads is
    // the QUERY head count (num_attention_heads), uniform across layer kinds. The 1.25
    // safety factor (kTransientSafety) covers the fused-kernel's tile working set.
    const std::uint64_t q = static_cast<std::uint64_t>(n_tokens);
    const std::uint64_t n_heads = static_cast<std::uint64_t>(geometry_.num_attention_heads);
    const std::uint64_t head_dim_local = static_cast<std::uint64_t>(geometry_.head_dim_local);
    const std::uint64_t head_dim_global = static_cast<std::uint64_t>(geometry_.head_dim_global);

    // Per-layer transient = the attention OUTPUT [q, head_dim] * n_heads * dtype.
    // Sliding layers use head_dim_local; global layers use head_dim_global.
    const std::uint64_t local_transient = q * head_dim_local * n_heads * kBf16Bytes;
    const std::uint64_t global_transient = q * head_dim_global * n_heads * kBf16Bytes;

    // Sum across all layers (each layer has its own attention output buffer).
    std::uint64_t total = 0;
    for (const auto& kind : geometry_.layer_types) {
        if (kind == LayerType::Sliding) {
            total += local_transient;
        } else {
            total += global_transient;
        }
    }
    // Apply the safety factor for the fused-kernel tile working set + graph buffering.
    const std::uint64_t scaled_sdpa = static_cast<std::uint64_t>(
        static_cast<double>(total) * kTransientSafety);

    // Sliding continuation prefill also materializes one bounded [prefix + q] K and V
    // assembly before SDPA. Charge the exact per-layer peak once; the 512 MiB workspace
    // reserve remains available for slice-update intermediates and graph buffers.
    return scaled_sdpa +
        continuation_local_kv_scratch(geometry_, n_tokens, offset, step_kind);
}

GovernorDecision Governor::evaluate(
    std::uint32_t n_tokens,
    std::uint32_t offset,
    const KvState& kvstate,
    StepKind step_kind) const {
    // settled_working_set = active + cache (live MLX counters).
    const std::uint64_t active = mx::get_active_memory();
    const std::uint64_t cache = mx::get_cache_memory();
    const std::uint64_t settled = active + cache;

    // kv_append: only global layers grow per token (16 KiB/token).
    // Local layers are flat (ring capacity doesn't grow).
    const std::uint64_t kv_append =
        static_cast<std::uint64_t>(n_tokens) * kGlobalKvBytesPerToken;

    // attention_transient: peak SDPA buffer during the step.
    const std::uint64_t transient = attention_transient(n_tokens, offset, step_kind);

    // predicted_peak = settled + kv_append + transient + workspace reserve.
    const std::uint64_t predicted_peak =
        settled + kv_append + transient + kWorkspaceReserveBytes;

    // Current KV byte counts (for telemetry).
    const std::uint64_t local_kv = local_kv_bytes(kvstate);
    const std::uint64_t global_kv = global_kv_bytes(kvstate);

    GovernorDecision decision;
    decision.predicted_peak_bytes = predicted_peak;
    decision.budget_ceiling_bytes = budget_ceiling_bytes_;
    decision.soft_watermark_bytes = soft_watermark_bytes_;
    decision.local_kv_bytes = local_kv;
    decision.global_kv_bytes = global_kv;

    if (predicted_peak > budget_ceiling_bytes_) {
        // Hard reject: even at the ceiling, this step won't fit.
        decision.admission = Admission::HardRejected;
        decision.reason = "predicted peak exceeds the 12.06 GiB budget ceiling";
        return decision;
    }

    if (predicted_peak > soft_watermark_bytes_) {
        // Soft pause: halve the chunk and retry.
        decision.admission = Admission::SoftPaused;
        decision.reason = "predicted peak exceeds the soft watermark; halve-chunk";
        return decision;
    }

    decision.admission = Admission::Accepted;
    decision.reason = "";
    return decision;
}

std::uint64_t predict_peak(
    std::uint32_t n_tokens,
    std::uint32_t offset,
    const KvState& /*kvstate*/,
    const Geometry& geometry,
    StepKind step_kind) {
    // This is the standalone prediction for telemetry fill (no admission).
    // Uses the same formula as Governor::evaluate but without the budget check.
    const std::uint64_t active = mx::get_active_memory();
    const std::uint64_t cache = mx::get_cache_memory();
    const std::uint64_t settled = active + cache;

    const std::uint64_t kv_append =
        static_cast<std::uint64_t>(n_tokens) * kGlobalKvBytesPerToken;

    // Recompute the attention transient inline (mirrors Governor::attention_transient):
    // the fused-SDPA OUTPUT [q, head_dim] per layer (the [q, ctx] scores are NOT
    // materialized by the fused kernel — see the M3 calibration report).
    const std::uint64_t q = static_cast<std::uint64_t>(n_tokens);
    const std::uint64_t n_heads = static_cast<std::uint64_t>(geometry.num_attention_heads);
    const std::uint64_t head_dim_local = static_cast<std::uint64_t>(geometry.head_dim_local);
    const std::uint64_t head_dim_global = static_cast<std::uint64_t>(geometry.head_dim_global);

    std::uint64_t total_transient = 0;
    for (const auto& kind : geometry.layer_types) {
        if (kind == LayerType::Sliding) {
            total_transient += q * head_dim_local * n_heads * kBf16Bytes;
        } else {
            total_transient += q * head_dim_global * n_heads * kBf16Bytes;
        }
    }
    total_transient = static_cast<std::uint64_t>(
        static_cast<double>(total_transient) * kTransientSafety);
    total_transient +=
        continuation_local_kv_scratch(geometry, n_tokens, offset, step_kind);

    return settled + kv_append + total_transient + kWorkspaceReserveBytes;
}

std::uint32_t throughput_optimum_context(std::uint64_t budget_ceiling_bytes) {
    // A4: the throughput-optimum context is where the marginal KV growth cost
    // equals the throughput gain. For the 12B dense, this is empirically ~4K tokens
    // on the 16 GB profile. Computed as: the context at which the global KV cost
    // reaches ~30% of the budget (the point where memory pressure starts to dominate
    // throughput). Not the ceiling — the governor targets this for scheduling.
    const std::uint64_t kv_budget = static_cast<std::uint64_t>(
        static_cast<double>(budget_ceiling_bytes) * 0.30);
    return static_cast<std::uint32_t>(kv_budget / kGlobalKvBytesPerToken);
}

bool peak_within_budget(
    std::uint32_t context_len,
    const Geometry& geometry,
    std::uint64_t budget_ceiling_bytes) {
    // G4 gate: predict the peak at ``context_len`` tokens (a full prefill of that
    // length) and check it stays ≤ budget. Uses the same formula but with the
    // full context as the prefill chunk (worst-case transient).
    // settled_working_set at context_len = model weights + full KV at context_len.
    // The model weights are estimated from the geometry (bf16 = 2 bytes/element):
    //   embedding: vocab_size * hidden_size (shared if tie_word_embeddings)
    //   per layer: 4 * hidden_size^2 (QKV+O) + 3 * hidden_size * intermediate (MLP)
    //   LM head: vocab_size * hidden_size (if not tied)
    // NOTE: this estimates DENSE bf16 weight bytes, but the 12B loads group-quantized
    // weights (g64/b4, ~6.7 GB vs ~24.7 GB bf16). For the `<=` gate the overestimate
    // is conservative (never falsely passes), but it means peak_within_budget would
    // reject the real 12B at the 8K sentinel — so this gate is NOT yet wired into the
    // load path (it is scaffold for a future quantization-aware G4 check). The
    // moe/use_double_wide_mlp/ple_* fields are also ignored; harmless for the 12B
    // (dense MLP, no MoE/PLE) but would need accounting for a MoE target.
    const std::uint64_t emb =
        static_cast<std::uint64_t>(geometry.vocab_size) * geometry.hidden_size;
    const std::uint64_t per_layer =
        static_cast<std::uint64_t>(geometry.hidden_size) * geometry.hidden_size * 4 +
        static_cast<std::uint64_t>(geometry.hidden_size) * geometry.intermediate_size * 3;
    const std::uint64_t lm_head = geometry.tie_word_embeddings ? 0 : emb;
    const std::uint64_t model_bytes =
        (emb + static_cast<std::uint64_t>(geometry.num_hidden_layers) * per_layer + lm_head) * 2;
    const std::uint64_t kv_bytes =
        static_cast<std::uint64_t>(context_len) * kGlobalKvBytesPerToken;

    // The transient for a full prefill of ``context_len`` tokens at offset 0:
    // q = context_len. The fused-SDPA OUTPUT [q, head_dim] per layer (the [q, ctx]
    // scores are NOT materialized — see Governor::attention_transient + the M3
    // calibration report).
    const std::uint64_t q = static_cast<std::uint64_t>(context_len);
    const std::uint64_t n_heads = static_cast<std::uint64_t>(geometry.num_attention_heads);
    const std::uint64_t head_dim_local = static_cast<std::uint64_t>(geometry.head_dim_local);
    const std::uint64_t head_dim_global = static_cast<std::uint64_t>(geometry.head_dim_global);

    std::uint64_t total_transient = 0;
    for (const auto& kind : geometry.layer_types) {
        if (kind == LayerType::Sliding) {
            total_transient += q * head_dim_local * n_heads * kBf16Bytes;
        } else {
            total_transient += q * head_dim_global * n_heads * kBf16Bytes;
        }
    }
    total_transient = static_cast<std::uint64_t>(
        static_cast<double>(total_transient) * kTransientSafety);

    const std::uint64_t predicted =
        model_bytes + kv_bytes + total_transient + kWorkspaceReserveBytes;

    return predicted <= budget_ceiling_bytes;
}

} // namespace hyperion::governor
