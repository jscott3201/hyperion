#include "governor.h"

#include <algorithm>
#include <limits>
#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace hyperion::governor {

using hyperion::model::LayerType;

namespace {

/// BF16 element size (bytes). The KV cache dtype is always bfloat16 in production.
constexpr std::size_t kBf16Bytes = 2;

/// Safety factor for fused-SDPA tiles and graph buffering (1.25 exactly).
constexpr std::uint64_t kTransientSafetyNumerator = 5;
constexpr std::uint64_t kTransientSafetyDenominator = 4;

constexpr std::uint64_t kMaxBytes = std::numeric_limits<std::uint64_t>::max();

std::uint64_t saturating_add(std::uint64_t lhs, std::uint64_t rhs) {
    return rhs > kMaxBytes - lhs ? kMaxBytes : lhs + rhs;
}

std::uint64_t saturating_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    return lhs > kMaxBytes / rhs ? kMaxBytes : lhs * rhs;
}

std::uint64_t scale_attention_transient(std::uint64_t bytes) {
    if (bytes > kMaxBytes / kTransientSafetyNumerator) {
        return kMaxBytes;
    }
    return bytes * kTransientSafetyNumerator / kTransientSafetyDenominator;
}

struct GlobalKvProjection {
    std::uint64_t persistent_bytes;
    std::uint64_t reallocation_bytes;
};

GlobalKvProjection fail_closed_global_projection() {
    return {kMaxBytes, kMaxBytes};
}

std::uint64_t arithmetic_series_sum(
    std::uint64_t first,
    std::uint64_t last,
    std::uint64_t count) {
    std::uint64_t endpoints = saturating_add(first, last);
    if (endpoints == kMaxBytes) {
        return kMaxBytes;
    }
    // Divide the even factor before multiplying to retain the exact integer sum.
    if (count % 2 == 0) {
        count /= 2;
    } else {
        endpoints /= 2;
    }
    return saturating_multiply(count, endpoints);
}

GlobalKvProjection project_global_kv(
    const KvState& kvstate,
    std::uint32_t n_tokens,
    StepKind step_kind) {
    std::uint64_t persistent_total = 0;
    std::uint64_t reallocation_total = 0;
    for (const auto& cache : kvstate.global) {
        const std::uint64_t current_capacity = cache.capacity();
        std::uint64_t projected_capacity = current_capacity;
        if (n_tokens != 0) {
            const std::uint64_t proposed_len =
                static_cast<std::uint64_t>(cache.committed_len()) + n_tokens;
            if (proposed_len > std::numeric_limits<std::uint32_t>::max()) {
                return fail_closed_global_projection();
            }
            const std::uint64_t step = cache.step();
            const std::uint64_t steps = proposed_len / step + (proposed_len % step != 0);
            const std::uint64_t required_capacity = saturating_multiply(steps, step);

            // MLX shape dimensions are signed ints (kv_shape casts capacity to int).
            // A larger rounded bucket cannot safely be submitted to the allocator.
            if (required_capacity >
                static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                return fail_closed_global_projection();
            }
            projected_capacity = std::max(current_capacity, required_capacity);
        }

        // Each cache owns distinct K and V BF16 buffers, including k_eq_v models.
        std::uint64_t bytes_per_capacity = saturating_multiply(
            cache.num_kv_heads(), cache.head_dim());
        bytes_per_capacity = saturating_multiply(bytes_per_capacity, kBf16Bytes);
        bytes_per_capacity = saturating_multiply(bytes_per_capacity, 2); // K + V
        const std::uint64_t cache_bytes =
            saturating_multiply(projected_capacity, bytes_per_capacity);
        if (cache_bytes == kMaxBytes) {
            return fail_closed_global_projection();
        }
        const std::uint64_t next_persistent =
            saturating_add(persistent_total, cache_bytes);
        if (next_persistent == kMaxBytes) {
            return fail_closed_global_projection();
        }
        persistent_total = next_persistent;

        if (projected_capacity > current_capacity) {
            std::uint64_t replacement_bytes = cache_bytes;
            if (step_kind == StepKind::Decode) {
                const std::uint64_t step = cache.step();
                const std::uint64_t growth = projected_capacity - current_capacity;
                if (current_capacity % step != 0 || growth % step != 0) {
                    return fail_closed_global_projection();
                }
                const std::uint64_t crossings = growth / step;
                const std::uint64_t first_capacity =
                    saturating_add(current_capacity, step);
                const std::uint64_t replacement_capacity_sum =
                    arithmetic_series_sum(
                        first_capacity, projected_capacity, crossings);
                replacement_bytes = saturating_multiply(
                    replacement_capacity_sum, bytes_per_capacity);
                if (replacement_bytes == kMaxBytes) {
                    return fail_closed_global_projection();
                }
            }

            // Prefill appends the full chunk once, so it allocates only the final
            // replacement. Decode appends q=1 sequentially, so every crossed bucket
            // allocates a larger replacement before the preceding buffer is released.
            const std::uint64_t next_reallocation =
                saturating_add(reallocation_total, replacement_bytes);
            if (next_reallocation == kMaxBytes) {
                return fail_closed_global_projection();
            }
            reallocation_total = next_reallocation;
        }
    }
    return {persistent_total, reallocation_total};
}

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
    std::uint64_t bytes = saturating_multiply(2, saturating_add(prefix, q));
    bytes = saturating_multiply(bytes, kv_heads);
    bytes = saturating_multiply(bytes, head_dim);
    return saturating_multiply(bytes, kBf16Bytes); // K + V
}

std::uint64_t attention_transient_bytes(
    const Geometry& geometry,
    std::uint32_t n_tokens,
    std::uint32_t offset,
    StepKind step_kind) {
    // Prefill is one q=n forward. Decode executes the admitted block as sequential
    // q=1 forwards, so its peak SDPA output is one token regardless of block length.
    const std::uint64_t q = step_kind == StepKind::Prefill
        ? n_tokens
        : (n_tokens == 0 ? 0ULL : 1ULL);
    const std::uint64_t n_heads = geometry.num_attention_heads;
    const std::uint64_t local_elements = saturating_multiply(
        saturating_multiply(q, geometry.head_dim_local), n_heads);
    const std::uint64_t global_elements = saturating_multiply(
        saturating_multiply(q, geometry.head_dim_global), n_heads);
    const std::uint64_t local_transient =
        saturating_multiply(local_elements, kBf16Bytes);
    const std::uint64_t global_transient =
        saturating_multiply(global_elements, kBf16Bytes);

    std::uint64_t total = 0;
    for (const auto& kind : geometry.layer_types) {
        total = saturating_add(
            total, kind == LayerType::Sliding ? local_transient : global_transient);
    }
    const std::uint64_t scaled_sdpa = scale_attention_transient(total);
    return saturating_add(
        scaled_sdpa,
        continuation_local_kv_scratch(geometry, n_tokens, offset, step_kind));
}

struct MemoryPrediction {
    std::uint64_t peak_bytes;
    std::uint64_t projected_global_bytes;
};

MemoryPrediction predict_memory(
    std::uint32_t n_tokens,
    std::uint32_t offset,
    const KvState& kvstate,
    const Geometry& geometry,
    StepKind step_kind) {
    const std::uint64_t settled = saturating_add(
        mx::get_active_memory(), mx::get_cache_memory());
    const GlobalKvProjection global =
        project_global_kv(kvstate, n_tokens, step_kind);

    std::uint64_t predicted = saturating_add(settled, global.reallocation_bytes);
    predicted = saturating_add(
        predicted,
        attention_transient_bytes(geometry, n_tokens, offset, step_kind));
    return {
        saturating_add(predicted, kWorkspaceReserveBytes),
        global.persistent_bytes,
    };
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

GovernorDecision Governor::evaluate(
    std::uint32_t n_tokens,
    std::uint32_t offset,
    const KvState& kvstate,
    StepKind step_kind) const {
    const MemoryPrediction prediction = predict_memory(
        n_tokens, offset, kvstate, geometry_, step_kind);

    // Local allocation is fixed; global telemetry is projected post-step allocation.
    const std::uint64_t local_kv = local_kv_bytes(kvstate);
    const std::uint64_t global_kv = prediction.projected_global_bytes;

    GovernorDecision decision;
    decision.predicted_peak_bytes = prediction.peak_bytes;
    decision.budget_ceiling_bytes = budget_ceiling_bytes_;
    decision.soft_watermark_bytes = soft_watermark_bytes_;
    decision.local_kv_bytes = local_kv;
    decision.global_kv_bytes = global_kv;

    if (prediction.peak_bytes == kMaxBytes ||
        prediction.peak_bytes > budget_ceiling_bytes_) {
        // Hard reject: even at the ceiling, this step won't fit.
        decision.admission = Admission::HardRejected;
        decision.reason = prediction.peak_bytes == kMaxBytes
            ? "memory projection exceeds the representable allocation range"
            : "predicted peak exceeds the 12.06 GiB budget ceiling";
        return decision;
    }

    if (prediction.peak_bytes > soft_watermark_bytes_) {
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
    const KvState& kvstate,
    const Geometry& geometry,
    StepKind step_kind) {
    return predict_memory(n_tokens, offset, kvstate, geometry, step_kind).peak_bytes;
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
    total_transient = scale_attention_transient(total_transient);

    const std::uint64_t predicted =
        model_bytes + kv_bytes + total_transient + kWorkspaceReserveBytes;

    return predicted <= budget_ceiling_bytes;
}

} // namespace hyperion::governor
