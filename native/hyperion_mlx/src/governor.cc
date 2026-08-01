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
constexpr std::size_t kBoolBytes = 1;

/// Safety factor for fused-SDPA tiles and graph buffering (1.25 exactly).
constexpr std::uint64_t kTransientSafetyNumerator = 5;
constexpr std::uint64_t kTransientSafetyDenominator = 4;

constexpr std::uint64_t kMaxBytes = std::numeric_limits<std::uint64_t>::max();

/// The pinned vector two-pass kernel selects at most 1024 blocks without an
/// MLX_SDPA_BLOCKS override. Hyperion does not set that debug override.
constexpr std::uint64_t kMaxFusedVectorBlocks = 1024;

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

bool has_layer_kind(const Geometry& geometry, LayerType kind) {
    return std::find(geometry.layer_types.begin(), geometry.layer_types.end(), kind) !=
        geometry.layer_types.end();
}

struct AttentionExecutionShape {
    std::uint64_t q;
    std::uint64_t local_k;
    std::uint64_t global_k;
};

AttentionExecutionShape attention_execution_shape(
    const Geometry& geometry,
    const AdmissionInput& input) {
    if (input.n_tokens == 0) {
        return {0, 0, 0};
    }

    if (input.step_kind == StepKind::Decode) {
        // Decode executes q=1 forwards sequentially. The last step has the largest
        // committed K axis and is therefore the admitted block's attention peak.
        const std::uint64_t committed = saturating_add(input.offset, input.n_tokens);
        return {
            1,
            std::min<std::uint64_t>(geometry.sliding_window, committed),
            committed,
        };
    }

    const std::uint64_t q = input.n_tokens;
    const std::uint64_t committed = saturating_add(input.offset, q);
    if (input.offset == 0) {
        // Fresh prefill attends to this chunk's own K/V for both layer kinds.
        return {q, q, q};
    }
    if (input.n_tokens == 1) {
        // The production single-query continuation reads the final ring window.
        return {
            q,
            std::min<std::uint64_t>(geometry.sliding_window, committed),
            committed,
        };
    }
    // Multi-query sliding continuation assembles retained-prefix + current chunk.
    return {
        q,
        saturating_add(
            std::min<std::uint64_t>(geometry.sliding_window, input.offset), q),
        committed,
    };
}

AttentionPath attention_path(
    std::uint64_t q,
    std::uint64_t k,
    std::uint64_t query_head_dim,
    std::uint64_t value_head_dim,
    std::uint64_t num_query_heads,
    std::uint64_t num_kv_heads) {
    if (q == 0) {
        return AttentionPath::None;
    }
    if (num_kv_heads == 0 || num_query_heads % num_kv_heads != 0 || q > k) {
        return AttentionPath::Fallback;
    }

    const std::uint64_t gqa = num_query_heads / num_kv_heads;
    if (q <= 8) {
        const bool equal_supported = query_head_dim == value_head_dim &&
            (query_head_dim == 64 || query_head_dim == 96 ||
                query_head_dim == 128 || query_head_dim == 256);
        const bool asymmetric_supported =
            query_head_dim == 192 && value_head_dim == 128;
        if ((equal_supported || asymmetric_supported) &&
            saturating_multiply(q, gqa) <= 32) {
            return AttentionPath::FusedVector;
        }
        return AttentionPath::Fallback;
    }

    const bool full_supported = query_head_dim == value_head_dim &&
        (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);
    return full_supported ? AttentionPath::FusedFull : AttentionPath::Fallback;
}

std::uint64_t tensor_bytes(
    std::uint64_t first,
    std::uint64_t second,
    std::uint64_t third,
    std::uint64_t element_bytes) {
    std::uint64_t bytes = saturating_multiply(first, second);
    bytes = saturating_multiply(bytes, third);
    return saturating_multiply(bytes, element_bytes);
}

std::uint64_t fused_vector_workspace_bytes(
    std::uint64_t q,
    std::uint64_t k,
    std::uint64_t num_query_heads,
    std::uint64_t value_head_dim) {
    if (k < 1024) {
        return 0;
    }

    // MLX 0.32.0's long-context vector path allocates:
    //   intermediate [B,Hq,q,blocks,Dv] in the query dtype, and
    //   sums/maxs    [B,Hq,q,blocks]    in float32.
    // Use the largest source-selected block count so the prediction is independent
    // of which pinned Metal architecture branch is active.
    std::uint64_t rows = saturating_multiply(num_query_heads, q);
    rows = saturating_multiply(rows, kMaxFusedVectorBlocks);
    const std::uint64_t intermediate =
        saturating_multiply(rows, saturating_multiply(value_head_dim, kBf16Bytes));
    const std::uint64_t reductions =
        saturating_multiply(rows, 2ULL * sizeof(float));
    return saturating_add(intermediate, reductions);
}

std::uint64_t attention_layer_phase_bytes(
    std::uint64_t q,
    std::uint64_t k,
    std::uint64_t num_query_heads,
    std::uint64_t value_head_dim,
    AttentionPath path) {
    if (path == AttentionPath::None) {
        return 0;
    }

    const std::uint64_t output = tensor_bytes(
        num_query_heads, q, value_head_dim, kBf16Bytes);
    const std::uint64_t mask = tensor_bytes(q, k, 1, kBoolBytes);
    std::uint64_t phase = saturating_add(output, mask);
    if (path == AttentionPath::Fallback) {
        const std::uint64_t scores = tensor_bytes(
            num_query_heads, q, k, kBf16Bytes);
        phase = saturating_add(phase, scores);
    } else if (path == AttentionPath::FusedVector) {
        phase = saturating_add(
            phase,
            fused_vector_workspace_bytes(q, k, num_query_heads, value_head_dim));
    }
    // Full fused attention has no allocator-backed intermediate in the pinned
    // source. The retained 1.25 factor is its conservative tile/graph allowance
    // and is also applied to fallback/vector phases.
    return scale_attention_transient(phase);
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
    StepKind step_kind,
    const hyperion::model::KvGrowthPlan& plan) {
    if (!plan.representable) {
        return fail_closed_global_projection();
    }
    std::uint64_t persistent_total = 0;
    std::uint64_t reallocation_total = 0;
    for (const auto& cache : kvstate.global) {
        const std::uint64_t current_capacity = cache.capacity();
        const std::uint64_t projected_capacity =
            hyperion::model::projected_global_capacity(cache, plan.n_tokens);

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

/// Copy-on-write allocation introduced by staging a whole growth operation.
/// Every local cache is rebound on the staged state. A global cache that does
/// not grow is also rebound at its current capacity. For sequential decode, a
/// crossing cache additionally needs its current-capacity candidate when the
/// block begins inside the bucket; later replacement candidates remain covered
/// by project_global_kv's replacement series.
std::uint64_t staging_cow_bytes(
    const KvState& kvstate,
    const hyperion::model::KvGrowthPlan& plan,
    StepKind step_kind) {
    if (!plan.representable) {
        return kMaxBytes;
    }
    if (!plan.requires_transaction) {
        return 0;
    }

    auto tensor_bytes = [](
        std::uint32_t capacity,
        std::uint32_t heads,
        std::uint32_t dim) {
        std::uint64_t bytes = saturating_multiply(capacity, heads);
        bytes = saturating_multiply(bytes, dim);
        return saturating_multiply(bytes, kBf16Bytes);
    };

    std::uint64_t total = 0;
    for (const auto& cache : kvstate.local) {
        const std::uint64_t per_tensor = tensor_bytes(
            cache.capacity(), cache.num_kv_heads(), cache.head_dim());
        total = saturating_add(total, saturating_multiply(per_tensor, 2));
        // Fresh mx::zeros inputs are lazy. Staging aliases those descriptors, so
        // the first functional rebind cannot donate and evaluation retains both
        // the original input and the staged candidate at peak. Once an original
        // is available, settled memory already includes it and no surcharge is due.
        if (!cache.keys().is_available()) {
            total = saturating_add(total, per_tensor);
        }
        if (!cache.values().is_available()) {
            total = saturating_add(total, per_tensor);
        }
    }
    for (const auto& cache : kvstate.global) {
        const bool grows = hyperion::model::projected_global_capacity(
            cache, plan.n_tokens) > cache.capacity();
        const bool decode_current_candidate =
            grows && step_kind == StepKind::Decode &&
            cache.committed_len() < cache.capacity();
        if (grows && !decode_current_candidate) {
            // Replacement accounting already covers the staged candidate.
        } else {
            const std::uint64_t candidate = tensor_bytes(
                cache.capacity(), cache.num_kv_heads(), cache.head_dim());
            total = saturating_add(
                total, saturating_multiply(candidate, 2)); // K + V
        }
        // A non-empty global append reads the old K/V whether it rebinds in the
        // current bucket or copies the prefix into a replacement. Charge each
        // unavailable retained input; capacity-zero buffers are not copied.
        if (cache.capacity() > 0) {
            const std::uint64_t retained = tensor_bytes(
                cache.capacity(), cache.num_kv_heads(), cache.head_dim());
            if (!cache.keys().is_available()) {
                total = saturating_add(total, retained);
            }
            if (!cache.values().is_available()) {
                total = saturating_add(total, retained);
            }
        }
    }
    return total;
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

TransientPrediction predict_transient_impl(
    const Geometry& geometry,
    const AdmissionInput& input) {
    const AttentionExecutionShape shape = attention_execution_shape(geometry, input);
    const std::uint64_t n_heads = geometry.num_attention_heads;
    const bool has_local = has_layer_kind(geometry, LayerType::Sliding);
    const bool has_global = has_layer_kind(geometry, LayerType::Full);

    const AttentionPath local_path = has_local
        ? attention_path(
              shape.q, shape.local_k,
              geometry.head_dim_local, geometry.head_dim_local,
              n_heads, geometry.num_kv_heads_local)
        : AttentionPath::None;
    const AttentionPath global_path = has_global
        ? attention_path(
              shape.q, shape.global_k,
              geometry.head_dim_global, geometry.head_dim_global,
              n_heads, geometry.num_kv_heads_global)
        : AttentionPath::None;

    std::uint64_t local_phase = attention_layer_phase_bytes(
        shape.q, shape.local_k, n_heads, geometry.head_dim_local, local_path);
    local_phase = saturating_add(
        local_phase,
        continuation_local_kv_scratch(
            geometry, input.n_tokens, input.offset, input.step_kind));
    const std::uint64_t global_phase = attention_layer_phase_bytes(
        shape.q, shape.global_k, n_heads, geometry.head_dim_global, global_path);
    const std::uint64_t attention_peak = std::max(local_phase, global_phase);

    const std::uint64_t epilogue_q = input.step_kind == StepKind::Prefill
        ? input.n_tokens
        : (input.n_tokens == 0 ? 0ULL : 1ULL);
    const std::uint64_t epilogue = input.include_epilogue
        ? scale_attention_transient(tensor_bytes(
              epilogue_q, geometry.vocab_size, 1, kBf16Bytes))
        : 0;
    return {
        local_phase,
        global_phase,
        attention_peak,
        epilogue,
        std::max(attention_peak, epilogue),
        shape.local_k,
        shape.global_k,
        local_path,
        global_path,
    };
}

struct MemoryPrediction {
    std::uint64_t peak_bytes;
    std::uint64_t projected_global_bytes;
};

MemoryPrediction predict_memory(
    const AdmissionInput& input,
    const KvState& kvstate,
    const Geometry& geometry,
    const hyperion::model::KvGrowthPlan* operation_plan) {
    const auto derived_plan =
        hyperion::model::plan_kv_growth(kvstate, input.n_tokens);
    const auto& plan = operation_plan == nullptr ? derived_plan : *operation_plan;
    const std::uint64_t settled = saturating_add(
        mx::get_active_memory(), mx::get_cache_memory());
    const GlobalKvProjection global =
        project_global_kv(kvstate, input.step_kind, derived_plan);

    std::uint64_t predicted = saturating_add(settled, global.reallocation_bytes);
    predicted = saturating_add(
        predicted, staging_cow_bytes(kvstate, plan, input.step_kind));
    predicted = saturating_add(
        predicted, predict_transient_impl(geometry, input).operation_peak_bytes);
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
        std::uint64_t bytes = saturating_multiply(
            cache.capacity(), cache.num_kv_heads());
        bytes = saturating_multiply(bytes, cache.head_dim());
        bytes = saturating_multiply(bytes, 2 * kBf16Bytes); // K + V
        total = saturating_add(total, bytes);
    }
    return total;
}

GovernorDecision Governor::evaluate(
    const AdmissionInput& input,
    const KvState& kvstate,
    const hyperion::model::KvGrowthPlan* operation_plan) const {
    const MemoryPrediction prediction = predict_memory(
        input, kvstate, geometry_, operation_plan);

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
        // The proposed shape breaches the hard ceiling. Prefill callers halve this
        // signal repeatedly; it becomes a terminal typed OOM only at one token.
        decision.admission = Admission::HardRejected;
        decision.reason = prediction.peak_bytes == kMaxBytes
            ? "memory projection exceeds the representable allocation range"
            : "predicted peak exceeds the device-derived budget ceiling";
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

PrefillAdmissionResult Governor::admit_prefill_to_fit(
    std::uint32_t proposed_n_tokens,
    std::uint32_t offset,
    std::uint32_t total_tokens,
    const KvState& kvstate,
    const hyperion::model::KvGrowthPlan* operation_plan) const {
    PrefillAdmissionResult result;
    if (proposed_n_tokens == 0) {
        result.decision = evaluate(
            AdmissionInput{0, offset, StepKind::Prefill, false},
            kvstate,
            operation_plan);
        result.attempts[0] = result.decision;
        result.attempt_count = 1;
        return result;
    }

    std::uint32_t take = proposed_n_tokens;
    while (true) {
        const bool final_chunk =
            offset <= total_tokens && take == total_tokens - offset;
        const GovernorDecision decision = evaluate(
            AdmissionInput{take, offset, StepKind::Prefill, final_chunk},
            kvstate,
            operation_plan);
        result.attempts[result.attempt_count++] = decision;
        result.n_tokens = take;
        result.decision = decision;

        if (decision.admission == Admission::Accepted || take == 1) {
            // At one token soft pressure is advisory and executable; hard pressure
            // is the sole terminal prefill rejection.
            return result;
        }
        take = std::max(std::uint32_t{1}, take / 2);
    }
}

std::uint64_t predict_peak(
    const AdmissionInput& input,
    const KvState& kvstate,
    const Geometry& geometry,
    const hyperion::model::KvGrowthPlan* operation_plan) {
    return predict_memory(input, kvstate, geometry, operation_plan).peak_bytes;
}

TransientPrediction predict_transient(
    const Geometry& geometry,
    const AdmissionInput& input) {
    return predict_transient_impl(geometry, input);
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
    // G4 scaffold: predict a production-shaped chunked prefill rather than one
    // impossible context-wide forward. It remains intentionally unwired until the
    // weight estimator becomes quantization-aware.
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
    const std::uint64_t emb = saturating_multiply(
        geometry.vocab_size, geometry.hidden_size);
    const std::uint64_t attention_weights = saturating_multiply(
        saturating_multiply(geometry.hidden_size, geometry.hidden_size), 4);
    const std::uint64_t mlp_weights = saturating_multiply(
        saturating_multiply(geometry.hidden_size, geometry.intermediate_size), 3);
    const std::uint64_t per_layer =
        saturating_add(attention_weights, mlp_weights);
    const std::uint64_t lm_head = geometry.tie_word_embeddings ? 0 : emb;
    std::uint64_t model_elements = saturating_add(
        emb,
        saturating_multiply(geometry.num_hidden_layers, per_layer));
    model_elements = saturating_add(model_elements, lm_head);
    const std::uint64_t model_bytes =
        saturating_multiply(model_elements, kBf16Bytes);

    const std::uint64_t local_layers = static_cast<std::uint64_t>(std::count(
        geometry.layer_types.begin(), geometry.layer_types.end(), LayerType::Sliding));
    const std::uint64_t global_layers = static_cast<std::uint64_t>(std::count(
        geometry.layer_types.begin(), geometry.layer_types.end(), LayerType::Full));
    std::uint64_t local_kv = saturating_add(
        geometry.sliding_window, hyperion::model::kDefaultGammaMax);
    local_kv = saturating_multiply(local_kv, geometry.num_kv_heads_local);
    local_kv = saturating_multiply(local_kv, geometry.head_dim_local);
    local_kv = saturating_multiply(local_kv, 2ULL * kBf16Bytes);
    local_kv = saturating_multiply(local_kv, local_layers);

    constexpr std::uint64_t kGlobalCapacityStep = 256;
    const std::uint64_t global_capacity = context_len == 0
        ? 0
        : saturating_multiply(
              (static_cast<std::uint64_t>(context_len) + kGlobalCapacityStep - 1) /
                  kGlobalCapacityStep,
              kGlobalCapacityStep);
    std::uint64_t global_kv = saturating_multiply(
        global_capacity, geometry.num_kv_heads_global);
    global_kv = saturating_multiply(global_kv, geometry.head_dim_global);
    global_kv = saturating_multiply(global_kv, 2ULL * kBf16Bytes);
    global_kv = saturating_multiply(global_kv, global_layers);
    const std::uint64_t kv_bytes = saturating_add(local_kv, global_kv);

    constexpr std::uint32_t kPrefillChunkSize = 2048;
    std::uint64_t operation_transient = 0;
    std::uint32_t offset = 0;
    while (offset < context_len) {
        const std::uint32_t take =
            std::min(kPrefillChunkSize, context_len - offset);
        const auto transient = predict_transient_impl(
            geometry,
            AdmissionInput{
                take,
                offset,
                StepKind::Prefill,
                take == context_len - offset,
            });
        operation_transient =
            std::max(operation_transient, transient.operation_peak_bytes);
        offset += take;
    }

    std::uint64_t predicted = saturating_add(model_bytes, kv_bytes);
    predicted = saturating_add(predicted, operation_transient);
    predicted = saturating_add(predicted, kWorkspaceReserveBytes);

    return predicted <= budget_ceiling_bytes;
}

} // namespace hyperion::governor
