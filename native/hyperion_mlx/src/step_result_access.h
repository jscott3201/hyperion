#pragma once

#include "hyperion_mlx.h"
#include "forward.h"
#include "governor.h"
#include "kv_cache.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace hyperion::model {

/// Test-only C++ accessor for a ``HypStepResult``'s fields.
///
/// The public C ABI now exposes ``hyp_step_result_fields`` for serving. Native C++ tests
/// retain this by-value helper for compact assertions; it is defined in ``model.cc`` where
/// ``HypStepResultOpaque`` is visible. NOT an ABI function (no ``extern "C"``, not counted
/// by ``check-abi-surface.sh``).
HypStepResultFields step_result_read(HypStepResult result) noexcept;

struct StepTelemetrySnapshot {
    std::uint64_t peak_mlx_bytes;
    std::uint64_t local_kv_bytes;
    std::uint64_t global_kv_bytes;
};

/// Source-private accumulator shared by every greedy/stochastic public step. Each
/// pre-admission prediction is observed exactly once; the terminal snapshot combines
/// their maximum with live allocated KV capacity. Active MLX memory is deliberately
/// absent: the result writer samples it at the terminal struct write.
class StepTelemetryAccumulator {
  public:
    [[nodiscard]] hyperion::governor::GovernorDecision observe(
        hyperion::governor::GovernorDecision decision) noexcept {
        peak_mlx_bytes_ = std::max(peak_mlx_bytes_, decision.predicted_peak_bytes);
        return decision;
    }

    [[nodiscard]] std::uint64_t peak_mlx_bytes() const noexcept {
        return peak_mlx_bytes_;
    }

    [[nodiscard]] StepTelemetrySnapshot snapshot(const KvState& kvstate) const noexcept {
        StepTelemetrySnapshot out{peak_mlx_bytes_, 0, 0};
        for (const auto& cache : kvstate.local) {
            out.local_kv_bytes = saturating_add(
                out.local_kv_bytes,
                cache_bytes(cache.capacity(), cache.num_kv_heads(), cache.head_dim()));
        }
        for (const auto& cache : kvstate.global) {
            out.global_kv_bytes = saturating_add(
                out.global_kv_bytes,
                cache_bytes(cache.capacity(), cache.num_kv_heads(), cache.head_dim()));
        }
        return out;
    }

  private:
    static std::uint64_t saturating_multiply(
        std::uint64_t lhs, std::uint64_t rhs) noexcept {
        constexpr auto kMax = std::numeric_limits<std::uint64_t>::max();
        if (lhs == 0 || rhs == 0) {
            return 0;
        }
        return lhs > kMax / rhs ? kMax : lhs * rhs;
    }

    static std::uint64_t saturating_add(
        std::uint64_t lhs, std::uint64_t rhs) noexcept {
        constexpr auto kMax = std::numeric_limits<std::uint64_t>::max();
        return rhs > kMax - lhs ? kMax : lhs + rhs;
    }

    static std::uint64_t cache_bytes(
        std::uint32_t capacity,
        std::uint32_t heads,
        std::uint32_t head_dim) noexcept {
        std::uint64_t bytes = saturating_multiply(capacity, heads);
        bytes = saturating_multiply(bytes, head_dim);
        // Production KV is BF16 and every cache owns distinct K and V buffers.
        return saturating_multiply(bytes, 2U * 2U);
    }

    std::uint64_t peak_mlx_bytes_ = 0;
};

/// Write the sampled token + terminal governor/KV telemetry to a validated result.
/// Samples active MLX memory at the result write. Defined in model.cc where the opaque
/// result body is visible.
void write_step_result(
    HypStepResult result,
    const ForwardPass::GreedySample& sample,
    std::uint32_t near_tie_events,
    HypGovernorState governor_state,
    const StepTelemetrySnapshot& telemetry);

/// M3 sampler: write the STOCHASTIC sample + governor telemetry + the top-k logprob
/// sidecar. Same telemetry fields as write_step_result, plus the ≤HYP_TOP_K_LOGPROBS
/// top-k ids/logprobs from the StochasticSample. near_tie_events is 0 for the sampled
/// path (near-tie is a greedy-only metric). Defined in model.cc.
void write_step_result_sampled(
    HypStepResult result,
    const ForwardPass::StochasticSample& sample,
    HypGovernorState governor_state,
    const StepTelemetrySnapshot& telemetry);

} // namespace hyperion::model
