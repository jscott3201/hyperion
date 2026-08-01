#pragma once

#include "dispatch.h"
#include "geometry.h"
#include "hyperion_mlx.h"
#include "kv_cache.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace mx = mlx::core;

namespace hyperion::governor {

using hyperion::model::Geometry;
using hyperion::model::KvState;

/// Fixed overhead added to each predicted peak before device-budget admission.
/// The 512 MiB reserve is the governor's safety margin for workspace + graph buffers
/// that MLX does not attribute to get_active_memory / get_cache_memory.
inline constexpr std::uint64_t kWorkspaceReserveBytes = 512ULL * 1024ULL * 1024ULL;

/// Planning/throughput global KV cost per logical token. Admission derives actual
/// capacity-stepped allocation from the live cache geometry instead.
inline constexpr std::uint64_t kGlobalKvBytesPerToken = 16 * 1024;

/// Sentinel context lengths for the G4 peak-≤-budget gate (tokens).
inline constexpr std::uint32_t kSentinel8K = 8192;
inline constexpr std::uint32_t kSentinel32K = 32768;

/// Governor admission outcome for a proposed step.
enum class Admission {
    /// The step is within budget; proceed.
    Accepted,
    /// The step would breach the soft watermark; halve the chunk and retry.
    SoftPaused,
    /// The proposed shape breaches the hard ceiling. Prefill halves this signal;
    /// only a one-token hard result is terminal.
    HardRejected,
};

/// Execution shape being admitted. ``n_tokens`` is a single forward query length for
/// prefill, but a count of sequential q=1 forwards for decode; the distinction keeps
/// continuation-prefill-only scratch out of decode admission and telemetry.
enum class StepKind {
    Prefill,
    Decode,
};

/// Exact execution shape proposed to admission. Prefill is one q=n forward;
/// decode is ``n_tokens`` sequential q=1 forwards. ``include_epilogue`` is true
/// only when this execution shape also evaluates lm_head + softcap: every non-empty
/// decode block and only the currently proposed final prefill chunk.
struct AdmissionInput {
    std::uint32_t n_tokens;
    std::uint32_t offset;
    StepKind step_kind;
    bool include_epilogue;
};

/// Pinned MLX 0.32.0 attention dispatch selected for one layer kind.
enum class AttentionPath {
    None,
    FusedVector,
    FusedFull,
    Fallback,
};

/// Per-phase transient prediction. Layer evaluation is sequential, so attention
/// is the maximum of the one-local-layer and one-global-layer phases. The logits
/// epilogue is sequential after the decoder stack, so the operation peak is the
/// maximum of attention and epilogue rather than their sum.
struct TransientPrediction {
    std::uint64_t local_phase_bytes;
    std::uint64_t global_phase_bytes;
    std::uint64_t attention_peak_bytes;
    std::uint64_t epilogue_phase_bytes;
    std::uint64_t operation_peak_bytes;
    std::uint64_t local_kv_length;
    std::uint64_t global_kv_length;
    AttentionPath local_path;
    AttentionPath global_path;
};

/// Governor state reported back to the caller + telemetry.
struct GovernorDecision {
    Admission admission = Admission::HardRejected;
    /// Predicted peak MLX bytes if the step proceeds (bytes).
    std::uint64_t predicted_peak_bytes = 0;
    /// The effective budget ceiling this decision was checked against (bytes).
    std::uint64_t budget_ceiling_bytes = 0;
    /// The soft watermark (bytes); breaching it triggers SoftPaused.
    std::uint64_t soft_watermark_bytes = 0;
    /// Predicted local KV bytes after the step (bytes).
    std::uint64_t local_kv_bytes = 0;
    /// Predicted global KV bytes after the step (bytes).
    std::uint64_t global_kv_bytes = 0;
    /// Human-readable reason for a rejection/pause (empty on Accepted).
    const char* reason = "";
};

/// Repeated halve-and-re-evaluate result for one prefill chunk. A uint32 proposal
/// reaches one in at most 32 attempts. Every decision is retained so step telemetry
/// can observe every attempted prediction without allocating on the request path.
struct PrefillAdmissionResult {
    std::uint32_t n_tokens = 0;
    GovernorDecision decision{};
    std::array<GovernorDecision, 32> attempts{};
    std::size_t attempt_count = 0;
};

/// Predictive admission governor for the M2 native forward pass.
///
/// Computes the predicted peak memory if a proposed prefill/decode step were to
/// proceed, and compares it against the device-derived budget ceiling + soft
/// watermark. The formula (from 05 §KV-and-memory, M2-2.6b):
///
/// ```text
/// predicted_peak = settled_working_set
///                + kv_reallocation_peak    [full replacement buffers on growth]
///                + staging_cow             [growth-transaction cache candidates]
///                + operation_transient     [max attention/epilogue phase]
///                + workspace + 512 MiB reserve
/// ```
///
/// where:
///   settled_working_set = mx::get_active_memory() + mx::get_cache_memory()
///   kv_reallocation_peak = full projected K+V bytes for each growing prefill cache,
///       or the sum of every crossed replacement capacity for sequential decode
///   staging_cow = all local K+V storage plus non-growing global K+V storage when
///       any global cache grows; sequential decode also charges a growing cache's
///       current candidate when no-growth appends precede its first crossing; each
///       unavailable retained K/V input is charged until settled memory contains it
///   operation_transient = max(attention_peak, final_logits_epilogue)
///   attention_peak = max(one local-layer phase, one global-layer phase)
///   fallback layer phase = score[B,Hq,q,k] + output[B,Hq,q,Dv] + bool mask[q,k]
///   fused layer phase = output + bool mask + pinned-kernel workspace allowance
///   local continuation prefill additionally keeps one assembled K/V scratch pair
///   alive with the local attention phase. All phase math is saturating and retains
///   the 1.25 transient safety factor. q is the chunk width for prefill and one for
///   sequential decode; decode K uses the last step in the admitted block.
///   final_logits_epilogue charges the full BF16 [1,q,vocab] tensor only for a final
///   prefill chunk (or q=1 decode), before the last-row slice.
///
/// The governor is stateless across steps (it reads live MLX memory counters each
/// call) but holds a reference to the geometry + budget for the per-layer math.
class Governor {
  public:
    /// ``budget_ceiling_bytes`` and ``soft_watermark_bytes`` come from
    /// ``platform::derive_budget`` (already clamped to the 12 GiB profile ceiling).
    Governor(
        const Geometry& geometry,
        std::uint64_t budget_ceiling_bytes,
        std::uint64_t soft_watermark_bytes);

    /// Predict admission for ``input`` at its committed offset. ``kvstate`` is read
    /// to compute current local/global KV bytes and replacement/staging lifetimes.
    [[nodiscard]] GovernorDecision evaluate(
        const AdmissionInput& input,
        const KvState& kvstate,
        const hyperion::model::KvGrowthPlan* operation_plan = nullptr) const;

    /// Repeatedly halve a prefill proposal after either hard or soft pressure.
    /// Recomputes K lengths and final-chunk epilogue inclusion at every shape. A
    /// hard decision is terminal only at one token; soft-at-one is executable.
    [[nodiscard]] PrefillAdmissionResult admit_prefill_to_fit(
        std::uint32_t proposed_n_tokens,
        std::uint32_t offset,
        std::uint32_t total_tokens,
        const KvState& kvstate,
        const hyperion::model::KvGrowthPlan* operation_plan = nullptr) const;

    /// The hard ceiling this governor checks against (bytes).
    [[nodiscard]] std::uint64_t budget_ceiling_bytes() const { return budget_ceiling_bytes_; }
    /// The soft watermark (bytes); breaching triggers SoftPaused (halve-chunk).
    [[nodiscard]] std::uint64_t soft_watermark_bytes() const { return soft_watermark_bytes_; }

  private:
    const Geometry& geometry_;
    std::uint64_t budget_ceiling_bytes_;
    std::uint64_t soft_watermark_bytes_;

    /// Current local KV bytes (sum of all LocalKvCache buffers, bf16).
    [[nodiscard]] std::uint64_t local_kv_bytes(const KvState& kvstate) const;
};

/// Compute the predicted peak for a step WITHOUT admission (for telemetry fill
/// in the step result). Reads live MLX memory counters.
[[nodiscard]] std::uint64_t predict_peak(
    const AdmissionInput& input,
    const KvState& kvstate,
    const Geometry& geometry,
    const hyperion::model::KvGrowthPlan* operation_plan = nullptr);

/// Geometry/dispatch-aware transient phase prediction used by admission and the
/// load-time G4 scaffold. It is independent of live MLX memory and KV allocation.
[[nodiscard]] TransientPrediction predict_transient(
    const Geometry& geometry,
    const AdmissionInput& input);

/// The throughput-optimum context length (A4) — the context at which throughput
/// peaks before memory pressure dominates. NOT the ceiling; the governor targets
/// this for scheduling, not the hard budget. Computed from the budget ceiling
/// and the per-token KV growth rate.
[[nodiscard]] std::uint32_t throughput_optimum_context(
    std::uint64_t budget_ceiling_bytes);

/// G4 scaffold: does a production-shaped chunked prefill at ``context_len`` stay
/// within budget? This remains unwired pending quantization-aware weight accounting.
[[nodiscard]] bool peak_within_budget(
    std::uint32_t context_len,
    const Geometry& geometry,
    std::uint64_t budget_ceiling_bytes);

} // namespace hyperion::governor
