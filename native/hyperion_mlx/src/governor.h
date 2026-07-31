#pragma once

#include "dispatch.h"
#include "geometry.h"
#include "hyperion_mlx.h"
#include "kv_cache.h"

#include <cstdint>

namespace mx = mlx::core;

namespace hyperion::governor {

using hyperion::model::Geometry;
using hyperion::model::KvState;

/// Fixed overheads (bytes) subtracted from the 12.06 GiB ceiling before admission.
/// The 512 MiB reserve is the governor's safety margin for workspace + graph buffers
/// that MLX does not attribute to get_active_memory / get_cache_memory.
inline constexpr std::uint64_t kWorkspaceReserveBytes = 512ULL * 1024ULL * 1024ULL;

/// KV append growth per global token (bytes). The global cache stores K and V
/// separately (M2-2.7 fix), each [num_kv_heads_global, head_dim_global] bf16 = 2 bytes.
/// 16 KiB/token is the documented per-token global KV cost (05 §KV-and-memory).
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
    /// The step would breach the hard ceiling even at 1 token; reject.
    HardRejected,
};

/// Governor state reported back to the caller + telemetry.
struct GovernorDecision {
    Admission admission;
    /// Predicted peak MLX bytes if the step proceeds (bytes).
    std::uint64_t predicted_peak_bytes;
    /// The effective budget ceiling this decision was checked against (bytes).
    std::uint64_t budget_ceiling_bytes;
    /// The soft watermark (bytes); breaching it triggers SoftPaused.
    std::uint64_t soft_watermark_bytes;
    /// Predicted local KV bytes after the step (bytes).
    std::uint64_t local_kv_bytes;
    /// Predicted global KV bytes after the step (bytes).
    std::uint64_t global_kv_bytes;
    /// Human-readable reason for a rejection/pause (empty on Accepted).
    const char* reason;
};

/// Predictive admission governor for the M2 native forward pass.
///
/// Computes the predicted peak memory if a proposed prefill/decode step were to
/// proceed, and compares it against the device-derived budget ceiling + soft
/// watermark. The formula (from 05 §KV-and-memory, M2-2.6b):
///
/// ```text
/// predicted_peak = settled_working_set
///                + kv_append               [16 KiB/token global]
///                + attention_transient     [SDPA outputs + continuation scratch]
///                + workspace + 512 MiB reserve
/// ```
///
/// where:
///   settled_working_set = mx::get_active_memory() + mx::get_cache_memory()
///   kv_append           = n_tokens * kGlobalKvBytesPerToken
///   attention_transient = (sum over layers of
///       q * head_dim_local  * n_heads * dtype  [sliding]
///       q * head_dim_global * n_heads * dtype  [global]) * safety
///       + one assembled local K/V buffer for continuation prefill
///   (the transient is the fused-SDPA OUTPUT [B, n_heads, q, head_dim] per layer —
///    MLX's mx::fast::scaled_dot_product_attention is a FUSED kernel that does NOT
///    materialize the [q, ctx] scores matrix (it streams it), so the transient is the
///    output buffer, scaled by num_attention_heads — the QUERY head count. The
///    continuation-prefill path additionally assembles the retained local prefix and
///    current chunk into bounded K and V buffers before SDPA; the governor charges one
///    exact per-layer peak for those buffers when ``offset > 0 && n_tokens > 1``. The
///    kTransientSafety factor (1.25) covers the fused-kernel tile working set. The
///    M2-2.6b governor modeled the full [q, ctx] scores and over-predicted by 2.3-3.6x
///    (M3 calibration, benchmarks/m3/governor-calibration.json); this is the fix.)
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

    /// Predict admission for a prefill/decode step of ``n_tokens`` at the given
    /// committed ``offset`` (the current prefix length; 0 for a fresh prefill).
    /// ``kvstate`` is read to compute the current local/global KV byte counts.
    [[nodiscard]] GovernorDecision evaluate(
        std::uint32_t n_tokens,
        std::uint32_t offset,
        const KvState& kvstate) const;

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
    /// Current global KV bytes (sum of all GlobalKvCache buffers, bf16).
    [[nodiscard]] std::uint64_t global_kv_bytes(const KvState& kvstate) const;
    /// Predicted attention transient for one step of ``n_tokens`` at ``offset`` (bytes).
    [[nodiscard]] std::uint64_t attention_transient(
        std::uint32_t n_tokens,
        std::uint32_t offset) const;
};

/// Compute the predicted peak for a step WITHOUT admission (for telemetry fill
/// in the step result). Reads live MLX memory counters.
[[nodiscard]] std::uint64_t predict_peak(
    std::uint32_t n_tokens,
    std::uint32_t offset,
    const KvState& kvstate,
    const Geometry& geometry);

/// The throughput-optimum context length (A4) — the context at which throughput
/// peaks before memory pressure dominates. NOT the ceiling; the governor targets
/// this for scheduling, not the hard budget. Computed from the budget ceiling
/// and the per-token KV growth rate.
[[nodiscard]] std::uint32_t throughput_optimum_context(
    std::uint64_t budget_ceiling_bytes);

/// G4 gate: does the predicted peak at ``context_len`` tokens stay ≤ budget?
/// Checked at the 8K/32K sentinels during model load.
[[nodiscard]] bool peak_within_budget(
    std::uint32_t context_len,
    const Geometry& geometry,
    std::uint64_t budget_ceiling_bytes);

} // namespace hyperion::governor
