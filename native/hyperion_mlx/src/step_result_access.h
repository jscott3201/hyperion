#pragma once

#include "hyperion_mlx.h"
#include "forward.h"

#include <cstdint>

namespace hyperion::model {

/// Test-only C++ accessor for a ``HypStepResult``'s fields.
///
/// The C ABI exposes only ``hyp_step_result_create`` / ``hyp_step_result_free`` (the
/// v1 ratchet is 11/25, M2-2.7 adds no new ABI signatures); the serving-path read accessor
/// (``hyp_step_result_fields``) is a later slice. The M2-2.7 native tests need to read the
/// sampled ``token_id`` / ``logit`` / ``near_tie_events`` back to assert token-exact parity
/// vs the oracle, so this private C++ helper — defined in ``model.cc`` where
/// ``HypStepResultOpaque`` is visible — threads the fields out without growing the public
/// ABI surface. NOT an ABI function (no ``extern "C"``, not counted by
/// ``check-abi-surface.sh``).
HypStepResultFields step_result_read(HypStepResult result) noexcept;

/// Write the sampled token + governor telemetry to a validated step-result handle.
/// Fills the M2-2.6b telemetry fields (peak/active MLX bytes, KV byte counts,
/// governor_state). Defined in model.cc where HypStepResultOpaque is visible.
void write_step_result(
    HypStepResult result,
    const ForwardPass::GreedySample& sample,
    std::uint32_t near_tie_events,
    HypGovernorState governor_state,
    std::uint64_t peak_mlx_bytes,
    std::uint64_t active_mlx_bytes,
    std::uint64_t local_kv_bytes,
    std::uint64_t global_kv_bytes);

} // namespace hyperion::model
