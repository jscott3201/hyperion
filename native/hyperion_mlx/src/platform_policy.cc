#include "platform_policy.h"

#include <algorithm>
#include <limits>

namespace hyperion::platform {
namespace {

bool version_at_least(Version actual, Version minimum) {
    if (actual.major != minimum.major) {
        return actual.major > minimum.major;
    }
    if (actual.minor != minimum.minor) {
        return actual.minor > minimum.minor;
    }
    return actual.patch >= minimum.patch;
}

bool version_equal(Version left, Version right) {
    return left.major == right.major && left.minor == right.minor && left.patch == right.patch;
}

std::uint64_t multiply_ratio_floor(
    std::uint64_t value,
    std::uint64_t numerator,
    std::uint64_t denominator) {
    return (value / denominator) * numerator +
        ((value % denominator) * numerator) / denominator;
}

} // namespace

Budget derive_budget(std::uint64_t recommended_working_set_bytes) {
    const std::uint64_t allocator_ceiling =
        multiply_ratio_floor(recommended_working_set_bytes, 949, 1000);
    const std::uint64_t effective = std::min(allocator_ceiling, kProfileCeilingBytes);
    return Budget{
        effective,
        multiply_ratio_floor(effective, 9, 10),
    };
}

Decision derive_device_budget(const DeviceInfo& device_info) {
    constexpr const char* kInvalidRecommendation =
        "device did not report a valid max recommended working set size";
    const auto found =
        device_info.find("max_recommended_working_set_size");
    if (found == device_info.end()) {
        return Decision{false, {}, kInvalidRecommendation};
    }
    const auto* recommended = std::get_if<std::size_t>(&found->second);
    if (recommended == nullptr || *recommended == 0) {
        return Decision{false, {}, kInvalidRecommendation};
    }
    const Budget budget = derive_budget(*recommended);
    if (budget.effective_bytes == 0 || budget.soft_watermark_bytes == 0) {
        return Decision{false, {}, kInvalidRecommendation};
    }
    return Decision{true, budget, {}};
}

RuntimeEnvironmentDecision evaluate_runtime_environment(
    bool has_mlx_sdpa_blocks_override) {
    if (has_mlx_sdpa_blocks_override) {
        return RuntimeEnvironmentDecision{
            false,
            "MLX_SDPA_BLOCKS is unsupported because it invalidates governor workspace accounting",
        };
    }
    return RuntimeEnvironmentDecision{true, ""};
}

Decision evaluate(
    bool has_gpu,
    std::uint32_t apple_gpu_family,
    Version macos,
    Version mlx_compile,
    std::string_view mlx_runtime,
    std::uint64_t recommended_working_set_bytes) {
    if (!has_gpu) {
        return Decision{false, {}, "no Metal device is available"};
    }
    if (apple_gpu_family < kMinimumAppleGpuFamily) {
        return Decision{false, {}, "requires Apple GPU family 10 (M5 generation) or newer"};
    }
    if (!version_at_least(
            macos,
            Version{kMinimumMacOsMajor, kMinimumMacOsMinor, 0})) {
        return Decision{false, {}, "requires macOS 26.2 or newer"};
    }
    if (!version_equal(
            mlx_compile,
            Version{kPinnedMlxMajor, kPinnedMlxMinor, kPinnedMlxPatch})) {
        return Decision{false, {}, "native wrapper was not compiled against MLX 0.32.0"};
    }
    if (mlx_runtime != "0.32.0") {
        return Decision{false, {}, "loaded MLX runtime is not exactly 0.32.0"};
    }
    if (recommended_working_set_bytes == 0) {
        return Decision{false, {}, "Metal reported a zero recommended working set"};
    }
    const Budget budget = derive_budget(recommended_working_set_bytes);
    if (budget.effective_bytes == 0 || budget.soft_watermark_bytes == 0) {
        return Decision{false, {}, "device-derived memory budget is zero"};
    }
    return Decision{true, budget, {}};
}

} // namespace hyperion::platform
