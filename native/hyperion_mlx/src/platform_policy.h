#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace hyperion::platform {

constexpr std::uint32_t kMinimumAppleGpuFamily = 1010;
constexpr std::uint32_t kMinimumMacOsMajor = 26;
constexpr std::uint32_t kMinimumMacOsMinor = 2;
constexpr std::uint32_t kPinnedMlxMajor = 0;
constexpr std::uint32_t kPinnedMlxMinor = 32;
constexpr std::uint32_t kPinnedMlxPatch = 0;
constexpr std::uint64_t kProfileCeilingBytes = 12ULL * 1024ULL * 1024ULL * 1024ULL;

struct Version {
    std::uint32_t major;
    std::uint32_t minor;
    std::uint32_t patch;
};

struct Budget {
    std::uint64_t effective_bytes;
    std::uint64_t soft_watermark_bytes;
};

struct Decision {
    bool supported;
    Budget budget;
    std::string reason;
};

struct RuntimeEnvironmentDecision {
    bool supported;
    const char* reason;
};

using DeviceInfo =
    std::unordered_map<std::string, std::variant<std::string, std::size_t>>;

[[nodiscard]] Budget derive_budget(std::uint64_t recommended_working_set_bytes);

/// Extract and validate MLX's public Metal recommendation, then derive the
/// effective/soft budget. Missing, wrong-typed, and zero device data fail closed.
[[nodiscard]] Decision derive_device_budget(const DeviceInfo& device_info);

/// MLX 0.32.0 accepts any positive ``MLX_SDPA_BLOCKS`` value for the two-pass
/// vector-attention workspace. The governor intentionally models the pinned
/// source-selected maximum instead, so an inherited override must fail closed.
[[nodiscard]] RuntimeEnvironmentDecision evaluate_runtime_environment(
    bool has_mlx_sdpa_blocks_override);

[[nodiscard]] Decision evaluate(
    bool has_gpu,
    std::uint32_t apple_gpu_family,
    Version macos,
    Version mlx_compile,
    std::string_view mlx_runtime,
    std::uint64_t recommended_working_set_bytes);

} // namespace hyperion::platform
