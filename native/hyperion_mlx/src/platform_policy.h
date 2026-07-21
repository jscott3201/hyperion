#pragma once

#include <cstdint>
#include <string>
#include <string_view>

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

[[nodiscard]] Budget derive_budget(std::uint64_t recommended_working_set_bytes);

[[nodiscard]] Decision evaluate(
    bool has_gpu,
    std::uint32_t apple_gpu_family,
    Version macos,
    Version mlx_compile,
    std::string_view mlx_runtime,
    std::uint64_t recommended_working_set_bytes);

} // namespace hyperion::platform
