#include "hyperion_mlx.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string_view>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "runtime_canary_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::uint64_t multiply_ratio_floor(
    std::uint64_t value,
    std::uint64_t numerator,
    std::uint64_t denominator) {
    return (value / denominator) * numerator +
        ((value % denominator) * numerator) / denominator;
}

} // namespace

int main() {
    HypCanaryInfo info{};
    const HypStatus status = hyp_runtime_canary(&info);
    if (status != HYP_STATUS_OK) {
        std::array<char, 1024> error{};
        (void)hyp_last_error(error.data(), error.size());
        std::fprintf(stderr, "%s\n", error.data());
        return 1;
    }
    require(info.abi_version == 3, "the ABI version changed");
    require(info.gpu_family >= 1010, "the runtime did not prove Apple10 support");
    require(
        info.macos_major > 26 || (info.macos_major == 26 && info.macos_minor >= 2),
        "the runtime accepted an operating system below macOS 26.2");
    require(
        info.mlx_compile_major == 0 && info.mlx_compile_minor == 32 &&
            info.mlx_compile_patch == 0,
        "the runtime was compiled against the wrong MLX headers");
    require(
        std::string_view(info.mlx_runtime_version) == "0.32.0",
        "the runtime loaded the wrong MLX dylib");
    require(info.recommended_working_set_bytes > 0, "Metal returned no working-set budget");
    const std::uint64_t expected_budget = std::min(
        multiply_ratio_floor(info.recommended_working_set_bytes, 949, 1000),
        12ULL * 1024ULL * 1024ULL * 1024ULL);
    require(
        info.effective_budget_bytes == expected_budget,
        "the effective budget does not match the public formula");
    require(
        info.soft_watermark_bytes == multiply_ratio_floor(expected_budget, 9, 10),
        "the soft watermark does not match the public formula");
    require(info.mlx_probe_value == 4.0F, "the MLX tensor probe returned the wrong value");
    require(
        info.metallib_probe_value == 42.0F,
        "the custom metallib probe returned the wrong value");
    require(info.gpu_name[0] != '\0', "the public GPU diagnostic is empty");
    return 0;
}
