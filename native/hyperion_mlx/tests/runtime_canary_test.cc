#include "hyperion_mlx.h"

#include <array>
#include <cassert>
#include <cstdio>

int main() {
    HypCanaryInfo info{};
    const HypStatus status = hyp_runtime_canary(&info);
    if (status != HYP_STATUS_OK) {
        std::array<char, 1024> error{};
        (void)hyp_last_error(error.data(), error.size());
        std::fprintf(stderr, "%s\n", error.data());
        return 1;
    }
    assert(info.abi_version == 1);
    assert(info.gpu_family >= 1010);
    assert(info.mlx_compile_major == 0);
    assert(info.mlx_compile_minor == 32);
    assert(info.mlx_compile_patch == 0);
    assert(info.recommended_working_set_bytes > 0);
    assert(info.effective_budget_bytes <= info.recommended_working_set_bytes);
    assert(info.soft_watermark_bytes < info.effective_budget_bytes);
    assert(info.mlx_probe_value == 4.0F);
    assert(info.metallib_probe_value == 42.0F);
    return 0;
}
