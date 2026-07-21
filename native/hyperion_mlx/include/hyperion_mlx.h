#ifndef HYPERION_MLX_H
#define HYPERION_MLX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Stable status values returned by every Hyperion native ABI function. */
typedef enum HypStatus {
    HYP_STATUS_OK = 0,
    HYP_STATUS_INVALID_ARGUMENT = 1,
    HYP_STATUS_NOT_FOUND = 2,
    HYP_STATUS_IO = 3,
    HYP_STATUS_OOM_GOVERNOR = 4,
    HYP_STATUS_UNSUPPORTED = 5,
    HYP_STATUS_INTERNAL = 6,
    HYP_STATUS_CANCELLED = 7
} HypStatus;

/** Results from the stateless M0 platform and native-execution canary. */
typedef struct HypCanaryInfo {
    uint32_t abi_version;
    uint32_t mlx_compile_major;
    uint32_t mlx_compile_minor;
    uint32_t mlx_compile_patch;
    uint32_t macos_major;
    uint32_t macos_minor;
    uint32_t macos_patch;
    uint32_t gpu_family;
    uint64_t recommended_working_set_bytes;
    uint64_t effective_budget_bytes;
    uint64_t soft_watermark_bytes;
    float mlx_probe_value;
    float metallib_probe_value;
    char mlx_runtime_version[32];
    char gpu_name[128];
} HypCanaryInfo;

/**
 * Enforce the M5/macOS 26.2/MLX 0.32.0 floor, execute real MLX and custom
 * metallib probes, and return the device-derived 16 GB-profile budget.
 */
HypStatus hyp_runtime_canary(HypCanaryInfo* out_info);

/** Copy the calling thread's last native error into a caller-owned buffer. */
HypStatus hyp_last_error(char* buffer, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif
