#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "hyperion_mlx.h"
#include "abi_error.h"
#include "platform_policy.h"

#include <mlx/mlx.h>
#include <mlx/version.h>

#include <mach-o/dyld.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef HYPERION_METALLIB_DEFAULT_PATH
#define HYPERION_METALLIB_DEFAULT_PATH "hyperion_canary.metallib"
#endif

namespace mx = mlx::core;

namespace {

constexpr std::uint32_t kAbiVersion = 2;

static_assert(MLX_VERSION_MAJOR == 0);
static_assert(MLX_VERSION_MINOR == 32);
static_assert(MLX_VERSION_PATCH == 0);
static_assert(sizeof(HypCanaryInfo) == 224);
static_assert(offsetof(HypCanaryInfo, recommended_working_set_bytes) == 32);
static_assert(offsetof(HypCanaryInfo, mlx_runtime_version) == 64);
static_assert(offsetof(HypCanaryInfo, gpu_name) == 96);

class NativeError final : public std::runtime_error {
  public:
    NativeError(HypStatus status, const std::string& message)
        : std::runtime_error(message), status_(status) {}

    [[nodiscard]] HypStatus status() const noexcept { return status_; }

  private:
    HypStatus status_;
};

void store_error(const char* message) noexcept {
    hyperion::abi::set_last_error(message);
}

HypStatus fail(HypStatus status, const char* message) noexcept {
    store_error(message);
    return status;
}

HypStatus ok() noexcept {
    hyperion::abi::clear_last_error();
    return HYP_STATUS_OK;
}

HypStatus fail_unexpected(const char* operation) noexcept {
    std::snprintf(
        hyperion::abi::g_last_error.data(),
        hyperion::abi::g_last_error.size(),
        "%s failed with a non-standard exception",
        operation != nullptr ? operation : "native operation");
    return HYP_STATUS_INTERNAL;
}

template <typename Function>
HypStatus abi_call(const char* operation, Function&& function) noexcept {
    try {
        function();
        return ok();
    } catch (const NativeError& error) {
        return fail(error.status(), error.what());
    } catch (const std::bad_alloc&) {
        return fail(
            HYP_STATUS_INTERNAL,
            "native allocation failed outside predictive governor admission");
    } catch (const std::filesystem::filesystem_error& error) {
        return fail(HYP_STATUS_IO, error.what());
    } catch (const std::exception& error) {
        return fail(HYP_STATUS_INTERNAL, error.what());
    } catch (...) {
        return fail_unexpected(operation);
    }
}

std::filesystem::path executable_directory() {
    std::vector<char> buffer(1024);
    std::uint32_t size = static_cast<std::uint32_t>(buffer.size());
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        buffer.resize(size);
        if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
            throw NativeError(HYP_STATUS_IO, "could not resolve the executable path");
        }
    }
    return std::filesystem::weakly_canonical(buffer.data()).parent_path();
}

std::filesystem::path resolve_metallib() {
    std::vector<std::filesystem::path> candidates;
    if (const char* configured = std::getenv("HYPERION_METALLIB_PATH");
        configured != nullptr && *configured != '\0') {
        candidates.emplace_back(configured);
    }
    candidates.push_back(executable_directory() / "hyperion_canary.metallib");
    candidates.emplace_back(HYPERION_METALLIB_DEFAULT_PATH);

    for (const auto& candidate : candidates) {
        if (std::filesystem::is_regular_file(candidate)) {
            return std::filesystem::canonical(candidate);
        }
    }
    throw NativeError(
        HYP_STATUS_NOT_FOUND,
        "hyperion_canary.metallib is missing; set HYPERION_METALLIB_PATH or place it beside the executable");
}

std::string ns_error(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    const char* text = error.localizedDescription.UTF8String;
    return text != nullptr ? text : "unknown Metal error";
}

float run_metallib_probe(id<MTLDevice> device) {
    const std::filesystem::path path = resolve_metallib();
    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (library == nil) {
        throw NativeError(
            HYP_STATUS_IO,
            "failed to load Hyperion metallib: " + ns_error(error));
    }
    id<MTLFunction> function = [library newFunctionWithName:@"hyperion_canary_add_one"];
    if (function == nil) {
        throw NativeError(HYP_STATUS_INTERNAL, "Hyperion metallib omitted its canary kernel");
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        throw NativeError(
            HYP_STATUS_INTERNAL,
            "failed to create Hyperion canary pipeline: " + ns_error(error));
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> buffer = [device newBufferWithLength:sizeof(float)
                                             options:MTLResourceStorageModeShared];
    if (queue == nil || buffer == nil) {
        throw NativeError(HYP_STATUS_INTERNAL, "failed to allocate Metal canary resources");
    }
    *static_cast<float*>(buffer.contents) = 41.0F;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        throw NativeError(
            HYP_STATUS_INTERNAL,
            "Hyperion canary command failed: " + ns_error(command.error));
    }
    const float value = *static_cast<float*>(buffer.contents);
    if (!std::isfinite(value) || std::fabs(value - 42.0F) > 0.0001F) {
        throw NativeError(HYP_STATUS_INTERNAL, "Hyperion metallib canary returned the wrong value");
    }
    return value;
}

float run_mlx_probe() {
    const mx::Device gpu = mx::Device::gpu;
    const mx::Stream stream = mx::new_stream(gpu);
    mx::array values = mx::ones({4}, mx::float32, stream);
    mx::array result = mx::sum(values, stream);
    mx::eval(result);
    mx::synchronize(stream);
    const float value = result.item<float>();
    if (!std::isfinite(value) || std::fabs(value - 4.0F) > 0.0001F) {
        throw NativeError(HYP_STATUS_INTERNAL, "MLX tensor canary returned the wrong value");
    }
    mx::clear_cache();
    return value;
}

std::uint32_t highest_supported_apple_family(id<MTLDevice> device) {
    std::uint32_t family = 0;
    for (std::uint32_t candidate = 1001; candidate <= 1010; ++candidate) {
        if ([device supportsFamily:static_cast<MTLGPUFamily>(candidate)]) {
            family = candidate;
        }
    }
    return family;
}

void copy_string(char* destination, std::size_t capacity, std::string_view source) {
    if (capacity == 0) {
        return;
    }
    const std::size_t count = std::min(capacity - 1, source.size());
    std::memcpy(destination, source.data(), count);
    destination[count] = '\0';
}

} // namespace

HypStatus hyp_runtime_canary(HypCanaryInfo* out_info) {
    return abi_call("hyp_runtime_canary", [&] {
        if (out_info == nullptr) {
            throw NativeError(
                HYP_STATUS_INVALID_ARGUMENT,
                "hyp_runtime_canary requires a non-null output pointer");
        }
        *out_info = HypCanaryInfo{};
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            const bool has_gpu = device != nil;
            const std::uint32_t gpu_family =
                has_gpu ? highest_supported_apple_family(device) : 0;
            const NSOperatingSystemVersion os = NSProcessInfo.processInfo.operatingSystemVersion;
            const char* runtime_text = mx::version();
            const std::string runtime_version =
                runtime_text != nullptr ? runtime_text : "<null>";
            const std::uint64_t recommended =
                has_gpu ? static_cast<std::uint64_t>(device.recommendedMaxWorkingSetSize) : 0;
            const auto decision = hyperion::platform::evaluate(
                has_gpu,
                gpu_family,
                hyperion::platform::Version{
                    static_cast<std::uint32_t>(os.majorVersion),
                    static_cast<std::uint32_t>(os.minorVersion),
                    static_cast<std::uint32_t>(os.patchVersion)},
                hyperion::platform::Version{
                    MLX_VERSION_MAJOR,
                    MLX_VERSION_MINOR,
                    MLX_VERSION_PATCH},
                runtime_version,
                recommended);
            if (!decision.supported) {
                throw NativeError(HYP_STATUS_UNSUPPORTED, decision.reason);
            }
            const float metallib_probe = run_metallib_probe(device);
            const float mlx_probe = run_mlx_probe();
            out_info->abi_version = kAbiVersion;
            out_info->mlx_compile_major = MLX_VERSION_MAJOR;
            out_info->mlx_compile_minor = MLX_VERSION_MINOR;
            out_info->mlx_compile_patch = MLX_VERSION_PATCH;
            out_info->macos_major = static_cast<std::uint32_t>(os.majorVersion);
            out_info->macos_minor = static_cast<std::uint32_t>(os.minorVersion);
            out_info->macos_patch = static_cast<std::uint32_t>(os.patchVersion);
            out_info->gpu_family = gpu_family;
            out_info->recommended_working_set_bytes = recommended;
            out_info->effective_budget_bytes = decision.budget.effective_bytes;
            out_info->soft_watermark_bytes = decision.budget.soft_watermark_bytes;
            out_info->mlx_probe_value = mlx_probe;
            out_info->metallib_probe_value = metallib_probe;
            copy_string(
                out_info->mlx_runtime_version,
                sizeof(out_info->mlx_runtime_version),
                runtime_version);
            const char* gpu_name = device.name.UTF8String;
            copy_string(
                out_info->gpu_name,
                sizeof(out_info->gpu_name),
                gpu_name != nullptr ? gpu_name : "<unknown>");
        }
    });
}

HypStatus hyp_last_error(char* buffer, size_t buffer_len) {
    try {
        if (buffer == nullptr || buffer_len == 0) {
            return fail(
                HYP_STATUS_INVALID_ARGUMENT,
                "hyp_last_error requires a writable non-empty buffer");
        }
        std::snprintf(buffer, buffer_len, "%s", hyperion::abi::g_last_error.data());
        return HYP_STATUS_OK;
    } catch (...) {
        return fail(HYP_STATUS_INTERNAL, "hyp_last_error failed unexpectedly");
    }
}
