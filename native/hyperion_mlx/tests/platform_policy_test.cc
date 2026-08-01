#include "platform_policy.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <variant>

using hyperion::platform::Version;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "platform_policy_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--negative-control") {
        require(false, "intentional negative control");
    }
    require(argc == 1, "unexpected command-line arguments");

    constexpr std::uint64_t recommended = 12'713'115'648ULL;
    const auto accepted = hyperion::platform::evaluate(
        true,
        1010,
        Version{26, 2, 0},
        Version{0, 32, 0},
        "0.32.0",
        recommended);
    require(accepted.supported, "the exact M5 floor should be accepted");
    require(
        accepted.budget.effective_bytes == 12'064'746'749ULL,
        "the device-derived effective budget changed");
    require(
        accepted.budget.soft_watermark_bytes == 10'858'272'074ULL,
        "the device-derived soft watermark changed");

    const auto capped = hyperion::platform::derive_budget(16ULL * 1024ULL * 1024ULL * 1024ULL);
    require(
        capped.effective_bytes == hyperion::platform::kProfileCeilingBytes,
        "the 16 GB profile must retain its 12 GiB hard ceiling");
    require(
        capped.soft_watermark_bytes == 11'596'411'699ULL,
        "the capped soft watermark must be floor(effective * 0.9)");

    hyperion::platform::DeviceInfo valid_device_info{
        {"max_recommended_working_set_size", static_cast<std::size_t>(recommended)},
    };
    const auto device_budget =
        hyperion::platform::derive_device_budget(valid_device_info);
    require(device_budget.supported,
        "a size_t device working-set recommendation must be accepted");
    require(device_budget.budget.effective_bytes == accepted.budget.effective_bytes,
        "device-info extraction must feed the existing budget derivation");

    const auto missing_device_budget =
        hyperion::platform::derive_device_budget({});
    const auto wrong_type_device_budget =
        hyperion::platform::derive_device_budget({
            {"max_recommended_working_set_size", std::string("12713115648")},
        });
    const auto zero_device_budget =
        hyperion::platform::derive_device_budget({
            {"max_recommended_working_set_size", std::size_t{0}},
        });
    require(
        !missing_device_budget.supported,
        "missing device recommendation must fail closed");
    require(
        !wrong_type_device_budget.supported,
        "wrong-typed device recommendation must fail closed");
    require(
        !zero_device_budget.supported,
        "zero device recommendation must fail closed");
    require(missing_device_budget.reason == wrong_type_device_budget.reason &&
            missing_device_budget.reason == zero_device_budget.reason,
        "invalid device recommendation data must use one generic rejection reason");

    const auto clean_runtime_environment =
        hyperion::platform::evaluate_runtime_environment(false);
    const auto overridden_runtime_environment =
        hyperion::platform::evaluate_runtime_environment(true);
    require(clean_runtime_environment.supported,
        "an unset MLX_SDPA_BLOCKS environment must be supported");
    require(!overridden_runtime_environment.supported &&
            std::string_view(overridden_runtime_environment.reason).find("MLX_SDPA_BLOCKS") !=
                std::string_view::npos,
        "an inherited MLX_SDPA_BLOCKS override must fail closed with a specific reason");

    require(
        !hyperion::platform::evaluate(
             true, 1009, Version{26, 6, 0}, Version{0, 32, 0}, "0.32.0", recommended)
             .supported,
        "Apple9/M4 must be rejected");
    require(
        !hyperion::platform::evaluate(
             true, 1010, Version{26, 1, 9}, Version{0, 32, 0}, "0.32.0", recommended)
             .supported,
        "macOS below 26.2 must be rejected");
    require(
        !hyperion::platform::evaluate(
             true, 1010, Version{26, 6, 0}, Version{0, 31, 3}, "0.32.0", recommended)
             .supported,
        "an MLX header-version mismatch must be rejected");
    require(
        !hyperion::platform::evaluate(
             true, 1010, Version{26, 6, 0}, Version{0, 32, 0}, "0.32.1", recommended)
             .supported,
        "an MLX dylib-version mismatch must be rejected");
    require(
        !hyperion::platform::evaluate(
             false, 0, Version{26, 6, 0}, Version{0, 32, 0}, "0.32.0", recommended)
             .supported,
        "a missing Metal device must be rejected");
    require(
        !hyperion::platform::evaluate(
             true, 1010, Version{26, 6, 0}, Version{0, 32, 0}, "0.32.0", 0)
             .supported,
        "a zero recommended working set must be rejected");
    return 0;
}
