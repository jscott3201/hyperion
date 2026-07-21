#include "hyperion_mlx.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "abi_contract_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    require(
        hyp_runtime_canary(nullptr) == HYP_STATUS_INVALID_ARGUMENT,
        "a null canary output must return INVALID_ARGUMENT");
    std::array<char, 256> error{};
    require(
        hyp_last_error(error.data(), error.size()) == HYP_STATUS_OK,
        "the thread-local diagnostic must be readable");
    require(
        std::string_view(error.data()).find("non-null output pointer") != std::string_view::npos,
        "the null-output diagnostic must survive the ABI boundary");

    char one_byte = 'x';
    require(
        hyp_last_error(&one_byte, 1) == HYP_STATUS_OK && one_byte == '\0',
        "a one-byte error buffer must remain NUL terminated");
    require(
        hyp_last_error(nullptr, 0) == HYP_STATUS_INVALID_ARGUMENT,
        "an invalid error buffer must return INVALID_ARGUMENT");
    return 0;
}
