#include "abi_error.h"

#include <cstdio>

namespace hyperion::abi {

thread_local std::array<char, kLastErrorCapacity> g_last_error{};

void set_last_error(const char* message) noexcept {
    std::snprintf(
        g_last_error.data(),
        g_last_error.size(),
        "%s",
        message != nullptr ? message : "unknown native error");
}

void clear_last_error() noexcept {
    g_last_error[0] = '\0';
}

} // namespace hyperion::abi
