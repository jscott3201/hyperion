#include "hyperion_mlx.h"

#include <array>
#include <cassert>
#include <string_view>

int main() {
    assert(hyp_runtime_canary(nullptr) == HYP_STATUS_INVALID_ARGUMENT);
    std::array<char, 256> error{};
    assert(hyp_last_error(error.data(), error.size()) == HYP_STATUS_OK);
    assert(std::string_view(error.data()).find("non-null output pointer") != std::string_view::npos);
    assert(hyp_last_error(nullptr, 0) == HYP_STATUS_INVALID_ARGUMENT);
    return 0;
}
