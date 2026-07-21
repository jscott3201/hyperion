#include "platform_policy.h"

#include <cassert>
#include <cstdint>

using hyperion::platform::Version;

int main() {
    constexpr std::uint64_t recommended = 12'713'115'648ULL;
    const auto accepted = hyperion::platform::evaluate(
        true,
        1010,
        Version{26, 2, 0},
        Version{0, 32, 0},
        "0.32.0",
        recommended);
    assert(accepted.supported);
    assert(accepted.budget.effective_bytes == 12'064'746'749ULL);
    assert(accepted.budget.soft_watermark_bytes == 10'858'272'074ULL);

    assert(!hyperion::platform::evaluate(
                true, 1009, Version{26, 6, 0}, Version{0, 32, 0}, "0.32.0", recommended)
                .supported);
    assert(!hyperion::platform::evaluate(
                true, 1010, Version{26, 1, 9}, Version{0, 32, 0}, "0.32.0", recommended)
                .supported);
    assert(!hyperion::platform::evaluate(
                true, 1010, Version{26, 6, 0}, Version{0, 31, 3}, "0.32.0", recommended)
                .supported);
    assert(!hyperion::platform::evaluate(
                true, 1010, Version{26, 6, 0}, Version{0, 32, 0}, "0.32.1", recommended)
                .supported);
    assert(!hyperion::platform::evaluate(
                false, 0, Version{26, 6, 0}, Version{0, 32, 0}, "0.32.0", recommended)
                .supported);
    return 0;
}
