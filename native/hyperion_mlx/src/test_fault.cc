#include "test_fault.h"

#include <stdexcept>

namespace hyperion::model::test {
namespace {

thread_local bool post_append_forward_fault_armed = false;

} // namespace

void arm_post_append_forward_fault() noexcept {
    post_append_forward_fault_armed = true;
}

void maybe_throw_post_append_forward_fault() {
    if (!post_append_forward_fault_armed) {
        return;
    }
    post_append_forward_fault_armed = false;
    throw std::runtime_error("injected post-append forward failure");
}

} // namespace hyperion::model::test
