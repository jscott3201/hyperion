#pragma once

namespace hyperion::model::test {

// Source-private, test-build-only one-shot fault at ForwardPass::forward's
// post-layer-append / pre-final-norm seam.
void arm_post_append_forward_fault() noexcept;
void maybe_throw_post_append_forward_fault();

} // namespace hyperion::model::test
