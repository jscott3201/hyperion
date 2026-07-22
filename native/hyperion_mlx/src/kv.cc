#include "kv.h"

#include <stdexcept>

namespace hyperion::model {

LocalRingIndex::LocalRingIndex(std::uint32_t window, std::uint32_t gamma_max)
    : window_(window),
      gamma_max_(gamma_max),
      committed_len_(0),
      speculative_len_(0) {
    if (window == 0) {
        throw std::invalid_argument("LocalRingIndex window must be non-zero");
    }
    // gamma_max may be zero for the non-speculative decode path; the ring is
    // still valid (capacity == window).
}

std::uint32_t LocalRingIndex::capacity() const {
    return window_ + gamma_max_;
}

std::uint32_t LocalRingIndex::next_write_pos() const {
    return committed_len_ + speculative_len_;
}

std::uint32_t LocalRingIndex::slot_for(std::uint32_t sequence_pos) const {
    return sequence_pos % capacity();
}

std::uint32_t LocalRingIndex::committed_len() const {
    return committed_len_;
}

std::uint32_t LocalRingIndex::speculative_len() const {
    return speculative_len_;
}

std::uint32_t LocalRingIndex::attention_len() const {
    return committed_len_;
}

bool LocalRingIndex::append_speculative(std::uint32_t n) {
    if (speculative_len_ + n > gamma_max_) {
        return false; // would overflow the slack; caller must commit/discard first
    }
    speculative_len_ += n;
    return true;
}

void LocalRingIndex::commit(std::uint32_t n) {
    if (n <= speculative_len_) {
        // Promote speculative tokens to committed.
        speculative_len_ -= n;
        committed_len_ += n;
        return;
    }
    // Commit all speculative, then the remainder as fresh committed appends
    // (the prefill path commits directly past the slack).
    const std::uint32_t fresh = n - speculative_len_;
    committed_len_ += speculative_len_ + fresh;
    speculative_len_ = 0;
}

void LocalRingIndex::discard_speculative() {
    speculative_len_ = 0; // A2: rollback by not committing; committed untouched
}

GlobalCacheIndex::GlobalCacheIndex(std::uint32_t step)
    : step_(step), committed_len_(0), step_count_(0) {
    if (step == 0) {
        throw std::invalid_argument("GlobalCacheIndex step must be non-zero");
    }
}

std::uint32_t GlobalCacheIndex::capacity() const {
    return step_ * step_count_;
}

std::uint32_t GlobalCacheIndex::committed_len() const {
    return committed_len_;
}

std::uint32_t GlobalCacheIndex::step_count() const {
    return step_count_;
}

void GlobalCacheIndex::append(std::uint32_t n) {
    committed_len_ += n;
    // Grow by whole steps at bucket boundaries only — never mid-write.
    while (committed_len_ > capacity()) {
        ++step_count_;
    }
}

} // namespace hyperion::model
