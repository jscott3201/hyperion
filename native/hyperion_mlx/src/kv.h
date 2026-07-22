#pragma once

#include <cstdint>

namespace hyperion::model {

/// Index arithmetic for one sliding-layer KV ring (A2/A3).
///
/// Capacity is ``window + gamma_max``. At any time the ring holds the last
/// ``window`` committed tokens plus up to ``gamma_max`` speculative (uncommitted)
/// tokens, so it never overwrites a still-needed token: writing at sequence
/// position ``committed_len + speculative_len`` overwrites the oldest
/// out-of-window token (safe — sliding attention reads only the last ``window``).
///
/// Speculative writes (MTP draft / verify, M7) live in the gamma slack and are
/// discarded by *not committing* — never by trimming committed state (A2). The
/// attention length excludes the speculative region (A3): a verify pass reads
/// only committed KV.
class LocalRingIndex {
  public:
    /// ``window`` is the sliding attention window; ``gamma_max`` the speculative slack.
    LocalRingIndex(std::uint32_t window, std::uint32_t gamma_max);

    /// Total ring capacity (``window + gamma_max``); flat regardless of context.
    [[nodiscard]] std::uint32_t capacity() const;

    /// Sequence position of the next write (== committed + speculative).
    [[nodiscard]] std::uint32_t next_write_pos() const;

    /// The ring slot for a sequence position (``pos % capacity``).
    [[nodiscard]] std::uint32_t slot_for(std::uint32_t sequence_pos) const;

    /// Tokens that are committed (cannot be rolled back).
    [[nodiscard]] std::uint32_t committed_len() const;

    /// Tokens currently in the speculative slack (0..gamma_max).
    [[nodiscard]] std::uint32_t speculative_len() const;

    /// What sliding attention reads: committed tokens only (excludes speculative).
    [[nodiscard]] std::uint32_t attention_len() const;

    /// Reserve ``n`` speculative slots (MTP draft). Returns false if it would
    /// exceed ``gamma_max`` — the caller must commit or discard first.
    [[nodiscard]] bool append_speculative(std::uint32_t n);

    /// Promote ``n`` speculative tokens to committed (decode accept / MTP accept).
    /// If ``n`` exceeds speculative_len, the remainder comes from fresh appends
    /// (prefill path: commit directly past speculative).
    void commit(std::uint32_t n);

    /// Discard all speculative tokens (MTP reject) — committed state is untouched.
    void discard_speculative();

  private:
    std::uint32_t window_;
    std::uint32_t gamma_max_;
    std::uint32_t committed_len_;
    std::uint32_t speculative_len_;
};

/// Index arithmetic for one global-layer capacity-stepped KV cache (K=V).
///
/// Capacity grows in ``step``-token increments (default 256, the bucket size) at
/// bucket boundaries only, never mid-token. M2 appends committed tokens only
/// (no speculative slack here — the MTP rollback slack lives in the local rings,
/// A2). M7 extends this for verify-pass discard.
class GlobalCacheIndex {
  public:
    explicit GlobalCacheIndex(std::uint32_t step = 256);

    /// Current allocated capacity (a multiple of ``step``).
    [[nodiscard]] std::uint32_t capacity() const;

    /// Tokens committed to the global cache.
    [[nodiscard]] std::uint32_t committed_len() const;

    /// Append ``n`` committed tokens, growing capacity by whole steps as needed.
    void append(std::uint32_t n);

    /// Number of capacity steps currently allocated.
    [[nodiscard]] std::uint32_t step_count() const;

  private:
    std::uint32_t step_;
    std::uint32_t committed_len_;
    std::uint32_t step_count_;
};

} // namespace hyperion::model
