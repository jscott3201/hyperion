#include "kv.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

using hyperion::model::GlobalCacheIndex;
using hyperion::model::LocalRingIndex;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "kv_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    // --- LocalRingIndex (12B: window 1024, gamma_max 8) ---
    LocalRingIndex ring(1024, 8);
    require(ring.capacity() == 1032, "capacity == window + gamma_max");
    require(ring.committed_len() == 0, "starts empty");
    require(ring.attention_len() == 0, "attention starts at 0");
    require(ring.slot_for(0) == 0, "slot 0 at pos 0");
    require(ring.slot_for(1032) == 0, "slot wraps at capacity");
    require(ring.slot_for(1033) == 1, "slot wraps + 1");

    // Prefill: commit a chunk directly (no speculative).
    ring.commit(2048);
    require(ring.committed_len() == 2048, "prefill commit advances committed");
    require(ring.attention_len() == 2048, "attention tracks committed");

    // The ring holds only `capacity` slots; committing past capacity wraps.
    // slot_for(2048) == 2048 % 1032 == 16.
    require(ring.slot_for(2048) == 2048 % 1032, "slot wraps modulo capacity");

    // MTP draft: append speculative tokens into the gamma slack.
    require(ring.append_speculative(4), "append 4 speculative within gamma_max");
    require(ring.speculative_len() == 4, "speculative tracked");
    require(ring.attention_len() == 2048, "attention EXCLUDES speculative (A3)");
    require(ring.next_write_pos() == 2052, "next write past speculative");

    // Reject overflow of the gamma slack.
    require(!ring.append_speculative(8), "overflow speculative is rejected (commit/discard first)");

    // Accept (commit) 2 of the 4 speculative — A2 append-only (promote, don't trim).
    ring.commit(2);
    require(ring.committed_len() == 2050, "commit promotes speculative to committed");
    require(ring.speculative_len() == 2, "remaining speculative after partial commit");
    require(ring.attention_len() == 2050, "attention tracks the new committed");

    // Reject the remaining speculative — A2 rollback by not committing.
    ring.discard_speculative();
    require(ring.committed_len() == 2050, "discard leaves committed untouched (A2)");
    require(ring.speculative_len() == 0, "discard clears speculative");
    require(ring.attention_len() == 2050, "attention unchanged after discard");

    // After discard, the slack is reusable.
    require(ring.append_speculative(8), "slack reusable after discard (up to gamma_max)");

    // --- GlobalCacheIndex (capacity-stepped, default step 256) ---
    GlobalCacheIndex global(256);
    require(global.capacity() == 0, "global starts unallocated");
    require(global.step_count() == 0, "zero steps initially");

    global.append(256);
    require(global.committed_len() == 256, "global append advances committed");
    require(global.capacity() == 256, "one step allocated at 256");
    require(global.step_count() == 1, "step count 1 at 256");

    // Growth happens at bucket boundaries only — never mid-write past capacity.
    global.append(1);
    require(global.committed_len() == 257, "global +1 committed");
    require(global.capacity() == 512, "grew by one whole step");
    require(global.step_count() == 2, "step count 2 at 257");

    // A large append grows by multiple whole steps.
    global.append(1024);
    require(global.committed_len() == 1281, "global +1024 committed");
    require(global.capacity() >= 1281, "capacity covers committed");
    require(global.capacity() % 256 == 0, "capacity stays a multiple of step");
    require((global.capacity() / 256) == global.step_count(), "step count consistent");

    // The growth invariant: capacity >= committed, always a whole number of steps.
    require(global.capacity() == 256 * ((1281 + 255) / 256), "capacity is the ceiling multiple");

    return 0;
}
