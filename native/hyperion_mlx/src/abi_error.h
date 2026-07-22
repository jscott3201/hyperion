#pragma once

#include <array>
#include <cstddef>

namespace hyperion::abi {

/// Capacity of the thread-local last-error buffer (matches the Rust wrapper).
constexpr std::size_t kLastErrorCapacity = 1024;

/// Shared thread-local last-error storage; written by every ABI function that
/// returns a non-OK status, read by hyp_last_error. One buffer per thread.
extern thread_local std::array<char, kLastErrorCapacity> g_last_error;

/// Store a message (NUL-terminated, truncated) into the thread-local buffer.
void set_last_error(const char* message) noexcept;

/// Clear the thread-local buffer (called on every OK return).
void clear_last_error() noexcept;

} // namespace hyperion::abi
