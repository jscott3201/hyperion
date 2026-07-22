#include "rope.h"

#include <cmath>
#include <stdexcept>

namespace mx = mlx::core;

namespace hyperion::model {

namespace {

// The proportionally-rotated region's frequencies, mirroring mlx-lm's
// ProportionalRoPE: exponents = arange(0, rotated, 2) / dims (the FULL head_dim,
// not rotated), finite freqs for the rotated pairs and inf for the rest (MLX treats
// inf freqs as identity — empirically verified on 0.32.0).
mx::array proportional_freqs(std::uint32_t dims, std::uint32_t rotated, double base, mx::Stream s) {
    if (rotated > dims || rotated % 2 != 0 || dims % 2 != 0) {
        throw std::invalid_argument("proportional RoPE: rotated/dims must be even and rotated <= dims");
    }
    const auto d = static_cast<int>(dims);
    const auto r = static_cast<int>(rotated);
    // exponents over the rotated pairs only: arange(0, rotated, 2) / dims.
    mx::array exponents = mx::divide(
        mx::arange(0, r, 2, mx::float32, s),
        mx::array(static_cast<float>(d), mx::float32),
        s);
    mx::array finite = mx::power(mx::array(static_cast<float>(base), mx::float32), exponents, s);
    // inf fill for the un-rotated pairs: dims/2 - rotated/2 values.
    const int fill = (d - r) / 2;
    mx::array infs = mx::full({fill}, std::numeric_limits<float>::infinity(), mx::float32, s);
    return mx::concatenate(std::vector<mx::array>{finite, infs}, 0, s);
}

} // namespace

Rope::Rope(const RopeSpec& spec, std::uint32_t head_dim, mx::Stream stream)
    : stream_(stream), dims_(head_dim) {
    if (head_dim == 0 || head_dim % 2 != 0) {
        throw std::invalid_argument("Rope: head_dim must be a positive even number");
    }
    if (spec.proportional) {
        // Global "proportional" scheme: partial rotation. partial_rotary_factor is
        // required (Geometry::validate enforces it).
        if (!spec.partial_rotary_factor.has_value()) {
            throw std::invalid_argument("Rope: proportional scheme requires partial_rotary_factor");
        }
        const float prf = *spec.partial_rotary_factor;
        // rotated_dims = floor(prf * head_dim), rounded to even.
        std::uint32_t rotated = static_cast<std::uint32_t>(prf * static_cast<float>(head_dim));
        rotated -= rotated % 2;
        rotated_dims_ = rotated;
        base_ = std::nullopt;
        freqs_ = proportional_freqs(head_dim, rotated, spec.theta, stream_);
    } else {
        // Local default scheme: full rotation, MLX derives freqs from base = theta.
        rotated_dims_ = head_dim;
        base_ = std::optional<float>(static_cast<float>(spec.theta));
        freqs_ = std::nullopt;
    }
}

mx::array Rope::apply(const mx::array& x, std::uint32_t offset) const {
    // base_=theta + freqs_=nullopt => default full-rotation path (MLX derives freqs).
    // base_=nullopt + freqs_=proportional => the proportional path.
    return mx::fast::rope(
        x,
        /*dims=*/static_cast<int>(dims_),
        /*traditional=*/false,
        /*base=*/base_,
        /*scale=*/1.0F,
        /*offset=*/static_cast<int>(offset),
        /*freqs=*/freqs_,
        stream_);
}

} // namespace hyperion::model
