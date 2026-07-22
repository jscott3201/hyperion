#pragma once

#include "geometry.h"

#include <cstdint>
#include <optional>

#include <mlx/mlx.h>

namespace hyperion::model {

/// The two RoPE schemes Gemma 4 uses, resolved once from ``Geometry`` (04).
///
/// * **Local (sliding):** full-rotation, default scheme, ``theta = 1e4`` — every
///   ``head_dim`` (256 on the 12B) element is rotated.
/// * **Global (full):** the "proportional" scheme — ``theta = 1e6``, partial rotation of
///   ``partial_rotary_factor * head_dim`` (128 of 512 on the 12B) dims. The un-rotated
///   region is left as identity by passing ``inf`` frequencies for those pairs to
///   ``mx::fast::rope`` (empirically verified on MLX 0.32.0: ``inf`` freqs yield finite,
///   identity output for those dims). This is NOT substitutable by ``nn.RoPE(rotated)``
///   because the proportional exponents are ``arange(0, rotated, 2) / dims`` (divided by
///   the FULL head_dim, not the rotated count) — a different frequency distribution.
class Rope {
  public:
    /// Build the RoPE for one attention kind from a ``RopeSpec`` + the layer's head_dim.
    /// ``head_dim`` is the full per-head dimension (256 local / 512 global on the 12B).
    Rope(const RopeSpec& spec, std::uint32_t head_dim, mlx::core::Stream stream);

    /// Apply RoPE to ``x`` (shape ``[..., n_heads, head_dim]``) at the given absolute
    /// ``offset`` (the cache length before this chunk). Rotates the first ``rotated_dims``
    /// pairs; the remainder is identity. Mirrors ``mlx-lm``'s ``ProportionalRoPE``/``nn.RoPE``
    /// ``__call__(x, offset)``.
    [[nodiscard]] mlx::core::array apply(const mlx::core::array& x, std::uint32_t offset) const;

    [[nodiscard]] std::uint32_t dims() const { return dims_; }
    [[nodiscard]] std::uint32_t rotated_dims() const { return rotated_dims_; }

  private:
    mlx::core::Stream stream_;
    std::uint32_t dims_;          // full head_dim
    std::uint32_t rotated_dims_;  // rotated pairs * 2
    // Exactly one is set per scheme: base_ for the default (full-rotation) scheme — MLX
    // derives freqs = base^(arange/dims); freqs_ for the proportional scheme (finite
    // freqs for the rotated pairs, inf for the rest → identity).
    std::optional<float> base_;
    std::optional<mlx::core::array> freqs_;
};

} // namespace hyperion::model
