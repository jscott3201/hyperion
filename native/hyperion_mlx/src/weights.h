#pragma once

#include <optional>
#include <cstdint>

#include <mlx/mlx.h>

namespace hyperion::model {

/// One quantized linear (proj) layer: ``(w, scales, biases)`` in MLX affine Q4.
///
/// The weights are stored transposed as ``[out_features, in_features]`` (the
/// mlx-lm conversion layout); ``quantized_matmul(x, w, ..., transpose=true)``
/// computes ``x @ wᵀ`` = ``[M, in] @ [in, out] → [M, out]``. MLX's backend
/// dispatches the qmv (M=1, decode) vs qmm (M>1, prefill) kernel internally;
/// the explicit mlx-lm generation-aware dispatch (qmv_quad/splitk/wide +
/// ``get_qmv_batch_limit``) is an M4 perf refinement, not needed for correctness.
struct QuantizedLinear {
    mlx::core::array w;
    mlx::core::array scales;
    std::optional<mlx::core::array> biases;
    int group_size;
    int bits;

    /// ``x`` is ``[M, in_features]``; returns ``[M, out_features]`` = ``x @ wᵀ``.
    [[nodiscard]] mlx::core::array apply(const mlx::core::array& x, mlx::core::Stream stream) const {
        return mlx::core::quantized_matmul(
            x,
            w,
            scales,
            biases,
            /*transpose=*/true,
            std::optional<int>(group_size),
            std::optional<int>(bits),
            /*mode=*/"affine",
            stream);
    }
};

} // namespace hyperion::model
