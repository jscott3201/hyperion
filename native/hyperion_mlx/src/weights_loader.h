#pragma once

#include "geometry.h"
#include "weights.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

namespace hyperion::model {

/// One decoder layer's loaded weights (dense 12B path). The quantized projections are
/// stored transposed ``[out, in]`` in the MLX affine layout (03 §Quantization); ``apply``
/// on ``QuantizedLinear`` computes ``x @ wᵀ``. ``v_proj`` is absent on global layers
/// (``attention_k_eq_v`` — K IS V), derived from the weight schema like
/// ``hyperion-model::WeightManifest::layer_kind``. The four layernorms are the Gemma 2
/// pre/post sandwich (04); ``q_norm``/``k_norm`` are scaled RMSNorm over ``head_dim``
/// while ``v_norm`` is parameterless (no tensor) — applied in ``forward.cc``.
struct LayerWeights {
    QuantizedLinear q_proj;
    QuantizedLinear k_proj;
    std::optional<QuantizedLinear> v_proj; ///< Absent on global (k_eq_v) layers.
    QuantizedLinear o_proj;
    QuantizedLinear gate_proj;
    QuantizedLinear up_proj;
    QuantizedLinear down_proj;

    mlx::core::array q_norm;       ///< [head_dim]
    mlx::core::array k_norm;       ///< [head_dim]
    mlx::core::array input_layernorm;            ///< [hidden]
    mlx::core::array post_attention_layernorm;  ///< [hidden]
    mlx::core::array pre_feedforward_layernorm;  ///< [hidden]
    mlx::core::array post_feedforward_layernorm; ///< [hidden]
    mlx::core::array layer_scalar;               ///< [] or [1]

    /// The attention kind derived from the weight schema (v_proj presence), parity-checked
    /// against ``Geometry::layer_types`` by the loader (a disagreement is a load error).
    LayerType kind;
};

/// All loaded weights for one model instance. The embedding table stays quantized; the
/// lookup gathers rows then dequantizes only the gathered rows (the production path for
/// the 262144×3840 12B embed — never dequantize the whole table residently).
struct ModelWeights {
    /// Quantized embedding (tied with the lm_head at 2.7). ``weight`` is packed
    /// ``[vocab, in]``, scales ``[vocab, groups]``, biases ``[vocab]``.
    ///@{
    mlx::core::array embed_weight;
    mlx::core::array embed_scales;
    mlx::core::array embed_biases;
    ///@}
    int embed_group_size{};
    int embed_bits{};

    mlx::core::array final_norm; ///< [hidden]
    std::vector<LayerWeights> layers;

    /// Embedding lookup: dequantize only the gathered rows of the tied table.
    /// ``ids`` is ``[L]`` int32; returns ``[L, hidden]`` bf16.
    [[nodiscard]] mlx::core::array embed(const mlx::core::array& ids, mlx::core::Stream stream) const;
};

/// Load all tensors from the safetensors shards in ``artifact_dir`` into a
/// ``ModelWeights``. Reads every ``model-*.safetensors`` in the directory (the
/// ``model.safetensors.index.json`` shard map is a Rust-side concern; the native loader
/// loads all shards and merges, since it has no JSON parser). Loads on the CPU stream
/// (MLX safetensors Load has no GPU kernel — the production CPU-load-then-GPU-compute
/// path); tensors transfer to the GPU on first use in the forward pass.
///
/// ``LayerType::kind`` per layer is derived from the weight schema (v_proj presence) and
/// parity-checked against ``geometry.layer_types`` — a mismatch is a load error, never a
/// coercion (mirrors ``hyperion-model::WeightManifest``).
///
/// Throws ``std::runtime_error`` on a missing/extra/mismatched tensor (the ABI layer
/// translates to ``HYP_STATUS_IO``/``HYP_STATUS_INTERNAL``).
[[nodiscard]] ModelWeights load_model_weights(
    const std::filesystem::path& artifact_dir,
    const Geometry& geometry,
    int group_size,
    int bits,
    mlx::core::Stream cpu_stream);

} // namespace hyperion::model
