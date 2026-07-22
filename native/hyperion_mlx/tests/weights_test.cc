// M5-gated smoke for the quantized weight-load + quantized_matmul wiring.
//
// Loads the REAL layer-0 q_proj (weight + scales + biases) from the M1-locked
// 12B artifact and checks mx::quantized_matmul against a dequantize + matmul
// reference on the same (w, scales, biases) triple. The most common real bug
// this catches is a quantized weight loaded in the wrong axis/transpose — the
// reference uses the same dequantize() MLX quantize() produces, so agreement
// proves the triple is in the shape quantized_matmul expects.
//
// Like hyperion_kv_cache / runtime_canary this is NOT in the --model-free ctest
// regex: it builds wherever MLX/Metal link, runs only on the self-hosted M5.
// It also needs the git-ignored ~6.7 GB artifact on disk; if
// HYPERION_12B_ARTIFACT is unset (CI), it self-skips (exit 0).

#include "weights.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

#include <mlx/mlx.h>

namespace mx = mlx::core;

using hyperion::model::QuantizedLinear;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "weights_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    const char* dir = std::getenv("HYPERION_12B_ARTIFACT");
    if (dir == nullptr || *dir == '\0') {
        std::cerr << "weights_test: HYPERION_12B_ARTIFACT unset; skipping "
                     "(M5-gated, needs the git-ignored 12B artifact)\n";
        return 0;
    }
    const std::string artifact(dir);

    // Lazily load both shards on the CPU (MLX's safetensors Load op has no GPU
    // kernel — weights load CPU-side, then transfer to the GPU for compute,
    // exactly the production path). mmap'd: the map is metadata, each array
    // materializes only on eval.
    const mx::Stream cpu = mx::default_stream(mx::Device::cpu);
    std::unordered_map<std::string, mx::array> tensors;
    for (const char* shard : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}) {
        auto loaded = mx::load_safetensors(artifact + "/" + shard, cpu);
        for (auto& [key, value] : loaded.first) {
            tensors.emplace(std::move(key), std::move(value));
        }
    }

    const std::string base = "language_model.model.layers.0.self_attn.q_proj";
    const auto wit = tensors.find(base + ".weight");
    const auto sit = tensors.find(base + ".scales");
    const auto bit = tensors.find(base + ".biases");
    require(
        wit != tensors.end() && sit != tensors.end() && bit != tensors.end(),
        "layer 0 q_proj {weight, scales, biases} present in the artifact");
    const mx::array w = wit->second;
    const mx::array scales = sit->second;
    const mx::array biases = bit->second;
    require(w.ndim() == 2, "q_proj weight is 2D (out x packed_in)");
    require(scales.ndim() == 2, "q_proj scales is 2D (out x groups)");

    const mx::Device gpu = mx::Device::gpu;
    const mx::Stream stream = mx::new_stream(gpu);
    // The packed weight is uint32 (8 values/word for 4-bit): its last dim is
    // in/8. The UNPACKED in_features is scales.shape(1) * group_size — that is
    // what quantized_matmul's x must match (hidden_size = 3840 on the 12B).
    constexpr int group_size = 64;
    constexpr int bits = 4;
    const int out_features = w.shape()[0];
    const int in_features = static_cast<int>(scales.shape()[1]) * group_size;
    std::cerr << "weights_test: q_proj weight " << w.shape()[0] << "x" << w.shape()[1]
              << " (packed), scales " << scales.shape()[0] << "x" << scales.shape()[1]
              << " -> out " << out_features << " x in " << in_features
              << " (affine Q4 g" << group_size << "/b" << bits << ")\n";

    const QuantizedLinear lin{
        w,
        scales,
        std::optional<mx::array>(biases),
        group_size,
        bits,
    };

    // Materialize the loaded CPU tensors (they are lazy until eval).
    mx::eval(w);
    mx::eval(scales);
    mx::eval(biases);

    // x: [M=4, in_features] f32 on the GPU — the matmuls below run on the GPU;
    // MLX transfers the CPU weights to the GPU (production load path).
    const int M = 4;
    mx::array x = mx::multiply(
        mx::ones({M, in_features}, mx::float32, stream),
        mx::array(0.1F, mx::float32),
        stream);

    // Reference: dequantize the same triple to f32, then x @ wᵀ (transpose=true).
    mx::array w_dequant = mx::dequantize(
        w,
        scales,
        std::optional<mx::array>(biases),
        std::optional<int>(group_size),
        std::optional<int>(bits),
        /*mode=*/"affine",
        std::nullopt,
        std::optional<mx::Dtype>(mx::float32),
        stream);
    mx::array ref = mx::matmul(x, mx::transpose(w_dequant, stream), stream); // [M, out]
    mx::array got = lin.apply(x, stream);                                     // [M, out]

    mx::eval(ref);
    mx::eval(got);
    mx::synchronize(stream);

    // max |got - ref| — quantized_matmul vs dequant+matmul differ only by
    // accumulation-order rounding, so this should be small.
    mx::array max_diff = mx::max(mx::abs(mx::subtract(got, ref, stream), stream), stream);
    mx::eval(max_diff);
    mx::synchronize(stream);
    std::cerr << "weights_test: max |got - ref| = " << max_diff.item<float>() << '\n';

    mx::array ok = mx::allclose(got, ref, /*rtol=*/1e-3, /*atol=*/1e-2, false, stream);
    mx::eval(ok);
    mx::synchronize(stream);
    require(
        ok.item<bool>(),
        "quantized_matmul matches dequantize+matmul reference within tolerance");
    return 0;
}
