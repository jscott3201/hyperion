#include "hyperion_mlx.h"
#include "abi_error.h"
#include "dispatch.h"
#include "forward.h"
#include "geometry.h"
#include "kv_cache.h"
#include "weights_loader.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include <mlx/mlx.h>

namespace mx = mlx::core;

// Opaque handle bodies. The C header forward-declares these as incomplete
// `struct HypModelOpaque*` in the global namespace, so the definitions live at
// global scope to match the tag (not inside hyperion::model).
struct HypModelOpaque {
    std::uint32_t magic;
    bool loaded = false;
    std::unique_ptr<hyperion::model::Geometry> geometry;
    std::unique_ptr<hyperion::model::DispatchTable> dispatch;
    std::unique_ptr<hyperion::model::ModelWeights> weights;
    std::optional<mx::Stream> stream;
};

struct HypKvStateOpaque {
    std::uint32_t magic;
    std::unique_ptr<hyperion::model::KvState> kv;
    std::optional<mx::Stream> stream;
};

struct HypStepResultOpaque {
    std::uint32_t magic;
    HypStepResultFields fields;
};

namespace hyperion::model {
namespace {

// Magic tags distinguish handle kinds and detect use-after-free (a freed
// handle's magic is cleared before delete; a dangling non-nulled pointer would
// fail the magic check). Distinct 32-bit values per kind.
constexpr std::uint32_t kModelMagic = 0x48504D4FU;        // "HPMO"
constexpr std::uint32_t kKvStateMagic = 0x48504B56U;      // "HPKV"
constexpr std::uint32_t kStepResultMagic = 0x48505352U;  // "HPSR"

// The M1-locked Gemma 4 12B quantization (hyperion-model::weights::LOCKED_*).
// The native loader has no JSON parser, so the locked group/bits are constants
// here; the Rust side re-derives them from config.json and parity-checks.
constexpr int kLockedGroupSize = 64;
constexpr int kLockedBits = 4;

} // namespace

HypStatus fail(HypStatus status, const char* message) noexcept {
    hyperion::abi::set_last_error(message);
    return status;
}

HypStatus ok() noexcept {
    hyperion::abi::clear_last_error();
    return HYP_STATUS_OK;
}

HypStatus validate_model(HypModel model, const char* /*operation*/) noexcept {
    if (model == nullptr) {
        return fail(HYP_STATUS_INVALID_ARGUMENT, "model handle is null");
    }
    if (model->magic != kModelMagic) {
        return fail(
            HYP_STATUS_INVALID_ARGUMENT,
            "model handle magic mismatch (wrong type, freed, or corrupted)");
    }
    return ok();
}

HypStatus validate_kvstate(HypKvState kvstate) noexcept {
    if (kvstate == nullptr || kvstate->magic != kKvStateMagic) {
        return fail(HYP_STATUS_INVALID_ARGUMENT, "kvstate handle is invalid");
    }
    return ok();
}

HypStatus validate_step_result(HypStepResult result) noexcept {
    if (result == nullptr || result->magic != kStepResultMagic) {
        return fail(HYP_STATUS_INVALID_ARGUMENT, "step-result handle is invalid");
    }
    return ok();
}

template <typename Handle>
HypStatus create_handle(Handle** out, std::uint32_t magic) noexcept {
    if (out == nullptr) {
        return fail(HYP_STATUS_INVALID_ARGUMENT, "create requires a non-null out handle");
    }
    try {
        auto* handle = new Handle{};
        handle->magic = magic;
        *out = handle;
        return ok();
    } catch (const std::bad_alloc&) {
        return fail(
            HYP_STATUS_INTERNAL,
            "native allocation failed outside predictive governor admission");
    } catch (...) {
        return fail(HYP_STATUS_INTERNAL, "unexpected exception in handle create");
    }
}

template <typename Handle>
HypStatus free_handle(Handle** handle, std::uint32_t magic) noexcept {
    if (handle == nullptr) {
        return fail(HYP_STATUS_INVALID_ARGUMENT, "free requires a non-null handle pointer");
    }
    Handle* ptr = *handle;
    if (ptr == nullptr) {
        return ok(); // idempotent: freeing a null handle is a no-op
    }
    if (ptr->magic != magic) {
        return fail(
            HYP_STATUS_INVALID_ARGUMENT,
            "handle magic mismatch (wrong type, freed, or corrupted)");
    }
    ptr->magic = 0; // invalidate; a dangling pointer would fail the magic check
    delete ptr;
    *handle = nullptr;
    return ok();
}

} // namespace hyperion::model

extern "C" {

HypStatus hyp_model_create(HypModel* out_model) {
    return hyperion::model::create_handle(out_model, hyperion::model::kModelMagic);
}

HypStatus hyp_model_free(HypModel* model) {
    return hyperion::model::free_handle(model, hyperion::model::kModelMagic);
}

HypStatus hyp_model_load(HypModel model,
                         const HypGeometryParams* geometry,
                         const char* weights_path) {
    auto status = hyperion::model::validate_model(model, "hyp_model_load");
    if (status != HYP_STATUS_OK) {
        return status;
    }
    if (geometry == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "geometry is null");
    }
    if (weights_path == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "weights_path is null");
    }
    if (model->loaded) {
        return hyperion::model::fail(
            HYP_STATUS_INVALID_ARGUMENT, "model is already loaded");
    }

    // Build + validate geometry from the ABI struct (a second line of defense —
    // Rust already validated, but never trust the caller across the ABI).
    hyperion::model::Geometry geo;
    if (auto err = hyperion::model::Geometry::from_abi(*geometry, geo); err.has_value()) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, err->c_str());
    }
    if (auto err = geo.validate(); err.has_value()) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, err->c_str());
    }

    const std::filesystem::path artifact(weights_path);
    if (!std::filesystem::is_directory(artifact)) {
        return hyperion::model::fail(HYP_STATUS_NOT_FOUND, "weights artifact directory not found");
    }

    try {
        auto dispatch = std::make_unique<hyperion::model::DispatchTable>(
            hyperion::model::build_dispatch(geo));
        // CPU stream for the safetensors Load (no GPU kernel); GPU stream for compute.
        const mx::Stream cpu = mx::default_stream(mx::Device::cpu);
        const mx::Stream gpu = mx::new_stream(mx::Device::gpu);
        auto weights = std::make_unique<hyperion::model::ModelWeights>(
            hyperion::model::load_model_weights(artifact, geo, hyperion::model::kLockedGroupSize, hyperion::model::kLockedBits, cpu));
        model->geometry = std::make_unique<hyperion::model::Geometry>(std::move(geo));
        model->dispatch = std::move(dispatch);
        model->weights = std::move(weights);
        model->stream = gpu;
        model->loaded = true;
        return hyperion::model::ok();
    } catch (const std::filesystem::filesystem_error& error) {
        return hyperion::model::fail(HYP_STATUS_IO, error.what());
    } catch (const std::bad_alloc&) {
        return hyperion::model::fail(
            HYP_STATUS_INTERNAL,
            "native allocation failed outside predictive governor admission");
    } catch (const std::exception& error) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, error.what());
    } catch (...) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, "hyp_model_load failed with a non-standard exception");
    }
}

HypStatus hyp_kvstate_create(HypModel model, HypKvState* out_kvstate) {
    auto status = hyperion::model::validate_model(model, "hyp_kvstate_create");
    if (status != HYP_STATUS_OK) {
        return status;
    }
    if (out_kvstate == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "out_kvstate is null");
    }
    if (!model->loaded) {
        return hyperion::model::fail(
            HYP_STATUS_INVALID_ARGUMENT, "model is not loaded; cannot allocate KV state");
    }
    try {
        // Real heterogeneous KV state, allocated from the loaded model's dispatch
        // table — one LocalKvCache per sliding layer, one GlobalKvCache per global
        // layer, in layer order. Ready for the 2.6 prefill read.
        auto kv = std::make_unique<hyperion::model::KvState>(
            hyperion::model::build_kv_state(*model->dispatch, hyperion::model::kDefaultGammaMax, mx::bfloat16, *model->stream));
        auto* handle = new HypKvStateOpaque{};
        handle->magic = hyperion::model::kKvStateMagic;
        handle->kv = std::move(kv);
        handle->stream = model->stream;
        handle->stream = model->stream;
        *out_kvstate = handle;
        return hyperion::model::ok();
    } catch (const std::bad_alloc&) {
        return hyperion::model::fail(
            HYP_STATUS_INTERNAL,
            "native allocation failed outside predictive governor admission");
    } catch (const std::exception& error) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, error.what());
    } catch (...) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, "hyp_kvstate_create failed with a non-standard exception");
    }
}

HypStatus hyp_kvstate_free(HypKvState* kvstate) {
    return hyperion::model::free_handle(kvstate, hyperion::model::kKvStateMagic);
}

HypStatus hyp_prefill_chunk(HypModel model,
                            HypKvState kvstate,
                            const HypTokenStream* tokens,
                            HypStepResult out_result) {
    auto status = hyperion::model::validate_model(model, "hyp_prefill_chunk");
    if (status != HYP_STATUS_OK) {
        return status;
    }
    status = hyperion::model::validate_kvstate(kvstate);
    if (status != HYP_STATUS_OK) {
        return status;
    }
    if (tokens == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "tokens is null");
    }
    status = hyperion::model::validate_step_result(out_result);
    if (status != HYP_STATUS_OK) {
        return status;
    }
    return hyperion::model::fail(
        HYP_STATUS_UNSUPPORTED,
        "hyp_prefill_chunk is not implemented until M2-2.6 (chunked prefill)");
}

HypStatus hyp_decode_block(HypModel model,
                            HypKvState kvstate,
                            uint32_t /*n_tokens*/,
                            HypStepResult out_result) {
    auto status = hyperion::model::validate_model(model, "hyp_decode_block");
    if (status != HYP_STATUS_OK) {
        return status;
    }
    status = hyperion::model::validate_kvstate(kvstate);
    if (status != HYP_STATUS_OK) {
        return status;
    }
    status = hyperion::model::validate_step_result(out_result);
    if (status != HYP_STATUS_OK) {
        return status;
    }
    return hyperion::model::fail(
        HYP_STATUS_UNSUPPORTED,
        "hyp_decode_block is not implemented until M2-2.7 (decode + greedy sampler)");
}

HypStatus hyp_step_result_create(HypStepResult* out_result) {
    return hyperion::model::create_handle(out_result, hyperion::model::kStepResultMagic);
}

HypStatus hyp_step_result_free(HypStepResult* result) {
    return hyperion::model::free_handle(result, hyperion::model::kStepResultMagic);
}

} // extern "C"
