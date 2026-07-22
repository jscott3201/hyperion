#include "hyperion_mlx.h"
#include "abi_error.h"

#include <cstdint>
#include <new>

// Opaque handle bodies. The C header forward-declares these as incomplete
// `struct HypModelOpaque*` in the global namespace, so the definitions live at
// global scope to match the tag (not inside hyperion::model).
struct HypModelOpaque {
    std::uint32_t magic;
    bool loaded;
};

struct HypKvStateOpaque {
    std::uint32_t magic;
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
    return hyperion::model::fail(
        HYP_STATUS_UNSUPPORTED,
        "hyp_model_load is not implemented until M2-2.2 (weight load + graph build)");
}

HypStatus hyp_kvstate_create(HypModel model, HypKvState* out_kvstate) {
    auto status = hyperion::model::validate_model(model, "hyp_kvstate_create");
    if (status != HYP_STATUS_OK) {
        return status;
    }
    if (out_kvstate == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "out_kvstate is null");
    }
    return hyperion::model::create_handle(out_kvstate, hyperion::model::kKvStateMagic);
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
        "hyp_decode_block is not implemented until M2-2.6/2.7 (decode + sampler)");
}

HypStatus hyp_step_result_create(HypStepResult* out_result) {
    return hyperion::model::create_handle(out_result, hyperion::model::kStepResultMagic);
}

HypStatus hyp_step_result_free(HypStepResult* result) {
    return hyperion::model::free_handle(result, hyperion::model::kStepResultMagic);
}

} // extern "C"
