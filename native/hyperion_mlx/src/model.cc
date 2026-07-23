#include "hyperion_mlx.h"
#include "abi_error.h"
#include "dispatch.h"
#include "forward.h"
#include "geometry.h"
#include "kv_cache.h"
#include "step_result_access.h"
#include "weights_loader.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

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
    std::unique_ptr<hyperion::model::ForwardPass> fwd; // M2-2.7: the per-layer math + epilogue.
    std::optional<mx::Stream> stream;
};

struct HypKvStateOpaque {
    std::uint32_t magic;
    std::unique_ptr<hyperion::model::KvState> kv;
    std::optional<mx::Stream> stream;
    // M2-2.7 autoregressive cursor: the offset = committed prefix length (passed
    // to forward() as the attention offset), and last_token = the id to embed for
    // the next decode step (set by prefill/decode from the sampler). Per-kvstate.
    std::uint32_t offset = 0;
    std::uint32_t last_token = 0;
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

// ── M2-2.7 generation epilogue + step-result plumbing ───────────────────────
// prefill_chunk/decode_block share: final-norm hidden (forward already applied
// final_norm) → tied lm_head → softcap → last-position slice → greedy argmax.

/// Run the lm_head + softcap + greedy sampler on the final-norm hidden state's
/// LAST position. ``h`` is ``[B, L, hidden]`` (already final-norm'd by forward()).
ForwardPass::GreedySample run_epilogue(ForwardPass& fwd, const mx::array& h, mx::Stream s) {
    const int L = static_cast<int>(h.shape(1));
    mx::array logits = fwd.lm_head(h);                 // [B, L, vocab]
    logits = fwd.softcap(logits);                      // [B, L, vocab]
    // last position → [1, 1, vocab] (a contiguous slice of the final row → sample_greedy
    // reads vocab contiguous floats; no reshape needed).
    mx::array last = mx::slice(
        logits,
        {0, L - 1, 0},
        {1, L, static_cast<int>(logits.shape(2))},
        {1, 1, 1},
        s);
    return fwd.sample_greedy(last);
}

/// Write the sampled token + telemetry to a validated step-result handle. Memory
/// fields are 0 (governor admission is 2.6); governor_state = READY.
void write_step_result(HypStepResult result, const ForwardPass::GreedySample& sample, std::uint32_t near_tie_events) {
    result->fields.token_id = sample.token_id;
    result->fields.logit = sample.logit;
    result->fields.near_tie_events = near_tie_events;
    result->fields.governor_state = HYP_GOVERNOR_READY;
    result->fields.peak_mlx_bytes = 0;        // 2.6 governor
    result->fields.active_mlx_bytes = 0;
    result->fields.phys_footprint_bytes = 0;
    result->fields.local_kv_bytes = 0;
    result->fields.global_kv_bytes = 0;
    result->fields.local_kv_eval_ms = 0.0F;
    result->fields.global_kv_eval_ms = 0.0F;
}

HypStepResultFields step_result_read(HypStepResult result) noexcept {
    if (result == nullptr || result->magic != kStepResultMagic) {
        return HypStepResultFields{};
    }
    return result->fields;
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
        // M2-2.7: the ForwardPass owns the per-layer math + the generation epilogue
        // (lm_head + softcap + greedy sampler). Built once at load; holds refs into the
        // geometry/dispatch/weights owned by this handle (lifetimes tied to the model).
        model->fwd = std::make_unique<hyperion::model::ForwardPass>(
            *model->geometry, *model->dispatch, *model->weights, gpu);
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
    if (tokens->count == 0) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "prefill token count is 0");
    }
    if (tokens->tokens == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "tokens->tokens is null");
    }
    status = hyperion::model::validate_step_result(out_result);
    if (status != HYP_STATUS_OK) {
        return status;
    }
    if (!model->loaded || model->fwd == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "model is not loaded");
    }
    // 2.6a prefill is chunked (2048-token chunks at the running offset; the rotation
    // read fires for sliding layers past the window). A single chunk for prompts ≤ 2048
    // (the 2.7 path, bit-identical). Re-prefilling a populated state still needs a fresh
    // kvstate (a reset is serving/2.6b); governor admission is 2.6b.
    if (kvstate->offset != 0) {
        return hyperion::model::fail(
            HYP_STATUS_INVALID_ARGUMENT, "prefill requires a fresh (offset 0) KV state");
    }

    try {
        const mx::Stream s = *model->stream;
        // Copy uint32 ids → int32 (MLX int32; ids < vocab < 2^31 so the cast is exact).
        std::vector<int32_t> host_ids;
        host_ids.reserve(tokens->count);
        for (std::uint32_t i = 0; i < tokens->count; ++i) {
            host_ids.push_back(static_cast<int32_t>(tokens->tokens[i]));
        }
        mx::array ids = mx::array(host_ids.data(), mx::Shape{static_cast<int>(tokens->count)}, mx::int32);

        // Chunked prefill (mlx-lm prefill_step_size=2048; 05 §Prefill chunking). Each chunk:
        // embed → forward(offset) (appends K/V via append_committed; the rotation read
        // fires for sliding past the window) → mx::eval(h) to materialize the chunk + the
        // cache writes before the next chunk (per mlx-lm mx.eval([c.state for c in cache])).
        // The epilogue runs ONLY on the last chunk (intermediate chunks' hidden is discarded).
        // Chunking is bitwise-invariant: each token's hidden attends causally to [0, token]
        // regardless of the chunk split, so the final-position hidden (→ token 1) is stable.
        constexpr std::uint32_t kPrefillChunkSize = 2048;
        hyperion::model::ForwardPass::GreedySample sample{0, 0.0F, false};
        std::uint32_t offset = 0;
        const std::uint32_t total = tokens->count;
        while (offset < total) {
            const std::uint32_t take = std::min(kPrefillChunkSize, total - offset);
            const int off0 = static_cast<int>(offset);
            const int off1 = static_cast<int>(offset + take);
            mx::array chunk_ids = mx::slice(ids, {off0}, {off1}, {1}, s); // [take]
            mx::array h = model->fwd->embed(chunk_ids);                   // [1, take, hidden]
            h = model->fwd->forward(h, *kvstate->kv, offset);             // final-norm'd; appends K/V
            if (offset + take == total) {
                sample = hyperion::model::run_epilogue(*model->fwd, h, s); // last chunk → token 1
            } else {
                mx::eval(h); // MLX lazy-graph materialization (NOT JS/Python eval): force this chunk + the cache writes before the next chunk
            }
            offset += take;
        }
        kvstate->offset = total;            // advance the autoregressive cursor
        kvstate->last_token = sample.token_id;
        hyperion::model::write_step_result(out_result, sample, sample.near_tie ? 1u : 0u);
        return hyperion::model::ok();
    } catch (const std::bad_alloc&) {
        return hyperion::model::fail(
            HYP_STATUS_INTERNAL, "native allocation failed outside predictive governor admission");
    } catch (const std::exception& error) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, error.what());
    } catch (...) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, "hyp_prefill_chunk failed with a non-standard exception");
    }
}

HypStatus hyp_decode_block(HypModel model,
                            HypKvState kvstate,
                            uint32_t n_tokens,
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
    if (!model->loaded || model->fwd == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "model is not loaded");
    }
    if (kvstate->offset == 0) {
        // decode needs a populated prefix — prefill first (or 2.6 chunked prefill).
        return hyperion::model::fail(
            HYP_STATUS_INVALID_ARGUMENT, "decode requires a populated KV state (prefill first)");
    }
    if (n_tokens == 0) {
        // Nothing to do; leave the step-result zeroed + READY. (A no-op decode is a
        // legal call shape — a verify block of size 0, M7.)
        hyperion::model::write_step_result(
            out_result, hyperion::model::ForwardPass::GreedySample{0, 0.0F, false}, 0u);
        return hyperion::model::ok();
    }

    try {
        const mx::Stream s = *model->stream;
        std::uint32_t near_tie_events = 0;
        hyperion::model::ForwardPass::GreedySample sample{0, 0.0F, false};
        // Autoregressive loop: embed last_token → forward at the current offset (q_len=1,
        // cached-prefix read) → epilogue → advance cursor. For greedy M2 the caller uses
        // n_tokens=1 per call; n_tokens>1 is the speculative-verify block shape (M7).
        for (std::uint32_t step = 0; step < n_tokens; ++step) {
            int32_t id = static_cast<int32_t>(kvstate->last_token);
            mx::array ids = mx::array(&id, mx::Shape{1}, mx::int32);
            mx::array h = model->fwd->embed(ids);             // [1, 1, hidden]
            h = model->fwd->forward(h, *kvstate->kv, kvstate->offset); // final-norm'd; appends 1 K/V
            sample = hyperion::model::run_epilogue(*model->fwd, h, s);
            if (sample.near_tie) {
                ++near_tie_events;
            }
            kvstate->last_token = sample.token_id;
            kvstate->offset += 1;
        }
        // The StepResult captures the LAST step's token + cumulative near_tie_events
        // across the block + peak memory (governor telemetry is 2.6).
        hyperion::model::write_step_result(out_result, sample, near_tie_events);
        return hyperion::model::ok();
    } catch (const std::bad_alloc&) {
        return hyperion::model::fail(
            HYP_STATUS_INTERNAL, "native allocation failed outside predictive governor admission");
    } catch (const std::exception& error) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, error.what());
    } catch (...) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, "hyp_decode_block failed with a non-standard exception");
    }
}

HypStatus hyp_step_result_create(HypStepResult* out_result) {
    return hyperion::model::create_handle(out_result, hyperion::model::kStepResultMagic);
}

HypStatus hyp_step_result_free(HypStepResult* result) {
    return hyperion::model::free_handle(result, hyperion::model::kStepResultMagic);
}

} // extern "C"
