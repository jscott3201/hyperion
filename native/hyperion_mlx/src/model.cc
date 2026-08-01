#include "hyperion_mlx.h"
#include "abi_error.h"
#include "dispatch.h"
#include "forward.h"
#include "geometry.h"
#include "governor.h"
#include "kv_cache.h"
#include "platform_policy.h"
#include "step_result_access.h"
#include "weights_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>
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
    std::unique_ptr<hyperion::governor::Governor> governor; // M2-2.6b: predictive admission.
    std::optional<mx::Stream> stream;
};

struct HypKvStateOpaque {
    std::uint32_t magic;
    bool poisoned = false;
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

// A direct decode aliases the live KV state, cursor, and last token. Once its
// first forward begins, an exception can therefore leave that handle partially
// advanced. Growth transactions use a staged owner and remain retryable.
class DirectKvExecutionGuard {
  public:
    DirectKvExecutionGuard(HypKvState kvstate, bool eligible) noexcept
        : kvstate_(kvstate), eligible_(eligible) {}

    DirectKvExecutionGuard(const DirectKvExecutionGuard&) = delete;
    DirectKvExecutionGuard& operator=(const DirectKvExecutionGuard&) = delete;

    ~DirectKvExecutionGuard() noexcept {
        if (eligible_ && armed_) {
            kvstate_->poisoned = true;
        }
    }

    void arm() noexcept {
        if (eligible_) {
            armed_ = true;
        }
    }

    void disarm() noexcept { armed_ = false; }

  private:
    HypKvState kvstate_;
    bool eligible_;
    bool armed_ = false;
};

governor::PrefillAdmissionResult admit_prefill_chunk(
    const governor::Governor& governor,
    std::uint32_t proposed_n_tokens,
    std::uint32_t offset,
    std::uint32_t total_tokens,
    const KvState& kvstate,
    const KvGrowthPlan* operation_plan,
    StepTelemetryAccumulator& telemetry) {
    auto result = governor.admit_prefill_to_fit(
        proposed_n_tokens,
        offset,
        total_tokens,
        kvstate,
        operation_plan);
    for (std::size_t i = 0; i < result.attempt_count; ++i) {
        (void)telemetry.observe(result.attempts[i]);
    }
    return result;
}

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
    if (kvstate->poisoned) {
        return fail(
            HYP_STATUS_INTERNAL,
            "KV state is unusable after an internal execution failure; free and recreate it");
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

/// M3 sampler: validate a HypSamplingConfig. Returns nullopt on OK, or a status message
/// on the 400-malformed cases (06 §error taxonomy): negative temperature, top_k > vocab,
/// top_p outside (0,1], min_p outside (0,1). NULL config OR temperature == 0 is VALID
/// (routes to the greedy default).
std::optional<std::string> validate_sampling_config(const HypSamplingConfig* config, std::uint32_t vocab_size) {
    if (config == nullptr || config->temperature == 0.0F) {
        return std::nullopt; // greedy default — always valid
    }
    if (config->temperature < 0.0F) {
        return "sampling temperature must be >= 0 (0 = greedy)";
    }
    if (config->top_k < 0) {
        return "sampling top_k must be >= 0 (0 = disabled)";
    }
    if (config->top_k > 0 && static_cast<std::uint64_t>(config->top_k) > vocab_size) {
        return "sampling top_k exceeds the vocabulary size";
    }
    if (!std::isnan(config->top_p) && (config->top_p < 0.0F || config->top_p > 1.0F)) {
        return "sampling top_p must be in [0, 1] (0 = disabled)";
    }
    if (config->min_p < 0.0F || config->min_p >= 1.0F) {
        return "sampling min_p must be in [0, 1) (0 = disabled)";
    }
    return std::nullopt;
}

/// M3 sampler: the sampled epilogue. Same lm_head + softcap + last-position slice as
/// run_epilogue, but routes to sample_stochastic when config != NULL and
/// config->temperature > 0. config == NULL OR temperature == 0 → the greedy path
/// (sample_greedy), byte-identical to run_epilogue — the G1 token-exact seal. The
/// seed is advanced per-call by the caller (per-request RNG state threaded across
/// the decode loop). Returns a StochasticSample (token + top-k logprob sidecar).
ForwardPass::StochasticSample run_epilogue_sampled(
    ForwardPass& fwd, const mx::array& h, mx::Stream s,
    const HypSamplingConfig* config, std::uint64_t rng_state) {
    const int L = static_cast<int>(h.shape(1));
    mx::array logits = fwd.lm_head(h);                 // [B, L, vocab]
    logits = fwd.softcap(logits);                      // [B, L, vocab]
    mx::array last = mx::slice(
        logits,
        {0, L - 1, 0},
        {1, L, static_cast<int>(logits.shape(2))},
        {1, 1, 1},
        s);
    if (config == nullptr || config->temperature == 0.0F) {
        // Greedy default — build a StochasticSample from the greedy argmax so the
        // caller's plumbing is uniform. The greedy path itself is byte-identical.
        auto g = fwd.sample_greedy(last);
        ForwardPass::StochasticSample out{};
        out.token_id = g.token_id;
        out.logit = g.logit;
        // top-k sidecar from the same last-position frame (the greedy top-1 is the
        // first entry; the rest are the next-highest — a partial_sort, same as the
        // sampled path).
        const mx::Stream cs = mx::default_stream(mx::Device::cpu);
        mx::array lf = mx::astype(mx::contiguous(last, false, cs), mx::float32, cs);
        mx::eval(lf);
        const float* p = lf.data<float>();
        const std::size_t n = lf.size();
        std::vector<std::uint32_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + HYP_TOP_K_LOGPROBS, idx.end(),
                          [&](std::uint32_t a, std::uint32_t b) { return p[a] > p[b]; });
        out.top_k_count = std::min<std::size_t>(HYP_TOP_K_LOGPROBS, n);
        for (std::size_t i = 0; i < out.top_k_count; ++i) {
            out.top_k_ids[i] = idx[i];
            out.top_k_logprobs[i] = p[idx[i]];
        }
        return out;
    }
    return fwd.sample_stochastic(last, *config, rng_state);
}

HypStepResultFields step_result_read(HypStepResult result) noexcept {
    if (result == nullptr || result->magic != kStepResultMagic) {
        return HypStepResultFields{};
    }
    return result->fields;
}

void write_step_result(
    HypStepResult result,
    const ForwardPass::GreedySample& sample,
    std::uint32_t near_tie_events,
    HypGovernorState governor_state,
    const StepTelemetrySnapshot& telemetry) {
    if (result == nullptr || result->magic != kStepResultMagic) {
        return; // invalid handle: no-op (the ABI call already validated, but never trust across the seam)
    }
    // Zero-init the whole struct first so the fields this helper does NOT set
    // (phys_footprint_bytes, local/global_kv_eval_ms) are deterministic zero,
    // not stale data from a prior step on this reused handle.
    result->fields = HypStepResultFields{};
    result->fields.token_id = sample.token_id;
    result->fields.logit = sample.logit;
    result->fields.peak_mlx_bytes = telemetry.peak_mlx_bytes;
    result->fields.active_mlx_bytes = mx::get_active_memory();
    result->fields.local_kv_bytes = telemetry.local_kv_bytes;
    result->fields.global_kv_bytes = telemetry.global_kv_bytes;
    result->fields.governor_state = governor_state;
    result->fields.near_tie_events = near_tie_events;
}

void write_step_result_sampled(
    HypStepResult result,
    const ForwardPass::StochasticSample& sample,
    HypGovernorState governor_state,
    const StepTelemetrySnapshot& telemetry) {
    if (result == nullptr || result->magic != kStepResultMagic) {
        return;
    }
    result->fields = HypStepResultFields{};
    result->fields.token_id = sample.token_id;
    result->fields.logit = sample.logit;
    result->fields.peak_mlx_bytes = telemetry.peak_mlx_bytes;
    result->fields.active_mlx_bytes = mx::get_active_memory();
    result->fields.local_kv_bytes = telemetry.local_kv_bytes;
    result->fields.global_kv_bytes = telemetry.global_kv_bytes;
    result->fields.governor_state = governor_state;
    result->fields.near_tie_events = 0u; // near-tie is a greedy-only metric
    result->fields.top_k_logprob_count = sample.top_k_count;
    for (std::size_t i = 0; i < sample.top_k_count; ++i) {
        result->fields.top_k_logprob_ids[i] = sample.top_k_ids[i];
        result->fields.top_k_logprob_values[i] = sample.top_k_logprobs[i];
    }
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

    const auto runtime_environment =
        hyperion::platform::evaluate_runtime_environment(
            std::getenv("MLX_SDPA_BLOCKS") != nullptr);
    if (!runtime_environment.supported) {
        return hyperion::model::fail(
            HYP_STATUS_UNSUPPORTED, runtime_environment.reason);
    }

    try {
        const auto device_budget = hyperion::platform::derive_device_budget(
            mx::device_info(mx::Device::gpu));
        if (!device_budget.supported) {
            return hyperion::model::fail(
                HYP_STATUS_UNSUPPORTED, device_budget.reason.c_str());
        }
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
        // M2-2.6b: build the predictive governor from MLX's public device data.
        // Extraction above fails closed before weight loading for missing,
        // wrong-typed, or zero recommendation data.
        model->governor = std::make_unique<hyperion::governor::Governor>(
            *model->geometry,
            device_budget.budget.effective_bytes,
            device_budget.budget.soft_watermark_bytes);
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
    if (!model->loaded || model->fwd == nullptr || model->governor == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "model is not loaded");
    }
    // 2.6a prefill is chunked (2048-token chunks at the running offset; the rotation
    // read fires for sliding layers past the window). A single chunk for prompts ≤ 2048
    // (the 2.7 path, bit-identical). Re-prefilling a populated state still needs a fresh
    // kvstate (a reset is serving/2.6b). Governor admission (2.6b) runs before each chunk.
    if (kvstate->offset != 0) {
        return hyperion::model::fail(
            HYP_STATUS_INVALID_ARGUMENT, "prefill requires a fresh (offset 0) KV state");
    }

    try {
        const mx::Stream s = *model->stream;
        hyperion::model::StepTelemetryAccumulator telemetry;
        const auto operation_plan =
            hyperion::model::plan_kv_growth(*kvstate->kv, tokens->count);
        auto settled_staging_plan = operation_plan;
        settled_staging_plan.requires_transaction = false;
        if (!operation_plan.representable) {
            const auto admission = hyperion::model::admit_prefill_chunk(
                *model->governor,
                std::min<std::uint32_t>(2048, tokens->count),
                0,
                tokens->count,
                *kvstate->kv,
                &operation_plan,
                telemetry);
            hyperion::model::write_step_result(
                out_result, hyperion::model::ForwardPass::GreedySample{0, 0.0F, false},
                0u, HYP_GOVERNOR_HARD_REJECT, telemetry.snapshot(*kvstate->kv));
            return hyperion::model::fail(
                HYP_STATUS_OOM_GOVERNOR, admission.decision.reason);
        }
        hyperion::model::KvGrowthTransaction transaction(
            kvstate->kv, operation_plan);
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
        // Governor admission (2.6b): repeatedly halve either a hard- or soft-pressure
        // proposal and re-evaluate its exact shape. Only hard pressure at one token
        // rejects; soft pressure at one token remains advisory.
        constexpr std::uint32_t kPrefillChunkSize = 2048;
        hyperion::model::ForwardPass::GreedySample sample{0, 0.0F, false};
        std::uint32_t offset = 0;
        const std::uint32_t total = tokens->count;
        while (offset < total) {
            std::uint32_t take = std::min(kPrefillChunkSize, total - offset);
            const auto admission = hyperion::model::admit_prefill_chunk(
                *model->governor,
                take,
                offset,
                total,
                transaction.state(),
                offset == 0 ? &operation_plan : &settled_staging_plan,
                telemetry);
            take = admission.n_tokens;
            if (admission.decision.admission ==
                hyperion::governor::Admission::HardRejected) {
                // The shared shrink loop returns hard only after one token fails.
                hyperion::model::write_step_result(
                    out_result, hyperion::model::ForwardPass::GreedySample{0, 0.0F, false},
                    0u, HYP_GOVERNOR_HARD_REJECT, telemetry.snapshot(*kvstate->kv));
                return hyperion::model::fail(
                    HYP_STATUS_OOM_GOVERNOR, admission.decision.reason);
            }
            const int off0 = static_cast<int>(offset);
            const int off1 = static_cast<int>(offset + take);
            mx::array chunk_ids = mx::slice(ids, {off0}, {off1}, {1}, s); // [take]
            mx::array h = model->fwd->embed(chunk_ids);                   // [1, take, hidden]
            h = model->fwd->forward(h, transaction.state(), offset);      // final-norm'd; appends K/V
            h = transaction.root_forward_result(h);
            if (offset + take == total) {
                sample = hyperion::model::run_epilogue(*model->fwd, h, s); // last chunk → token 1
            } else {
                mx::eval(h); // MLX lazy-graph materialization (NOT JS/Python eval): force this chunk + the cache writes before the next chunk
            }
            offset += take;
        }
        transaction.materialize();
        transaction.publish();
        kvstate->offset = total;            // advance the autoregressive cursor
        kvstate->last_token = sample.token_id;
        hyperion::model::write_step_result(
            out_result, sample, sample.near_tie ? 1u : 0u,
            telemetry.successful_governor_state(), telemetry.snapshot(*kvstate->kv));
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
    if (!model->loaded || model->fwd == nullptr || model->governor == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "model is not loaded");
    }
    if (kvstate->offset == 0) {
        // decode needs a populated prefix — prefill first (or 2.6 chunked prefill).
        return hyperion::model::fail(
            HYP_STATUS_INVALID_ARGUMENT, "decode requires a populated KV state (prefill first)");
    }
    hyperion::model::StepTelemetryAccumulator telemetry;
    if (n_tokens == 0) {
        // A no-op decode attempts no admission. It still reports the live cache
        // allocation and active MLX bytes at this terminal write.
        hyperion::model::write_step_result(
            out_result, hyperion::model::ForwardPass::GreedySample{0, 0.0F, false}, 0u,
            HYP_GOVERNOR_READY, telemetry.snapshot(*kvstate->kv));
        return hyperion::model::ok();
    }

    // Governor admission (2.6b): predict the peak for the full decode block at the
    // current offset. Decode is 1 token/step (q_len=1), so continuation-prefill scratch
    // does not apply and the attention transient is one q=1 forward; global KV
    // admission sums every sequential capacity-bucket replacement in the block.
    const auto operation_plan =
        hyperion::model::plan_kv_growth(*kvstate->kv, n_tokens);
    auto decision = telemetry.observe(model->governor->evaluate(
        hyperion::governor::AdmissionInput{
            n_tokens,
            kvstate->offset,
            hyperion::governor::StepKind::Decode,
            true,
        },
        *kvstate->kv,
        &operation_plan));
    if (decision.admission == hyperion::governor::Admission::HardRejected) {
        hyperion::model::write_step_result(
            out_result, hyperion::model::ForwardPass::GreedySample{0, 0.0F, false}, 0u,
            HYP_GOVERNOR_HARD_REJECT, telemetry.snapshot(*kvstate->kv));
        return hyperion::model::fail(HYP_STATUS_OOM_GOVERNOR, decision.reason);
    }

    try {
        const mx::Stream s = *model->stream;
        hyperion::model::KvGrowthTransaction transaction(
            kvstate->kv, operation_plan);
        const bool transactional = transaction.active();
        hyperion::model::DirectKvExecutionGuard execution_guard(
            kvstate, !transactional);
        std::uint32_t near_tie_events = 0;
        hyperion::model::ForwardPass::GreedySample sample{0, 0.0F, false};
        std::uint32_t staged_offset = kvstate->offset;
        std::uint32_t staged_last_token = kvstate->last_token;
        std::uint32_t& execution_offset =
            transactional ? staged_offset : kvstate->offset;
        std::uint32_t& execution_last_token =
            transactional ? staged_last_token : kvstate->last_token;
        // Autoregressive loop: embed last_token → forward at the current offset (q_len=1,
        // cached-prefix read) → epilogue → advance cursor. For greedy M2 the caller uses
        // n_tokens=1 per call; n_tokens>1 is the speculative-verify block shape (M7).
        for (std::uint32_t step = 0; step < n_tokens; ++step) {
            int32_t id = static_cast<int32_t>(execution_last_token);
            mx::array ids = mx::array(&id, mx::Shape{1}, mx::int32);
            mx::array h = model->fwd->embed(ids);             // [1, 1, hidden]
            if (step == 0) {
                execution_guard.arm();
            }
            h = model->fwd->forward(h, transaction.state(), execution_offset); // final-norm'd; appends 1 K/V
            h = transaction.root_forward_result(h);
            sample = hyperion::model::run_epilogue(*model->fwd, h, s);
            if (sample.near_tie) {
                ++near_tie_events;
            }
            execution_last_token = sample.token_id;
            execution_offset += 1;
        }
        transaction.materialize();
        transaction.publish();
        if (transactional) {
            kvstate->offset = staged_offset;
            kvstate->last_token = staged_last_token;
        }
        // Preserve the pre-step admission prediction: recomputing after KV mutation
        // would treat the just-allocated bucket as settled and charge it a second time.
        hyperion::model::write_step_result(
            out_result, sample, near_tie_events,
            telemetry.successful_governor_state(), telemetry.snapshot(*kvstate->kv));
        execution_guard.disarm();
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

// ── M3 sampler surface: the sampled ABI pair (03 §Sampling). config == NULL OR ──
// temperature == 0 routes to the greedy path byte-identically (the G1 token-exact
// seal). Non-NULL with temperature > 0 routes to sample_stochastic. The greedy
// hyp_prefill_chunk/hyp_decode_block above are UNCHANGED (the seal binds to them).
HypStatus hyp_prefill_chunk_sampled(HypModel model,
                                    HypKvState kvstate,
                                    const HypTokenStream* tokens,
                                    const HypSamplingConfig* config,
                                    HypStepResult out_result) {
    // Validate the model handle FIRST (validate_sampling_config reads
    // model->geometry->vocab_size, so the model must be valid before the config check).
    auto status = hyperion::model::validate_model(model, "hyp_prefill_chunk_sampled");
    if (status != HYP_STATUS_OK) return status;
    status = hyperion::model::validate_kvstate(kvstate);
    if (status != HYP_STATUS_OK) return status;
    status = hyperion::model::validate_step_result(out_result);
    if (status != HYP_STATUS_OK) return status;
    // Validate the sampling config (the 400-malformed bucket, 06 §error taxonomy).
    if (auto err = hyperion::model::validate_sampling_config(config, model->geometry->vocab_size); err.has_value()) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, err->c_str());
    }
    // Greedy default: delegate to hyp_prefill_chunk (byte-identical, G1 seal). The greedy
    // path fills near_tie_events; the sampled path does not — delegating keeps the greedy
    // semantics exact without re-running the epilogue.
    if (config == nullptr || config->temperature == 0.0F) {
        return hyp_prefill_chunk(model, kvstate, tokens, out_result);
    }
    if (tokens == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "tokens is null");
    }
    if (!model->loaded || model->fwd == nullptr || model->governor == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "model is not loaded");
    }
    if (kvstate->offset != 0) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "prefill requires a fresh (offset 0) KV state");
    }
    try {
        const mx::Stream s = *model->stream;
        hyperion::model::StepTelemetryAccumulator telemetry;
        const auto operation_plan =
            hyperion::model::plan_kv_growth(*kvstate->kv, tokens->count);
        auto settled_staging_plan = operation_plan;
        settled_staging_plan.requires_transaction = false;
        if (!operation_plan.representable) {
            const auto admission = hyperion::model::admit_prefill_chunk(
                *model->governor,
                std::min<std::uint32_t>(2048, tokens->count),
                0,
                tokens->count,
                *kvstate->kv,
                &operation_plan,
                telemetry);
            hyperion::model::ForwardPass::StochasticSample rejected{};
            hyperion::model::write_step_result_sampled(
                out_result, rejected, HYP_GOVERNOR_HARD_REJECT,
                telemetry.snapshot(*kvstate->kv));
            return hyperion::model::fail(
                HYP_STATUS_OOM_GOVERNOR, admission.decision.reason);
        }
        hyperion::model::KvGrowthTransaction transaction(
            kvstate->kv, operation_plan);
        std::vector<int32_t> host_ids;
        host_ids.reserve(tokens->count);
        for (std::uint32_t i = 0; i < tokens->count; ++i) {
            host_ids.push_back(static_cast<int32_t>(tokens->tokens[i]));
        }
        mx::array ids = mx::array(host_ids.data(), mx::Shape{static_cast<int>(tokens->count)}, mx::int32);
        constexpr std::uint32_t kPrefillChunkSize = 2048;
        hyperion::model::ForwardPass::StochasticSample sample{};
        std::uint32_t offset = 0;
        const std::uint32_t total = tokens->count;
        while (offset < total) {
            std::uint32_t take = std::min(kPrefillChunkSize, total - offset);
            const auto admission = hyperion::model::admit_prefill_chunk(
                *model->governor,
                take,
                offset,
                total,
                transaction.state(),
                offset == 0 ? &operation_plan : &settled_staging_plan,
                telemetry);
            take = admission.n_tokens;
            if (admission.decision.admission ==
                hyperion::governor::Admission::HardRejected) {
                hyperion::model::ForwardPass::StochasticSample rejected{};
                hyperion::model::write_step_result_sampled(
                    out_result, rejected, HYP_GOVERNOR_HARD_REJECT,
                    telemetry.snapshot(*kvstate->kv));
                return hyperion::model::fail(
                    HYP_STATUS_OOM_GOVERNOR, admission.decision.reason);
            }
            const int off0 = static_cast<int>(offset);
            const int off1 = static_cast<int>(offset + take);
            mx::array chunk_ids = mx::slice(ids, {off0}, {off1}, {1}, s);
            mx::array h = model->fwd->embed(chunk_ids);
            h = model->fwd->forward(h, transaction.state(), offset);
            h = transaction.root_forward_result(h);
            if (offset + take == total) {
                // last chunk → the sampled epilogue. Seed advances per prefill (the request seed).
                sample = hyperion::model::run_epilogue_sampled(*model->fwd, h, s, config, config->seed);
            } else {
                mx::eval(h);
            }
            offset += take;
        }
        transaction.materialize();
        transaction.publish();
        kvstate->offset = total;
        kvstate->last_token = sample.token_id;
        hyperion::model::write_step_result_sampled(
            out_result, sample, telemetry.successful_governor_state(),
            telemetry.snapshot(*kvstate->kv));
        return hyperion::model::ok();
    } catch (const std::bad_alloc&) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, "native allocation failed outside predictive governor admission");
    } catch (const std::exception& error) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, error.what());
    } catch (...) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, "hyp_prefill_chunk_sampled failed with a non-standard exception");
    }
}

HypStatus hyp_decode_block_sampled(HypModel model,
                                   HypKvState kvstate,
                                   uint32_t n_tokens,
                                   const HypSamplingConfig* config,
                                   HypStepResult out_result) {
    // Validate the model handle FIRST (validate_sampling_config reads
    // model->geometry->vocab_size, so the model must be valid before the config check).
    auto status = hyperion::model::validate_model(model, "hyp_decode_block_sampled");
    if (status != HYP_STATUS_OK) return status;
    status = hyperion::model::validate_kvstate(kvstate);
    if (status != HYP_STATUS_OK) return status;
    status = hyperion::model::validate_step_result(out_result);
    if (status != HYP_STATUS_OK) return status;
    if (auto err = hyperion::model::validate_sampling_config(config, model->geometry->vocab_size); err.has_value()) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, err->c_str());
    }
    // Greedy default: delegate to hyp_decode_block (byte-identical, G1 seal).
    if (config == nullptr || config->temperature == 0.0F) {
        return hyp_decode_block(model, kvstate, n_tokens, out_result);
    }
    if (!model->loaded || model->fwd == nullptr || model->governor == nullptr) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "model is not loaded");
    }
    if (kvstate->offset == 0) {
        return hyperion::model::fail(HYP_STATUS_INVALID_ARGUMENT, "decode requires a populated KV state (prefill first)");
    }
    hyperion::model::StepTelemetryAccumulator telemetry;
    if (n_tokens == 0) {
        hyperion::model::ForwardPass::StochasticSample sample{};
        hyperion::model::write_step_result_sampled(
            out_result, sample, telemetry.successful_governor_state(),
            telemetry.snapshot(*kvstate->kv));
        return hyperion::model::ok();
    }
    const auto operation_plan =
        hyperion::model::plan_kv_growth(*kvstate->kv, n_tokens);
    auto decision = telemetry.observe(model->governor->evaluate(
        hyperion::governor::AdmissionInput{
            n_tokens,
            kvstate->offset,
            hyperion::governor::StepKind::Decode,
            true,
        },
        *kvstate->kv,
        &operation_plan));
    if (decision.admission == hyperion::governor::Admission::HardRejected) {
        hyperion::model::ForwardPass::StochasticSample sample{};
        hyperion::model::write_step_result_sampled(
            out_result, sample, HYP_GOVERNOR_HARD_REJECT,
            telemetry.snapshot(*kvstate->kv));
        return hyperion::model::fail(HYP_STATUS_OOM_GOVERNOR, decision.reason);
    }
    try {
        const mx::Stream s = *model->stream;
        hyperion::model::KvGrowthTransaction transaction(
            kvstate->kv, operation_plan);
        const bool transactional = transaction.active();
        hyperion::model::DirectKvExecutionGuard execution_guard(
            kvstate, !transactional);
        hyperion::model::ForwardPass::StochasticSample sample{};
        // Per-request RNG: seed advances each step so a single seed yields a reproducible
        // stream (same seed → same tokens). std::mt19937_64 advanced by a per-step salt.
        std::uint64_t rng_state = config->seed;
        std::uint32_t staged_offset = kvstate->offset;
        std::uint32_t staged_last_token = kvstate->last_token;
        std::uint32_t& execution_offset =
            transactional ? staged_offset : kvstate->offset;
        std::uint32_t& execution_last_token =
            transactional ? staged_last_token : kvstate->last_token;
        for (std::uint32_t step = 0; step < n_tokens; ++step) {
            int32_t id = static_cast<int32_t>(execution_last_token);
            mx::array ids = mx::array(&id, mx::Shape{1}, mx::int32);
            mx::array h = model->fwd->embed(ids);
            if (step == 0) {
                execution_guard.arm();
            }
            h = model->fwd->forward(h, transaction.state(), execution_offset);
            h = transaction.root_forward_result(h);
            sample = hyperion::model::run_epilogue_sampled(*model->fwd, h, s, config, rng_state);
            // Advance the RNG state per step (a fixed salt — deterministic per-request).
            rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
            execution_last_token = sample.token_id;
            execution_offset += 1;
        }
        transaction.materialize();
        transaction.publish();
        if (transactional) {
            kvstate->offset = staged_offset;
            kvstate->last_token = staged_last_token;
        }
        // Successful decode telemetry reports the exact pre-step admission decision.
        hyperion::model::write_step_result_sampled(
            out_result, sample, HYP_GOVERNOR_READY, telemetry.snapshot(*kvstate->kv));
        execution_guard.disarm();
        return hyperion::model::ok();
    } catch (const std::bad_alloc&) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, "native allocation failed outside predictive governor admission");
    } catch (const std::exception& error) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, error.what());
    } catch (...) {
        return hyperion::model::fail(HYP_STATUS_INTERNAL, "hyp_decode_block_sampled failed with a non-standard exception");
    }
}

HypStatus hyp_step_result_create(HypStepResult* out_result) {
    return hyperion::model::create_handle(out_result, hyperion::model::kStepResultMagic);
}

HypStatus hyp_step_result_free(HypStepResult* result) {
    return hyperion::model::free_handle(result, hyperion::model::kStepResultMagic);
}

// M3 serving: read a step result's fields back across the ABI (13 -> 14,
// abi_version 2 -> 3; see ADR 0003). Copies the struct out — the caller still
// owns the handle and must free it separately. Aliases the existing private
// `step_result_read` helper's magic-tag check (no double-trust across the seam:
// validate the handle here, the same way `step_result_read` does, rather than
// calling it and trusting a default-constructed struct).
HypStatus hyp_step_result_fields(HypStepResult result,
                                 HypStepResultFields* out_fields) {
    if (out_fields == nullptr) {
        return HYP_STATUS_INVALID_ARGUMENT;
    }
    if (result == nullptr || result->magic != hyperion::model::kStepResultMagic) {
        return HYP_STATUS_INVALID_ARGUMENT;
    }
    *out_fields = result->fields;
    return HYP_STATUS_OK;
}

} // extern "C"
