#include "hyperion_mlx.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "abi_contract_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    require(
        hyp_runtime_canary(nullptr) == HYP_STATUS_INVALID_ARGUMENT,
        "a null canary output must return INVALID_ARGUMENT");
    std::array<char, 256> error{};
    require(
        hyp_last_error(error.data(), error.size()) == HYP_STATUS_OK,
        "the thread-local diagnostic must be readable");
    require(
        std::string_view(error.data()).find("non-null output pointer") != std::string_view::npos,
        "the null-output diagnostic must survive the ABI boundary");

    char one_byte = 'x';
    require(
        hyp_last_error(&one_byte, 1) == HYP_STATUS_OK && one_byte == '\0',
        "a one-byte error buffer must remain NUL terminated");
    require(
        hyp_last_error(nullptr, 0) == HYP_STATUS_INVALID_ARGUMENT,
        "an invalid error buffer must return INVALID_ARGUMENT");

    // M2 magic-tagged handle lifecycle.
    HypModel model = nullptr;
    require(hyp_model_create(&model) == HYP_STATUS_OK, "model create must succeed");
    require(model != nullptr, "model handle must be non-null after create");

    // hyp_model_load is real (M2-2.3a): a minimal geometry (num_hidden_layers=0,
    // empty layer_types) is rejected by from_abi + Geometry::validate BEFORE any MLX or
    // I/O — so this stays model-free (no GPU / no artifact needed). The successful load
    // path + prefill/decode stubs are exercised by the M5-gated hyperion_forward_test.
    HypGeometryParams geometry{};
    geometry.hidden_size = 3840;
    require(
        hyp_model_load(model, &geometry, "weights") == HYP_STATUS_INVALID_ARGUMENT,
        "hyp_model_load rejects an invalid (empty-layer_types) geometry");

    // kvstate_create requires a loaded model; the model above never loaded.
    HypKvState kv = nullptr;
    require(
        hyp_kvstate_create(model, &kv) == HYP_STATUS_INVALID_ARGUMENT,
        "kvstate create on an unloaded model must reject");
    require(kv == nullptr, "a rejected kvstate create must not produce a handle");

    HypStepResult result = nullptr;
    require(hyp_step_result_create(&result) == HYP_STATUS_OK, "step result create must succeed");
    require(result != nullptr, "step result handle must be non-null");

    // prefill/decode with a null kvstate hit the kvstate validation guard (the stubs
    // never reach the UNSUPPORTED return without a valid kvstate; the full stub path is
    // in hyperion_forward_test's test_abi_load).
    HypTokenStream tokens{nullptr, 0, 1};
    require(
        hyp_prefill_chunk(model, nullptr, &tokens, result) == HYP_STATUS_INVALID_ARGUMENT,
        "prefill with a null kvstate must reject, not UB");
    require(
        hyp_decode_block(model, nullptr, 1, result) == HYP_STATUS_INVALID_ARGUMENT,
        "decode with a null kvstate must reject, not UB");

    require(hyp_step_result_free(&result) == HYP_STATUS_OK, "step result free must succeed");
    require(result == nullptr, "step result handle must be nulled after free");
    require(
        hyp_step_result_free(&result) == HYP_STATUS_OK,
        "double-free of a nulled handle is an idempotent no-op (no UB)");
    require(hyp_model_free(&model) == HYP_STATUS_OK, "model free must succeed");
    require(model == nullptr, "model handle must be nulled after free");

    // Null handle-pointer arguments -> INVALID_ARGUMENT (no UB).
    require(
        hyp_model_free(nullptr) == HYP_STATUS_INVALID_ARGUMENT,
        "free(nullptr*) must reject, not UB");
    require(
        hyp_kvstate_create(nullptr, &kv) == HYP_STATUS_INVALID_ARGUMENT,
        "kvstate create on a null model must reject");

    // Wrong type: a step-result handle where a model is expected -> magic mismatch.
    HypStepResult wrong = nullptr;
    require(hyp_step_result_create(&wrong) == HYP_STATUS_OK, "create wrong-type handle");
    require(
        hyp_model_load(reinterpret_cast<HypModel>(wrong), &geometry, "weights") ==
            HYP_STATUS_INVALID_ARGUMENT,
        "a wrong-type handle must be rejected by the magic tag");
    require(hyp_step_result_free(&wrong) == HYP_STATUS_OK, "free wrong-type handle");
    return 0;
}
