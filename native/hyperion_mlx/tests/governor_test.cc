#include "governor.h"
#include "geometry.h"
#include "kv_cache.h"
#include "platform_policy.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;
using hyperion::model::Geometry;
using hyperion::model::LayerType;
using hyperion::model::build_kv_state;
using hyperion::model::KvState;
using hyperion::governor::Admission;
using hyperion::governor::AdmissionInput;
using hyperion::governor::AttentionPath;
using hyperion::governor::Governor;
using hyperion::governor::GovernorDecision;
using hyperion::governor::StepKind;
using hyperion::governor::kGlobalKvBytesPerToken;
using hyperion::governor::kWorkspaceReserveBytes;
using hyperion::governor::peak_within_budget;
using hyperion::governor::predict_peak;
using hyperion::governor::predict_transient;
using hyperion::governor::throughput_optimum_context;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "governor_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

/// Build a 5:1 tiny geometry mirroring the committed gemma4-unified-tiny fixture
/// (5 sliding + 1 global, sliding_window=8).
Geometry build_tiny_geometry() {
    Geometry g;
    g.model_type = hyperion::model::TextModelType::Gemma4UnifiedText;
    g.hidden_size = 128;
    g.intermediate_size = 256;
    g.num_hidden_layers = 6;
    g.layer_types = {
        LayerType::Sliding, LayerType::Sliding, LayerType::Sliding,
        LayerType::Sliding, LayerType::Sliding, LayerType::Full,
    };
    g.num_attention_heads = 4;
    g.head_dim_local = 64;
    g.head_dim_global = 128;
    g.num_kv_heads_local = 2;
    g.num_kv_heads_global = 1;
    g.attention_k_eq_v_global = true;
    g.num_kv_shared_layers = 0;
    g.sliding_window = 8;
    g.rope_local = {10000.0, std::nullopt, false};
    g.rope_global = {1000000.0, 0.25F, true};
    g.final_logit_softcapping = 30.0F;
    g.rms_norm_eps = 1e-6F;
    g.attention_bias = false;
    g.vocab_size = 128;
    g.max_position_embeddings = 256;
    g.tie_word_embeddings = true;
    g.ple_hidden_per_layer_input = 0;
    g.ple_vocab_per_layer_input = 0;
    g.use_double_wide_mlp = false;
    g.moe = std::nullopt;
    return g;
}

void test_transient_model() {
    const auto geometry = build_tiny_geometry();
    const AdmissionInput fresh_two{2, 0, StepKind::Prefill, false};
    const auto base = predict_transient(geometry, fresh_two);
    require(base.global_phase_bytes > base.local_phase_bytes &&
            base.attention_peak_bytes == base.global_phase_bytes,
        "attention peak must select the max local/global phase");
    require(base.operation_peak_bytes == base.attention_peak_bytes,
        "non-final prefill must have no epilogue phase");

    auto duplicated = geometry;
    duplicated.layer_types.insert(
        duplicated.layer_types.end(),
        geometry.layer_types.begin(), geometry.layer_types.end());
    duplicated.num_hidden_layers =
        static_cast<std::uint32_t>(duplicated.layer_types.size());
    const auto duplicate_prediction = predict_transient(duplicated, fresh_two);
    require(duplicate_prediction.local_phase_bytes == base.local_phase_bytes &&
            duplicate_prediction.global_phase_bytes == base.global_phase_bytes &&
            duplicate_prediction.attention_peak_bytes == base.attention_peak_bytes,
        "duplicating same-kind layers must not change a sequential attention transient");

    auto local_only = geometry;
    local_only.layer_types = {LayerType::Sliding};
    local_only.num_hidden_layers = 1;

    for (const std::uint32_t dim : {64u, 96u, 128u, 256u}) {
        local_only.head_dim_local = dim;
        require(
            predict_transient(
                local_only, AdmissionInput{8, 0, StepKind::Prefill, false})
                    .local_path == AttentionPath::FusedVector,
            "every pinned equal-dim vector boundary must dispatch fused");
    }
    for (const std::uint32_t dim : {64u, 80u, 128u}) {
        local_only.head_dim_local = dim;
        require(
            predict_transient(
                local_only, AdmissionInput{9, 0, StepKind::Prefill, false})
                    .local_path == AttentionPath::FusedFull,
            "every pinned equal-dim full boundary must dispatch fused");
    }
    local_only.head_dim_local = 512;
    require(
        predict_transient(
            local_only, AdmissionInput{1, 0, StepKind::Prefill, false})
                .local_path == AttentionPath::Fallback &&
            predict_transient(
                local_only, AdmissionInput{9, 0, StepKind::Prefill, false})
                .local_path == AttentionPath::Fallback,
        "unsupported equal head dimensions must fall back in vector and full modes");

    local_only.head_dim_local = 96;
    require(
        predict_transient(
            local_only, AdmissionInput{8, 0, StepKind::Prefill, false})
                .local_path == AttentionPath::FusedVector,
        "q<=8 head-dim 96 must use the pinned vector fused path");
    require(
        predict_transient(
            local_only, AdmissionInput{9, 0, StepKind::Prefill, false})
                .local_path == AttentionPath::Fallback,
        "q>8 head-dim 96 must fall back (not full-fused)");
    local_only.head_dim_local = 80;
    require(
        predict_transient(
            local_only, AdmissionInput{8, 0, StepKind::Prefill, false})
                .local_path == AttentionPath::Fallback,
        "head-dim 80 must not use the vector fused path");
    require(
        predict_transient(
            local_only, AdmissionInput{9, 0, StepKind::Prefill, false})
                .local_path == AttentionPath::FusedFull,
        "q>8 head-dim 80 must use the full fused path");
    local_only.head_dim_local = 256;
    require(
        predict_transient(
            local_only, AdmissionInput{8, 0, StepKind::Prefill, false})
                .local_path == AttentionPath::FusedVector &&
            predict_transient(
                local_only, AdmissionInput{9, 0, StepKind::Prefill, false})
                .local_path == AttentionPath::Fallback,
        "head-dim 256 must cross from vector fused to fallback at q=9");

    local_only.num_attention_heads = 8;
    local_only.num_kv_heads_local = 1;
    require(
        predict_transient(
            local_only, AdmissionInput{4, 0, StepKind::Prefill, false})
                .local_path == AttentionPath::FusedVector &&
            predict_transient(
                local_only, AdmissionInput{5, 0, StepKind::Prefill, false})
                .local_path == AttentionPath::Fallback,
        "vector fused dispatch must enforce q*gqa <= 32");

    local_only.num_attention_heads = 4;
    local_only.num_kv_heads_local = 2;
    const auto fallback_nine = predict_transient(
        local_only, AdmissionInput{9, 0, StepKind::Prefill, false});
    constexpr std::uint64_t kFallbackNineBytes =
        (4ULL * 9 * 9 * 2 + 4ULL * 9 * 256 * 2 + 9ULL * 9) * 5 / 4;
    require(fallback_nine.local_phase_bytes == kFallbackNineBytes,
        "fallback phase must include BF16 scores/output and the explicit bool mask");
    const auto fallback_ten = predict_transient(
        local_only, AdmissionInput{10, 0, StepKind::Prefill, false});
    require(fallback_ten.local_phase_bytes > fallback_nine.local_phase_bytes,
        "fallback score and mask storage must grow with q and k");

    local_only.sliding_window = 4096;
    const auto long_vector = predict_transient(
        local_only, AdmissionInput{1, 1023, StepKind::Decode, true});
    constexpr std::uint64_t kVectorRows = 4ULL * 1 * 1024;
    constexpr std::uint64_t kVectorWorkspace =
        kVectorRows * (256ULL * 2 + 2ULL * sizeof(float));
    constexpr std::uint64_t kLongVectorPhase =
        (4ULL * 1 * 256 * 2 + 1ULL * 1024 + kVectorWorkspace) * 5 / 4;
    require(long_vector.local_path == AttentionPath::FusedVector &&
            long_vector.local_phase_bytes == kLongVectorPhase,
        "long-context fused vector phase must include the pinned two-pass workspace");

    auto local_dominates = geometry;
    local_dominates.head_dim_local = 256;
    const auto local_max = predict_transient(
        local_dominates, AdmissionInput{9, 0, StepKind::Prefill, false});
    require(local_max.local_phase_bytes > local_max.global_phase_bytes &&
            local_max.attention_peak_bytes == local_max.local_phase_bytes,
        "attention peak must select local when its fallback phase is larger");

    auto epilogue_geometry = geometry;
    epilogue_geometry.vocab_size = 1'000'000;
    const auto non_final = predict_transient(
        epilogue_geometry, AdmissionInput{4, 0, StepKind::Prefill, false});
    const auto final = predict_transient(
        epilogue_geometry, AdmissionInput{4, 0, StepKind::Prefill, true});
    require(non_final.epilogue_phase_bytes == 0,
        "non-final prefill chunk must not charge the logits epilogue");
    require(final.epilogue_phase_bytes == 4ULL * 1'000'000 * 2 * 5 / 4 &&
            final.operation_peak_bytes == final.epilogue_phase_bytes,
        "final prefill must charge the full q*vocab BF16 logits phase");

    const auto fresh_lengths = predict_transient(
        geometry, AdmissionInput{5, 0, StepKind::Prefill, false});
    require(fresh_lengths.local_kv_length == 5 &&
            fresh_lengths.global_kv_length == 5,
        "fresh prefill K length must equal q for both kinds");
    const auto continuation_lengths = predict_transient(
        geometry, AdmissionInput{5, 10, StepKind::Prefill, false});
    require(continuation_lengths.local_kv_length == 13 &&
            continuation_lengths.global_kv_length == 15,
        "continuation K lengths must match assembled local and full global axes");
    const auto decode_lengths = predict_transient(
        geometry, AdmissionInput{5, 10, StepKind::Decode, true});
    require(decode_lengths.local_kv_length == 8 &&
            decode_lengths.global_kv_length == 15,
        "sequential decode must use the final q=1 step's K lengths");

    auto overflow_geometry = local_only;
    overflow_geometry.num_attention_heads =
        std::numeric_limits<std::uint32_t>::max();
    overflow_geometry.num_kv_heads_local = 1;
    overflow_geometry.head_dim_local = 512;
    const auto saturated = predict_transient(
        overflow_geometry,
        AdmissionInput{
            std::numeric_limits<std::uint32_t>::max(),
            0,
            StepKind::Prefill,
            false,
        });
    require(
        saturated.operation_peak_bytes ==
            std::numeric_limits<std::uint64_t>::max(),
        "unrepresentable transient geometry must saturate upward and fail closed");
}

} // namespace

int main() {
    test_transient_model();

    // Self-skip on CI (no GPU): the governor reads mx::get_active_memory() which
    // requires a live MLX device. If there's no GPU, exit 0 (like the other M5-gated
    // tests).
    if (!mx::is_available(mx::Device::gpu)) {
        std::cout << "governor_test: no GPU available, skipping\n";
        return 0;
    }

    const auto geometry = build_tiny_geometry();
    const auto budget = hyperion::platform::derive_budget(16ULL * 1024 * 1024 * 1024);
    Governor gov(geometry, budget.effective_bytes, budget.soft_watermark_bytes);

    // ── Constants ─────────────────────────────────────────────────────────────

    // The global KV growth rate is 16 KiB/token (documented in 05 §KV-and-memory).
    require(kGlobalKvBytesPerToken == 16 * 1024, "global KV bytes per token must be 16 KiB");

    // The workspace reserve is 512 MiB.
    require(kWorkspaceReserveBytes == 512ULL * 1024 * 1024, "workspace reserve must be 512 MiB");

    // ── Budget derivation ─────────────────────────────────────────────────────

    // The 16 GB profile ceiling is 12 GiB (the platform_policy hard clamp).
    require(budget.effective_bytes == 12ULL * 1024 * 1024 * 1024,
        "the 16 GB profile effective budget must be the 12 GiB ceiling");
    // The soft watermark is 90% of the effective budget.
    require(budget.soft_watermark_bytes == 12ULL * 1024 * 1024 * 1024 * 9 / 10,
        "the soft watermark must be 90% of the effective budget");

    // ── Throughput optimum ────────────────────────────────────────────────────

    // The throughput-optimum context is where the global KV cost reaches ~30% of
    // the budget. For a 12 GiB ceiling: 0.30 * 12 GiB / 16 KiB ≈ 245,760 tokens.
    // This is NOT the ceiling — it's the scheduling target (A4).
    const auto opt = throughput_optimum_context(budget.effective_bytes);
    require(opt > 0, "throughput optimum context must be positive");
    require(opt < 1'000'000, "throughput optimum context must be reasonable");
    // Verify the formula: 0.30 * budget / 16 KiB.
    const auto expected_opt = static_cast<std::uint32_t>(
        static_cast<double>(budget.effective_bytes) * 0.30 / kGlobalKvBytesPerToken);
    require(opt == expected_opt, "throughput optimum must match the 30% formula");

    // ── Geometry-only G4 scaffold ─────────────────────────────────────────────

    // This dense-BF16 geometry estimator is intentionally unwired from real quantized
    // admission. These assertions cover its arithmetic only; they do not claim that
    // the production 12B 8K/32K G4 evidence gate is closed.
    require(peak_within_budget(8192, geometry, budget.effective_bytes),
        "8K context must pass the geometry-only budget scaffold");
    require(peak_within_budget(32768, geometry, budget.effective_bytes),
        "32K context must pass the geometry-only budget scaffold");

    // ── Governor admission ────────────────────────────────────────────────────

    // Build a real KV state for the tiny geometry.
    const mx::Stream s = mx::default_stream(mx::Device::gpu);
    auto kvstate = build_kv_state(
        hyperion::model::build_dispatch(geometry),
        hyperion::model::kDefaultGammaMax,
        mx::bfloat16,
        s);

    constexpr std::uint64_t kGlobalBucketBytes =
        2ULL * 256 * 1 * 128 * 2; // K+V * capacity * heads * dim * BF16
    constexpr std::uint64_t kLocalStagingCowBytes =
        5ULL * 2 * 16 * 2 * 64 * 2; // layers * K+V * cap * heads * dim * BF16
    constexpr std::uint64_t kLocalLazyOriginalBytes = kLocalStagingCowBytes;
    const std::uint64_t kOneTokenTransient = predict_transient(
        geometry, AdmissionInput{1, 0, StepKind::Prefill, false})
        .operation_peak_bytes;

    // At offset 0 with 0 tokens, the prediction reports the current zero-byte global
    // allocation and charges no allocation or attention transient.
    const auto empty_zero = gov.evaluate(
        AdmissionInput{0, 0, StepKind::Prefill, false}, kvstate);
    require(empty_zero.admission == Admission::Accepted,
        "zero-token step at offset 0 must be accepted");
    require(empty_zero.predicted_peak_bytes >= kWorkspaceReserveBytes,
        "predicted peak must include the workspace reserve");
    require(empty_zero.global_kv_bytes == 0,
        "zero-token probe must report the current zero-byte global allocation");

    // The first token allocates one full 256-token K+V bucket. Persistent telemetry
    // reports that projected bucket, and admission charges the full new allocation.
    auto decision = gov.evaluate(
        AdmissionInput{1, 0, StepKind::Prefill, false}, kvstate);
    require(decision.admission == Admission::Accepted,
        "single prefill token at offset 0 must be accepted");
    require(decision.global_kv_bytes == kGlobalBucketBytes,
        "first token must project one full global KV bucket");
    require(
        decision.predicted_peak_bytes ==
            empty_zero.predicted_peak_bytes + kGlobalBucketBytes +
                kLocalStagingCowBytes + kLocalLazyOriginalBytes +
                kOneTokenTransient,
        "first allocation must charge replacement, local candidate, and lazy originals");

    Governor soft_gov(
        geometry, std::numeric_limits<std::uint64_t>::max(),
        empty_zero.predicted_peak_bytes);
    const auto soft_first_bucket = soft_gov.evaluate(
        AdmissionInput{1, 0, StepKind::Prefill, false}, kvstate);
    require(soft_first_bucket.admission == Admission::SoftPaused,
        "first bucket must soft-pause when it exceeds only the soft watermark");
    require(soft_first_bucket.global_kv_bytes == kGlobalBucketBytes,
        "soft-pause telemetry must report projected post-step global KV bytes");

    // More tokens in the same proposed bucket do not add another KV allocation;
    // only the query-width attention transient changes.
    const auto two_token_first_bucket = gov.evaluate(
        AdmissionInput{2, 0, StepKind::Prefill, false}, kvstate);
    require(two_token_first_bucket.global_kv_bytes == kGlobalBucketBytes,
        "within-bucket proposal must retain one projected global KV bucket");
    require(
        two_token_first_bucket.predicted_peak_bytes ==
            empty_zero.predicted_peak_bytes + kGlobalBucketBytes +
                kLocalStagingCowBytes + kLocalLazyOriginalBytes +
                predict_transient(
                    geometry, AdmissionInput{2, 0, StepKind::Prefill, false})
                    .operation_peak_bytes,
        "within-bucket proposal must not charge per-logical-token KV growth");

    // The lazy-original term is conditional, not a permanent doubling. Pin the
    // incremental prediction before and after explicitly materializing the same
    // fresh local inputs: settled memory rises, while the lazy surcharge disappears.
    auto materialized_originals_state = build_kv_state(
        hyperion::model::build_dispatch(geometry),
        hyperion::model::kDefaultGammaMax, mx::bfloat16, s);
    const auto lazy_originals_zero = gov.evaluate(
        AdmissionInput{0, 0, StepKind::Prefill, false},
        materialized_originals_state);
    const auto lazy_originals_growth = gov.evaluate(
        AdmissionInput{1, 0, StepKind::Prefill, false},
        materialized_originals_state);
    require(
        lazy_originals_growth.predicted_peak_bytes -
                lazy_originals_zero.predicted_peak_bytes ==
            kGlobalBucketBytes + kLocalStagingCowBytes +
                kLocalLazyOriginalBytes + kOneTokenTransient,
        "lazy fresh inputs must add retained originals to the growth increment");
    std::vector<mx::array> local_originals;
    for (const auto& cache : materialized_originals_state.local) {
        local_originals.push_back(cache.keys());
        local_originals.push_back(cache.values());
    }
    mx::eval(local_originals);
    // The exact predictor-parity checks below read MLX's live allocator counters.
    // Wait for this materialization and its allocator bookkeeping to settle first;
    // otherwise a cold Metal process can observe different baselines in two
    // back-to-back predictions even though their shape accounting is identical.
    mx::synchronize(s);
    for (const auto& cache : materialized_originals_state.local) {
        require(cache.keys().is_available() && cache.values().is_available(),
            "explicit local-original eval must make every retained input available");
    }
    const auto available_originals_zero = gov.evaluate(
        AdmissionInput{0, 0, StepKind::Prefill, false},
        materialized_originals_state);
    const auto available_originals_growth = gov.evaluate(
        AdmissionInput{1, 0, StepKind::Prefill, false},
        materialized_originals_state);
    require(
        available_originals_growth.predicted_peak_bytes -
                available_originals_zero.predicted_peak_bytes ==
            kGlobalBucketBytes + kLocalStagingCowBytes + kOneTokenTransient,
        "available originals must be represented by settled memory, not double charged");

    // Decode appends sequential q=1 forwards. Crossing three capacity steps therefore
    // allocates replacement sizes 1 + 2 + 3 buckets, while persistent telemetry reports
    // only the final three-bucket allocation.
    auto multistep_kvstate = build_kv_state(
        hyperion::model::build_dispatch(geometry),
        hyperion::model::kDefaultGammaMax,
        mx::bfloat16,
        s);
    const auto multistep_zero = gov.evaluate(
        AdmissionInput{0, 0, StepKind::Decode, false}, multistep_kvstate);
    require(
        multistep_zero.predicted_peak_bytes ==
            gov.evaluate(
                AdmissionInput{0, 0, StepKind::Prefill, false},
                multistep_kvstate).predicted_peak_bytes,
        "zero-token decode must retain a zero-width attention transient");
    const auto multistep = gov.evaluate(
        AdmissionInput{513, 0, StepKind::Decode, true}, multistep_kvstate);
    require(multistep.global_kv_bytes == 3 * kGlobalBucketBytes,
        "513-token proposal must project three global KV buckets");
    require(
        multistep.predicted_peak_bytes ==
            multistep_zero.predicted_peak_bytes + 6 * kGlobalBucketBytes +
                kLocalStagingCowBytes + kLocalLazyOriginalBytes +
                predict_transient(
                    geometry, AdmissionInput{513, 0, StepKind::Decode, true})
                    .operation_peak_bytes,
        "multi-step decode must charge sequential replacements and one q=1 transient");
    require(
        predict_peak(
            AdmissionInput{513, 0, StepKind::Decode, true},
            multistep_kvstate,
            geometry) ==
            multistep.predicted_peak_bytes,
        "predict_peak must use the same sequential decode replacement sum");

    // Prefill appends the same 513 tokens once and therefore allocates only the final
    // three-bucket replacement. This StepKind distinction is load-bearing.
    const auto multistep_prefill = gov.evaluate(
        AdmissionInput{513, 0, StepKind::Prefill, false}, multistep_kvstate);
    require(multistep_prefill.global_kv_bytes == 3 * kGlobalBucketBytes,
        "513-token prefill must project the same final three-bucket allocation");
    require(
        multistep_prefill.predicted_peak_bytes ==
            multistep_zero.predicted_peak_bytes + 3 * kGlobalBucketBytes +
                kLocalStagingCowBytes + kLocalLazyOriginalBytes +
                predict_transient(
                    geometry, AdmissionInput{513, 0, StepKind::Prefill, false})
                    .operation_peak_bytes,
        "multi-step prefill must charge only its one final replacement allocation");

    // A whole-public-call plan controls the shallow transaction and its staging/COW
    // lifetime, but replacement projection remains the current chunk's derived plan.
    // The first token of a 513-token operation must therefore project one bucket,
    // never pre-charge the operation's final three-bucket global capacity.
    const auto whole_prefill_plan =
        hyperion::model::plan_kv_growth(multistep_kvstate, 513);
    const auto first_chunk_with_whole_plan = gov.evaluate(
        AdmissionInput{1, 0, StepKind::Prefill, false},
        multistep_kvstate,
        &whole_prefill_plan);
    const auto first_chunk_derived = gov.evaluate(
        AdmissionInput{1, 0, StepKind::Prefill, false},
        multistep_kvstate);
    require(first_chunk_with_whole_plan.global_kv_bytes == kGlobalBucketBytes,
        "whole-operation staging must not replace per-chunk global growth projection");
    require(
        first_chunk_with_whole_plan.predicted_peak_bytes ==
            first_chunk_derived.predicted_peak_bytes,
        "whole-operation and current-chunk growth plans must retain distinct lifetimes");

    // An unavailable non-empty global input is retained by the replacement graph
    // even at an exact boundary. It is not yet represented by settled memory, so
    // admission must charge its current K+V in addition to the replacement.
    auto lazy_global_state = build_kv_state(
        hyperion::model::build_dispatch(geometry),
        hyperion::model::kDefaultGammaMax, mx::bfloat16, s);
    const mx::array lazy_bucket = mx::ones({256, 1, 128}, mx::bfloat16, s);
    lazy_global_state.global[0].append(lazy_bucket, lazy_bucket, 256);
    require(!lazy_global_state.global[0].keys().is_available() &&
            !lazy_global_state.global[0].values().is_available(),
        "lazy global negative control must remain unavailable before admission");
    const auto lazy_global_zero = gov.evaluate(
        AdmissionInput{0, 256, StepKind::Decode, false}, lazy_global_state);
    const auto lazy_global_growth = gov.evaluate(
        AdmissionInput{1, 256, StepKind::Decode, true}, lazy_global_state);
    require(
        lazy_global_growth.predicted_peak_bytes -
                lazy_global_zero.predicted_peak_bytes ==
            2 * kGlobalBucketBytes + kGlobalBucketBytes +
                kLocalStagingCowBytes + kLocalLazyOriginalBytes +
                predict_transient(
                    geometry, AdmissionInput{1, 256, StepKind::Decode, true})
                    .operation_peak_bytes,
        "unavailable old global K+V must be charged at exact growth");

    // Multi-token continuation prefill assembles a bounded local K+V buffer from the
    // retained prefix plus the current chunk. It coexists with one local attention
    // phase; neither component is multiplied by the number of sliding layers.
    const auto fresh_transient = predict_transient(
        geometry, AdmissionInput{2, 0, StepKind::Prefill, false});
    const auto continuation_transient = predict_transient(
        geometry, AdmissionInput{2, 8, StepKind::Prefill, false});
    constexpr std::uint64_t kExpectedContinuationCharge =
        2ULL * (8 + 2) * 2 * 64 * 2; // K+V * len * kv_heads * dim * bf16
    require(
        continuation_transient.local_phase_bytes ==
            (4ULL * 2 * 64 * 2 + 2ULL * (8 + 2)) * 5 / 4 +
                kExpectedContinuationCharge,
        "continuation local phase must coexist with one assembled K+V scratch pair");
    require(continuation_transient.local_phase_bytes > fresh_transient.local_phase_bytes,
        "continuation local phase must grow beyond fresh-prefill attention");
    constexpr std::uint64_t kExpectedPartialPrefixCharge =
        2ULL * (4 + 2) * 2 * 64 * 2;
    const auto partial_transient = predict_transient(
        geometry, AdmissionInput{2, 4, StepKind::Prefill, false});
    require(
        partial_transient.local_phase_bytes ==
            (4ULL * 2 * 64 * 2 + 2ULL * (4 + 2)) * 5 / 4 +
                kExpectedPartialPrefixCharge,
        "continuation scratch must use the available prefix below the window");
    require(
        predict_transient(
            geometry, AdmissionInput{1, 12, StepKind::Prefill, false})
                .local_phase_bytes ==
            (4ULL * 1 * 64 * 2 + 1ULL * geometry.sliding_window) * 5 / 4,
        "single-token prefill must not charge continuation-prefill scratch");
    require(
        predict_transient(
            geometry, AdmissionInput{2, 8, StepKind::Decode, true})
                .global_kv_length == 10 &&
            predict_transient(
                geometry, AdmissionInput{1, 8, StepKind::Decode, true})
                .global_kv_length == 9,
        "multi-token decode peak must use the final sequential q=1 K length");

    // ── Halve-chunk behavior ──────────────────────────────────────────────────

    // A very large direct proposal must signal pressure rather than silently pass.
    decision = gov.evaluate(
        AdmissionInput{1'000'000, 0, StepKind::Prefill, false}, kvstate);
    require(decision.admission != Admission::Accepted ||
            decision.predicted_peak_bytes <= budget.soft_watermark_bytes,
        "a 1M-token chunk must not be accepted if it breaches the soft watermark");

    const auto prefill_one = gov.evaluate(
        AdmissionInput{1, 0, StepKind::Prefill, false}, kvstate);
    const auto prefill_two = gov.evaluate(
        AdmissionInput{2, 0, StepKind::Prefill, false}, kvstate);

    Governor hard_then_fit(
        geometry,
        prefill_two.predicted_peak_bytes,
        prefill_two.predicted_peak_bytes);
    const auto hard_fit = hard_then_fit.admit_prefill_to_fit(
        8, 0, 16, kvstate);
    require(hard_fit.n_tokens == 2 &&
            hard_fit.decision.admission == Admission::Accepted &&
            hard_fit.attempt_count == 3,
        "hard pressure above one token must repeatedly halve until a shape fits");
    require(hard_fit.attempts[0].admission == Admission::HardRejected &&
            hard_fit.attempts[1].admission == Admission::HardRejected,
        "every oversized hard proposal must be retained as a shrink attempt");

    Governor soft_to_one(
        geometry,
        std::numeric_limits<std::uint64_t>::max(),
        prefill_one.predicted_peak_bytes - 1);
    const auto soft_fit = soft_to_one.admit_prefill_to_fit(
        8, 0, 16, kvstate);
    require(soft_fit.n_tokens == 1 &&
            soft_fit.decision.admission == Admission::SoftPaused &&
            soft_fit.attempt_count == 4,
        "repeated soft pressure must reach an executable advisory one-token shape");

    Governor hard_at_one(
        geometry,
        prefill_one.predicted_peak_bytes - 1,
        prefill_one.predicted_peak_bytes - 1);
    const auto hard_reject = hard_at_one.admit_prefill_to_fit(
        4, 0, 16, kvstate);
    require(hard_reject.n_tokens == 1 &&
            hard_reject.decision.admission == Admission::HardRejected &&
            hard_reject.attempt_count == 3,
        "prefill may terminally reject only after the one-token shape is hard over budget");

    auto large_vocab_geometry = geometry;
    large_vocab_geometry.vocab_size = 1'000'000;
    Governor large_vocab_probe(
        large_vocab_geometry,
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max());
    const auto non_final_two = large_vocab_probe.evaluate(
        AdmissionInput{2, 0, StepKind::Prefill, false}, kvstate);
    Governor final_chunk_shrink(
        large_vocab_geometry,
        non_final_two.predicted_peak_bytes,
        non_final_two.predicted_peak_bytes);
    const auto final_recheck = final_chunk_shrink.admit_prefill_to_fit(
        4, 0, 4, kvstate);
    require(final_recheck.n_tokens == 2 &&
            final_recheck.decision.admission == Admission::Accepted &&
            final_recheck.attempt_count == 2,
        "halving a rejected final chunk must re-evaluate the smaller shape as non-final");

    // ── Telemetry fields ──────────────────────────────────────────────────────

    // Allocate the first bucket with four committed tokens, then verify another token
    // inside that already-allocated bucket charges no KV allocation transient.
    {
        const std::uint32_t n = 4;
        const mx::array k_up = mx::ones({n, 1, 128}, mx::bfloat16, s);
        const mx::array v_up = mx::ones({n, 1, 128}, mx::bfloat16, s);
        kvstate.global[0].append(k_up, v_up, n);
        mx::eval(kvstate.global[0].keys(), kvstate.global[0].values());
        mx::synchronize(s);
    }
    const auto four_token_zero = gov.evaluate(
        AdmissionInput{0, 4, StepKind::Decode, false}, kvstate);
    const auto within_allocated_bucket = gov.evaluate(
        AdmissionInput{1, 4, StepKind::Decode, true}, kvstate);
    require(within_allocated_bucket.global_kv_bytes == kGlobalBucketBytes,
        "within allocated bucket must retain current persistent global KV bytes");
    require(
        within_allocated_bucket.predicted_peak_bytes ==
            four_token_zero.predicted_peak_bytes +
                predict_transient(
                    geometry, AdmissionInput{1, 4, StepKind::Decode, true})
                    .operation_peak_bytes,
        "within allocated bucket must charge zero KV allocation transient");

    // A sequential block that starts inside the bucket performs staged no-growth
    // writes before its later crossing. Charge the current-capacity COW candidate
    // in addition to the eventual two-bucket replacement. One-shot prefill grows
    // before writing and therefore has no analogous current-capacity candidate.
    const auto later_decode_crossing = gov.evaluate(
        AdmissionInput{253, 4, StepKind::Decode, true}, kvstate);
    require(
        later_decode_crossing.predicted_peak_bytes ==
            four_token_zero.predicted_peak_bytes + 3 * kGlobalBucketBytes +
                kLocalStagingCowBytes + kLocalLazyOriginalBytes +
                predict_transient(
                    geometry, AdmissionInput{253, 4, StepKind::Decode, true})
                    .operation_peak_bytes,
        "later decode crossing must charge current candidate plus replacement and local COW");
    const auto one_shot_prefill_crossing = gov.evaluate(
        AdmissionInput{253, 4, StepKind::Prefill, false}, kvstate);
    require(
        one_shot_prefill_crossing.predicted_peak_bytes ==
            four_token_zero.predicted_peak_bytes + 2 * kGlobalBucketBytes +
                kLocalStagingCowBytes + kLocalLazyOriginalBytes +
                predict_transient(
                    geometry, AdmissionInput{253, 4, StepKind::Prefill, false})
                    .operation_peak_bytes,
        "one-shot prefill crossing must not charge decode's current-capacity candidate");

    // Fill the remainder to the exact end of the first 256-token bucket.
    {
        const std::uint32_t n = 252;
        const mx::array k_up = mx::ones({n, 1, 128}, mx::bfloat16, s);
        const mx::array v_up = mx::ones({n, 1, 128}, mx::bfloat16, s);
        kvstate.global[0].append(k_up, v_up, n);
        mx::eval(kvstate.global[0].keys(), kvstate.global[0].values());
        mx::synchronize(s);
    }

    // At committed==capacity, zero tokens retain the current bucket. The next token
    // crosses the exact boundary and projects a two-bucket persistent allocation.
    const auto boundary_zero = gov.evaluate(
        AdmissionInput{0, 256, StepKind::Decode, false}, kvstate);
    require(boundary_zero.global_kv_bytes == kGlobalBucketBytes,
        "zero-token probe at an exact boundary must retain current capacity");
    const auto boundary_crossing = gov.evaluate(
        AdmissionInput{1, 256, StepKind::Decode, true}, kvstate);
    require(boundary_crossing.global_kv_bytes == 2 * kGlobalBucketBytes,
        "first token beyond a full bucket must project the next capacity step");
    require(
        boundary_crossing.predicted_peak_bytes ==
            boundary_zero.predicted_peak_bytes + 2 * kGlobalBucketBytes +
                kLocalStagingCowBytes + kLocalLazyOriginalBytes +
                predict_transient(
                    geometry, AdmissionInput{1, 256, StepKind::Decode, true})
                    .operation_peak_bytes,
        "256 to 257 must charge replacement K+V plus local staging COW");

    // Starting at one full bucket, 513 sequential decode tokens cross replacements
    // of 2 + 3 + 4 buckets and finish with four persistent buckets.
    const auto boundary_multicross = gov.evaluate(
        AdmissionInput{513, 256, StepKind::Decode, true}, kvstate);
    require(boundary_multicross.global_kv_bytes == 4 * kGlobalBucketBytes,
        "boundary multi-cross decode must project four persistent buckets");
    require(
        boundary_multicross.predicted_peak_bytes ==
            boundary_zero.predicted_peak_bytes + 9 * kGlobalBucketBytes +
                kLocalStagingCowBytes + kLocalLazyOriginalBytes +
                predict_transient(
                    geometry, AdmissionInput{513, 256, StepKind::Decode, true})
                    .operation_peak_bytes,
        "boundary multi-cross decode must charge replacements plus local staging COW");

    // A ceiling that would admit delta-only accounting must reject full replacement
    // accounting. Old-buffer bytes are already part of settled MLX memory.
    const std::uint64_t delta_only_ceiling =
        boundary_zero.predicted_peak_bytes + kGlobalBucketBytes +
            kLocalStagingCowBytes + kLocalLazyOriginalBytes +
            predict_transient(
                geometry, AdmissionInput{1, 256, StepKind::Decode, true})
                .operation_peak_bytes;
    Governor tight_gov(geometry, delta_only_ceiling, delta_only_ceiling);
    const auto replacement_rejection = tight_gov.evaluate(
        AdmissionInput{1, 256, StepKind::Decode, true}, kvstate);
    require(replacement_rejection.admission == Admission::HardRejected,
        "full replacement allocation must reject a ceiling that delta-only charging would admit");
    require(replacement_rejection.global_kv_bytes == 2 * kGlobalBucketBytes,
        "hard rejection telemetry must still report projected post-step global KV bytes");

    // If one global layer grows, another non-growing global layer is still staged
    // and loses donation on its first rebind. Pin that full current K+V candidate.
    auto two_global_geometry = geometry;
    two_global_geometry.layer_types.push_back(LayerType::Full);
    two_global_geometry.num_hidden_layers += 1;
    auto two_global_state = build_kv_state(
        hyperion::model::build_dispatch(two_global_geometry),
        hyperion::model::kDefaultGammaMax, mx::bfloat16, s);
    const mx::array full_bucket = mx::ones({256, 1, 128}, mx::bfloat16, s);
    const mx::array one_token = mx::ones({1, 1, 128}, mx::bfloat16, s);
    two_global_state.global[0].append(full_bucket, full_bucket, 256);
    two_global_state.global[1].append(one_token, one_token, 1);
    mx::eval(
        two_global_state.global[0].keys(), two_global_state.global[0].values(),
        two_global_state.global[1].keys(), two_global_state.global[1].values());
    Governor two_global_gov(
        two_global_geometry, budget.effective_bytes, budget.soft_watermark_bytes);
    const auto two_global_zero = two_global_gov.evaluate(
        AdmissionInput{0, 256, StepKind::Decode, false}, two_global_state);
    const auto one_grows = two_global_gov.evaluate(
        AdmissionInput{1, 256, StepKind::Decode, true}, two_global_state);
    const std::uint64_t kTwoGlobalOneTokenTransient = predict_transient(
        two_global_geometry,
        AdmissionInput{1, 256, StepKind::Decode, true})
        .operation_peak_bytes;
    require(kTwoGlobalOneTokenTransient == predict_transient(
            geometry, AdmissionInput{1, 256, StepKind::Decode, true})
            .operation_peak_bytes,
        "duplicating a global layer must not change the per-phase transient");
    require(
        one_grows.predicted_peak_bytes ==
            two_global_zero.predicted_peak_bytes + 2 * kGlobalBucketBytes +
                kGlobalBucketBytes + kLocalStagingCowBytes +
                kLocalLazyOriginalBytes + kTwoGlobalOneTokenTransient,
        "growth transaction must charge a non-growing global layer's full K+V COW");

    // Proposals that exceed uint32 committed length / signed-int MLX capacity fail
    // closed instead of wrapping into a small capacity prediction.
    const auto overflow_rejection = gov.evaluate(
        AdmissionInput{
            std::numeric_limits<std::uint32_t>::max(),
            256,
            StepKind::Decode,
            true,
        },
        kvstate);
    require(overflow_rejection.admission == Admission::HardRejected,
        "unrepresentable capacity proposal must fail closed");
    require(
        overflow_rejection.predicted_peak_bytes ==
            std::numeric_limits<std::uint64_t>::max(),
        "unrepresentable capacity proposal must saturate peak upward");
    require(
        overflow_rejection.global_kv_bytes ==
            std::numeric_limits<std::uint64_t>::max(),
        "unrepresentable capacity proposal must saturate projected telemetry upward");

    // MLX reads MLX_SDPA_BLOCKS at attention evaluation time. A value that appears
    // after model construction must therefore reject at each admission, not rely
    // solely on the model-load environment check.
    require(std::getenv("MLX_SDPA_BLOCKS") == nullptr,
        "governor test requires a clean MLX_SDPA_BLOCKS environment");
    require(::setenv("MLX_SDPA_BLOCKS", "1048576", 1) == 0,
        "governor test must install the runtime override negative control");
    const auto runtime_override_rejection = gov.evaluate(
        AdmissionInput{1, 1023, StepKind::Decode, true}, kvstate);
    require(runtime_override_rejection.admission == Admission::HardRejected &&
            runtime_override_rejection.predicted_peak_bytes ==
                std::numeric_limits<std::uint64_t>::max(),
        "a post-load MLX_SDPA_BLOCKS override must fail admission closed");
    require(::unsetenv("MLX_SDPA_BLOCKS") == 0,
        "governor test must restore the runtime environment");

    decision = boundary_crossing;

    // The decision must report the budget ceiling + soft watermark.
    require(decision.budget_ceiling_bytes == budget.effective_bytes,
        "decision must report the budget ceiling");
    require(decision.soft_watermark_bytes == budget.soft_watermark_bytes,
        "decision must report the soft watermark");

    // The decision must report local + global KV byte counts.
    require(decision.local_kv_bytes > 0, "local KV bytes must be non-zero (5 sliding layers)");
    require(decision.global_kv_bytes == 2 * kGlobalBucketBytes,
        "decision must report projected post-step global KV bytes");

    // ── predict_peak standalone ───────────────────────────────────────────────

    // predict_peak must match Governor::evaluate's predicted_peak_bytes.
    const auto standalone = predict_peak(
        AdmissionInput{1, 0, StepKind::Prefill, false}, kvstate, geometry);
    decision = gov.evaluate(
        AdmissionInput{1, 0, StepKind::Prefill, false}, kvstate);
    require(standalone == decision.predicted_peak_bytes,
        "predict_peak must match Governor::evaluate's predicted_peak_bytes");

    const auto continuation_standalone = predict_peak(
        AdmissionInput{2, 8, StepKind::Prefill, false}, kvstate, geometry);
    const auto continuation_decision = gov.evaluate(
        AdmissionInput{2, 8, StepKind::Prefill, false}, kvstate);
    require(continuation_standalone == continuation_decision.predicted_peak_bytes,
        "predict_peak must include the same continuation-prefill scratch charge");

    std::cout << "governor_test: all assertions passed\n";
    return 0;
}
