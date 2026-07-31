#include "governor.h"
#include "geometry.h"
#include "kv_cache.h"
#include "platform_policy.h"

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
using hyperion::governor::Governor;
using hyperion::governor::GovernorDecision;
using hyperion::governor::StepKind;
using hyperion::governor::kGlobalKvBytesPerToken;
using hyperion::governor::kWorkspaceReserveBytes;
using hyperion::governor::peak_within_budget;
using hyperion::governor::predict_peak;
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

} // namespace

int main() {
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

    // ── G4 gate ───────────────────────────────────────────────────────────────

    // At 8K context, the predicted peak must be within budget (the G4 gate passes).
    require(peak_within_budget(8192, geometry, budget.effective_bytes),
        "8K context must pass the G4 peak-≤-budget gate");
    // At 32K context, the predicted peak must also be within budget (the G4 gate passes).
    require(peak_within_budget(32768, geometry, budget.effective_bytes),
        "32K context must pass the G4 peak-≤-budget gate");

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
    constexpr std::uint64_t kOneTokenTransient =
        (5ULL * 1 * 64 * 4 * 2 + 1ULL * 1 * 128 * 4 * 2) * 5 / 4;

    // At offset 0 with 0 tokens, the prediction reports the current zero-byte global
    // allocation and charges no allocation or attention transient.
    const auto empty_zero = gov.evaluate(0, 0, kvstate, StepKind::Prefill);
    require(empty_zero.admission == Admission::Accepted,
        "zero-token step at offset 0 must be accepted");
    require(empty_zero.predicted_peak_bytes >= kWorkspaceReserveBytes,
        "predicted peak must include the workspace reserve");
    require(empty_zero.global_kv_bytes == 0,
        "zero-token probe must report the current zero-byte global allocation");

    // The first token allocates one full 256-token K+V bucket. Persistent telemetry
    // reports that projected bucket, and admission charges the full new allocation.
    auto decision = gov.evaluate(1, 0, kvstate, StepKind::Prefill);
    require(decision.admission == Admission::Accepted,
        "single prefill token at offset 0 must be accepted");
    require(decision.global_kv_bytes == kGlobalBucketBytes,
        "first token must project one full global KV bucket");
    require(
        decision.predicted_peak_bytes ==
            empty_zero.predicted_peak_bytes + kGlobalBucketBytes + kOneTokenTransient,
        "first allocation must charge the full projected global KV bucket");

    Governor soft_gov(
        geometry, std::numeric_limits<std::uint64_t>::max(),
        empty_zero.predicted_peak_bytes);
    const auto soft_first_bucket =
        soft_gov.evaluate(1, 0, kvstate, StepKind::Prefill);
    require(soft_first_bucket.admission == Admission::SoftPaused,
        "first bucket must soft-pause when it exceeds only the soft watermark");
    require(soft_first_bucket.global_kv_bytes == kGlobalBucketBytes,
        "soft-pause telemetry must report projected post-step global KV bytes");

    // More tokens in the same proposed bucket do not add another KV allocation;
    // only the query-width attention transient changes.
    const auto two_token_first_bucket = gov.evaluate(2, 0, kvstate, StepKind::Prefill);
    require(two_token_first_bucket.global_kv_bytes == kGlobalBucketBytes,
        "within-bucket proposal must retain one projected global KV bucket");
    require(
        two_token_first_bucket.predicted_peak_bytes ==
            empty_zero.predicted_peak_bytes + kGlobalBucketBytes + 2 * kOneTokenTransient,
        "within-bucket proposal must not charge per-logical-token KV growth");

    // Decode appends sequential q=1 forwards. Crossing three capacity steps therefore
    // allocates replacement sizes 1 + 2 + 3 buckets, while persistent telemetry reports
    // only the final three-bucket allocation.
    auto multistep_kvstate = build_kv_state(
        hyperion::model::build_dispatch(geometry),
        hyperion::model::kDefaultGammaMax,
        mx::bfloat16,
        s);
    const auto multistep_zero = gov.evaluate(0, 0, multistep_kvstate, StepKind::Decode);
    require(
        multistep_zero.predicted_peak_bytes ==
            gov.evaluate(0, 0, multistep_kvstate, StepKind::Prefill).predicted_peak_bytes,
        "zero-token decode must retain a zero-width attention transient");
    const auto multistep = gov.evaluate(513, 0, multistep_kvstate, StepKind::Decode);
    require(multistep.global_kv_bytes == 3 * kGlobalBucketBytes,
        "513-token proposal must project three global KV buckets");
    require(
        multistep.predicted_peak_bytes ==
            multistep_zero.predicted_peak_bytes + 6 * kGlobalBucketBytes +
                kOneTokenTransient,
        "multi-step decode must charge sequential replacements and one q=1 transient");
    require(
        predict_peak(513, 0, multistep_kvstate, geometry, StepKind::Decode) ==
            multistep.predicted_peak_bytes,
        "predict_peak must use the same sequential decode replacement sum");

    // Prefill appends the same 513 tokens once and therefore allocates only the final
    // three-bucket replacement. This StepKind distinction is load-bearing.
    const auto multistep_prefill =
        gov.evaluate(513, 0, multistep_kvstate, StepKind::Prefill);
    require(multistep_prefill.global_kv_bytes == 3 * kGlobalBucketBytes,
        "513-token prefill must project the same final three-bucket allocation");
    require(
        multistep_prefill.predicted_peak_bytes ==
            multistep_zero.predicted_peak_bytes + 3 * kGlobalBucketBytes +
                513 * kOneTokenTransient,
        "multi-step prefill must charge only its one final replacement allocation");

    // Multi-token continuation prefill assembles a bounded local K+V buffer from the
    // retained prefix plus the current chunk. At offset 8 the tiny fixture retains the
    // full window, so q=2 charges exactly 5 KiB once (not once per sliding layer).
    const auto fresh_prefill = gov.evaluate(2, 0, kvstate, StepKind::Prefill);
    const auto continuation_prefill = gov.evaluate(2, 8, kvstate, StepKind::Prefill);
    constexpr std::uint64_t kExpectedContinuationCharge =
        2ULL * (8 + 2) * 2 * 64 * 2; // K+V * len * kv_heads * dim * bf16
    require(
        continuation_prefill.predicted_peak_bytes ==
            fresh_prefill.predicted_peak_bytes + kExpectedContinuationCharge,
        "continuation prefill must charge the assembled local K+V scratch");
    constexpr std::uint64_t kExpectedPartialPrefixCharge =
        2ULL * (4 + 2) * 2 * 64 * 2;
    require(
        gov.evaluate(2, 4, kvstate, StepKind::Prefill).predicted_peak_bytes ==
            fresh_prefill.predicted_peak_bytes + kExpectedPartialPrefixCharge,
        "continuation scratch must use the available prefix below the window");
    require(
        gov.evaluate(1, 12, kvstate, StepKind::Prefill).predicted_peak_bytes ==
            gov.evaluate(1, 0, kvstate, StepKind::Prefill).predicted_peak_bytes,
        "single-token prefill must not charge continuation-prefill scratch");
    require(
        gov.evaluate(2, 8, kvstate, StepKind::Decode).predicted_peak_bytes ==
            gov.evaluate(1, 8, kvstate, StepKind::Decode).predicted_peak_bytes,
        "multi-token decode must retain the q=1 transient execution shape");

    // ── Halve-chunk behavior ──────────────────────────────────────────────────

    // A very large prefill chunk should trigger SoftPaused (halve-chunk) or
    // HardRejected if even 1 token breaches the ceiling. For the tiny geometry
    // on the 12 GiB ceiling, a 1M-token chunk should at least SoftPause.
    decision = gov.evaluate(1'000'000, 0, kvstate, StepKind::Prefill);
    require(decision.admission != Admission::Accepted ||
            decision.predicted_peak_bytes <= budget.soft_watermark_bytes,
        "a 1M-token chunk must not be accepted if it breaches the soft watermark");

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
    const auto four_token_zero = gov.evaluate(0, 4, kvstate, StepKind::Decode);
    const auto within_allocated_bucket = gov.evaluate(1, 4, kvstate, StepKind::Decode);
    require(within_allocated_bucket.global_kv_bytes == kGlobalBucketBytes,
        "within allocated bucket must retain current persistent global KV bytes");
    require(
        within_allocated_bucket.predicted_peak_bytes ==
            four_token_zero.predicted_peak_bytes + kOneTokenTransient,
        "within allocated bucket must charge zero KV allocation transient");

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
    const auto boundary_zero = gov.evaluate(0, 256, kvstate, StepKind::Decode);
    require(boundary_zero.global_kv_bytes == kGlobalBucketBytes,
        "zero-token probe at an exact boundary must retain current capacity");
    const auto boundary_crossing = gov.evaluate(1, 256, kvstate, StepKind::Decode);
    require(boundary_crossing.global_kv_bytes == 2 * kGlobalBucketBytes,
        "first token beyond a full bucket must project the next capacity step");
    require(
        boundary_crossing.predicted_peak_bytes ==
            boundary_zero.predicted_peak_bytes + 2 * kGlobalBucketBytes +
                kOneTokenTransient,
        "bucket growth must charge full replacement K+V, not only the capacity delta");

    // Starting at one full bucket, 513 sequential decode tokens cross replacements
    // of 2 + 3 + 4 buckets and finish with four persistent buckets.
    const auto boundary_multicross =
        gov.evaluate(513, 256, kvstate, StepKind::Decode);
    require(boundary_multicross.global_kv_bytes == 4 * kGlobalBucketBytes,
        "boundary multi-cross decode must project four persistent buckets");
    require(
        boundary_multicross.predicted_peak_bytes ==
            boundary_zero.predicted_peak_bytes + 9 * kGlobalBucketBytes +
                kOneTokenTransient,
        "boundary multi-cross decode must charge 2+3+4 replacements and one q=1 transient");

    // A ceiling that would admit delta-only accounting must reject full replacement
    // accounting. Old-buffer bytes are already part of settled MLX memory.
    const std::uint64_t delta_only_ceiling =
        boundary_zero.predicted_peak_bytes + kGlobalBucketBytes + kOneTokenTransient;
    Governor tight_gov(geometry, delta_only_ceiling, delta_only_ceiling);
    const auto replacement_rejection =
        tight_gov.evaluate(1, 256, kvstate, StepKind::Decode);
    require(replacement_rejection.admission == Admission::HardRejected,
        "full replacement allocation must reject a ceiling that delta-only charging would admit");
    require(replacement_rejection.global_kv_bytes == 2 * kGlobalBucketBytes,
        "hard rejection telemetry must still report projected post-step global KV bytes");

    // Proposals that exceed uint32 committed length / signed-int MLX capacity fail
    // closed instead of wrapping into a small capacity prediction.
    const auto overflow_rejection = gov.evaluate(
        std::numeric_limits<std::uint32_t>::max(), 256, kvstate, StepKind::Decode);
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
    const auto standalone = predict_peak(1, 0, kvstate, geometry, StepKind::Prefill);
    decision = gov.evaluate(1, 0, kvstate, StepKind::Prefill);
    require(standalone == decision.predicted_peak_bytes,
        "predict_peak must match Governor::evaluate's predicted_peak_bytes");

    const auto continuation_standalone =
        predict_peak(2, 8, kvstate, geometry, StepKind::Prefill);
    const auto continuation_decision =
        gov.evaluate(2, 8, kvstate, StepKind::Prefill);
    require(continuation_standalone == continuation_decision.predicted_peak_bytes,
        "predict_peak must include the same continuation-prefill scratch charge");

    std::cout << "governor_test: all assertions passed\n";
    return 0;
}
