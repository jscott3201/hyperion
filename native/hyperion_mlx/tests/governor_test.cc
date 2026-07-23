#include "governor.h"
#include "geometry.h"
#include "kv_cache.h"
#include "platform_policy.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
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

    // At offset 0 with 0 tokens, the predicted peak should be the settled working set
    // + workspace reserve (no KV growth, no transient).
    auto decision = gov.evaluate(0, 0, kvstate);
    require(decision.admission == Admission::Accepted,
        "zero-token step at offset 0 must be accepted");
    require(decision.predicted_peak_bytes >= kWorkspaceReserveBytes,
        "predicted peak must include the workspace reserve");

    // A single decode token at offset 0 should be accepted (minimal growth).
    decision = gov.evaluate(1, 0, kvstate);
    require(decision.admission == Admission::Accepted,
        "single decode token at offset 0 must be accepted");

    // The predicted peak must include the 16 KiB/token KV growth.
    const auto prev_peak = decision.predicted_peak_bytes;
    decision = gov.evaluate(2, 0, kvstate);
    require(decision.predicted_peak_bytes >= prev_peak + kGlobalKvBytesPerToken,
        "predicted peak must grow by at least 16 KiB per token");

    // ── Halve-chunk behavior ──────────────────────────────────────────────────

    // A very large prefill chunk should trigger SoftPaused (halve-chunk) or
    // HardRejected if even 1 token breaches the ceiling. For the tiny geometry
    // on the 12 GiB ceiling, a 1M-token chunk should at least SoftPause.
    decision = gov.evaluate(1'000'000, 0, kvstate);
    require(decision.admission != Admission::Accepted ||
            decision.predicted_peak_bytes <= budget.soft_watermark_bytes,
        "a 1M-token chunk must not be accepted if it breaches the soft watermark");

    // ── Telemetry fields ──────────────────────────────────────────────────────

    // Populate the KV state with a few tokens so the global cache has allocated
    // buffers (it starts at capacity 0 and grows in 256-token steps).
    {
        const std::uint32_t n = 4;
        const mx::array k_up = mx::ones({n, 1, 128}, mx::bfloat16, s);
        const mx::array v_up = mx::ones({n, 1, 128}, mx::bfloat16, s);
        kvstate.global[0].append(k_up, v_up, n);
        mx::eval(k_up, v_up); // MLX lazy-graph materialization (NOT Python/JS eval)
        mx::synchronize(s);
    }

    // Re-evaluate at offset 4 so the governor reads the populated KV state.
    decision = gov.evaluate(1, 4, kvstate);

    // The decision must report the budget ceiling + soft watermark.
    require(decision.budget_ceiling_bytes == budget.effective_bytes,
        "decision must report the budget ceiling");
    require(decision.soft_watermark_bytes == budget.soft_watermark_bytes,
        "decision must report the soft watermark");

    // The decision must report local + global KV byte counts.
    require(decision.local_kv_bytes > 0, "local KV bytes must be non-zero (5 sliding layers)");
    require(decision.global_kv_bytes > 0, "global KV bytes must be non-zero (1 global layer)");

    // ── predict_peak standalone ───────────────────────────────────────────────

    // predict_peak must match Governor::evaluate's predicted_peak_bytes.
    const auto standalone = predict_peak(1, 0, kvstate, geometry);
    decision = gov.evaluate(1, 0, kvstate);
    require(standalone == decision.predicted_peak_bytes,
        "predict_peak must match Governor::evaluate's predicted_peak_bytes");

    std::cout << "governor_test: all assertions passed\n";
    return 0;
}
