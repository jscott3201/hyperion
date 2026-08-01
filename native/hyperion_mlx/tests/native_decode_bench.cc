// native_decode_bench — a LEAN, non-ledger decode throughput measurement for the
// M2-2.6d "decode within 0.9x of M1 stock" gate (10-milestones-and-gates.md:35).
// NOT ledger-compliant (no A-C-C-A, no manifest SHAs, no controller handshake) — a
// defensible 0.9x SIGNAL only. The ledger-grade arm is a dedicated future slice
// (BENCHMARKS.md HYP-M1-DEFER-001).
//
// Loads the REAL 12B via the native ForwardPass (same path as forward_12b_decode_test),
// runs prefill (offset 0) + N-1 decode steps, and records token_offsets_ns (monotonic,
// per-token materialization — mirrors the M1 worker's timing model so the two are
// directly comparable). Prints prefill_tok_s + decode_tok_s + ITL p50/p95/p99.
//
// MEASURED RESULT (M5, mlx 0.32.0, 12B g64/b4, 24 tokens, 5 trials): the UNCOMPILED
// native decode = ~16 tok/s ≈ 0.999x of M1 stock mlx-lm (~15.4 tok/s) — PASSES the
// 0.9x gate WITHOUT mx::compile. A Tier-1 rms_norm shapeless compile was measured
// at ~3x SLOWER (per-call dispatch overhead beats fusion for a cheap elementwise
// op when the weight is passed as a per-call varying input) and was REVERTED; the
// mx::compile throughput work is deferred to a per-layer-graph design (Tier 2/3).
// The 0.9x gate is met by the uncompiled bit-exact path. See stock_decode_bench.py
// for the M1-stock arm.
//
// M5-gated (needs the 12B artifact). Self-skips (exit 0) when
// HYPERION_12B_ARTIFACT is unset or the golden (for the prompt ids) is absent.
//
// Usage:
//   HYPERION_12B_ARTIFACT=$repo/artifacts/models/gemma4-12b-qat-mlx-g64-b4 \
//   HYPERION_REPO_ROOT=$repo ./build/native/native_decode_bench [--tokens N] [--trials T]
//
// Defaults: 24 tokens (matches the golden), 5 trials + 2 warmups.

#include "dispatch.h"
#include "forward.h"
#include "geometry.h"
#include "governor.h"
#include "kv_cache.h"
#include "platform_policy.h"
#include "weights_loader.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

using hyperion::model::build_dispatch;
using hyperion::model::build_kv_state;
using hyperion::model::ForwardPass;
using hyperion::model::Geometry;
using hyperion::model::LayerType;
using hyperion::model::load_model_weights;
using hyperion::model::ModelWeights;
using hyperion::model::RopeSpec;
using hyperion::model::TextModelType;

namespace {

Geometry make_12b_geometry() {
    Geometry g{};
    g.model_type = TextModelType::Gemma4UnifiedText;
    g.hidden_size = 3840;
    g.intermediate_size = 15360;
    g.num_hidden_layers = 48;
    g.layer_types.reserve(48);
    for (std::uint32_t i = 0; i < 48; ++i) {
        g.layer_types.push_back(((i + 1) % 6 == 0) ? LayerType::Full : LayerType::Sliding);
    }
    g.num_attention_heads = 16;
    g.head_dim_local = 256;
    g.head_dim_global = 512;
    g.num_kv_heads_local = 8;
    g.num_kv_heads_global = 1;
    g.attention_k_eq_v_global = true;
    g.num_kv_shared_layers = 0;
    g.sliding_window = 1024;
    g.rope_local = RopeSpec{10000.0, std::nullopt, false};
    g.rope_global = RopeSpec{1000000.0, std::optional<float>(0.25F), true};
    g.final_logit_softcapping = 30.0F;
    g.rms_norm_eps = 1e-6F;
    g.attention_bias = false;
    g.vocab_size = 262144;
    g.max_position_embeddings = 262144;
    g.tie_word_embeddings = true;
    g.ple_hidden_per_layer_input = 0;
    g.ple_vocab_per_layer_input = 0;
    g.use_double_wide_mlp = false;
    g.moe = std::nullopt;
    return g;
}

double nearest_rank(const std::vector<std::uint64_t>& sorted, int pct) {
    if (sorted.empty()) return 0.0;
    // nearest-rank percentile (the m1.rs validator's method).
    std::size_t idx = (sorted.size() * pct + 99) / 100; // ceil(pct/100 * n)
    if (idx == 0) idx = 1;
    if (idx > sorted.size()) idx = sorted.size();
    return static_cast<double>(sorted[idx - 1]);
}

} // namespace

int main(int argc, char** argv) {
    std::uint32_t n_tokens = 24;
    std::uint32_t trials = 5;
    std::uint32_t warmups = 2;
    bool calibrate_sentinels = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--tokens") n_tokens = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--trials") trials = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--warmups") warmups = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--calibrate-sentinels") calibrate_sentinels = true;
    }

    const char* dir = std::getenv("HYPERION_12B_ARTIFACT");
    if (dir == nullptr || *dir == '\0') {
        std::cerr << "native_decode_bench: HYPERION_12B_ARTIFACT unset; skipping (M5-gated)\n";
        return 0;
    }
    const std::filesystem::path artifact(dir);
    if (!std::filesystem::exists(artifact / "model-00001-of-00002.safetensors")) {
        std::cerr << "native_decode_bench: 12B artifact shards absent; skipping\n";
        return 0;
    }
    const char* env_root = std::getenv("HYPERION_REPO_ROOT");
    std::string root = (env_root != nullptr && *env_root != '\0') ? std::string(env_root) : std::string("../../..");
    const std::filesystem::path golden =
        std::filesystem::path(root) / "native/hyperion_mlx/tests/fixtures/12b_greedy_golden.safetensors";
    if (!std::filesystem::exists(golden)) {
        std::cerr << "native_decode_bench: golden absent (" << golden << "); skipping\n";
        return 0;
    }

    try {
        const Geometry g = make_12b_geometry();
        const auto dispatch = build_dispatch(g);
        const mx::Stream cpu = mx::default_stream(mx::Device::cpu);
        const mx::Stream gpu = mx::new_stream(mx::Device::gpu);
        const auto runtime_environment =
            hyperion::platform::evaluate_runtime_environment(
                std::getenv("MLX_SDPA_BLOCKS") != nullptr);
        if (!runtime_environment.supported) {
            throw std::runtime_error(runtime_environment.reason);
        }

        auto golden_map = mx::load_safetensors(golden.string(), cpu).first;
        auto ids_it = golden_map.find("ids");
        if (ids_it == golden_map.end()) {
            std::cerr << "native_decode_bench: golden has no ids; skipping\n";
            return 0;
        }
        mx::array ids = ids_it->second; // [L] int32
        const int L = static_cast<int>(ids.shape(0));
        mx::eval(ids);

        ModelWeights weights = load_model_weights(artifact, g, 64, 4, cpu);
        ForwardPass fwd(g, dispatch, weights, gpu);

        // The production epilogue: tied lm_head -> softcap -> final-position slice ->
        // host greedy scan. Keeping it above calibration lets the sentinel path measure
        // the same final-chunk lifetime that the governor predicts.
        auto epilogue = [&](const mx::array& h, int Lh) -> std::uint32_t {
            mx::array logits = fwd.softcap(fwd.lm_head(h)); // [1, Lh, vocab]
            mx::array last = mx::slice(
                logits,
                {0, Lh - 1, 0},
                {1, Lh, static_cast<int>(logits.shape(2))},
                {1, 1, 1},
                gpu); // [1, 1, vocab]
            auto sample = fwd.sample_greedy(last);
            return sample.token_id;
        };

        // ── M3 governor calibration mode (--calibrate-sentinels, 10:45-46). Runs a full ──
        // prefill at the 8K + 32K sentinels (offset 0), using the same request-wide
        // transaction, repeated shrink admission, rooted forward outputs, final epilogue,
        // and materialize/publish sequence as hyp_prefill_chunk. The error band compares
        // the prediction for shapes that actually executed against mx::get_peak_memory();
        // rejected proposal predictions are reported separately. Synthetic ids (zeros)
        // are defensible because peak MLX memory for dense matmuls is shape-driven.
        if (calibrate_sentinels) {
            constexpr std::uint32_t kChunk = 2048;
            const std::array<std::uint32_t, 2> sentinels = {
                hyperion::governor::kSentinel8K, hyperion::governor::kSentinel32K};
            const auto device_budget = hyperion::platform::derive_device_budget(
                mx::device_info(mx::Device::gpu));
            if (!device_budget.supported) {
                throw std::runtime_error(device_budget.reason);
            }
            hyperion::governor::Governor governor(
                g,
                device_budget.budget.effective_bytes,
                device_budget.budget.soft_watermark_bytes);
            std::cout << "{\n  \"schema\": \"hyperion.m3-governor-calibration.v2\",\n";
            std::cout << "  \"model\": \"gemma4-12b-qat-mlx-g64-b4\",\n";
            std::cout << "  \"prefill_chunk_size\": " << kChunk << ",\n";
            std::cout << "  \"geometry\": {\"hidden_size\": " << g.hidden_size
                      << ", \"num_hidden_layers\": " << g.num_hidden_layers
                      << ", \"num_attention_heads\": " << g.num_attention_heads
                      << ", \"sliding_window\": " << g.sliding_window << "},\n";
            std::cout << "  \"governor\": {\"workspace_reserve_bytes\": "
                      << hyperion::governor::kWorkspaceReserveBytes
                      << ", \"transient_safety\": 1.25"
                      << ", \"budget_ceiling_bytes\": "
                      << device_budget.budget.effective_bytes
                      << ", \"soft_watermark_bytes\": "
                      << device_budget.budget.soft_watermark_bytes << "},\n";
            std::cout << "  \"sentinels\": [\n";
            for (std::size_t s = 0; s < sentinels.size(); ++s) {
                const std::uint32_t ctx = sentinels[s];
                std::vector<int32_t> host_ids(ctx, 0);
                mx::array ids_ctx = mx::array(host_ids.data(), mx::Shape{static_cast<int>(ctx)}, mx::int32);
                auto live_kvstate = std::make_unique<hyperion::model::KvState>(
                    build_kv_state(
                        dispatch,
                        hyperion::model::kDefaultGammaMax,
                        mx::bfloat16,
                        gpu));
                const auto operation_plan =
                    hyperion::model::plan_kv_growth(*live_kvstate, ctx);
                if (!operation_plan.representable) {
                    throw std::runtime_error(
                        "sentinel prefill exceeds representable KV dimensions");
                }
                auto settled_staging_plan = operation_plan;
                settled_staging_plan.requires_transaction = false;
                hyperion::model::KvGrowthTransaction transaction(
                    live_kvstate, operation_plan);

                mx::reset_peak_memory();
                std::uint64_t max_executed_predicted = 0;
                std::uint64_t max_attempted_predicted = 0;
                std::uint32_t admission_attempts = 0;
                std::uint32_t executed_chunks = 0;
                std::uint32_t smallest_executed_chunk =
                    std::numeric_limits<std::uint32_t>::max();
                std::uint32_t offset = 0;
                bool controlled_hard_reject = false;
                while (offset < ctx) {
                    const std::uint32_t proposed = std::min(kChunk, ctx - offset);
                    const auto admission = governor.admit_prefill_to_fit(
                        proposed,
                        offset,
                        ctx,
                        transaction.state(),
                        offset == 0 ? &operation_plan : &settled_staging_plan);
                    for (std::size_t i = 0; i < admission.attempt_count; ++i) {
                        max_attempted_predicted = std::max(
                            max_attempted_predicted,
                            admission.attempts[i].predicted_peak_bytes);
                        ++admission_attempts;
                    }
                    if (admission.decision.admission ==
                        hyperion::governor::Admission::HardRejected) {
                        // The shrink loop returns hard only after the one-token shape
                        // is rejected. This is a successful calibration outcome: stop
                        // before constructing that chunk's graph and let transaction
                        // destruction discard the already-staged partial prefill.
                        controlled_hard_reject = true;
                        break;
                    }
                    const std::uint32_t take = admission.n_tokens;
                    max_executed_predicted = std::max(
                        max_executed_predicted,
                        admission.decision.predicted_peak_bytes);
                    smallest_executed_chunk = std::min(
                        smallest_executed_chunk, take);
                    ++executed_chunks;

                    const int off0 = static_cast<int>(offset);
                    const int off1 = static_cast<int>(offset + take);
                    mx::array chunk_ids = mx::slice(ids_ctx, {off0}, {off1}, {1}, gpu);
                    mx::array h = fwd.forward(
                        fwd.embed(chunk_ids), transaction.state(), offset);
                    h = transaction.root_forward_result(h);
                    if (offset + take == ctx) {
                        (void)epilogue(h, static_cast<int>(take));
                    } else {
                        mx::eval(h); // force this chunk + cache writes before the next
                    }
                    offset += take;
                }
                if (!controlled_hard_reject) {
                    transaction.materialize();
                    transaction.publish();
                }
                mx::synchronize(gpu);
                const std::uint64_t measured = mx::get_peak_memory();
                if (controlled_hard_reject) {
                    std::cout << "    {\"context_len\": " << ctx
                              << ", \"outcome\": \"controlled_hard_reject\""
                              << ", \"reason_code\": \"governor_hard_reject_at_one_token\""
                              << ", \"completed_tokens\": " << offset
                              << ", \"max_predicted_bytes\": "
                              << max_executed_predicted
                              << ", \"max_attempted_predicted_bytes\": "
                              << max_attempted_predicted
                              << ", \"measured_bytes\": " << measured
                              << ", \"admission_attempts\": " << admission_attempts
                              << ", \"executed_chunks\": " << executed_chunks
                              << ", \"smallest_executed_chunk\": ";
                    if (executed_chunks == 0) {
                        std::cout << "null";
                    } else {
                        std::cout << smallest_executed_chunk;
                    }
                    std::cout << ", \"uncontrolled_oom\": false}";
                    if (s + 1 < sentinels.size()) std::cout << ",";
                    std::cout << "\n";
                    std::cerr << "  [sentinel " << (ctx / 1024)
                              << "K] controlled governor rejection after " << offset
                              << " tokens; max_executed_predicted="
                              << (max_executed_predicted / (1024 * 1024))
                              << " MiB  max_attempted_predicted="
                              << (max_attempted_predicted / (1024 * 1024))
                              << " MiB  measured_so_far="
                              << (measured / (1024 * 1024)) << " MiB\n";
                    continue;
                }
                const std::int64_t error_band =
                    static_cast<std::int64_t>(max_executed_predicted) -
                    static_cast<std::int64_t>(measured);
                const double rel_error = measured > 0
                    ? static_cast<double>(error_band) / static_cast<double>(measured) : 0.0;
                const bool within_guard = measured > 0 && std::fabs(rel_error) < 0.20;
                std::cout << "    {\"context_len\": " << ctx
                          << ", \"outcome\": \"completed\""
                          << ", \"completed_tokens\": " << offset
                          << ", \"max_predicted_bytes\": " << max_executed_predicted
                          << ", \"max_attempted_predicted_bytes\": "
                          << max_attempted_predicted
                          << ", \"measured_bytes\": " << measured
                          << ", \"error_band_bytes\": " << error_band
                          << ", \"relative_error\": " << rel_error
                          << ", \"admission_attempts\": " << admission_attempts
                          << ", \"executed_chunks\": " << executed_chunks
                          << ", \"smallest_executed_chunk\": "
                          << smallest_executed_chunk
                          << ", \"within_20pct_guard\": " << (within_guard ? "true" : "false")
                          << ", \"uncontrolled_oom\": false"
                          << "}";
                if (s + 1 < sentinels.size()) std::cout << ",";
                std::cout << "\n";
                std::cerr << "  [sentinel " << (ctx / 1024)
                          << "K] max_executed_predicted="
                          << (max_executed_predicted / (1024 * 1024))
                          << " MiB  max_attempted_predicted="
                          << (max_attempted_predicted / (1024 * 1024))
                          << " MiB  measured="
                          << (measured / (1024 * 1024)) << " MiB  error_band="
                          << (error_band / (1024 * 1024)) << " MiB  rel="
                          << (rel_error * 100.0) << "%"
                          << (within_guard ? "" : "  *** OUTSIDE 20% GUARD ***") << "\n";
            }
            std::cout << "  ]\n}\n";
            return 0;
        }

        // A full generate (prefill + n-1 decode) returning per-token monotonic offsets.
        // offsets[0] = TTFT (prefill → token 0); offsets[1..] = decode tokens.
        auto generate = [&](std::uint32_t n, std::vector<std::uint64_t>& offsets) -> std::uint32_t {
            auto kvstate = build_kv_state(dispatch, hyperion::model::kDefaultGammaMax, mx::bfloat16, gpu);
            mx::synchronize(gpu);
            const auto start = std::chrono::steady_clock::now();
            // Prefill (offset 0).
            mx::array h = fwd.forward(fwd.embed(ids), kvstate, 0);
            std::uint32_t tok = epilogue(h, L);
            mx::synchronize(gpu);
            offsets.push_back(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count()));
            std::uint32_t offset = static_cast<std::uint32_t>(L);
            // Decode n-1 steps.
            for (std::uint32_t i = 1; i < n; ++i) {
                int32_t v = static_cast<int32_t>(tok);
                mx::array one = mx::array(&v, mx::Shape{1}, mx::int32);
                h = fwd.forward(fwd.embed(one), kvstate, offset);
                tok = epilogue(h, 1);
                mx::synchronize(gpu);
                offsets.push_back(static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start).count()));
                offset += 1;
            }
            return tok;
        };

        // Warmups + measured trials.
        for (std::uint32_t w = 0; w < warmups; ++w) {
            std::vector<std::uint64_t> dummy;
            generate(n_tokens, dummy);
            (void)dummy;
        }
        std::vector<std::vector<std::uint64_t>> trial_offsets;
        std::vector<double> prefill_tok_s;
        std::vector<double> decode_tok_s;
        for (std::uint32_t t = 0; t < trials; ++t) {
            std::vector<std::uint64_t> offsets;
            generate(n_tokens, offsets);
            // prefill_tok_s = L / offsets[0]; decode_tok_s = (n-1) / (last - first).
            prefill_tok_s.push_back(static_cast<double>(L) * 1e9 / static_cast<double>(offsets[0]));
            const double decode_dur = static_cast<double>(offsets.back() - offsets.front());
            decode_tok_s.push_back(static_cast<double>(n_tokens - 1) * 1e9 / decode_dur);
            // ITLs (offsets windows 2) for percentiles.
            std::vector<std::uint64_t> itls;
            for (std::size_t i = 1; i < offsets.size(); ++i) itls.push_back(offsets[i] - offsets[i - 1]);
            std::sort(itls.begin(), itls.end());
            trial_offsets.push_back(offsets);
        }

        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        std::cout << "native_decode_bench mode=uncompiled-bit-exact"
                  << " prompt_tokens=" << L << " generated_tokens=" << n_tokens
                  << " trials=" << trials << " warmups=" << warmups << "\n";
        std::cout << "  prefill_tok_s median=" << median(prefill_tok_s)
                  << " min=" << *std::min_element(prefill_tok_s.begin(), prefill_tok_s.end())
                  << " max=" << *std::max_element(prefill_tok_s.begin(), prefill_tok_s.end()) << "\n";
        std::cout << "  decode_tok_s  median=" << median(decode_tok_s)
                  << " min=" << *std::min_element(decode_tok_s.begin(), decode_tok_s.end())
                  << " max=" << *std::max_element(decode_tok_s.begin(), decode_tok_s.end()) << "\n";
        // Aggregate ITLs across all trials for percentiles.
        std::vector<std::uint64_t> all_itls;
        for (const auto& o : trial_offsets)
            for (std::size_t i = 1; i < o.size(); ++i) all_itls.push_back(o[i] - o[i - 1]);
        std::sort(all_itls.begin(), all_itls.end());
        std::cout << "  itl_ns p50=" << nearest_rank(all_itls, 50)
                  << " p95=" << nearest_rank(all_itls, 95)
                  << " p99=" << nearest_rank(all_itls, 99) << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native_decode_bench: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "native_decode_bench: non-standard exception\n";
        return 1;
    }
}
