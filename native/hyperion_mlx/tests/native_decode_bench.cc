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
#include "kv_cache.h"
#include "weights_loader.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
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
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--tokens") n_tokens = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--trials") trials = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--warmups") warmups = static_cast<std::uint32_t>(std::stoul(next()));
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

        // The epilogue (final-norm → tied lm_head → softcap → last-position → greedy).
        // Mirrors forward_12b_decode_test. sample_greedy host-scans (the 2.3b lesson).
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
