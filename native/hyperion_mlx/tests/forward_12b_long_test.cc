// M5-gated real-12B long-context token-exact seal (M2-2.6a).
//
// The 2.7 decode test sealed a 21-token prompt (single chunk, no rotation). This seals
// LONG context: a 2436-token chat-templated prompt that crosses one 2048 chunk boundary
// AND pushes the sliding ring past the 1024 window (the chunk-2 forward fires the rotation
// read). Prefill is chunked (mirrors the native hyp_prefill_chunk + the oracle's chunked
// prefill in gen_12b_long_golden.py); decode N tokens. Hard gate: greedy tokens exact vs
// the mlx-lm oracle. Soft report: peak MLX bytes (mx::get_peak_memory — the 2.6b governor
// gate input; not gated here).
//
// Uses ForwardPass directly (one load, gets the tokens AND the logit frame; the ABI
// chunked path hyp_prefill_chunk is exercised on the tiny fixture in forward_test). NOT
// in --model-free ctest. Self-skips when HYPERION_12B_ARTIFACT is unset or golden absent.

#include "dispatch.h"
#include "forward.h"
#include "geometry.h"
#include "kv_cache.h"
#include "weights_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
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

constexpr std::uint32_t kPrefillChunkSize = 2048;

// The 12B primary geometry (identical to forward_12b_test.cc / forward_12b_decode_test.cc).
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

void require(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "forward_12b_long_test: " << msg << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    const char* dir = std::getenv("HYPERION_12B_ARTIFACT");
    if (dir == nullptr || *dir == '\0') {
        std::cerr << "forward_12b_long_test: HYPERION_12B_ARTIFACT unset; skipping (M5-gated)\n";
        return 0;
    }
    const std::filesystem::path artifact(dir);
    if (!std::filesystem::exists(artifact / "model-00001-of-00002.safetensors")) {
        std::cerr << "forward_12b_long_test: 12B artifact shards absent; skipping\n";
        return 0;
    }
    const char* env_root = std::getenv("HYPERION_REPO_ROOT");
    std::string root = (env_root != nullptr && *env_root != '\0') ? std::string(env_root) : std::string("../../..");
    const std::filesystem::path golden =
        std::filesystem::path(root) / "native/hyperion_mlx/tests/fixtures/12b_long_greedy_golden.safetensors";
    if (!std::filesystem::exists(golden)) {
        std::cerr << "forward_12b_long_test: golden absent (" << golden
                  << "); regenerate via gen_12b_long_golden.py; skipping\n";
        return 0;
    }

    try {
        const Geometry g = make_12b_geometry();
        require(!g.validate().has_value(), "12B geometry validates");
        const auto dispatch = build_dispatch(g);

        const mx::Stream cpu = mx::default_stream(mx::Device::cpu);
        const mx::Stream gpu = mx::new_stream(mx::Device::gpu);

        auto golden_map = mx::load_safetensors(golden.string(), cpu).first;
        auto ids_it = golden_map.find("ids");
        auto tok_it = golden_map.find("greedy_tokens");
        auto logits_it = golden_map.find("prefill_logits");
        require(ids_it != golden_map.end(), "golden has ids");
        require(tok_it != golden_map.end(), "golden has greedy_tokens");
        require(logits_it != golden_map.end(), "golden has prefill_logits");
        mx::array ids = ids_it->second;                 // [L] int32
        mx::array golden_tokens = tok_it->second;       // [N] int32
        mx::array golden_prefill_logits = logits_it->second;
        const int L = static_cast<int>(ids.shape(0));
        const int N = static_cast<int>(golden_tokens.shape(0));
        std::cerr << "forward_12b_long_test: " << L << " prompt tokens, " << N << " greedy tokens\n";

        ModelWeights weights = load_model_weights(artifact, g, 64, 4, cpu);
        auto kvstate = build_kv_state(dispatch, hyperion::model::kDefaultGammaMax, mx::bfloat16, gpu);
        ForwardPass fwd(g, dispatch, weights, gpu);

        const mx::Stream cs = mx::default_stream(mx::Device::cpu);
        auto sig_rel = [&](const mx::array& a, const mx::array& b) -> float {
            mx::array af = mx::astype(mx::contiguous(a, false, cs), mx::float32, cs);
            mx::array bf = mx::astype(mx::contiguous(b, false, cs), mx::float32, cs);
            mx::eval(af);
            mx::eval(bf);
            const float* pa = af.data<float>();
            const float* pb = bf.data<float>();
            const std::size_t n = af.size();
            float md = 0.0F, mo = 0.0F;
            for (std::size_t i = 0; i < n; ++i) {
                md = std::max(md, std::fabs(pa[i] - pb[i]));
                mo = std::max(mo, std::fabs(pb[i]));
            }
            return mo > 0.0F ? md / mo : md;
        };

        auto epilogue = [&](const mx::array& h, int Lh) -> std::tuple<std::uint32_t, mx::array, bool> {
            mx::array logits = fwd.softcap(fwd.lm_head(h)); // [1, Lh, vocab]
            mx::array last = mx::slice(
                logits, {0, Lh - 1, 0}, {1, Lh, static_cast<int>(logits.shape(2))}, {1, 1, 1}, gpu);
            auto sample = fwd.sample_greedy(last);
            return {sample.token_id, last, sample.near_tie};
        };

        mx::eval(golden_tokens);
        const std::int32_t* gt = golden_tokens.data<int32_t>();
        bool pass = true;
        std::uint32_t near_tie_events = 0;

        // ── Chunked prefill (mirrors hyp_prefill_chunk + the oracle): 2048-token chunks
        // at the running offset; the chunk-2 forward fires the rotation read past window.
        // Epilogue on the last chunk only; mx::eval(h) after each intermediate chunk.
        std::uint32_t offset = 0;
        std::uint32_t prefill_token = 0;
        std::optional<mx::array> prefill_logits;
        while (offset < static_cast<std::uint32_t>(L)) {
            const std::uint32_t take = std::min(kPrefillChunkSize, static_cast<std::uint32_t>(L) - offset);
            const int off0 = static_cast<int>(offset);
            const int off1 = static_cast<int>(offset + take);
            mx::array chunk_ids = mx::slice(ids, {off0}, {off1}, {1}, gpu);
            mx::array h = fwd.forward(fwd.embed(chunk_ids), kvstate, offset); // appends K/V; rotation read past window
            if (offset + take == static_cast<std::uint32_t>(L)) {
                auto [tok, lg, nt] = epilogue(h, static_cast<int>(take));
                prefill_token = tok;
                prefill_logits = std::move(lg);
                near_tie_events += nt ? 1 : 0;
            } else {
                mx::eval(h); // MLX lazy-graph materialization (NOT JS/Python eval): force the chunk + cache writes
            }
            offset += take;
        }
        const bool tok0_ok = (prefill_token == static_cast<std::uint32_t>(gt[0]));
        pass &= tok0_ok;
        const float prefill_logit_rel = sig_rel(*prefill_logits, golden_prefill_logits);
        std::cerr << "  [prefill chunked] token " << prefill_token << (tok0_ok ? " == " : " != ") << gt[0]
                  << (tok0_ok ? "  OK" : "  FAIL")
                  << "  | prefill-logit max|Δ|/max|oracle|=" << prefill_logit_rel << '\n';

        // ── Decode N-1 (offset>0 cached-prefix read; rotation read continues past window).
        std::uint32_t last_token = prefill_token;
        for (int i = 1; i < N; ++i) {
            int32_t v = static_cast<int32_t>(last_token);
            mx::array one = mx::array(&v, mx::Shape{1}, mx::int32);
            mx::array h = fwd.forward(fwd.embed(one), kvstate, offset);
            auto [dtok, _, nti] = epilogue(h, 1);
            (void)_;
            near_tie_events += nti ? 1 : 0;
            const bool ok = (dtok == static_cast<std::uint32_t>(gt[i]));
            pass &= ok;
            if (!ok) {
                std::cerr << "  [decode " << i << "] token " << dtok << " != oracle " << gt[i] << "  FAIL\n";
            }
            last_token = dtok;
            offset += 1;
        }
        const std::size_t peak_mlx = mx::get_peak_memory();
        const std::size_t active_mlx = mx::get_active_memory();
        std::cerr << "  near_tie_events=" << near_tie_events
                  << "  peak_mlx=" << (peak_mlx / (1024 * 1024)) << " MiB"
                  << "  active_mlx=" << (active_mlx / (1024 * 1024)) << " MiB\n";

        if (pass) {
            std::cerr << "forward_12b_long_test: PASS — long-context greedy tokens exact vs the oracle "
                      << "(prefill-logit max|Δ|/max|oracle|=" << prefill_logit_rel
                      << "; peak_mlx=" << (peak_mlx / (1024 * 1024)) << " MiB)\n";
            return 0;
        }
        std::cerr << "forward_12b_long_test: FAIL — see the token mismatches above\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "forward_12b_long_test: uncaught exception: " << e.what() << '\n';
        return 1;
    }
}
