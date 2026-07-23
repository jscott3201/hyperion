// M5-gated real-12B G1 token-exact seal (M2-2.7).
//
// The 2.3b test sealed the per-layer MATH (sliding bit-exact hidden states). This is the
// PARITY seal the milestone is named for: the native epilogue (tied lm_head + 30·tanh(x/30)
// softcap + greedy argmax) + the offset>0 cached-prefix attention read produce the SAME
// greedy tokens as the pinned mlx-lm oracle on the chat-templated prompt.
//
// The 12B is Gemma4UnifiedForConditionalGeneration — an instruct/THINKING model that
// REQUIRES the chat template. Raw-prompt greedy is degenerate (argmax lands on
// near-vocab-edge byte tokens); the chat-templated prompt produces coherent tokens
// through the <|channel>thought channel ("The user is asking for the capital of France.
// The capital of France is Paris."). The oracle golden (gen_12b_greedy_golden.py) feeds
// the SAME templated ids, so token-exact parity isolates the engine (the forward math is
// faithful, proven by 2.3b). The thinking-mode output is a STRONG test — diverse tokens
// (special tokens, newlines, words, punctuation, "Paris").
//
// G1 kernel-change tier (08): bit-exact logits aren't achievable (the global quantized-
// kernel noise is real), so the hard gate is greedy TOKENS exact; the prefill-logit
// max-abs is a REPORTED number (fault-boundary calibration — inject a real single-layer
// fault, sit the gate between clean and fault — is a G1 refinement, not 2.7).
//
// Uses ForwardPass directly (one load, gets both the tokens AND the logit frame; the ABI
// wrapper hyp_prefill_chunk/hyp_decode_block is sealed on the tiny fixture in forward_test
// where prefill token == inline ref). NOT in --model-free ctest. Self-skips when
// HYPERION_12B_ARTIFACT is unset or the golden is absent.

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

// The 12B primary geometry (identical to forward_12b_test.cc — the v1-primary model, fixed;
// the hyperion-model geometry tests assert these against the real config.json).
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
        std::cerr << "forward_12b_decode_test: " << msg << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    const char* dir = std::getenv("HYPERION_12B_ARTIFACT");
    if (dir == nullptr || *dir == '\0') {
        std::cerr << "forward_12b_decode_test: HYPERION_12B_ARTIFACT unset; skipping (M5-gated)\n";
        return 0;
    }
    const std::filesystem::path artifact(dir);
    if (!std::filesystem::exists(artifact / "model-00001-of-00002.safetensors")) {
        std::cerr << "forward_12b_decode_test: 12B artifact shards absent; skipping\n";
        return 0;
    }

    const char* env_root = std::getenv("HYPERION_REPO_ROOT");
    std::string root = (env_root != nullptr && *env_root != '\0') ? std::string(env_root) : std::string("../../..");
    const std::filesystem::path golden =
        std::filesystem::path(root) / "native/hyperion_mlx/tests/fixtures/12b_greedy_golden.safetensors";
    if (!std::filesystem::exists(golden)) {
        std::cerr << "forward_12b_decode_test: golden absent (" << golden
                  << "); regenerate via gen_12b_greedy_golden.py; skipping\n";
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
        mx::array golden_prefill_logits = logits_it->second; // [1, vocab] bf16
        const int L = static_cast<int>(ids.shape(0));
        const int N = static_cast<int>(golden_tokens.shape(0));
        std::cerr << "forward_12b_decode_test: " << L << " prompt tokens, " << N << " greedy tokens\n";

        ModelWeights weights = load_model_weights(artifact, g, 64, 4, cpu);
        auto kvstate = build_kv_state(dispatch, hyperion::model::kDefaultGammaMax, mx::bfloat16, gpu);
        ForwardPass fwd(g, dispatch, weights, gpu);

        // Host-side signal-relative metric (max|Δ|/max|oracle|) — the 2.3b comparator.
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

        // The epilogue: final-norm hidden (forward already applied final_norm) → tied
        // lm_head → softcap → last-position slice → greedy argmax (host-side, the 2.3b
        // stale-scalar lesson). Returns the token, the last-position logits frame, and the
        // near_tie flag (top-2 gap < 0.5, A1).
        auto epilogue = [&](const mx::array& h, int Lh) -> std::tuple<std::uint32_t, mx::array, bool> {
            mx::array logits = fwd.softcap(fwd.lm_head(h)); // [1, Lh, vocab]
            mx::array last = mx::slice(
                logits,
                {0, Lh - 1, 0},
                {1, Lh, static_cast<int>(logits.shape(2))},
                {1, 1, 1},
                gpu); // [1, 1, vocab]
            auto sample = fwd.sample_greedy(last);
            return {sample.token_id, last, sample.near_tie};
        };

        bool pass = true;
        std::uint32_t near_tie_events = 0;
        mx::eval(golden_tokens); // MLX lazy-graph materialization (NOT JS/Python eval).
        const std::int32_t* gt = golden_tokens.data<int32_t>();

        // ── Prefill (offset 0): embed → forward → epilogue. Token 1 + the logit frame. ──
        mx::array h = fwd.forward(fwd.embed(ids), kvstate, 0); // final-norm'd; appends K/V
        auto [tok, prefill_logits, nt0] = epilogue(h, L);
        near_tie_events += nt0 ? 1 : 0;
        const bool tok0_ok = (tok == static_cast<std::uint32_t>(gt[0]));
        pass &= tok0_ok;
        const float prefill_logit_rel = sig_rel(prefill_logits, golden_prefill_logits);
        std::cerr << "  [prefill] token " << tok << (tok0_ok ? " == " : " != ") << gt[0]
                  << (tok0_ok ? "  OK" : "  FAIL")
                  << "  | prefill-logit max|Δ|/max|oracle|=" << prefill_logit_rel << '\n';

        // ── Decode N-1 steps (offset>0 cached-prefix read): embed last token → forward
        // at the running offset → epilogue. Each token must match the oracle exactly. ──
        std::uint32_t last_token = tok;
        std::uint32_t offset = static_cast<std::uint32_t>(L);
        for (int i = 1; i < N; ++i) {
            int32_t v = static_cast<int32_t>(last_token);
            mx::array one = mx::array(&v, mx::Shape{1}, mx::int32);
            h = fwd.forward(fwd.embed(one), kvstate, offset); // appends 1 K/V; cached-prefix read
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
        std::cerr << "  near_tie_events=" << near_tie_events << " (top-2 gap < 0.5 across " << N << " steps)\n";

        if (pass) {
            std::cerr << "forward_12b_decode_test: PASS — greedy tokens exact vs the mlx-lm oracle "
                      << "(prefill-logit max|Δ|/max|oracle|=" << prefill_logit_rel << ")\n";
            return 0;
        }
        std::cerr << "forward_12b_decode_test: FAIL — see the token mismatches above\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "forward_12b_decode_test: uncaught exception: " << e.what() << '\n';
        return 1;
    }
}
