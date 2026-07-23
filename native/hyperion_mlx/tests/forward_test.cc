// M5-gated model-free test for the native forward pass (M2-2.3a).
//
// Loads the committed tiny gemma4_unified fixture (REAL quantized g64/b4 weights),
// runs the native ``ForwardPass`` over a short prompt, and compares the final hidden
// state to an INLINE REFERENCE that re-implements the same Gemma 4 math independently
// (erf-based gelu vs the production tanh approximation; independently assembled
// norm/attention/MLP ordering). Agreement proves the production math has no
// transcription/ordering bugs. Conceptual correctness (scale=1.0, QK-norm-before-rope,
// v_norm-no-scale, proportional RoPE) is sealed by 2.3b against the mlx-lm oracle.
//
// Like hyperion_kv_cache / hyperion_weights this is NOT in the --model-free ctest
// regex: it builds wherever MLX/Metal link, runs only on the self-hosted M5. It
// self-skips (exit 0) when the tiny fixture dir is absent (CI).
//
// Also exercises the real ``hyp_model_load`` / ``hyp_kvstate_create`` ABI path on the
// fixture, and the per-kind attention mask (A3 — no broadcast_shapes crash once a
// sliding layer rotates past its window).

#include "dispatch.h"
#include "forward.h"
#include "geometry.h"
#include "kv_cache.h"
#include "step_result_access.h"
#include "weights_loader.h"

#include "hyperion_mlx.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
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

// Must match crates/hyperion-model/fixtures/build_gemma4_unified_tiny.py.
constexpr std::uint32_t kHidden = 128;
constexpr std::uint32_t kInter = 256;
constexpr std::uint32_t kLayers = 6;
constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kHdLocal = 64;
constexpr std::uint32_t kHdGlobal = 128;
constexpr std::uint32_t kKvLocal = 2;
constexpr std::uint32_t kKvGlobal = 1;
constexpr std::uint32_t kWindow = 8;
constexpr std::uint32_t kVocab = 128;

void require(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "forward_test: " << msg << '\n';
        std::exit(EXIT_FAILURE);
    }
}

Geometry make_tiny_geometry() {
    Geometry g{};
    g.model_type = TextModelType::Gemma4UnifiedText;
    g.hidden_size = kHidden;
    g.intermediate_size = kInter;
    g.num_hidden_layers = kLayers;
    g.layer_types = {LayerType::Sliding, LayerType::Sliding, LayerType::Sliding,
                     LayerType::Sliding, LayerType::Sliding, LayerType::Full};
    g.num_attention_heads = kHeads;
    g.head_dim_local = kHdLocal;
    g.head_dim_global = kHdGlobal;
    g.num_kv_heads_local = kKvLocal;
    g.num_kv_heads_global = kKvGlobal;
    g.attention_k_eq_v_global = true;
    g.num_kv_shared_layers = 0;
    g.sliding_window = kWindow;
    g.rope_local = RopeSpec{10000.0, std::nullopt, false};
    g.rope_global = RopeSpec{1000000.0, std::optional<float>(0.25F), true};
    g.final_logit_softcapping = 30.0F;
    g.rms_norm_eps = 1e-6F;
    g.attention_bias = false;
    g.vocab_size = kVocab;
    g.max_position_embeddings = 256;
    g.tie_word_embeddings = true;
    g.ple_hidden_per_layer_input = 0;
    g.ple_vocab_per_layer_input = 0;
    g.use_double_wide_mlp = false;
    g.moe = std::nullopt;
    return g;
}

// Build the ABI geometry mirror (for the hyp_model_load path test) from the same dims.
HypGeometryParams make_tiny_abi_geometry(const std::vector<HypLayerType>& layer_types_abi) {
    HypGeometryParams p{};
    p.model_type = HYP_GEMMA4_UNIFIED_TEXT;
    p.hidden_size = kHidden;
    p.intermediate_size = kInter;
    p.num_hidden_layers = kLayers;
    p.layer_types = layer_types_abi.data();
    p.num_attention_heads = kHeads;
    p.head_dim_local = kHdLocal;
    p.head_dim_global = kHdGlobal;
    p.num_kv_heads_local = kKvLocal;
    p.num_kv_heads_global = kKvGlobal;
    p.attention_k_eq_v_global = 1;
    p.num_kv_shared_layers = 0;
    p.sliding_window = kWindow;
    p.rope_local = HypRopeSpec{10000.0, 0, 0.0F, 0};
    p.rope_global = HypRopeSpec{1000000.0, 1, 0.25F, 1};
    p.final_logit_softcapping = 30.0F;
    p.rms_norm_eps = 1e-6F;
    p.attention_bias = 0;
    p.vocab_size = kVocab;
    p.max_position_embeddings = 256;
    p.tie_word_embeddings = 1;
    p.ple_hidden_per_layer_input = 0;
    p.ple_vocab_per_layer_input = 0;
    p.use_double_wide_mlp = 0;
    p.has_moe = 0;
    p.moe = HypMoeConfig{0, 0, 0};
    return p;
}

bool allclose(const mx::array& a, const mx::array& b, const mx::Stream& s, float rtol = 1e-2, float atol = 1e-2) {
    mx::array eq = mx::allclose(a, b, rtol, atol, false, s);
    mx::eval(eq);
    mx::synchronize(s);
    return eq.item<bool>();
}

// ---------------------------------------------------------------------------
// Inline reference: the same Gemma 4 math, assembled independently + an erf-based
// gelu (vs the production tanh approximation). Agreement with the production
// ForwardPass proves no transcription/ordering bug.
// ---------------------------------------------------------------------------

mx::array ref_rms(const mx::array& x, const std::optional<mx::array>& w, float eps, const mx::Stream& s) {
    return mx::fast::rms_norm(x, w, eps, s);
}

mx::array ref_gelu_erf(const mx::array& x, const mx::Stream& s) {
    // Exact GELU (erf), independent of the production tanh approximation.
    mx::array half = mx::array(0.5F, x.dtype());
    mx::array one = mx::array(1.0F, x.dtype());
    mx::array inv = mx::array(0.7071067811865476F, x.dtype()); // 1/sqrt(2)
    mx::array erf = mx::erf(mx::multiply(inv, x, s), s);
    return mx::multiply(mx::multiply(half, x, s), mx::add(one, erf, s), s);
}

mx::array ref_rope(const mx::array& x, const RopeSpec& spec, std::uint32_t head_dim, std::uint32_t offset, const mx::Stream& s) {
    const auto hd = static_cast<int>(head_dim);
    if (spec.proportional) {
        const float prf = *spec.partial_rotary_factor;
        int rotated = static_cast<int>(prf * static_cast<float>(head_dim));
        rotated -= rotated % 2;
        mx::array exponents = mx::divide(
            mx::arange(0, rotated, 2, mx::float32, s),
            mx::array(static_cast<float>(hd), mx::float32), s);
        mx::array finite = mx::power(mx::array(static_cast<float>(spec.theta), mx::float32), exponents, s);
        mx::array infs = mx::full({(hd - rotated) / 2}, std::numeric_limits<float>::infinity(), mx::float32, s);
        mx::array freqs = mx::concatenate(std::vector<mx::array>{finite, infs}, 0, s);
        return mx::fast::rope(x, hd, false, std::nullopt, 1.0F, static_cast<int>(offset), freqs, s);
    }
    return mx::fast::rope(x, hd, false, std::optional<float>(static_cast<float>(spec.theta)), 1.0F, static_cast<int>(offset), std::nullopt, s);
}

mx::array ref_attention(const mx::array& x, const ModelWeights& w, std::size_t layer,
                         const Geometry& g, const mx::Stream& s) {
    const auto& lw = w.layers[layer];
    const bool global = (layer == 5);
    const RopeSpec& rs = global ? g.rope_global : g.rope_local;
    const std::uint32_t hd = global ? g.head_dim_global : g.head_dim_local;
    const std::uint32_t nkvh = global ? g.num_kv_heads_global : g.num_kv_heads_local;
    const int B = 1, L = static_cast<int>(x.shape(1));
    const int nh = static_cast<int>(g.num_attention_heads);
    const int hd2 = static_cast<int>(hd);
    const int nkvh2 = static_cast<int>(nkvh);

    mx::array q = mx::reshape(lw.q_proj.apply(x, s), {B, L, nh, hd2}, s);
    q = ref_rms(q, std::optional<mx::array>(lw.q_norm), g.rms_norm_eps, s);
    mx::array k_raw = lw.k_proj.apply(x, s);
    mx::array k = mx::reshape(k_raw, {B, L, nkvh2, hd2}, s);
    k = ref_rms(k, std::optional<mx::array>(lw.k_norm), g.rms_norm_eps, s);
    // V: k_eq_v (global) -> v_norm(same k_proj); sliding -> v_norm(v_proj). No rope on V.
    mx::array v_raw = global ? k_raw : lw.v_proj->apply(x, s);
    mx::array v = ref_rms(mx::reshape(v_raw, {B, L, nkvh2, hd2}, s), std::nullopt, g.rms_norm_eps, s);
    q = ref_rope(mx::transpose(q, {0, 2, 1, 3}, s), rs, hd, 0, s);
    k = ref_rope(mx::transpose(k, {0, 2, 1, 3}, s), rs, hd, 0, s);
    v = mx::transpose(v, {0, 2, 1, 3}, s);
    // causal mask [L,L]
    mx::array qi = mx::reshape(mx::arange(0, L, mx::int32, s), {L, 1}, s);
    mx::array ki = mx::reshape(mx::arange(0, L, mx::int32, s), {1, L}, s);
    mx::array mask = mx::less_equal(ki, qi, s);
    mx::array out = mx::fast::scaled_dot_product_attention(
        q, k, v, 1.0F, std::string(""), std::optional<mx::array>(mask), std::nullopt, s);
    out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}, s), {B, L, nh * hd2}, s);
    return lw.o_proj.apply(out, s);
}

mx::array ref_forward(const mx::array& ids, const ModelWeights& w, const Geometry& g, const mx::Stream& s) {
    const float scale = std::sqrt(static_cast<float>(g.hidden_size));
    mx::array h = mx::multiply(w.embed(ids, s), mx::array(scale, mx::bfloat16), s);
    h = mx::reshape(h, {1, static_cast<int>(ids.shape(0)), static_cast<int>(g.hidden_size)}, s);
    for (std::size_t layer = 0; layer < w.layers.size(); ++layer) {
        const auto& lw = w.layers[layer];
        mx::array residual = h;
        mx::array a = ref_rms(h, std::optional<mx::array>(lw.input_layernorm), g.rms_norm_eps, s);
        a = ref_attention(a, w, layer, g, s);
        a = ref_rms(a, std::optional<mx::array>(lw.post_attention_layernorm), g.rms_norm_eps, s);
        a = mx::add(residual, a, s);
        residual = a;
        mx::array m = ref_rms(a, std::optional<mx::array>(lw.pre_feedforward_layernorm), g.rms_norm_eps, s);
        m = lw.down_proj.apply(mx::multiply(ref_gelu_erf(lw.gate_proj.apply(m, s), s), lw.up_proj.apply(m, s), s), s);
        m = ref_rms(m, std::optional<mx::array>(lw.post_feedforward_layernorm), g.rms_norm_eps, s);
        h = mx::add(mx::multiply(mx::add(residual, m, s), lw.layer_scalar, s), mx::array(0.0F, mx::bfloat16), s);
    }
    return ref_rms(h, std::optional<mx::array>(w.final_norm), g.rms_norm_eps, s);
}

// Inline reference for the greedy epilogue: tied lm_head + softcap + last-position
// argmax (host-side, per the 2.3b stale-scalar lesson). Mirrors
// ForwardPass::lm_head/softcap/sample_greedy so the ABI prefill token can be checked
// against an independent assembly. Transcription parity — weak vs the 12B oracle in
// forward_12b_decode_test, but exercises the full epilogue path on CI.
std::uint32_t ref_greedy(const mx::array& ids, const ModelWeights& w, const Geometry& g, const mx::Stream& s) {
    mx::array h = ref_forward(ids, w, g, s); // [1, L, hidden] final-norm'd
    mx::array logits = mx::quantized_matmul(
        h, w.embed_weight, w.embed_scales, std::optional<mx::array>(w.embed_biases),
        /*transpose=*/true, std::optional<int>(w.embed_group_size), std::optional<int>(w.embed_bits),
        /*mode=*/"affine", s); // [1, L, vocab]
    const float sc = g.final_logit_softcapping;
    logits = mx::multiply(
        mx::tanh(mx::divide(logits, mx::array(sc, logits.dtype()), s), s),
        mx::array(sc, logits.dtype()), s);
    const int L = static_cast<int>(h.shape(1));
    const int vocab = static_cast<int>(logits.shape(2));
    const mx::Stream cs = mx::default_stream(mx::Device::cpu);
    mx::array last = mx::astype(
        mx::contiguous(mx::slice(logits, {0, L - 1, 0}, {1, L, vocab}, {1, 1, 1}, s), false, cs),
        mx::float32, cs);
    mx::eval(last);
    const float* p = last.data<float>();
    float top1 = -std::numeric_limits<float>::infinity();
    std::uint32_t arg = 0;
    for (int i = 0; i < vocab; ++i) {
        if (p[i] > top1) {
            top1 = p[i];
            arg = static_cast<std::uint32_t>(i);
        }
    }
    return arg;
}

// ---------------------------------------------------------------------------

void test_forward_parity(const std::filesystem::path& fixture, const mx::Stream& gpu) {
    const Geometry g = make_tiny_geometry();
    require(!g.validate().has_value(), "tiny geometry validates");
    const auto dispatch = build_dispatch(g);
    const mx::Stream cpu = mx::default_stream(mx::Device::cpu);
    ModelWeights weights = load_model_weights(fixture, g, 64, 4, cpu);
    auto kvstate = build_kv_state(dispatch, hyperion::model::kDefaultGammaMax, mx::bfloat16, gpu);
    ForwardPass fwd(g, dispatch, weights, gpu);

    // A short prompt (L=4 <= window=8, offset 0).
    mx::array ids = mx::array({7, 3, 40, 100}, mx::Shape{4}, mx::int32);
    mx::array native = fwd.forward(fwd.embed(ids), kvstate, 0);
    mx::array ref = ref_forward(ids, weights, g, gpu);
    mx::eval(native);
    mx::eval(ref);
    mx::synchronize(gpu);
    require(allclose(native, ref, gpu), "native forward == inline reference (transcription parity)");
    std::cerr << "forward_test: native vs ref max |Δ| within tolerance (6-layer tiny fixture)\n";

    // The KV state advanced: each global layer's committed_len == L; each local ring
    // committed_len == L (< window, no rotation yet).
    require(kvstate.global[0].committed_len() == 4, "global cache advanced");
    require(kvstate.local[0].committed_len() == 4, "local ring advanced");

    // A3 rotation smoke: build a sliding mask whose query window has rotated past the
    // ring window (offset 12, kv_len 8 == window) — must not crash / broadcast_shapes.
    mx::array rot_mask = fwd.build_mask(LayerType::Sliding, 4, kWindow, 12);
    mx::array g_mask = fwd.build_mask(LayerType::Full, 4, 16, 12);
    mx::eval(rot_mask);
    mx::eval(g_mask);
    mx::synchronize(gpu);
    require(rot_mask.shape(0) == 4 && rot_mask.shape(1) == static_cast<int>(kWindow), "sliding mask shape [q,kv]");
    require(g_mask.shape(0) == 4 && g_mask.shape(1) == 16, "global mask shape [q,kv]");
}

void test_abi_load(const std::filesystem::path& fixture, const mx::Stream& gpu) {
    // The real hyp_model_load + hyp_prefill_chunk + hyp_decode_block path on the tiny
    // fixture (M2-2.7): single-chunk prefill → first token; decode 1 → next token. The
    // prefill token is checked against the inline reference argmax (ref_greedy) —
    // transcription parity on the epilogue (the 12B oracle in forward_12b_decode_test
    // is the real seal). Chunked 2048 prefill + governor admission is 2.6.
    const Geometry g = make_tiny_geometry();
    std::vector<HypLayerType> layer_types_abi = {
        HYP_LAYER_SLIDING, HYP_LAYER_SLIDING, HYP_LAYER_SLIDING,
        HYP_LAYER_SLIDING, HYP_LAYER_SLIDING, HYP_LAYER_FULL,
    };
    HypGeometryParams abi = make_tiny_abi_geometry(layer_types_abi);

    HypModel model = nullptr;
    require(hyp_model_create(&model) == HYP_STATUS_OK, "model create");
    require(
        hyp_model_load(model, &abi, fixture.string().c_str()) == HYP_STATUS_OK,
        "hyp_model_load succeeds on the tiny fixture");
    HypKvState kv = nullptr;
    require(hyp_kvstate_create(model, &kv) == HYP_STATUS_OK, "kvstate create on loaded model");

    HypStepResult result = nullptr;
    require(hyp_step_result_create(&result) == HYP_STATUS_OK, "step result create");

    // Prefill a short prompt (L=4 <= window=8) → first generated token.
    std::vector<std::uint32_t> prompt = {7u, 3u, 40u, 100u};
    HypTokenStream tokens{prompt.data(), static_cast<std::uint32_t>(prompt.size()), 1};
    require(
        hyp_prefill_chunk(model, kv, &tokens, result) == HYP_STATUS_OK,
        "hyp_prefill_chunk succeeds (single-chunk prefill)");
    HypStepResultFields pf = hyperion::model::step_result_read(result);
    require(pf.token_id < g.vocab_size, "prefill token within vocab");
    require(pf.governor_state == HYP_GOVERNOR_READY, "governor READY (governor is 2.6)");

    // Cross-check the prefill token vs the inline reference argmax (same weights, same prompt).
    const mx::Stream cpu = mx::default_stream(mx::Device::cpu);
    ModelWeights weights = load_model_weights(fixture, g, 64, 4, cpu);
    mx::array ids = mx::array({7, 3, 40, 100}, mx::Shape{4}, mx::int32);
    const std::uint32_t ref_tok = ref_greedy(ids, weights, g, gpu);
    require(
        pf.token_id == ref_tok,
        "prefill token == inline reference argmax (transcription parity on the epilogue)");

    // Decode 1 token (offset>0, cached-prefix read) → next token.
    require(
        hyp_decode_block(model, kv, 1, result) == HYP_STATUS_OK,
        "hyp_decode_block succeeds (offset>0 cached-prefix read)");
    HypStepResultFields dc = hyperion::model::step_result_read(result);
    require(dc.token_id < g.vocab_size, "decode token within vocab");

    std::cerr << "forward_test: ABI prefill+decode OK on the tiny fixture "
              << "(prefill token " << pf.token_id << " == ref " << ref_tok
              << "; decode token " << dc.token_id
              << "; near_tie_events=" << dc.near_tie_events << ")\n";

    require(hyp_step_result_free(&result) == HYP_STATUS_OK, "step result free");
    require(hyp_kvstate_free(&kv) == HYP_STATUS_OK, "kvstate free");
    require(hyp_model_free(&model) == HYP_STATUS_OK, "model free");
}

void test_abi_chunked_prefill(const std::filesystem::path& fixture, const mx::Stream& /*gpu*/) {
    // M2-2.6a: the chunked hyp_prefill_chunk path + the rotation read on the tiny fixture
    // (window=8, cap=16). A >2048-token prompt crosses the 2048 chunk boundary AND rotates
    // the ring ~150× (committed/16), exercising append_committed's cap-chunking + read_window's
    // rotation heavily on CI (model-free). Asserts no crash + a valid token (no inline ref —
    // ref_forward is offset-0 only; the 12B forward_12b_long_test is the real seal).
    const Geometry g = make_tiny_geometry();
    std::vector<HypLayerType> layer_types_abi = {
        HYP_LAYER_SLIDING, HYP_LAYER_SLIDING, HYP_LAYER_SLIDING,
        HYP_LAYER_SLIDING, HYP_LAYER_SLIDING, HYP_LAYER_FULL,
    };
    HypGeometryParams abi = make_tiny_abi_geometry(layer_types_abi);

    HypModel model = nullptr;
    require(hyp_model_create(&model) == HYP_STATUS_OK, "chunked: model create");
    require(hyp_model_load(model, &abi, fixture.string().c_str()) == HYP_STATUS_OK, "chunked: load");
    HypKvState kv = nullptr;
    require(hyp_kvstate_create(model, &kv) == HYP_STATUS_OK, "chunked: kvstate create");
    HypStepResult result = nullptr;
    require(hyp_step_result_create(&result) == HYP_STATUS_OK, "chunked: step result create");

    // 2500 tokens (ids i%128) — 2 chunks ([0:2048) + [2048:2500)), rotation past window=8.
    std::vector<std::uint32_t> prompt;
    prompt.resize(2500);
    for (std::uint32_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = i % g.vocab_size;
    }
    HypTokenStream tokens{prompt.data(), static_cast<std::uint32_t>(prompt.size()), 1};
    require(
        hyp_prefill_chunk(model, kv, &tokens, result) == HYP_STATUS_OK,
        "chunked: hyp_prefill_chunk succeeds on a >2048-token prompt");
    HypStepResultFields pf = hyperion::model::step_result_read(result);
    require(pf.token_id < g.vocab_size, "chunked: prefill token within vocab");
    require(
        hyp_decode_block(model, kv, 1, result) == HYP_STATUS_OK,
        "chunked: decode after a long prefill succeeds (rotation read past window)");
    HypStepResultFields dc = hyperion::model::step_result_read(result);
    require(dc.token_id < g.vocab_size, "chunked: decode token within vocab");

    std::cerr << "forward_test: chunked prefill OK (2500 tokens → 2 chunks; rotation ~150×; "
              << "prefill token " << pf.token_id << ", decode token " << dc.token_id << ")\n";

    require(hyp_step_result_free(&result) == HYP_STATUS_OK, "chunked: step result free");
    require(hyp_kvstate_free(&kv) == HYP_STATUS_OK, "chunked: kvstate free");
    require(hyp_model_free(&model) == HYP_STATUS_OK, "chunked: model free");
}

void test_mask_by_kind(const Geometry& g, const mx::Stream& s) {
    const auto dispatch = build_dispatch(g);
    // A3: each mask is sourced from the FIRST layer of its kind — never layer 0
    // unconditionally (a rotating sliding ring clamps its offset to window-1, so a mask
    // built from a sliding cache truncates/crashes past the window).
    require(dispatch.sliding.mask_source_layer == g.first_sliding_layer(), "sliding mask source = first sliding");
    require(dispatch.global.mask_source_layer == g.first_global_layer(), "global mask source = first global");
    require(dispatch.global.window == 0, "global window is 0 (full causal)");
    require(dispatch.sliding.window == kWindow, "sliding window from geometry");
    (void)s;
    std::cerr << "forward_test: A3 mask-by-kind sources verified (sliding/global first-of-kind)\n";
}

} // namespace

int main() {
    const char* dir = std::getenv("HYPERION_TINY_FIXTURE");
    if (dir == nullptr || *dir == '\0') {
        // Fall back to the committed fixture path relative to the repo root.
        const char* env_root = std::getenv("HYPERION_REPO_ROOT");
        std::string d = (env_root != nullptr && *env_root != '\0')
                             ? std::string(env_root)
                             : std::string("../../..");
        dir = nullptr; // resolved below
        std::string fallback = d + "/crates/hyperion-model/fixtures/gemma4-unified-tiny";
        const std::filesystem::path fixture(fallback);
        if (!std::filesystem::exists(fixture / "model-00001-of-00002.safetensors")) {
            std::cerr << "forward_test: tiny fixture absent; skipping (set HYPERION_TINY_FIXTURE or run on the M5)\n";
            return 0;
        }
        try {
            const mx::Stream gpu = mx::new_stream(mx::Device::gpu);
            test_forward_parity(fixture, gpu);
            test_abi_load(fixture, gpu);
            test_abi_chunked_prefill(fixture, gpu);
            test_mask_by_kind(make_tiny_geometry(), gpu);
        } catch (const std::exception& e) {
            std::cerr << "forward_test: uncaught exception: " << e.what() << '\n';
            return 1;
        }
        return 0;
    }
    const std::filesystem::path fixture(dir);
    try {
        const mx::Stream gpu = mx::new_stream(mx::Device::gpu);
        test_forward_parity(fixture, gpu);
        test_abi_load(fixture, gpu);
        test_abi_chunked_prefill(fixture, gpu);
        test_mask_by_kind(make_tiny_geometry(), gpu);
    } catch (const std::exception& e) {
        std::cerr << "forward_test: uncaught exception: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
