// forward_12b_fault_boundary_test — the G1 logit fault-boundary gate (M2 completion).
//
// The M2 gate (10-milestones-and-gates.md:33) is "G1 parity (token-exact + two-sided
// logit thresholds, DERIVED)". The token-exact half is sealed by forward_12b_decode_test
// + forward_12b_long_test (M2-2.7/2.6a). This test seals the DERIVED two-sided logit
// threshold half — the fault-boundary calibration 08-correctness-and-verification.md
// §18-22 prescribes: "measure clean max-abs; inject a real single-layer fault; the gate
// sits between measured-clean and measured-fault. Re-derive when MLX/quant/kernels
// change." No magic tolerances.
//
// ── The fault (derive, don't port — 08:21) ──────────────────────────────────────
// The 08 "e.g." fault is "RoPE offset +1 on one layer". Measured on the Gemma-4 12B it
// produces a logit delta AT OR BELOW the engine noise floor (sig_rel ~0.019 vs clean
// ~0.021 — no separation; the residual stream + QK-norm absorb a one-position rotation).
// The SEPARATING fault is a per-layer residual-scale amplification: layer 0's
// layer_scalar × 5 → sig_rel ~0.41 (~20× the noise floor, argmax-stable). This is a real
// single-layer structural perturbation, NOT a quantization-noise proxy (08:23-25).
// Both the oracle (gen_12b_faulted_golden.py: layer_scalar *= 5) and the native arm
// (ForwardPass::forward_faulted(..., fault_layer=0, layer_scalar_factor=5.0)) inject the
// SAME fault, so the faulted delta isolates the engine's FAITHFULNESS to the fault.
//
// ── The three gates ────────────────────────────────────────────────────────────
//   1. CLEAN  : sig_rel(native_clean,   oracle_clean)   < THRESHOLD
//       — a correct unfused impl sits at the engine noise floor (well under the gate).
//   2. FAULTED: sig_rel(native_faulted, oracle_faulted) < THRESHOLD
//       — a correct engine REPRODUCES the fault (native_faulted ≈ oracle_faulted).
//         A broken engine that silently DROPS the fault has native_faulted ≈ native_clean,
//         so this delta ≈ the fault magnitude (≫ THRESHOLD) → FAILS. This is the gate
//         the simple native-only-fault approach CANNOT catch.
//   3. FLOOR  : fault_magnitude = sig_rel(oracle_faulted, oracle_clean) > 10 × CLEAN_FLOOR
//       — confirms the fault actually moved the output beyond the noise floor (guards a
//         too-weak fault that would let a broken impl pass gate 2 by coincidence).
//
// ── The threshold ──────────────────────────────────────────────────────────────
//   THRESHOLD = sqrt(CLEAN × FAULT_MAGNITUDE) — the geometric mean, equal multiplicative
//   margin on both sides (the clean/fault scales are ~20× apart). Committed as a constant
//   below with the measured values + the re-derive mandate. NOT computed per-run.
//
// CALIBRATION (re-run when MLX/quant/kernels change — 08:21):
//   1. clean : run forward_12b_decode_test, read stderr "prefill-logit max|Δ|/max|oracle|=X".
//   2. fault : sig_rel(oracle_faulted, oracle_clean) — from gen_12b_faulted_golden.py +
//      gen_12b_greedy_golden.py (both committed). This test prints it.
//   3. threshold = sqrt(clean × fault). Update kG1FaultBoundaryThreshold + the comment.
//   (The FLOOR check uses the runtime clean_delta — no second constant to update.)
//
// DISCRIMINATION LIMIT: the geometric-mean threshold (~4.4× the clean floor) catches a
// fault that is DROPPED or grossly mis-scaled (×0..×3) — gate 2 fails because
// native_faulted lands far from oracle_faulted. A near-miss fault factor (e.g. the engine
// applies ×4.5 where the oracle applies ×5) can fall under the threshold and escape; the
// gate is calibrated for drop/scramble detection, not fault-factor constant-exactness.
// Tightening that would require a per-factor differential gate (out of scope for the M2
// G1 two-sided threshold; M4's kernel gate consumes this boundary as-is, 10:53).
//
// M5-gated (needs the live MLX memory counter + the 12B artifact). Self-skips (exit 0)
// when HYPERION_12B_ARTIFACT is unset or a golden is absent (CI). NOT in --model-free ctest.

#include "dispatch.h"
#include "forward.h"
#include "geometry.h"
#include "kv_cache.h"
#include "weights_loader.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

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

// The derived fault: layer 0's per-layer residual scale × 5 (see gen_12b_faulted_golden.py
// + the file header). MUST match the faulted golden's FAULT_LAYER + LAYER_SCALAR_FACTOR.
constexpr std::size_t kFaultLayer = 0;
constexpr float kLayerScalarFactor = 5.0F;

// ── The committed G1 fault-boundary threshold (DERIVED 2026-07-23 on the M5) ─────
//   CLEAN noise floor (native_clean vs oracle_clean)   : sig_rel = 0.020833
//   FAULT magnitude  (oracle_faulted vs oracle_clean)  : sig_rel = 0.412281
//   THRESHOLD = sqrt(CLEAN × FAULT) = sqrt(0.020833 × 0.412281) ≈ 0.092637
//   Separation: fault is ~19.8× the clean floor; the gate sits at ~4.4× clean (≈22% of
//   the fault), giving ~4.4× headroom below and ~4.4× headroom above (geometric mean).
// Environment: mlx 0.32.0 / mlx-lm 0.31.3 (oracle/.venv, VCS 8239c72), Python 3.12.13;
//   12B = gemma4_unified, group-quantized g64/b4, bf16 compute; unfused (stock
//   mx::quantized_matmul + mx::fast::scaled_dot_product_attention).
// RE-DERIVE when MLX/quant/kernels change (08:21): re-run gen_12b_greedy_golden.py +
//   gen_12b_faulted_golden.py, re-measure clean + fault, recompute sqrt(clean×fault),
//   update this constant + the comment. The clean floor is also reported by
//   forward_12b_decode_test's stderr "prefill-logit max|Δ|/max|oracle|".
constexpr float kG1FaultBoundaryThreshold = 0.092637F;
// The floor check: the fault magnitude must exceed 10× the RUNTIME clean_delta (not a
// committed constant), confirming the fault separates from the CURRENT engine noise
// (08:20 "the gate sits between measured-clean and measured-fault" — the two must
// actually be separable). Self-calibrating: if the noise floor widens, the floor
// demands a stronger fault rather than silently passing a too-weak one. 10× is the
// minimum; the derived fault gives ~19.8× at the measured clean 0.0208.
constexpr float kFloorSeparation = 10.0F;

// The 12B primary geometry (identical to forward_12b_decode_test.cc — the v1-primary
// model, fixed; the hyperion-model geometry tests assert these against the real config).
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
        std::cerr << "forward_12b_fault_boundary_test: " << msg << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// Host-side signal-relative metric (max|Δ|/max|oracle|) — the 2.3b/2.7 comparator.
// Both arrays are cast to f32 on the CPU stream, forced contiguous, then host-scanned.
float sig_rel(const mx::array& a, const mx::array& b, const mx::Stream& cs) {
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
}

} // namespace

int main() {
    const char* dir = std::getenv("HYPERION_12B_ARTIFACT");
    if (dir == nullptr || *dir == '\0') {
        std::cerr << "forward_12b_fault_boundary_test: HYPERION_12B_ARTIFACT unset; skipping (M5-gated)\n";
        return 0;
    }
    const std::filesystem::path artifact(dir);
    if (!std::filesystem::exists(artifact / "model-00001-of-00002.safetensors")) {
        std::cerr << "forward_12b_fault_boundary_test: 12B artifact shards absent; skipping\n";
        return 0;
    }

    const char* env_root = std::getenv("HYPERION_REPO_ROOT");
    std::string root = (env_root != nullptr && *env_root != '\0') ? std::string(env_root) : std::string("../../..");
    const std::filesystem::path fixtures =
        std::filesystem::path(root) / "native/hyperion_mlx/tests/fixtures";
    const std::filesystem::path clean_golden = fixtures / "12b_greedy_golden.safetensors";
    const std::filesystem::path faulted_golden = fixtures / "12b_faulted_greedy_golden.safetensors";
    if (!std::filesystem::exists(clean_golden)) {
        std::cerr << "forward_12b_fault_boundary_test: clean golden absent (" << clean_golden
                  << "); regenerate via gen_12b_greedy_golden.py; skipping\n";
        return 0;
    }
    if (!std::filesystem::exists(faulted_golden)) {
        std::cerr << "forward_12b_fault_boundary_test: faulted golden absent (" << faulted_golden
                  << "); regenerate via gen_12b_faulted_golden.py; skipping\n";
        return 0;
    }

    try {
        const Geometry g = make_12b_geometry();
        require(!g.validate().has_value(), "12B geometry validates");
        const auto dispatch = build_dispatch(g);

        const mx::Stream cpu = mx::default_stream(mx::Device::cpu);
        const mx::Stream gpu = mx::new_stream(mx::Device::gpu);

        // Load both goldens (ids + prefill_logits). The faulted golden shares the clean
        // golden's ids (the fault doesn't change the prompt), but it carries its own
        // prefill_logits (the faulted logit frame) + greedy_tokens.
        auto clean_map = mx::load_safetensors(clean_golden.string(), cpu).first;
        auto faulted_map = mx::load_safetensors(faulted_golden.string(), cpu).first;
        auto ids_it = clean_map.find("ids");
        auto clean_logits_it = clean_map.find("prefill_logits");
        auto faulted_logits_it = faulted_map.find("prefill_logits");
        require(ids_it != clean_map.end(), "clean golden has ids");
        require(clean_logits_it != clean_map.end(), "clean golden has prefill_logits");
        require(faulted_logits_it != faulted_map.end(), "faulted golden has prefill_logits");
        mx::array ids = ids_it->second;                              // [L] int32
        mx::array oracle_clean_logits = clean_logits_it->second;    // [1, vocab] bf16
        mx::array oracle_faulted_logits = faulted_logits_it->second; // [1, vocab] bf16
        const int L = static_cast<int>(ids.shape(0));
        std::cerr << "forward_12b_fault_boundary_test: " << L << " prompt tokens\n";

        ModelWeights weights = load_model_weights(artifact, g, 64, 4, cpu);
        ForwardPass fwd(g, dispatch, weights, gpu);
        const mx::Stream cs = mx::default_stream(mx::Device::cpu);

        // The last-position prefill logit frame: embed → forward / forward_faulted →
        // lm_head → softcap → last-position slice. Mirrors forward_12b_decode_test's
        // epilogue so the sig_rel is directly comparable to the clean-seal's reported
        // number. Each run gets its OWN KvState (the prefill appends to the cache, so the
        // clean + faulted arms must not share one).
        auto prefill_logits = [&](bool faulted) -> mx::array {
            auto kvstate = build_kv_state(
                dispatch, hyperion::model::kDefaultGammaMax, mx::bfloat16, gpu);
            mx::array state = faulted
                ? fwd.forward_faulted(fwd.embed(ids), kvstate, 0, kFaultLayer, kLayerScalarFactor)
                : fwd.forward(fwd.embed(ids), kvstate, 0);
            mx::array logits = fwd.softcap(fwd.lm_head(state)); // [1, L, vocab]
            const int Lh = static_cast<int>(logits.shape(1));
            return mx::slice(
                logits,
                {0, Lh - 1, 0},
                {1, Lh, static_cast<int>(logits.shape(2))},
                {1, 1, 1},
                gpu); // [1, 1, vocab]
        };

        // ── The three measurements ──────────────────────────────────────────────
        const mx::array native_clean = prefill_logits(false);
        const mx::array native_faulted = prefill_logits(true);

        const float clean_delta = sig_rel(native_clean, oracle_clean_logits, cs);
        const float faulted_delta = sig_rel(native_faulted, oracle_faulted_logits, cs);
        const float fault_magnitude = sig_rel(oracle_faulted_logits, oracle_clean_logits, cs);

        std::cerr << "  [clean]    sig_rel(native_clean,    oracle_clean)    = " << clean_delta
                  << "  (gate: < " << kG1FaultBoundaryThreshold << ")\n";
        std::cerr << "  [faulted]  sig_rel(native_faulted,  oracle_faulted)  = " << faulted_delta
                  << "  (gate: < " << kG1FaultBoundaryThreshold << ")\n";
        std::cerr << "  [magnitude] sig_rel(oracle_faulted, oracle_clean)   = " << fault_magnitude
                  << "  (floor: > " << (kFloorSeparation * clean_delta) << " = "
                  << kFloorSeparation << "x runtime clean=" << clean_delta << ")\n";

        // ── The three gates ─────────────────────────────────────────────────────
        require(clean_delta < kG1FaultBoundaryThreshold,
                 "G1 fault-boundary CLEAN gate: native_clean vs oracle_clean exceeds the "
                 "threshold — the engine noise floor widened; re-derive if MLX/quant/"
                 "kernels changed (08:21)");
        require(faulted_delta < kG1FaultBoundaryThreshold,
                 "G1 fault-boundary FAULTED gate: native_faulted vs oracle_faulted exceeds "
                 "the threshold — the engine does NOT faithfully reproduce the fault (it is "
                 "dropping or scrambling the layer-scalar perturbation); re-derive if "
                 "MLX/quant/kernels changed (08:21)");
        // FLOOR: the fault magnitude must exceed kFloorSeparation × the RUNTIME clean_delta
        // (not a stale committed constant), confirming the fault separates from the CURRENT
        // engine noise (08:20 — "the gate sits between measured-clean and measured-fault";
        // the two must actually be separable). Using the runtime clean_delta makes the floor
        // self-calibrating: if the noise floor widened, the floor demands a stronger fault
        // rather than silently passing a too-weak one. A failure here means EITHER the fault
        // is too weak (re-derive with a stronger fault, 08:20) OR the clean noise widened
        // (re-measure clean + re-derive kG1FaultBoundaryThreshold per the header procedure).
        require(fault_magnitude > kFloorSeparation * clean_delta,
                 "G1 fault-boundary FLOOR check: the fault magnitude does not separate from "
                 "the runtime noise floor (need > 10x clean) — either the fault is too weak "
                 "to gate (re-derive with a stronger fault, 08:20) or the clean noise widened "
                 "(re-measure clean + re-derive kG1FaultBoundaryThreshold per the header)");

        std::cerr << "forward_12b_fault_boundary_test: PASS — G1 two-sided logit fault-"
                     "boundary gate holds (clean=" << clean_delta << " < "
                  << kG1FaultBoundaryThreshold << ", faulted=" << faulted_delta << " < "
                  << kG1FaultBoundaryThreshold << ", magnitude=" << fault_magnitude
                  << " > " << (kFloorSeparation * clean_delta) << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "forward_12b_fault_boundary_test: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "forward_12b_fault_boundary_test: non-standard exception\n";
        return 1;
    }
}
