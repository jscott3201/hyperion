// M5-gated content-verification tests for the mx::array-backed KV caches.
//
// These exercises the REAL MLX runtime (mx::array + slice_update + GPU eval),
// so — like hyperion_runtime_canary — they are NOT in the model-free ctest
// regex. They build everywhere MLX/Metal link, but run only on the self-hosted
// M5 (or any Apple-Silicon box with a Metal device). 2.1a proved the index
// arithmetic; these prove the bytes actually land in the right slots.

#include "kv_cache.h"
#include "dispatch.h"
#include "geometry.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mx = mlx::core;

using hyperion::model::build_dispatch;
using hyperion::model::build_kv_state;
using hyperion::model::Geometry;
using hyperion::model::GlobalKvCache;
using hyperion::model::LayerType;
using hyperion::model::LocalKvCache;
using hyperion::model::RopeSpec;
using hyperion::model::TextModelType;

namespace {

constexpr std::uint32_t kH = 2;
constexpr std::uint32_t kD = 3;
constexpr std::uint32_t kWin = 4;
constexpr std::uint32_t kGamma = 4;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "kv_cache_test: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

/// A distinct-valued update block for n tokens: token i holds the contiguous
/// range [base*elt + i*elt, ...) so every element is unique and readback can
/// pin a physical slot to the logical token that wrote it.
mx::array make_update(
    std::uint32_t n,
    std::uint32_t base,
    const mx::Stream& s,
    std::uint32_t h = kH,
    std::uint32_t d = kD) {
    const std::uint32_t elt = h * d;
    const double start = static_cast<double>(base * elt);
    const double stop = static_cast<double>((base + n) * elt);
    return mx::reshape(mx::arange(start, stop, s), {static_cast<int>(n), static_cast<int>(h), static_cast<int>(d)}, s);
}

/// True if buf[start : start+count] == update[upd_start : upd_start+count].
bool region_matches(
    const mx::array& buf,
    std::uint32_t start,
    const mx::array& update,
    std::uint32_t upd_start,
    std::uint32_t count,
    const mx::Stream& s,
    std::uint32_t h = kH,
    std::uint32_t d = kD) {
    const auto hi = static_cast<int>(h);
    const auto di = static_cast<int>(d);
    mx::array buf_region = mx::slice(
        buf,
        {static_cast<int>(start), 0, 0},
        {static_cast<int>(start + count), hi, di},
        {1, 1, 1},
        s);
    mx::array upd_region = mx::slice(
        update,
        {static_cast<int>(upd_start), 0, 0},
        {static_cast<int>(upd_start + count), hi, di},
        {1, 1, 1},
        s);
    mx::array eq = mx::allclose(buf_region, upd_region, 1e-5, 1e-5, false, s);
    mx::eval(eq);
    mx::synchronize(s);
    return eq.item<bool>();
}

/// Prefill n committed tokens in <= gamma chunks, writing distinct base..base+n
/// token values. Exercises the append+commit prefill path that rotates the ring.
void prefill(
    LocalKvCache& cache,
    std::uint32_t n,
    std::uint32_t base,
    const mx::Stream& s) {
    for (std::uint32_t written = 0; written < n;) {
        const std::uint32_t take = std::min<std::uint32_t>(kGamma, n - written);
        const mx::array upd = make_update(take, base + written, s);
        require(cache.append(upd, upd, take), "prefill append within gamma");
        cache.commit(take);
        written += take;
    }
}

void test_local_basic_and_straddle(const mx::Stream& s) {
    LocalKvCache cache(kWin, kGamma, kH, kD, mx::float32, s);
    require(cache.capacity() == kWin + kGamma, "ring capacity = window + gamma");
    require(cache.attention_len() == 0, "starts empty");

    // Prefill 7 tokens (capacity 8): token i -> slot i for i in 0..6. The next
    // write lands at slot 7, setting up a straddle on the next append.
    prefill(cache, 7, 0, s);
    require(cache.committed_len() == 7, "prefill advances committed");
    require(cache.attention_len() == 7, "attention tracks committed");
    require(cache.slot_for(7) == 7, "next write lands at slot 7");

    // Append 4 SPECULATIVE at slot 7. first = min(4, 8-7) = 1 -> slot 7 gets
    // upd[0]; remaining 3 -> slots 0,1,2 get upd[1,2,3] (wrapped).
    const mx::array spec = make_update(4, 100, s);
    require(cache.append(spec, spec, 4), "speculative append within gamma");
    require(cache.speculative_len() == 4, "speculative tracked");
    require(cache.attention_len() == 7, "attention EXCLUDES speculative (A3)");

    // Straddle verification: slot 7 holds spec token 0; slots 0,1,2 hold spec
    // tokens 1,2,3 (the wrap).
    require(region_matches(cache.keys(), 7, spec, 0, 1, s), "slot 7 == spec[0]");
    require(region_matches(cache.values(), 7, spec, 0, 1, s), "v slot 7 == spec[0]");
    require(region_matches(cache.keys(), 0, spec, 1, 3, s), "slots 0..2 == spec[1..3] (wrap)");

    // Discard the speculative (MTP reject, A2): committed is untouched. The
    // in-window committed region (tokens 3..6, slots 3..6) must survive; the
    // out-of-window slots 0,1,2 were overwritten by the draft and stay stale
    // (correct sliding-window behavior — they are not read).
    cache.discard_speculative();
    require(cache.committed_len() == 7, "discard leaves committed untouched (A2)");
    require(cache.speculative_len() == 0, "discard clears speculative");
    require(cache.attention_len() == 7, "attention unchanged after discard");
    // Reconstruct the committed block to verify the in-window prefix survived.
    const mx::array committed = make_update(7, 0, s);
    require(region_matches(cache.keys(), 3, committed, 3, 4, s), "in-window committed KV survived discard");

    // Slack is reusable after discard.
    require(cache.append(spec, spec, 4), "slack reusable after discard");
    cache.discard_speculative();
}

void test_local_speculative_exclusion(const mx::Stream& s) {
    LocalKvCache cache(kWin, kGamma, kH, kD, mx::float32, s);
    prefill(cache, 2, 0, s);
    require(cache.attention_len() == 2, "prefill 2 committed");

    require(cache.append(make_update(4, 10, s), make_update(4, 10, s), 4), "fill gamma slack");
    require(cache.speculative_len() == 4, "speculative == gamma_max");
    require(cache.attention_len() == 2, "attention still excludes speculative");

    // Overflow is rejected; caller must commit or discard first.
    require(!cache.append(make_update(1, 20, s), make_update(1, 20, s), 1), "gamma overflow rejected");

    cache.discard_speculative();
    require(cache.attention_len() == 2, "discard restores attention");
    require(cache.append(make_update(4, 30, s), make_update(4, 30, s), 4), "slack reusable");
}

void test_local_partial_commit(const mx::Stream& s) {
    LocalKvCache cache(kWin, kGamma, kH, kD, mx::float32, s);
    // MTP draft 4, accept 2 (A2: promote, don't trim).
    require(cache.append(make_update(4, 0, s), make_update(4, 0, s), 4), "draft 4");
    cache.commit(2);
    require(cache.committed_len() == 2, "partial commit promotes");
    require(cache.speculative_len() == 2, "remainder stays speculative");
    require(cache.attention_len() == 2, "attention tracks promoted");
    // Draft 2 more into the freed slack, then accept all.
    require(cache.append(make_update(2, 4, s), make_update(2, 4, s), 2), "draft into freed slack");
    cache.commit(4);
    require(cache.committed_len() == 6, "commit past speculative (fresh appends)");
    require(cache.speculative_len() == 0, "all speculative consumed");
    require(cache.attention_len() == 6, "attention tracks full commit");
}

void test_global_grow_and_preserve(const mx::Stream& s) {
    // step=4, 1 head, dim 2, K=V. Tiny step to exercise bucket-boundary growth.
    GlobalKvCache cache(4, 1, 2, true, mx::float32, s);
    require(cache.capacity() == 0, "global starts unallocated");
    require(cache.step_count() == 0, "zero steps initially");
    require(cache.k_eq_v(), "global K=V");

    const mx::array u0 = make_update(4, 0, s, 1, 2); // 4 tokens, base 0, h=1 d=2
    cache.append(u0, u0, 4);
    require(cache.committed_len() == 4, "append advances committed");
    require(cache.capacity() == 4, "one step at 4");
    require(cache.step_count() == 1, "step count 1");
    require(region_matches(cache.keys(), 0, u0, 0, 4, s, 1, 2), "tokens 0..4 in place");
    require(region_matches(cache.values(), 0, u0, 0, 4, s, 1, 2), "values alias keys (K=V)");

    // Cross a bucket boundary: 4 more -> capacity grows to 8 (step 2). The old
    // committed prefix must survive the reallocation.
    const mx::array u1 = make_update(4, 4, s, 1, 2);
    cache.append(u1, u1, 4);
    require(cache.committed_len() == 8, "committed 8");
    require(cache.capacity() == 8, "grew by one whole step");
    require(cache.step_count() == 2, "step count 2");
    require(region_matches(cache.keys(), 0, u0, 0, 4, s, 1, 2), "old prefix preserved across grow");
    require(region_matches(cache.keys(), 4, u1, 0, 4, s, 1, 2), "new tokens in place");

    // A sub-step append grows by one step but only partially fills it.
    const mx::array u2 = make_update(2, 8, s, 1, 2);
    cache.append(u2, u2, 2);
    require(cache.committed_len() == 10, "committed 10");
    require(cache.capacity() == 12, "capacity ceiling multiple of step");
    require(cache.step_count() == 3, "step count 3");
    require(cache.capacity() % 4 == 0, "capacity stays a multiple of step");
    require(region_matches(cache.keys(), 0, u0, 0, 4, s, 1, 2), "prefix still preserved");
    require(region_matches(cache.keys(), 8, u2, 0, 2, s, 1, 2), "tail tokens in place");
}

void test_global_stores_distinct_v(const mx::Stream& s) {
    // Regression guard for the V=K aliasing bug (M2-2.7): the GlobalKvCache MUST store V
    // separately from K even when k_eq_v — gemma4's V = v_norm(k_proj) is DISTINCT from
    // K = rope(k_norm(k_proj)). test_global_grow_and_preserve feeds IDENTICAL K and V, so
    // it cannot catch the aliasing. Feed DISTINCT K (base 0) and V (base 100) and verify
    // values() holds V, not K.
    GlobalKvCache cache(4, 1, 2, true, mx::float32, s);
    const mx::array k_up = make_update(4, 0, s, 1, 2);   // [4,1,2] values 0..7
    const mx::array v_up = make_update(4, 100, s, 1, 2); // [4,1,2] values 200..207 (distinct)
    cache.append(k_up, v_up, 4);
    require(cache.committed_len() == 4, "distinct-v: committed 4");
    require(region_matches(cache.keys(), 0, k_up, 0, 4, s, 1, 2), "distinct-v: keys == K (base 0)");
    require(region_matches(cache.values(), 0, v_up, 0, 4, s, 1, 2), "distinct-v: values == V (base 100, NOT K)");
}

void test_k_append_construction(const mx::Stream& s) {
    // Regression guard for the k_append reshape-scramble bug (M2-2.7): forward.cc builds
    //   k_append = reshape(transpose(k, {0,2,1,3}), {L, nkv, hd})
    // from k = [1, nkv, L, hd] (post-transpose) to write into the cache's [cap, nkv, hd]
    // slots. The transpose is LOAD-BEARING: a plain reshape(k, {L, nkv, hd}) reinterprets
    // the flat buffer and swaps the nkv/L axes (the bug). Verify k_append[i, j, k] == k's
    // value at [0, j, i, k] (position i, head j, dim k) — and that the plain reshape
    // scrambles (negative control proving the transpose is necessary).
    constexpr int B = 1, Nkv = 3, L = 4, Hd = 2;
    auto val = [](int j, int i, int k) { return static_cast<float>(j * 100 + i * 10 + k); };
    std::vector<float> host(static_cast<std::size_t>(B * Nkv * L * Hd));
    for (int j = 0; j < Nkv; ++j) {
        for (int i = 0; i < L; ++i) {
            for (int k = 0; k < Hd; ++k) {
                host[static_cast<std::size_t>((j * L + i) * Hd + k)] = val(j, i, k); // [1, nkv, L, hd]
            }
        }
    }
    const mx::array k = mx::array(host.data(), mx::Shape{B, Nkv, L, Hd}, mx::float32);
    const mx::array k_append = mx::reshape(mx::transpose(k, {0, 2, 1, 3}, s), {L, Nkv, Hd}, s);
    mx::eval(k_append);
    const float* p = k_append.data<float>();
    bool ok = true;
    for (int i = 0; i < L && ok; ++i) {
        for (int j = 0; j < Nkv && ok; ++j) {
            for (int k = 0; k < Hd && ok; ++k) {
                if (p[static_cast<std::size_t>((i * Nkv + j) * Hd + k)] != val(j, i, k)) {
                    ok = false; // [L, nkv, hd] row-major
                }
            }
        }
    }
    require(ok, "k_append construction: [i,j,k] == k[0,j,i,k] (transpose prevents scramble)");

    // Negative control: the BUGGY plain reshape DOES scramble (proves the transpose is
    // load-bearing, not decorative).
    const mx::array scrambled = mx::reshape(k, {L, Nkv, Hd}, s);
    mx::eval(scrambled);
    const float* sp = scrambled.data<float>();
    bool scrambled_matches = true;
    for (int i = 0; i < L && scrambled_matches; ++i) {
        for (int j = 0; j < Nkv && scrambled_matches; ++j) {
            for (int k = 0; k < Hd && scrambled_matches; ++k) {
                if (sp[static_cast<std::size_t>((i * Nkv + j) * Hd + k)] != val(j, i, k)) {
                    scrambled_matches = false;
                }
            }
        }
    }
    require(!scrambled_matches, "negative control: plain reshape scrambles (the bug)");
}

void test_local_append_committed_wrap(const mx::Stream& s) {
    // Regression guard for LocalKvCache::append_committed (M2-2.6): writing > cap tokens
    // chunks by cap internally (each write_ring ≤ cap, no OOB) and the ring rotates —
    // the last `cap` committed tokens land in the correct (wrapped) physical slots.
    // cap = kWin + kGamma = 4 + 4 = 8, window = 4. Write 12 tokens (> cap) with distinct
    // K (base 0) and V (base 100); the buffer holds the last 8 committed (logical 4..11):
    //   slots 0..3 = logical 8..11 (wrapped), slots 4..7 = logical 4..7.
    LocalKvCache cache(kWin, kGamma, kH, kD, mx::float32, s);
    const mx::array k_up = make_update(12, 0, s);    // [12, h, d] base 0
    const mx::array v_up = make_update(12, 100, s);   // distinct V
    cache.append_committed(k_up, v_up, 12);
    require(cache.committed_len() == 12, "wrap: committed 12");
    require(cache.attention_len() == 12, "wrap: attention 12");
    require(cache.capacity() == 8, "wrap: cap 8");
    require(region_matches(cache.keys(), 0, make_update(4, 8, s), 0, 4, s), "wrap: slots 0..3 == logical 8..11 (K)");
    require(region_matches(cache.keys(), 4, make_update(4, 4, s), 0, 4, s), "wrap: slots 4..7 == logical 4..7 (K)");
    require(region_matches(cache.values(), 0, make_update(4, 108, s), 0, 4, s), "wrap: slots 0..3 == logical 8..11 (V, not K)");
    require(region_matches(cache.values(), 4, make_update(4, 104, s), 0, 4, s), "wrap: slots 4..7 == logical 4..7 (V)");
}

void test_local_rotation_read(const mx::Stream& s) {
    // Regression guard for LocalKvCache::read_window (M2-2.6): after writing > cap tokens,
    // the rotation read returns the last min(window, committed) tokens in LOGICAL order
    // (the ring has rotated; the linear [0:n] slice would be wrong). Distinct K/V also
    // catches any K/V confusion in the read.
    LocalKvCache cache(kWin, kGamma, kH, kD, mx::float32, s); // cap=8, window=4
    const mx::array k_up = make_update(12, 0, s);    // base 0
    const mx::array v_up = make_update(12, 100, s);   // distinct V
    cache.append_committed(k_up, v_up, 12);
    const int h = static_cast<int>(kH);
    const int d = static_cast<int>(kD);
    const mx::array kr = cache.read_window(cache.keys(), kWin, h, d, s);   // [1, h, 4, d]
    const mx::array vr = cache.read_window(cache.values(), kWin, h, d, s);
    // Expected: the last 4 written (logical 8..11) in logical order.
    const mx::array k_exp = make_update(4, 8, s);     // [4, h, d] = logical 8..11 of base-0
    const mx::array v_exp = make_update(4, 108, s);   // logical 8..11 of base-100
    // read_window returns [1, h, n, d]; transpose → [1, n, h, d] → reshape [n, h, d] for region_matches.
    auto to_nhd = [&](const mx::array& r) {
        return mx::reshape(mx::transpose(r, {0, 2, 1, 3}, s), {4, h, d}, s);
    };
    require(region_matches(to_nhd(kr), 0, k_exp, 0, 4, s), "rotation: keys read == last 4 written (logical order)");
    require(region_matches(to_nhd(vr), 0, v_exp, 0, 4, s), "rotation: values read == last 4 written (logical order, not K)");
}

Geometry make_small_geometry() {
    // 6 layers in a 5:1 layout (5 sliding + 1 global), tiny dims.
    Geometry geometry{};
    geometry.model_type = TextModelType::Gemma4UnifiedText;
    geometry.hidden_size = 16;
    geometry.intermediate_size = 32;
    geometry.num_hidden_layers = 6;
    geometry.layer_types = {
        LayerType::Sliding, LayerType::Sliding, LayerType::Sliding,
        LayerType::Sliding, LayerType::Sliding, LayerType::Full,
    };
    geometry.num_attention_heads = 2;
    geometry.head_dim_local = kD;
    geometry.head_dim_global = 4;
    geometry.num_kv_heads_local = kH;
    geometry.num_kv_heads_global = 1;
    geometry.attention_k_eq_v_global = true;
    geometry.num_kv_shared_layers = 0;
    geometry.sliding_window = kWin;
    geometry.rope_local = RopeSpec{10000.0, std::optional<float>(), false};
    geometry.rope_global = RopeSpec{1000000.0, std::optional<float>(0.25F), true};
    geometry.final_logit_softcapping = 30.0F;
    geometry.rms_norm_eps = 1e-6F;
    geometry.attention_bias = false;
    geometry.vocab_size = 256;
    geometry.max_position_embeddings = 256;
    geometry.tie_word_embeddings = true;
    geometry.ple_hidden_per_layer_input = 0;
    geometry.ple_vocab_per_layer_input = 0;
    geometry.use_double_wide_mlp = false;
    geometry.moe = std::optional<hyperion::model::MoeConfig>();
    return geometry;
}

void test_build_kv_state(const mx::Stream& s) {
    const auto geometry = make_small_geometry();
    const auto validation = geometry.validate();
    require(!validation.has_value(), "small geometry validates");

    const auto table = build_dispatch(geometry);
    auto state = build_kv_state(table, kGamma, mx::float32, s);

    require(state.local.size() == 5, "one local cache per sliding layer");
    require(state.global.size() == 1, "one global cache per global layer");

    const LocalKvCache& local0 = state.local[0];
    require(local0.capacity() == kWin + kGamma, "local capacity from ArchSpec");
    require(local0.num_kv_heads() == kH, "local kv heads from ArchSpec");
    require(local0.head_dim() == kD, "local head_dim from ArchSpec");
    require(local0.attention_len() == 0, "local cache starts empty");

    const GlobalKvCache& global0 = state.global[0];
    require(global0.k_eq_v(), "global K=V from ArchSpec");
    require(global0.num_kv_heads() == 1, "global kv heads from ArchSpec");
    require(global0.head_dim() == 4, "global head_dim from ArchSpec");
    require(global0.capacity() == 0, "global cache starts unallocated");
    require(global0.step() == 256, "global step is the 256-token bucket");

    // The caches are usable end-to-end from the builder. Local dims (kH,kD)
    // and global dims (1, 4) differ — separate updates per kind.
    const mx::array upd_local = make_update(4, 0, s);
    require(state.local[0].append(upd_local, upd_local, 4), "builder-built local cache appends");
    state.local[0].commit(4);
    require(state.local[0].attention_len() == 4, "builder-built local cache writes");
    const mx::array upd_global = make_update(4, 0, s, 1, 4);
    state.global[0].append(upd_global, upd_global, 4);
    require(state.global[0].committed_len() == 4, "builder-built global cache writes");
    require(state.global[0].capacity() == 256, "builder-built global grew one step");
}

} // namespace

int main() {
    try {
        const mx::Device gpu = mx::Device::gpu;
        const mx::Stream stream = mx::new_stream(gpu);
        test_local_basic_and_straddle(stream);
        test_local_speculative_exclusion(stream);
        test_local_partial_commit(stream);
        test_global_grow_and_preserve(stream);
        test_global_stores_distinct_v(stream);
        test_k_append_construction(stream);
        test_local_append_committed_wrap(stream);
        test_local_rotation_read(stream);
        test_build_kv_state(stream);
    } catch (const std::exception& error) {
        std::cerr << "kv_cache_test: uncaught exception: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
