#ifndef HYPERION_MLX_H
#define HYPERION_MLX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Stable status values returned by every Hyperion native ABI function. */
typedef enum HypStatus {
    HYP_STATUS_OK = 0,
    HYP_STATUS_INVALID_ARGUMENT = 1,
    HYP_STATUS_NOT_FOUND = 2,
    HYP_STATUS_IO = 3,
    HYP_STATUS_OOM_GOVERNOR = 4,
    HYP_STATUS_UNSUPPORTED = 5,
    HYP_STATUS_INTERNAL = 6,
    HYP_STATUS_CANCELLED = 7
} HypStatus;

/** Results from the stateless M0 platform and native-execution canary. */
typedef struct HypCanaryInfo {
    uint32_t abi_version;
    uint32_t mlx_compile_major;
    uint32_t mlx_compile_minor;
    uint32_t mlx_compile_patch;
    uint32_t macos_major;
    uint32_t macos_minor;
    uint32_t macos_patch;
    uint32_t gpu_family;
    uint64_t recommended_working_set_bytes;
    uint64_t effective_budget_bytes;
    uint64_t soft_watermark_bytes;
    float mlx_probe_value;
    float metallib_probe_value;
    char mlx_runtime_version[32];
    char gpu_name[128];
} HypCanaryInfo;

/**
 * Enforce the M5/macOS 26.2/MLX 0.32.0 floor, execute real MLX and custom
 * metallib probes, and return the device-derived 16 GB-profile budget.
 */
HypStatus hyp_runtime_canary(HypCanaryInfo* out_info);

/** Copy the calling thread's last native error into a caller-owned buffer. */
HypStatus hyp_last_error(char* buffer, size_t buffer_len);

/* --- M2 model + step ABI (opaque, magic-tagged handles) ------------------ */

/** Opaque handle typedefs; the struct bodies are private to the native library. */
typedef struct HypModelOpaque* HypModel;
typedef struct HypKvStateOpaque* HypKvState;
typedef struct HypStepResultOpaque* HypStepResult;

typedef enum HypTextModelType {
    HYP_GEMMA4_TEXT = 0,
    HYP_GEMMA4_UNIFIED_TEXT = 1
} HypTextModelType;

typedef enum HypLayerType {
    HYP_LAYER_SLIDING = 0,
    HYP_LAYER_FULL = 1
} HypLayerType;

typedef enum HypGovernorState {
    HYP_GOVERNOR_READY = 0,
    HYP_GOVERNOR_SOFT_PAUSED = 1,
    HYP_GOVERNOR_HARD_REJECT = 2
} HypGovernorState;

/** RoPE scheme for one attention kind (mirrors hyperion::model::RopeSpec). */
typedef struct HypRopeSpec {
    double theta;
    int has_partial_rotary_factor; /* 0 = full rotary, 1 = partial */
    float partial_rotary_factor;
    int proportional;
} HypRopeSpec;

typedef struct HypMoeConfig {
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t moe_intermediate_size;
} HypMoeConfig;

/** Validated geometry crossing the ABI from Rust's hyperion-model::Geometry. */
typedef struct HypGeometryParams {
    HypTextModelType model_type;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_hidden_layers;
    const HypLayerType* layer_types; /* length == num_hidden_layers */
    uint32_t num_attention_heads;
    uint32_t head_dim_local;
    uint32_t head_dim_global;
    uint32_t num_kv_heads_local;
    uint32_t num_kv_heads_global;
    int attention_k_eq_v_global;
    uint32_t num_kv_shared_layers;
    uint32_t sliding_window;
    HypRopeSpec rope_local;
    HypRopeSpec rope_global;
    float final_logit_softcapping;
    float rms_norm_eps;
    int attention_bias;
    uint32_t vocab_size;
    uint32_t max_position_embeddings;
    int tie_word_embeddings;
    uint32_t ple_hidden_per_layer_input;
    uint32_t ple_vocab_per_layer_input;
    int use_double_wide_mlp;
    int has_moe;
    HypMoeConfig moe;
} HypGeometryParams;

/** A stream of token ids fed to prefill or decode. */
typedef struct HypTokenStream {
    const uint32_t* tokens;
    uint32_t count;
    int is_prompt; /* 1 = prompt tokens (prefill), 0 = decode continuation */
} HypTokenStream;

/** M3 sampling configuration (the sampled-mode epilogue). Pass NULL (or
 *  ``temperature == 0.0``) to route to the greedy argmax path bit-identically — the
 *  G1 token-exact seal (M2-2.7) depends on this default. Non-NULL with
 *  ``temperature > 0.0`` routes to ``ForwardPass::sample_stochastic`` (a host-side
 *  softmax + seeded categorical draw over the fp32 logit vector, 03 §Sampling).
 *  The RNG is a per-request seeded host PRNG (statistical-faithfulness bar per
 *  08 §35-37 + the M2-2.6c fault-boundary threshold; NOT bit-exact vs mlx-lm —
 *  C++/Metal vs MLX FP reduction order differs, so seeded-token-exact draws
 *  against the oracle are out of scope by design). */
typedef struct HypSamplingConfig {
    float  temperature;   /* t > 0.0 enables sampling; t == 0.0 → greedy. t < 0 → INVALID_ARG. */
    int32_t top_k;        /* <=0 = disabled; >0 = keep the top-k logits (capped at vocab). */
    float  top_p;         /* <=0.0 or >1.0 = disabled; (0,1] = nucleus sampling. */
    float  min_p;         /* <=0.0 = disabled; (0,1) = min-relative-mass filtering. */
    uint64_t seed;        /* per-request RNG seed (host PRNG); same seed → same token stream. */
} HypSamplingConfig;

/** The number of top-k logprobs returned per step when sampling (03:105-106: "ship the
 *  top-k logprobs requested"). A small fixed sidecar — NOT the full 262144-wide logit
 *  frame (~1 MB/token), which MUST NOT cross the ABI. */
#define HYP_TOP_K_LOGPROBS 8

/** Per-step telemetry + sampled token; caller-owned handle, reused across steps. */
typedef struct {
    uint32_t token_id;
    float logit;
    uint64_t peak_mlx_bytes;
    uint64_t active_mlx_bytes;
    uint64_t phys_footprint_bytes;
    uint64_t local_kv_bytes;
    uint64_t global_kv_bytes;
    float local_kv_eval_ms;
    float global_kv_eval_ms;
    HypGovernorState governor_state;
    uint32_t near_tie_events;
    /* M3 sampler: the ≤HYP_TOP_K_LOGPROBS top logprobs for the step (greedy or sampled).
     * ``top_k_logprob_count`` is the number filled (0 if none requested; up to
     * HYP_TOP_K_LOGPROBS). ids are the token ids, logprobs the post-softcap logprobs,
     * sorted descending. */
    uint32_t top_k_logprob_count;
    uint32_t top_k_logprob_ids[HYP_TOP_K_LOGPROBS];
    float top_k_logprob_values[HYP_TOP_K_LOGPROBS];
} HypStepResultFields;

/** Allocate an empty magic-tagged model handle; free with hyp_model_free. */
HypStatus hyp_model_create(HypModel* out_model);

/** Free a model handle and set *model to NULL. A NULL handle is a no-op (OK). */
HypStatus hyp_model_free(HypModel* model);

/** Load weights and build the graph from validated geometry (stub -> UNSUPPORTED at M2-1.3). */
HypStatus hyp_model_load(HypModel model,
                         const HypGeometryParams* geometry,
                         const char* weights_path);

/** Allocate a KV-state handle bound to the model (stub -> UNSUPPORTED at M2-1.3). */
HypStatus hyp_kvstate_create(HypModel model, HypKvState* out_kvstate);

/** Free a KV-state handle and set *kvstate to NULL. A NULL handle is a no-op. */
HypStatus hyp_kvstate_free(HypKvState* kvstate);

/** Prefill a prompt chunk into the KV state (stub -> UNSUPPORTED at M2-2.6). */
HypStatus hyp_prefill_chunk(HypModel model,
                            HypKvState kvstate,
                            const HypTokenStream* tokens,
                            HypStepResult out_result);

/** Decode n_tokens into the KV state (stub -> UNSUPPORTED at M2-2.6/2.7). */
HypStatus hyp_decode_block(HypModel model,
                           HypKvState kvstate,
                           uint32_t n_tokens,
                           HypStepResult out_result);

/** M3 sampler surface: prefill a prompt chunk with a sampling config (NULL = greedy).
 *  Identical to ``hyp_prefill_chunk`` except the epilogue routes to ``sample_stochastic``
 *  when ``config != NULL`` and ``config->temperature > 0``. The greedy default (NULL or
 *  temperature == 0) is byte-identical to ``hyp_prefill_chunk`` (the G1 token-exact seal). */
HypStatus hyp_prefill_chunk_sampled(HypModel model,
                                    HypKvState kvstate,
                                    const HypTokenStream* tokens,
                                    const HypSamplingConfig* config,
                                    HypStepResult out_result);

/** M3 sampler surface: decode n_tokens with a sampling config (NULL = greedy). Identical
 *  to ``hyp_decode_block`` except the per-step epilogue routes to ``sample_stochastic``
 *  when ``config != NULL`` and ``config->temperature > 0``. The greedy default is
 *  byte-identical to ``hyp_decode_block`` (the G1 token-exact seal). */
HypStatus hyp_decode_block_sampled(HypModel model,
                                   HypKvState kvstate,
                                   uint32_t n_tokens,
                                   const HypSamplingConfig* config,
                                   HypStepResult out_result);

/** Allocate an empty magic-tagged step-result handle (caller-owned, reused). */
HypStatus hyp_step_result_create(HypStepResult* out_result);

/** Free a step-result handle and set *result to NULL. A NULL handle is a no-op. */
HypStatus hyp_step_result_free(HypStepResult* result);

/** Copy a step result's fields into a caller-owned ``HypStepResultFields``.
 *  The serving path reads the sampled ``token_id`` / ``logit`` / top-k logprobs
 *  back across the ABI with this accessor (M3 serving: 13 -> 14, abi_version
 *  2 -> 3; see ADR 0003). Does NOT consume the handle — the caller still owns
 *  the handle and must free it with ``hyp_step_result_free``. The struct is
 *  copied out, so there is no aliasing across the seam. Returns OK on a valid
 *  handle, INVALID_ARGUMENT on a null handle, bad magic, or null out_fields. */
HypStatus hyp_step_result_fields(HypStepResult result,
                                 HypStepResultFields* out_fields);

#ifdef __cplusplus
}
#endif

#endif
