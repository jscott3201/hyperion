# 06 — Serving & agentic API

The HTTP/SSE surfaces are shared. Rendering, tokenization, stop rules, generation defaults,
thinking/tool grammar, and incremental parsing belong to an exact checkpoint conversation
profile. A Qwen request must never be rendered or parsed by the current Gemma profile.

## Surfaces (carry mlx-bonsai's dual-dialect core)

Single axum router, localhost-first (`127.0.0.1` bind default; non-loopback REQUIRES a
bearer token or refuses to start — fail-closed, bonsai rule), 32 MiB body limit.

| Endpoint | Dialect |
|---|---|
| `POST /v1/messages` (+ `count_tokens`) | Anthropic-native, real SSE event sequence (message_start → content_block_* → message_delta → message_stop) |
| `POST /v1/chat/completions`, `GET /v1/models[/{id}]` | OpenAI-native, chunked SSE + `[DONE]` |
| `GET /control/health`, `/control/stats`, `POST /control/reload`, `/control/shutdown` | Ops |

Internally both dialects normalize to one `PreparedPrompt` through the selected conversation
profile (same checkpoint template render, same
`EngineRequest`); dialect only selects SSE framing + error envelope. Session identity for the
conversation cache: honor an explicit `session_id` metadata field on both dialects, else
fall back to prefix-chain fingerprint.

## Error taxonomy (carried verbatim — hard-won)

400 malformed / bad tool schema / unsupported tool_choice; 401 constant-time auth;
409 reload conflict; **413** context overflow (`prompt + max_tokens > context`) AND body
limit; **429** single-flight busy; **529** governor rejection pre-stream, SSE `error`
event mid-stream (headers already flushed); 503 not-ready; 500 internal (opaque,
detail server-side only). Contract tests for every code (bonsai has them; port).

## Streaming

Token-incremental real SSE (Helios's 3-frame post-hoc "streaming" is banned). Decode loop
emits per-step deltas through the bounded channel; cancel-on-client-drop propagates to the
engine thread between steps/chunks. Usage accounting in the terminal frame (both dialects).

## Sampler surface

Greedy (`temperature: 0`) and sampled modes. Defaults come from the exact checkpoint generation
profile. The current Gemma profile uses t=1.0, top_p=0.95, top_k=64. min_p, presence/frequency/repetition
penalties supported. Per-request `seed` honored and echoed. Deterministic tagging runs use
greedy intent; interactive assistant workloads use sampled defaults — both are first-class
(greedy-only was a bonsai limitation, dropped).

## Tool calling (current Gemma 4 conversation profile)

- Declarations rendered into the template's `<|tool|>declaration:` block from OpenAI
  `tools` / Anthropic `tools` schemas.
- Model emits `<|tool_call>call:name{args}<tool_call|>` blocks (string literals
  `<|"|>`-delimited).
  hyperion's incremental parser (adapt bonsai `tool_call.rs`): tolerant assembly from
  streamed fragments, schema validation, argument-JSON coercion **degrade-never-drop**
  (a malformed-but-recoverable call surfaces as a call + `repaired: true` telemetry, never
  silently dropped text), server-minted IDs.
- **Parallel calls: supported, bounded.** Gemma 4 emits multiple calls per turn; known
  in-the-wild failure mode is duplicate identical calls (LM Studio bug reports). Policy:
  accept up to `max_tool_calls_per_turn` (default 8), **dedupe exact-duplicate
  name+canonicalized-args within a turn** (dedupe recorded in telemetry, surfaced once),
  `tool_choice: auto | none` (forced-tool 400s in v1, matching bonsai).
- `<|tool_response|>` rendering on the way back in, per template.
- Raw-fidelity telemetry: parsed/wellformed/repaired counters per request → feeds agent-eval
  "server-repaired upper bound" honesty (bonsai discipline).

Qwen's profile uses its own XML-like function/parameter call grammar, tool-response folding,
stop IDs, and thinking defaults. It requires separate golden and incremental-parser fixtures;
none of the Gemma markers above are shared protocol.

## Thinking mode (current Gemma profile)

- Enable via template (`<|think|>` in system turn / `enable_thinking`); 12B-it template's
  empty-thinking stabilization token honored (ghost-thought suppression).
- Transcript policy enforced by `hyperion-tokenizer` at render time: **thought blocks
  stripped from prior turns; retained across tool-call chains within the current turn**
  (Google's documented rule; community templates get this wrong — golden fixtures pin ours).
- `<|channel|>thought` deltas streamed as a distinct SSE lane (Anthropic: `thinking` content
  block type; OpenAI: `reasoning_content`-style extension field) so agent frameworks can
  render/budget them.
- **Thinking budget:** per-request `max_thinking_tokens` (default unlimited; deployment
  profiles may set 512–2048). On budget hit: inject the thought-close sequence and continue
  to the answer — policy lives Rust-side, no native knowledge. (Budget policy is our design;
  the mode itself is Google-documented.)

## Constrained JSON (optional shared agentic lane; legacy M5)

Reality: neither predecessor had it; both post-hoc repaired. For the target tool-driven
workloads, tool-argument validity is the reliability boundary, and "validator disposes" plus repair
telemetry already de-risks v0. So:

- **Implemented Gemma baseline (legacy M3):** post-hoc parser + repair + `validator disposes`
  downstream.
- **candidate:** schema-guided token masking for tool-argument JSON: compile the active tool's
  JSON-Schema to a token-mask automaton over the selected conversation profile's exact
  vocabulary (Gemma 262144; Qwen3.8 248320). A Swift port suggests roughly 3–10% decode overhead.
  Scope the candidate masks to: object
  structure, key names, string/number/bool/null, enums. NOT full JSON-Schema (no regex
  patterns, no oneOf recursion) — bounded, testable, honest.
- Gate: constrained mode must strictly dominate post-hoc on malformed-rate at ≤10% decode
  overhead on the agent-eval suite, else stays opt-in.

## Concurrency contract

Single-flight generation (Semaphore(1) → 429), bounded engine channel (8), cancel-on-drop,
reset-around-every-generation. `/control/reload` swaps checkpoints atomically behind 409
protection. Multi-request continuous batching is explicitly post-v1 (local single-operator
workload; bonsai evidence: no localhost transport tax worth batching for at n=1).
