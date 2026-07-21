# Gemma 4 canonical/custom chat-template analysis

- Date: 2026-07-21
- Status: verified design input for M3
- Scope: rendering semantics only; no serving implementation is pulled into M0

## Pins and result

Hyperion will use the checkpoint's current canonical Gemma 4 template as the M3 source of
truth, pinned by model revision and checksum. It will not carry a project-owned fork of the
template.

- Canonical source: [`google/gemma-4-12B-it` `chat_template.jinja`][official-template], whose
  current template change is commit
  `711c1368e39f1712f48ff0eb7bcdbbb760d52db0`.
- M0 checkpoint source: `google/gemma-4-12B-it-qat-q4_0-unquantized` at immutable revision
  `b6ed86275a6a5735884e208bfed95b445a684ca2`.
- Local source and converted copies: 18,683 bytes, SHA-256
  `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`; both are byte-identical
  to the canonical file reviewed on this date.
- Compared custom source: [`custom_pub_chat_template_gemma4.jinja`][custom-template] at gist
  revision `0b7d5b52e9beed4997b0e65b6366e21d616fbd64`, 31,881 bytes, SHA-256
  `b20f7b8784dbbcccf9f8ecf5d695578c1ec64693ce500b03faebfe8ba3dca8f9`.
- Validation environment: Transformers 5.14.1 and Jinja2 3.1.6 from the locked project oracle.

The [July canonical change][canonical-change] now contains the custom iteration's most
important serializer fixes: JSON `null`, fail-loud rejection of stringified tool arguments,
balanced continuation turns, and safer optional-input handling. The remaining differences
make the canonical template the preferable pinned wire reference. Its reasoning-retention
policy follows [Google's prompt-format guidance][google-prompt]—strip thinking across
ordinary turns while retaining it within a live function-calling turn—but that policy still
requires a real-model behavioral gate before Hyperion enables thinking in production.

## Semantic comparison

| Area | Current canonical template | Earlier custom iteration | Hyperion ruling |
|---|---|---|---|
| Thinking defaults | `enable_thinking=false`; `preserve_thinking=false` | Both default `true` | Pass policy explicitly per request; default to no cross-turn thought retention. |
| Thought retention | Keeps current-turn reasoning; historical reasoning is restored only when `preserve_thinking=true` and that message has tool calls | Requires tool calls even for current-turn reasoning and preserves historical tool reasoning by default | Keep canonical bytes as the reference, but gate the production policy on the real-model loop test below. |
| Post-tool generation | With thinking enabled, resumes with `<\|channel>thought\n` after a terminal tool response | Does not add that continuation cue | Freeze the canonical suffix in a golden fixture and exercise it in the behavioral gate. |
| Tool arguments | Renders null as `null`; rejects stringified/non-mapping arguments | Same core fixes | Deserialize once at ingress and return a deterministic client error before rendering. |
| Optional inputs | Guards empty messages, uses `.get()`, recognizes falsy names as `unknown` | More direct indexing and weaker empty/falsy handling | Normalize and validate first, while remaining byte-identical to canonical output for valid input. |
| Turn closure | Balances tool/assistant continuations and closes a tool-response turn before a following user | Can omit that close in this history shape | Canonical bytes are authoritative. |
| Consecutive assistant chunks | Concatenates chunks without an inserted newline | Inserts a newline | Normalize unusual histories deliberately, then freeze canonical concatenation. |
| OpenAI media aliases | Recognizes `image_url` and `input_audio` as well as native names | Recognizes only `image` and `audio` | Preserve renderer parity; Hyperion v1 still rejects unsupported multimodal requests. |
| Tool-response fallback | Missing, empty, or unmatched names render as `unknown` | An empty name may remain empty | Reject broken correlation before rendering; do not depend on fallback recovery. |

The canonical template also replaced a backward scan with tracked previous-role state, handles
tool-call-plus-response continuations, and avoids an extra terminal turn close. These are
implementation differences whose output is observable in malformed or fragmented histories.

## Open behavioral risk: reasoning reinjection

Canonical rendering is not itself evidence that the resulting agent loop is stable. An open
[upstream report][loop-report] attributes verbatim repetition or runaway generation in
multi-step tool use to reinjecting earlier raw thought blocks from the same live turn. Its
reported 24-run harness saw failures in 7 runs with the merged change and
`preserve_thinking=true`, 9 with it false, and none when reinjection was disabled. That is
material counterevidence, not a Hyperion result: the report's directly validated model was a
third-party derivative, and the pinned 12B QAT checkpoint has not yet been measured here.

Simply removing live-turn thoughts is not an assumed fix. Google's guidance explicitly says
they must remain between function calls inside one model turn. M3 must therefore run the
pinned 12B model through a deterministic, stubbed multi-step tool scenario under both
`preserve_thinking=false` and `true`, across fixed seeds. Record exact rendered bytes, token
IDs, seeds, generated tokens, stop reasons, and a repetition metric. Every run must terminate
within fixed tool-step and generation-token limits, without verbatim thought replay or runaway
output. A failure blocks thinking-mode enablement until a separate ADR chooses and validates a
mitigation; it does not authorize an unrecorded template fork. Thinking-off remains the
default baseline.

## Required normalization outside the template

The template is a serializer, not a conversation validator. Hyperion must enforce these rules
before rendering:

1. Consolidate OpenAI `system`/`developer` and Anthropic system input into one initial system
   turn. Only the first such message is hoisted; a later developer message would otherwise
   serialize as unsupported `<|turn>developer`.
2. Mint unique external tool-call IDs, reject duplicate/missing/orphan IDs, and correlate every
   response before IDs disappear from Gemma's prompt wire format.
3. Preserve tool-call list order and reorder responses to that call order. Same-name parallel
   calls cannot be distinguished by name after serialization.
4. Dedupe exact `name + canonicalized arguments` calls at the server policy boundary, not in
   the template.
5. Validate tool identifiers, schema text, keys, descriptions, and argument strings against
   reserved Gemma control-token injection. The template does not escape a literal `<|"|>` or
   other control tokens embedded in user-controlled values, so ambiguous input must fail
   closed.

## M3 golden/negative fixtures and behavioral gate

Every positive fixture compares rendered bytes and token IDs against pinned Transformers;
every negative fixture must fail before generation.

1. Default thinking-off user prompt, including the exact empty-thought primer.
2. Thinking-on system turn with `<|think|>` before system text and tool declarations.
3. Current-turn text reasoning retained; identical reasoning stripped after a later user turn.
4. Multi-step tool-chain reasoning retained only inside the current turn.
5. Terminal tool response with thinking on ends in `<|channel>thought\n`.
6. Nested schema with enum, nullable, arrays, required fields, and sorted properties.
7. Null, boolean, number, list, and nested arguments; stringified arguments produce a stable
   client error.
8. Parallel calls with reversed response arrival normalize back to call order.
9. Same-name parallel calls retain unique external IDs even though IDs vanish from prompt
   bytes.
10. Duplicate, missing, and orphan tool IDs fail before rendering.
11. Native embedded responses, OpenAI role-`tool`, and Anthropic-normalized results render
    identically.
12. Tool-to-final-assistant and tool-to-new-user histories have balanced turn tags.
13. Consecutive assistant continuation matches canonical byte concatenation.
14. System/developer normalization emits one system turn and never `<|turn>developer`.
15. Raw thought blocks in assistant content are stripped without duplicating structured
    reasoning.
16. Reserved control-token injection in tool names, schema keys/descriptions, or argument
    strings fails closed.
17. Template revision/checksum drift fails cumulative tier 1.
18. Empty messages, missing content, media aliases, empty tool names, and unmatched response
    IDs exercise canonical fallback bytes even when the public API rejects the same shapes.
19. The real pinned 12B checkpoint completes a bounded multi-step tool loop without verbatim
    thought repetition or runaway generation under both preservation modes and fixed seeds,
    with the full evidence named in the behavioral-risk section.

[official-template]: https://huggingface.co/google/gemma-4-12B-it/blob/main/chat_template.jinja
[custom-template]: https://gist.github.com/jscott3201/ad69c4ffbd79f18b11a0f6a94c94fadf
[canonical-change]: https://huggingface.co/google/gemma-4-12B-it/discussions/35
[loop-report]: https://huggingface.co/google/gemma-4-12B-it/discussions/38
[google-prompt]: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
