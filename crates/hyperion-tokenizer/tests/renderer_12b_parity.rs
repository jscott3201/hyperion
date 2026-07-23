//! M3 chat-template renderer parity seal against the REAL 12B oracle (M5-gated).
//!
//! The deferred half of the M3 "in-process tokenizer + template golden
//! fixtures" deliverable (`10:40`). The tokenizer half (PR #20) sealed
//! encode/decode; this seals the RENDERER: the gemma4 `chat_template.jinja`
//! rendered via minijinja must produce the byte-identical prompt string AND
//! token-id sequence that the pinned mlx-lm oracle's
//! `tokenizer.apply_chat_template(...)` + `tokenizer.encode(...)` produces.
//!
//! This is the 21-id golden parity the M3 state handoff named. The probe that
//! established it: `examples/render_probe.rs` (a dev tool, not committed to the
//! gate); this test is the committed, self-skipping gate evidence.
//!
//! Self-skips (skip) when `HYPERION_12B_ARTIFACT` is unset (CI runs model-free).

use hyperion_tokenizer::TokenizerHandle;
use hyperion_tokenizer::renderer::{ChatMessage, ChatTemplate, RenderOptions};
use std::path::Path;

/// The exact conversation the M2-2.7 greedy golden (`gen_12b_greedy_golden.py`)
/// renders: a single user turn with the "The capital of France is" prompt and
/// the generation prompt appended. The oracle produces 21 templated ids; the
/// renderer must match every one.
#[test]
fn renderer_matches_oracle_21_id_greedy_golden_prompt() {
    let dir = match std::env::var("HYPERION_12B_ARTIFACT") {
        Ok(d) if !d.is_empty() => d,
        _ => {
            eprintln!("renderer parity test: HYPERION_12B_ARTIFACT unset; skipping (M5-gated)");
            return;
        }
    };
    let artifact = Path::new(&dir);
    let tok = TokenizerHandle::from_file(&artifact.join("tokenizer.json"))
        .expect("12B tokenizer.json loads");
    let tpl = ChatTemplate::from_artifact(artifact, Some(&tok))
        .expect("canonical chat_template.jinja compiles");

    let prompt = tpl
        .render(
            &[ChatMessage {
                role: "user".to_string(),
                content: Some(serde_json::Value::String(
                    "The capital of France is".to_string(),
                )),
                tool_calls: None,
                tool_responses: None,
                reasoning: None,
                reasoning_content: None,
                tool_call_id: None,
                name: None,
            }],
            &RenderOptions {
                add_generation_prompt: true,
                ..Default::default()
            },
        )
        .expect("render succeeds");

    // The 21-id golden: the exact sequence `gen_12b_greedy_golden.py` emits
    // (verified against the pinned mlx-lm oracle, 2026-07-23, mlx 0.32.0 /
    // mlx-lm 0.31.3, 12B g64/b4). BOS (id 2) + the templated body. If this
    // breaks, the renderer diverged from the oracle — re-derive, don't port.
    let golden_ids: [u32; 21] = [
        2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107,
        105, 4368, 107,
    ];
    let rendered_ids = tok.encode(&prompt, false);
    assert_eq!(
        rendered_ids,
        golden_ids,
        "renderer id sequence must match the oracle byte-for-byte (got {} ids)",
        rendered_ids.len()
    );
    assert_eq!(
        prompt,
        "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nThe capital of France is<turn|>\n<|turn>model\n",
        "rendered prompt string must match the oracle byte-for-byte"
    );
}

/// The `enable_thinking` override flips the thinking block off and changes the
/// generation-prompt suffix (the `<|channel>thought\n<channel|>` form). This
/// must match `apply_chat_template(enable_thinking=False)`. Guards the
/// `Option<bool>`-default-resolution path.
#[test]
fn renderer_enable_thinking_false_matches_oracle() {
    let dir = match std::env::var("HYPERION_12B_ARTIFACT") {
        Ok(d) if !d.is_empty() => d,
        _ => {
            eprintln!("renderer thinking test: HYPERION_12B_ARTIFACT unset; skipping (M5-gated)");
            return;
        }
    };
    let artifact = Path::new(&dir);
    let tok = TokenizerHandle::from_file(&artifact.join("tokenizer.json"))
        .expect("12B tokenizer.json loads");
    let tpl = ChatTemplate::from_artifact(artifact, Some(&tok))
        .expect("canonical chat_template.jinja compiles");

    let prompt = tpl
        .render(
            &[ChatMessage {
                role: "user".to_string(),
                content: Some(serde_json::Value::String("hi".to_string())),
                tool_calls: None,
                tool_responses: None,
                reasoning: None,
                reasoning_content: None,
                tool_call_id: None,
                name: None,
            }],
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: Some(false),
                ..Default::default()
            },
        )
        .expect("render succeeds");
    // No <|think|> system block; the trailing <|channel>thought\n<channel|>
    // generation prompt (the enable_thinking=False path, template line 384-386).
    assert!(
        !prompt.contains("<|think|>"),
        "enable_thinking=False suppresses the think block: {prompt:?}"
    );
    assert!(
        prompt.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
        "enable_thinking=False ends with the thought-channel generation prompt: {prompt:?}"
    );
}
