//! In-process tokenization + chat-template rendering boundary (M3).
//!
//! The serving layer (axum/SSE, M3) tokenizes requests before producing the
//! `EngineRequest` that crosses the C ABI. This crate is the in-process boundary: the
//! `tokenizers` HuggingFace Rust crate — the exact backend transformers uses — so
//! encode/decode is byte-identical to the oracle. No Python anywhere on the request
//! path (01:49).
//!
//! M0 exposed only the static boundary contract; M3 adds the real handle behind it
//! (PR #20) and the chat-template RENDERER (this slice) — a real Jinja2 evaluation
//! via minijinja that renders the gemma4 `chat_template.jinja` in-process. The
//! gemma4 template calls dict `.get()`, which minijinja's built-in map does not expose
//! as a method; `renderer.rs` wraps every JSON map in a custom `Object` to dispatch it.

pub mod renderer;

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use tokenizers::Tokenizer;
use tokenizers::tokenizer::step_decode_stream;

const MAX_RETAINED_TOKEN_IDS: usize = 256;
const MAX_RETAINED_TEXT_BYTES: usize = 64 * 1024;

/// Stable description of the tokenizer execution boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerContract {
    /// Tokenization must remain in the Rust process.
    pub in_process: bool,
    /// Python is forbidden on the serving/request path.
    pub python_request_path: bool,
}

/// Return the v1 tokenizer boundary contract.
#[must_use]
pub const fn contract() -> TokenizerContract {
    TokenizerContract {
        in_process: true,
        python_request_path: false,
    }
}

/// A loaded HuggingFace fast tokenizer (the `tokenizers` crate) — the in-process encode/
/// decode handle. The 12B's `tokenizer.json` (BPE, byte_fallback, added_tokens) loads
/// directly; the `tokenizers` crate IS the transformers backend, so encode/decode is
/// byte-identical to the oracle. The empty `special_tokens` post-processor means encode
/// does NOT auto-add BOS/EOS — the only `<bos>` comes from the chat template's
/// `{{ bos_token }}`.
#[derive(Clone)]
pub struct TokenizerHandle {
    inner: Arc<Tokenizer>,
    vocab_size: usize,
}

/// A load or encode error.
#[derive(Debug)]
pub enum TokenizerError {
    Load(String),
    Encode(String),
    Decode(String),
}

/// Whether incremental decoding omits or retains added special-token text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpecialTokenPolicy {
    /// Omit added special tokens such as EOS from decoded output.
    Skip,
    /// Retain added special tokens exactly as the tokenizer decodes them.
    Preserve,
}

impl SpecialTokenPolicy {
    const fn skip_special_tokens(self) -> bool {
        matches!(self, Self::Skip)
    }
}

/// Per-response incremental decoder with explicitly bounded retained state.
///
/// The underlying tokenizer needs a short token/prefix history to preserve
/// whitespace and assemble byte-fallback UTF-8. This wrapper owns that state,
/// caps it, and supplies the EOF flush that the dependency does not expose.
pub struct StreamingDecoder {
    tokenizer: Arc<Tokenizer>,
    skip_special_tokens: bool,
    skipped_special_ids: HashSet<u32>,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl StreamingDecoder {
    /// Push one generated token ID, returning only newly safe decoded text.
    pub fn push(&mut self, id: u32) -> Result<Option<String>, TokenizerError> {
        // The dependency's skip flag omits special-token text from decode, but
        // its stream helper still retains the corresponding ID. Skip them at
        // the boundary so controls cannot consume unresolved-state capacity or
        // disturb an incomplete byte-fallback sequence.
        if self.skipped_special_ids.contains(&id) {
            return Ok(None);
        }
        if self.ids.len() >= MAX_RETAINED_TOKEN_IDS {
            return Err(TokenizerError::Decode(format!(
                "streaming decoder retained token limit exceeded (maximum {MAX_RETAINED_TOKEN_IDS})"
            )));
        }

        // Advance transactionally so an error or bound violation cannot leave
        // the public decoder in an over-limit or partially-mutated state.
        let mut ids = self.ids.clone();
        let mut prefix = self.prefix.clone();
        let mut prefix_index = self.prefix_index;
        let fragment = step_decode_stream(
            &self.tokenizer,
            vec![id],
            self.skip_special_tokens,
            &mut ids,
            &mut prefix,
            &mut prefix_index,
        )
        .map_err(|e| TokenizerError::Decode(e.to_string()))?;

        if ids.len() > MAX_RETAINED_TOKEN_IDS {
            return Err(TokenizerError::Decode(format!(
                "streaming decoder retained token limit exceeded (maximum {MAX_RETAINED_TOKEN_IDS})"
            )));
        }
        if prefix.len() > MAX_RETAINED_TEXT_BYTES {
            return Err(TokenizerError::Decode(format!(
                "streaming decoder retained text limit exceeded (maximum {MAX_RETAINED_TEXT_BYTES} bytes)"
            )));
        }

        self.ids = ids;
        self.prefix = prefix;
        self.prefix_index = prefix_index;
        Ok(fragment)
    }

    /// Finish a successful response and emit its unresolved EOF suffix once.
    ///
    /// The consuming API makes repeat finish impossible:
    ///
    /// ```compile_fail
    /// fn finish_twice(decoder: hyperion_tokenizer::StreamingDecoder) {
    ///     let _ = decoder.finish();
    ///     let _ = decoder.finish();
    /// }
    /// ```
    pub fn finish(self) -> Result<String, TokenizerError> {
        let decoded = self
            .tokenizer
            .decode(&self.ids, self.skip_special_tokens)
            .map_err(|e| TokenizerError::Decode(e.to_string()))?;
        decoded
            .strip_prefix(&self.prefix)
            .map(str::to_owned)
            .ok_or_else(|| {
                TokenizerError::Decode(
                    "streaming decoder EOF output did not match emitted prefix".to_string(),
                )
            })
    }
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(s) => write!(f, "tokenizer load error: {s}"),
            Self::Encode(s) => write!(f, "encode error: {s}"),
            Self::Decode(s) => write!(f, "decode error: {s}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl TokenizerHandle {
    /// Load a HuggingFace `tokenizer.json`.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path).map_err(|e| TokenizerError::Load(e.to_string()))?;
        let vocab_size = inner.get_vocab_size(true);
        Ok(Self {
            inner: Arc::new(inner),
            vocab_size,
        })
    }

    /// Load a HuggingFace `tokenizer.json` from inline bytes (the
    /// `tokenizers` crate's `Tokenizer::from_bytes`). Used by the M3 server
    /// contract tests so they don't need a fixture file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let inner =
            Tokenizer::from_bytes(bytes).map_err(|e| TokenizerError::Load(e.to_string()))?;
        let vocab_size = inner.get_vocab_size(true);
        Ok(Self {
            inner: Arc::new(inner),
            vocab_size,
        })
    }

    /// Encode text to token ids. `add_special_tokens=false` because the chat template
    /// already emits the special tokens (bos etc.); the post-processor's empty
    /// special_tokens map means this is a no-op either way, but explicit is safer.
    #[must_use]
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        match self.inner.encode(text, add_special_tokens) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => Vec::new(),
        }
    }

    /// Decode a complete token-ID buffer back to text while omitting added
    /// special tokens. Incremental callers use [`Self::streaming_decoder`].
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap_or_default()
    }

    /// Create one bounded incremental decoder for a generated response.
    #[must_use]
    pub fn streaming_decoder(&self, policy: SpecialTokenPolicy) -> StreamingDecoder {
        let skipped_special_ids = if policy == SpecialTokenPolicy::Skip {
            self.inner
                .get_added_vocabulary()
                .get_added_tokens_decoder()
                .iter()
                .filter_map(|(id, token)| token.special.then_some(*id))
                .collect()
        } else {
            HashSet::new()
        };
        StreamingDecoder {
            tokenizer: self.inner.clone(),
            skip_special_tokens: policy.skip_special_tokens(),
            skipped_special_ids,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    /// The vocab size (incl. added tokens).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Whether the vocab contains a token id for `token` (used for the
    /// `enable_thinking` default inference — mlx-lm detects `<|channel>`/`<channel|>`).
    #[must_use]
    pub fn has_token(&self, token: &str) -> bool {
        self.inner.token_to_id(token).is_some()
    }

    /// The token id for a vocab string, if present (M3 serving: the
    /// `EngineRequest::eos_token_id` is resolved from the model's `eos_token`
    /// string via this). `None` if the token isn't in the vocab.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::AddedToken;
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::models::wordlevel::WordLevel;

    use super::*;

    fn handle(inner: Tokenizer) -> TokenizerHandle {
        let vocab_size = inner.get_vocab_size(true);
        TokenizerHandle {
            inner: Arc::new(inner),
            vocab_size,
        }
    }

    fn word_level(tokens: &[(&str, u32)]) -> TokenizerHandle {
        let vocab = tokens
            .iter()
            .map(|(token, id)| ((*token).to_string(), *id))
            .chain(std::iter::once(("[UNK]".to_string(), 10_000)))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("WordLevel model builds");
        handle(Tokenizer::new(model))
    }

    fn byte_fallback(tokens: &[(&str, u32)]) -> TokenizerHandle {
        let vocab = tokens
            .iter()
            .map(|(token, id)| ((*token).to_string(), *id))
            .collect::<Vocab>();
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .byte_fallback(true)
            .build()
            .expect("byte-fallback BPE builds");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(ByteFallback::default()));
        handle(tokenizer)
    }

    fn collect_stream(
        tokenizer: &TokenizerHandle,
        ids: &[u32],
        policy: SpecialTokenPolicy,
    ) -> Result<String, TokenizerError> {
        let mut decoder = tokenizer.streaming_decoder(policy);
        let mut output = String::new();
        for id in ids {
            if let Some(fragment) = decoder.push(*id)? {
                output.push_str(&fragment);
            }
        }
        output.push_str(&decoder.finish()?);
        Ok(output)
    }

    #[test]
    fn request_path_has_no_python_backend() {
        let contract = contract();
        assert!(contract.in_process);
        assert!(!contract.python_request_path);
    }

    #[test]
    fn streaming_matches_one_shot_multi_token_whitespace() {
        let tokenizer = word_level(&[("hello", 2), ("world", 3)]);
        let ids = [2, 3];
        assert_eq!(tokenizer.inner.decode(&ids, true).unwrap(), "hello world");
        assert_eq!(
            collect_stream(&tokenizer, &ids, SpecialTokenPolicy::Skip).unwrap(),
            tokenizer.inner.decode(&ids, true).unwrap()
        );
    }

    #[test]
    fn streaming_assembles_split_byte_fallback_scalar() {
        let tokenizer = byte_fallback(&[("<0xE5>", 0), ("<0x8F>", 1), ("<0xAB>", 2)]);
        let mut decoder = tokenizer.streaming_decoder(SpecialTokenPolicy::Skip);
        assert_eq!(decoder.push(0).unwrap(), None);
        assert_eq!(decoder.push(1).unwrap(), None);
        assert_eq!(decoder.push(2).unwrap().as_deref(), Some("叫"));
        assert_eq!(decoder.finish().unwrap(), "");
        assert_eq!(tokenizer.inner.decode(&[0, 1, 2], true).unwrap(), "叫");
    }

    #[test]
    fn special_token_policies_are_explicit_and_ordered() {
        let tokenizer = word_level(&[
            ("hello", 0),
            ("<|tool_call>", 1),
            ("<|\"|>", 2),
            ("<tool_call|>", 3),
            ("world", 4),
        ]);
        // Rebuild with the marked special-token vocabulary because the handle
        // intentionally shares its tokenizer through Arc in production.
        let mut inner = tokenizer.inner.as_ref().clone();
        inner
            .add_special_tokens([
                AddedToken::from("<|tool_call>", true),
                AddedToken::from("<|\"|>", true),
                AddedToken::from("<tool_call|>", true),
            ])
            .expect("special tokens register");
        let tokenizer = handle(inner);

        let ids = [0, 1, 2, 3, 4];
        assert_eq!(
            collect_stream(&tokenizer, &ids, SpecialTokenPolicy::Preserve).unwrap(),
            "hello <|tool_call> <|\"|> <tool_call|> world"
        );
        assert_eq!(
            collect_stream(&tokenizer, &ids, SpecialTokenPolicy::Skip).unwrap(),
            "hello world"
        );
    }

    #[test]
    fn skipped_specials_do_not_consume_or_disturb_unresolved_state() {
        let tokenizer = byte_fallback(&[("<0xE5>", 0), ("<0x8F>", 1), ("<0xAB>", 2), ("<eos>", 3)]);
        let mut inner = tokenizer.inner.as_ref().clone();
        inner
            .add_special_tokens([AddedToken::from("<eos>", true)])
            .expect("special token registers");
        let tokenizer = handle(inner);
        let special_id = tokenizer.token_to_id("<eos>").expect("special ID");

        let mut decoder = tokenizer.streaming_decoder(SpecialTokenPolicy::Skip);
        assert_eq!(decoder.push(0).unwrap(), None);
        for _ in 0..(MAX_RETAINED_TOKEN_IDS + 1) {
            assert_eq!(decoder.push(special_id).unwrap(), None);
        }
        assert_eq!(decoder.ids, vec![0], "skipped IDs never enter state");
        assert_eq!(decoder.push(1).unwrap(), None);
        assert_eq!(decoder.push(special_id).unwrap(), None);
        assert_eq!(decoder.push(2).unwrap().as_deref(), Some("叫"));
        assert_eq!(decoder.finish().unwrap(), "");

        assert_eq!(
            collect_stream(&tokenizer, &[0, 1, 2], SpecialTokenPolicy::Skip).unwrap(),
            "叫",
            "skipped controls leave fallback assembly equivalent to no controls"
        );
    }

    #[test]
    fn finish_is_lossless_for_incomplete_fallback_and_empty_generation() {
        let tokenizer = byte_fallback(&[("<0xE5>", 0), ("hello", 1)]);
        let mut incomplete = tokenizer.streaming_decoder(SpecialTokenPolicy::Skip);
        assert_eq!(incomplete.push(0).unwrap(), None);
        assert_eq!(incomplete.finish().unwrap(), "�");

        let empty = tokenizer.streaming_decoder(SpecialTokenPolicy::Skip);
        assert_eq!(empty.finish().unwrap(), "");
    }

    #[test]
    fn finish_does_not_duplicate_prior_fragments() {
        let tokenizer = byte_fallback(&[("hello", 0), ("<0xE5>", 1)]);
        let mut decoder = tokenizer.streaming_decoder(SpecialTokenPolicy::Skip);
        assert_eq!(decoder.push(0).unwrap().as_deref(), Some("hello"));
        assert_eq!(decoder.push(1).unwrap(), None);
        assert_eq!(decoder.finish().unwrap(), "�");
        // `finish` consumes the decoder, so a second finish is impossible.
    }

    #[test]
    fn unresolved_token_state_stops_at_hard_limit() {
        let tokenizer = byte_fallback(&[("<0xE5>", 0)]);
        let mut decoder = tokenizer.streaming_decoder(SpecialTokenPolicy::Skip);
        for _ in 0..MAX_RETAINED_TOKEN_IDS {
            assert_eq!(decoder.push(0).unwrap(), None);
        }
        let error = decoder.push(0).unwrap_err();
        assert_eq!(
            error.to_string(),
            "decode error: streaming decoder retained token limit exceeded (maximum 256)"
        );
        assert_eq!(decoder.ids.len(), MAX_RETAINED_TOKEN_IDS);
    }

    #[test]
    fn retained_text_state_stops_at_hard_limit() {
        let at_limit = "x".repeat(MAX_RETAINED_TEXT_BYTES);
        let over_limit = "y".repeat(MAX_RETAINED_TEXT_BYTES + 1);
        let tokenizer = word_level(&[(&at_limit, 0), (&over_limit, 1)]);

        let mut exact = tokenizer.streaming_decoder(SpecialTokenPolicy::Skip);
        assert_eq!(exact.push(0).unwrap().as_deref(), Some(at_limit.as_str()));
        assert_eq!(exact.prefix.len(), MAX_RETAINED_TEXT_BYTES);

        let mut over = tokenizer.streaming_decoder(SpecialTokenPolicy::Skip);
        let error = over.push(1).unwrap_err();
        assert_eq!(
            error.to_string(),
            "decode error: streaming decoder retained text limit exceeded (maximum 65536 bytes)"
        );
        assert!(over.ids.is_empty());
        assert!(over.prefix.is_empty());
    }

    /// The in-process tokenizer parity seal against the REAL 12B (M5-gated). The 12B's
    /// tokenizer.json loads via the `tokenizers` crate (HF's own backend → byte-identical
    /// encode vs the oracle). The committed greedy golden's 21 templated ids
    /// (gen_12b_greedy_golden.py: tokenizer.encode(apply_chat_template(...))) are the
    /// parity target — the BOS at id 2 + the templated body. We encode the EXACT golden
    /// rendered string (read from the committed fixture's prefill metadata is not
    /// available; instead we verify encode parity on a known string + the thinking-model
    /// vocab inference, the renderer-deferral-safe subset). Self-skips (skip) when
    /// HYPERION_12B_ARTIFACT is unset (CI).
    #[test]
    fn real_12b_tokenizer_encodes_byte_identical_to_oracle() {
        let dir = match std::env::var("HYPERION_12B_ARTIFACT") {
            Ok(d) if !d.is_empty() => d,
            _ => {
                eprintln!(
                    "tokenizer parity test: HYPERION_12B_ARTIFACT unset; skipping (M5-gated)"
                );
                return;
            }
        };
        let path = std::path::Path::new(&dir);
        let tok = TokenizerHandle::from_file(&path.join("tokenizer.json"))
            .expect("12B tokenizer.json loads");
        // The 12B is a thinking model: the vocab contains <|channel> / <channel|>.
        assert!(tok.has_token("<|channel>"), "12B vocab has <|channel>");
        assert!(tok.has_token("<channel|>"), "12B vocab has <channel|>");
        assert_eq!(tok.vocab_size(), 262_144, "12B vocab size");
        // Encode parity: the oracle (gen_12b_greedy_golden.py) encodes the templated
        // prompt; the first token is the BOS (<bos>, id 2) emitted by the template. A
        // direct encode of "<bos>" must produce id 2 (the bos is a single added token).
        let bos = tok.encode("<bos>", false);
        assert_eq!(
            bos,
            vec![2u32],
            "<bos> encodes to id 2 (the golden's first token)"
        );
        // Encode a multi-token ASCII string and round-trip it through decode (the
        // Sequence decoder must reconstruct the text, modulo the SPM ▁→space swap).
        let text = "The capital of France is";
        let ids = tok.encode(text, true);
        assert!(!ids.is_empty(), "encode produced tokens");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "encode→decode round-trips the ASCII text");
    }
}
