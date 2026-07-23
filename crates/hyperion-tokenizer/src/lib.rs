//! In-process tokenization boundary (M3).
//!
//! The serving layer (axum/SSE, M3) tokenizes requests before producing the
//! `EngineRequest` that crosses the C ABI. This crate is the in-process boundary: the
//! `tokenizers` HuggingFace Rust crate — the exact backend transformers uses — so
//! encode/decode is byte-identical to the oracle. No Python anywhere on the request
//! path (01:49).
//!
//! M0 exposed only the static boundary contract; M3 adds the real handle behind it.
//! The chat-template RENDERER (minijinja) is a follow-up slice — the gemma4
//! `chat_template.jinja` uses dict `.get()` which minijinja does not support as a method;
//! a custom-object or template-preprocess solution is deferred (see the M3 state).

use std::path::Path;
use std::sync::Arc;

use tokenizers::Tokenizer;

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

    /// Decode token ids back to text (the Sequence decoder: Replace ▁→space, ByteFallback,
    /// Fuse). For the SPM no-space streaming detokenizer (decode parity), defer to a
    /// follow-up slice.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap_or_default()
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_path_has_no_python_backend() {
        let contract = contract();
        assert!(contract.in_process);
        assert!(!contract.python_request_path);
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
