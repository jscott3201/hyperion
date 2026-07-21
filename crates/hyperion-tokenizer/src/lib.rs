//! In-process tokenization and transcript policy boundary.
//!
//! M3 adds the real tokenizer, template, and tool-call wire format. M0 exposes
//! only the invariant consumed by the live server path; there is no helper
//! subprocess or alternate backend.

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_path_has_no_python_backend() {
        let contract = contract();
        assert!(contract.in_process);
        assert!(!contract.python_request_path);
    }
}
