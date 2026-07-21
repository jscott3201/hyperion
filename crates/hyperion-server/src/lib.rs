//! M0 server entry boundary. HTTP serving intentionally arrives at M3.

use hyperion_core::EngineIdentity;

/// Information used by the binary's honest M0 build report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuildInfo {
    /// Engine identity consumed on the future serving path.
    pub engine: EngineIdentity,
    /// Whether M3's HTTP surface is implemented.
    pub serving_available: bool,
}

/// Return build state without advertising a stub server as functional.
#[must_use]
pub fn build_info() -> BuildInfo {
    let tokenizer = hyperion_tokenizer::contract();
    let engine = hyperion_core::identity();
    debug_assert_eq!(engine.tokenizer, tokenizer);
    BuildInfo {
        engine,
        serving_available: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m0_does_not_claim_a_serving_backend() {
        let info = build_info();
        assert_eq!(info.engine.native_backend_count, 1);
        assert!(!info.serving_available);
    }
}
