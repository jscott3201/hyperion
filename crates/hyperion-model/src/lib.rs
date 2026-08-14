//! Model-family identity, load dispatch, and family-specific validated geometry.

pub mod geometry;
pub mod load;
pub mod weights;

/// Stable architecture-family identity, independent of checkpoint marketing names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamily {
    /// Google's Gemma 4 family.
    Gemma4,
    /// Qwen's `qwen3_5` hybrid Gated DeltaNet/full-attention architecture.
    Qwen35Hybrid,
}

impl ModelFamily {
    /// Stable family name used in diagnostics and evidence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gemma4 => "gemma-4",
            Self::Qwen35Hybrid => "qwen3_5",
        }
    }
}

/// Primary M0/M1 checkpoint from which the on-device Q4 artifact is derived.
pub const PRIMARY_SOURCE_MODEL_ID: &str = "google/gemma-4-12B-it-qat-q4_0-unquantized";

/// Immutable Hugging Face revision reviewed for M0 acquisition.
pub const PRIMARY_SOURCE_REVISION: &str = "b6ed86275a6a5735884e208bfed95b445a684ca2";

/// First Qwen checkpoint targeted by the dual-family program.
pub const QWEN38_SOURCE_MODEL_ID: &str = "Qwen/Qwen3.8-27B";

/// Immutable Qwen3.8 revision reviewed for architecture and artifact identity.
pub const QWEN38_SOURCE_REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_and_source_identities_are_stable() {
        assert_eq!(ModelFamily::Gemma4.as_str(), "gemma-4");
        assert_eq!(ModelFamily::Qwen35Hybrid.as_str(), "qwen3_5");
        assert!(PRIMARY_SOURCE_MODEL_ID.starts_with("google/gemma-4-"));
        assert_eq!(PRIMARY_SOURCE_REVISION.len(), 40);
        assert_eq!(QWEN38_SOURCE_MODEL_ID, "Qwen/Qwen3.8-27B");
        assert_eq!(QWEN38_SOURCE_REVISION.len(), 40);
    }
}
