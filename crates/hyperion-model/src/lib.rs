//! Gemma 4 model identity and, from M2 onward, validated geometry.

pub mod geometry;
pub mod weights;

/// The only model family accepted by Hyperion v1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamily {
    /// Google's Gemma 4 family.
    Gemma4,
}

impl ModelFamily {
    /// Stable family name used in diagnostics and evidence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gemma4 => "gemma-4",
        }
    }
}

/// Primary M0/M1 checkpoint from which the on-device Q4 artifact is derived.
pub const PRIMARY_SOURCE_MODEL_ID: &str = "google/gemma-4-12B-it-qat-q4_0-unquantized";

/// Immutable Hugging Face revision reviewed for M0 acquisition.
pub const PRIMARY_SOURCE_REVISION: &str = "b6ed86275a6a5735884e208bfed95b445a684ca2";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_is_gemma_four_only() {
        assert_eq!(ModelFamily::Gemma4.as_str(), "gemma-4");
        assert!(PRIMARY_SOURCE_MODEL_ID.starts_with("google/gemma-4-"));
        assert_eq!(PRIMARY_SOURCE_REVISION.len(), 40);
    }
}
