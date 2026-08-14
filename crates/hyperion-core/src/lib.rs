//! Engine-policy spine shared by the server and evidence harness.

pub mod geometry_abi;

pub use hyperion_ffi::{CanaryInfo, Error as NativeError};
use hyperion_model::ModelFamily;
use hyperion_tokenizer::TokenizerContract;

/// Static identity of the currently executable engine boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineIdentity {
    /// Families with a native executable graph in this build. Architecture
    /// recognition alone never adds an entry here.
    pub executable_model_families: &'static [ModelFamily],
    /// The serving tokenizer execution invariant.
    pub tokenizer: TokenizerContract,
    /// Number of native backends compiled into Hyperion.
    pub native_backend_count: u8,
}

/// Return the engine's compile-time scope and boundary invariants.
#[must_use]
pub const fn identity() -> EngineIdentity {
    EngineIdentity {
        executable_model_families: &[ModelFamily::Gemma4],
        tokenizer: hyperion_tokenizer::contract(),
        native_backend_count: 1,
    }
}

/// Enforce the platform floor and exercise both MLX and Hyperion's metallib.
pub fn startup_canary() -> Result<CanaryInfo, NativeError> {
    hyperion_ffi::runtime_canary()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_has_one_in_process_native_path() {
        let identity = identity();
        assert_eq!(identity.executable_model_families, &[ModelFamily::Gemma4]);
        assert!(
            !identity
                .executable_model_families
                .contains(&ModelFamily::Qwen35Hybrid)
        );
        assert_eq!(identity.native_backend_count, 1);
        assert!(identity.tokenizer.in_process);
        assert!(!identity.tokenizer.python_request_path);
    }
}
