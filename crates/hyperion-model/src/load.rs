//! Family-aware model configuration dispatch at the serving load boundary.
//!
//! Recognizing an architecture is deliberately separate from being able to
//! execute it. The current native graph is Gemma 4-only; Qwen's `qwen3_5`
//! discriminator is recognized so it cannot accidentally fall through the
//! Gemma parser, tokenizer, or native ABI while that graph is being built.

use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ModelFamily;
use crate::geometry::{Geometry, GeometryError};

/// A family-specific executable model borrowed from a validated load plan.
///
/// This view is intentionally separate from [`ModelLoadPlan`]. Runtime callers
/// receive it from an opaque plan; the engine never accepts a caller-created
/// view or an independently supplied weight directory.
#[derive(Clone, Copy, Debug)]
pub enum ExecutableModelRef<'a> {
    /// The implemented Gemma 4 graph and state topology.
    Gemma4(&'a Geometry),
}

/// An executable model configuration bound to the artifact directory whose
/// `config.json` it came from.
///
/// Fields are private so callers cannot bypass top-level family dispatch or
/// independently pair a validated config with another weight root. This first
/// seam validates the family config, not the full weight inventory or artifact
/// digest; family-specific manifest authority is a later artifact-contract
/// slice.
#[derive(Clone, Debug)]
pub struct ModelLoadPlan {
    root: String,
    config: ExecutableModelConfig,
}

#[derive(Clone, Debug)]
enum ExecutableModelConfig {
    Gemma4(Geometry),
}

impl ModelLoadPlan {
    /// Canonicalize an artifact directory, read its `config.json`, and produce
    /// an opaque family-specific load plan.
    ///
    /// Qwen's architecture is recognized but intentionally rejected before
    /// tokenizer/template loading or native initialization. Adding a family
    /// here requires its own strict parser and executable graph; no family is
    /// coerced through Gemma's geometry.
    pub fn from_directory(root: impl AsRef<Path>) -> Result<Self, ModelLoadError> {
        let requested_root = root.as_ref();
        let canonical_root =
            requested_root
                .canonicalize()
                .map_err(|source| ModelLoadError::CanonicalizeRoot {
                    path: requested_root.to_path_buf(),
                    source,
                })?;
        let root = canonical_root
            .to_str()
            .ok_or_else(|| ModelLoadError::NonUtf8Root {
                path: canonical_root.clone(),
            })?
            .to_string();
        let config_path = canonical_root.join("config.json");
        let config_metadata = std::fs::symlink_metadata(&config_path).map_err(|source| {
            ModelLoadError::ReadConfig {
                path: config_path.clone(),
                source,
            }
        })?;
        if !config_metadata.file_type().is_file() {
            return Err(ModelLoadError::UnsafeConfigFile { path: config_path });
        }
        let json =
            std::fs::read_to_string(&config_path).map_err(|source| ModelLoadError::ReadConfig {
                path: config_path.clone(),
                source,
            })?;
        let config = parse_config_str(&json).map_err(ModelLoadError::Config)?;
        Ok(Self { root, config })
    }

    /// Canonical artifact root used for config, tokenizer, template, and
    /// native weight loading.
    #[must_use]
    pub fn root(&self) -> &Path {
        Path::new(&self.root)
    }

    /// UTF-8 canonical artifact root required by the native ABI.
    #[must_use]
    pub fn root_str(&self) -> &str {
        &self.root
    }

    /// Borrow the validated family-specific executable config.
    #[must_use]
    pub const fn executable(&self) -> ExecutableModelRef<'_> {
        match &self.config {
            ExecutableModelConfig::Gemma4(geometry) => ExecutableModelRef::Gemma4(geometry),
        }
    }

    /// Stable architecture family for evidence and dispatch.
    #[must_use]
    pub const fn family(&self) -> ModelFamily {
        match self.config {
            ExecutableModelConfig::Gemma4(_) => ModelFamily::Gemma4,
        }
    }

    /// Maximum context declared by the validated family-specific config.
    #[must_use]
    pub const fn max_context(&self) -> u32 {
        match &self.config {
            ExecutableModelConfig::Gemma4(geometry) => geometry.max_position_embeddings,
        }
    }
}

fn parse_config_str(json: &str) -> Result<ExecutableModelConfig, ModelConfigError> {
    let root: Value =
        serde_json::from_str(json).map_err(|error| ModelConfigError::MalformedConfig {
            detail: error.to_string(),
        })?;
    let root = root
        .as_object()
        .ok_or_else(|| ModelConfigError::MalformedConfig {
            detail: "config root must be a JSON object".to_string(),
        })?;
    let model_type = root
        .get("model_type")
        .ok_or(ModelConfigError::MissingModelType)?
        .as_str()
        .ok_or_else(|| ModelConfigError::MalformedConfig {
            detail: "config model_type must be a string".to_string(),
        })?;
    let family =
        outer_model_family(model_type).ok_or_else(|| ModelConfigError::UnknownArchitecture {
            model_type: model_type.to_string(),
        })?;

    if let Some(text_model_type) = root
        .get("text_config")
        .and_then(Value::as_object)
        .and_then(|text| text.get("model_type"))
        .and_then(Value::as_str)
        && let Some(text_family) = nested_model_family(text_model_type)
        && text_family != family
    {
        return Err(ModelConfigError::ConflictingArchitecture {
            outer_model_type: model_type.to_string(),
            text_model_type: text_model_type.to_string(),
        });
    }

    match family {
        ModelFamily::Gemma4 => Geometry::from_config_str(json)
            .map(ExecutableModelConfig::Gemma4)
            .map_err(ModelConfigError::Gemma4),
        ModelFamily::Qwen35Hybrid => Err(ModelConfigError::RecognizedButUnsupported { family }),
    }
}

fn outer_model_family(model_type: &str) -> Option<ModelFamily> {
    match model_type {
        "gemma4" | "gemma4_unified" => Some(ModelFamily::Gemma4),
        "qwen3_5" => Some(ModelFamily::Qwen35Hybrid),
        _ => None,
    }
}

fn nested_model_family(model_type: &str) -> Option<ModelFamily> {
    match model_type {
        "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => {
            Some(ModelFamily::Gemma4)
        }
        "qwen3_5" | "qwen3_5_text" => Some(ModelFamily::Qwen35Hybrid),
        _ => None,
    }
}

/// Failure to classify or validate a model configuration for execution.
#[derive(Debug)]
pub enum ModelConfigError {
    /// JSON syntax, root shape, or discriminator type is malformed.
    MalformedConfig { detail: String },
    /// The required top-level architecture discriminator is absent.
    MissingModelType,
    /// The top-level architecture is not recognized by this build.
    UnknownArchitecture { model_type: String },
    /// Outer and nested text discriminators name different known families.
    ConflictingArchitecture {
        outer_model_type: String,
        text_model_type: String,
    },
    /// The family is a roadmap target but has no executable graph in this build.
    RecognizedButUnsupported { family: ModelFamily },
    /// Gemma-specific strict parsing or validation failed.
    Gemma4(GeometryError),
}

impl fmt::Display for ModelConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedConfig { detail } => write!(formatter, "malformed config: {detail}"),
            Self::MissingModelType => {
                formatter.write_str("config is missing required top-level model_type")
            }
            Self::UnknownArchitecture { model_type } => {
                write!(formatter, "unknown model architecture {model_type:?}")
            }
            Self::ConflictingArchitecture {
                outer_model_type,
                text_model_type,
            } => write!(
                formatter,
                "conflicting model architectures: outer model_type {outer_model_type:?}, text_config.model_type {text_model_type:?}"
            ),
            Self::RecognizedButUnsupported { family } => write!(
                formatter,
                "recognized model family {} is not executable in this build",
                family.as_str()
            ),
            Self::Gemma4(error) => write!(formatter, "Gemma 4 config: {error}"),
        }
    }
}

impl std::error::Error for ModelConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gemma4(error) => Some(error),
            _ => None,
        }
    }
}

/// Failure to bind an artifact root and classify its family config.
#[derive(Debug)]
pub enum ModelLoadError {
    /// The requested artifact root did not resolve.
    CanonicalizeRoot {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The canonical root cannot cross the native UTF-8 path boundary.
    NonUtf8Root { path: PathBuf },
    /// The artifact's `config.json` could not be read.
    ReadConfig {
        path: PathBuf,
        source: std::io::Error,
    },
    /// Load-bearing config metadata is not an in-root regular file.
    UnsafeConfigFile { path: PathBuf },
    /// The config was malformed, unknown, conflicting, or not executable.
    Config(ModelConfigError),
}

impl fmt::Display for ModelLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CanonicalizeRoot { path, source } => {
                write!(formatter, "resolve artifact root {path:?}: {source}")
            }
            Self::NonUtf8Root { path } => {
                write!(formatter, "artifact root is not valid UTF-8: {path:?}")
            }
            Self::ReadConfig { path, source } => {
                write!(formatter, "read model config {path:?}: {source}")
            }
            Self::UnsafeConfigFile { path } => write!(
                formatter,
                "model config must be a regular non-symlink file inside the artifact root: {path:?}"
            ),
            Self::Config(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ModelLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CanonicalizeRoot { source, .. } | Self::ReadConfig { source, .. } => Some(source),
            Self::Config(error) => Some(error),
            Self::NonUtf8Root { .. } | Self::UnsafeConfigFile { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn gemma_text_config() -> Value {
        json!({
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attention_k_eq_v": true,
            "bos_token_id": 2,
            "dtype": "bfloat16",
            "enable_moe_block": false,
            "eos_token_id": 1,
            "final_logit_softcapping": 30.0,
            "global_head_dim": 512,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 3840,
            "hidden_size_per_layer_input": 0,
            "initializer_range": 0.02,
            "intermediate_size": 15360,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ],
            "max_position_embeddings": 262144,
            "model_type": "gemma4_unified_text",
            "moe_intermediate_size": null,
            "num_attention_heads": 16,
            "num_experts": null,
            "num_global_key_value_heads": 1,
            "num_hidden_layers": 6,
            "num_key_value_heads": 8,
            "num_kv_shared_layers": 0,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                },
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                }
            },
            "sliding_window": 1024,
            "tie_word_embeddings": true,
            "top_k_experts": null,
            "use_bidirectional_attention": "vision",
            "use_cache": true,
            "use_double_wide_mlp": false,
            "vocab_size": 262144,
            "vocab_size_per_layer_input": 0
        })
    }

    fn gemma_config() -> Value {
        json!({
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4_unified",
            "text_config": gemma_text_config()
        })
    }

    #[test]
    fn routes_gemma_through_its_strict_parser() {
        let config = parse_config_str(&gemma_config().to_string()).unwrap();
        let ExecutableModelConfig::Gemma4(geometry) = config;
        assert_eq!(geometry.hidden_size, 3840);
        assert_eq!(geometry.num_hidden_layers, 6);
        assert_eq!(geometry.max_position_embeddings, 262_144);
    }

    #[test]
    fn routes_e_series_gemma_discriminator_through_the_same_family() {
        let mut config = gemma_config();
        config["model_type"] = json!("gemma4");
        config["text_config"]["model_type"] = json!("gemma4_text");
        let parsed = parse_config_str(&config.to_string()).unwrap();
        let ExecutableModelConfig::Gemma4(geometry) = parsed;
        assert_eq!(
            geometry.model_type,
            crate::geometry::TextModelType::Gemma4Text
        );
    }

    #[test]
    fn recognizes_qwen_before_any_gemma_or_native_path() {
        let qwen = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text"}
        });
        let error = parse_config_str(&qwen.to_string()).unwrap_err();
        assert!(matches!(
            error,
            ModelConfigError::RecognizedButUnsupported {
                family: ModelFamily::Qwen35Hybrid
            }
        ));
        assert_eq!(
            error.to_string(),
            "recognized model family qwen3_5 is not executable in this build"
        );
    }

    #[test]
    fn rejects_missing_unknown_and_non_string_discriminators() {
        assert!(matches!(
            parse_config_str(r#"{"text_config": {}}"#),
            Err(ModelConfigError::MissingModelType)
        ));
        assert!(matches!(
            parse_config_str(r#"{"model_type": "future_model"}"#),
            Err(ModelConfigError::UnknownArchitecture { .. })
        ));
        assert!(matches!(
            parse_config_str(r#"{"model_type": 7}"#),
            Err(ModelConfigError::MalformedConfig { .. })
        ));
    }

    #[test]
    fn rejects_conflicting_known_outer_and_text_families() {
        let mut config = gemma_config();
        config["text_config"]["model_type"] = json!("qwen3_5_text");
        let error = parse_config_str(&config.to_string()).unwrap_err();
        assert!(matches!(
            error,
            ModelConfigError::ConflictingArchitecture { .. }
        ));

        let qwen = json!({
            "model_type": "qwen3_5",
            "text_config": {"model_type": "gemma4_text"}
        });
        assert!(matches!(
            parse_config_str(&qwen.to_string()),
            Err(ModelConfigError::ConflictingArchitecture { .. })
        ));
    }

    #[test]
    fn public_plan_routes_the_committed_gemma_fixture() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("gemma4-unified-tiny");
        let plan = ModelLoadPlan::from_directory(&fixture).expect("tiny artifact load plan");
        assert_eq!(plan.family(), ModelFamily::Gemma4);
        assert_eq!(plan.max_context(), 256);
        assert_eq!(plan.root(), fixture.canonicalize().unwrap());
        let ExecutableModelRef::Gemma4(geometry) = plan.executable();
        assert_eq!(geometry.hidden_size, 128);
        assert_eq!(geometry.num_hidden_layers, 6);
    }

    #[test]
    fn public_plan_returns_typed_qwen_unavailable_error() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("qwen35-recognized-only");
        assert!(matches!(
            ModelLoadPlan::from_directory(fixture),
            Err(ModelLoadError::Config(
                ModelConfigError::RecognizedButUnsupported {
                    family: ModelFamily::Qwen35Hybrid
                }
            ))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn public_plan_rejects_config_symlink_escape() {
        use std::os::unix::fs::symlink;
        use std::time::{SystemTime, UNIX_EPOCH};

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let scratch = std::env::temp_dir().join(format!(
            "hyperion-config-symlink-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&scratch).expect("create isolated scratch directory");
        let external_config = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("gemma4-unified-tiny")
            .join("config.json");
        symlink(&external_config, scratch.join("config.json")).expect("create config symlink");

        let result = ModelLoadPlan::from_directory(&scratch);
        std::fs::remove_file(scratch.join("config.json")).expect("remove test symlink");
        std::fs::remove_dir(&scratch).expect("remove isolated scratch directory");

        assert!(matches!(
            result,
            Err(ModelLoadError::UnsafeConfigFile { .. })
        ));
    }
}
