//! Quantized-weight manifest for the M1-locked Gemma 4 artifact.
//!
//! From M2-2.2 onward this is the Rust-side view of the converted MLX affine
//! Q4 artifact: which tensor lives in which safetensors shard, the quantization
//! config (validated against the M1 lock), and the per-layer attention kind
//! derived *from the weight schema itself*. Two disciplines carry over from
//! ``geometry``:
//!
//! * No silent default-through on the load-bearing quant block — it is parsed
//!   with ``#[serde(deny_unknown_fields)]`` and every field required, so a
//!   drift in the locked Q4 (g64/b4/affine) surfaces as a parse error.
//! * The layer kind is derived from the tensors that actually exist, not from
//!   a hard-coded index rule. A global (full) layer in the 12B stores **no
//!   ``v_proj``** — the K projection output IS the value tensor
//!   (``attention_k_eq_v``). So ``v_proj`` presence ⇒ sliding, absence ⇒ full.
//!   This is the same ``layer_types`` array ``geometry`` parses from
//!   ``text_config``; the manifest re-derives it from the weights as a parity
//!   cross-check (a manifest/config disagreement is a bug, never a coercion).

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use serde::Deserialize;

use crate::geometry::LayerType;

// ---------------------------------------------------------------------------
// Locked quantization (M1 decision 0002)
// ---------------------------------------------------------------------------

/// The only quantization Hyperion v1 accepts for the Gemma 4 12B: MLX affine
/// Q4, group size 64, bits 4. The converted artifact (``gemma4-12B-qat-mlx-g64-b4``)
/// is locked to these; any drift is rejected.
pub const LOCKED_GROUP_SIZE: u32 = 64;
pub const LOCKED_BITS: u32 = 4;
pub const LOCKED_MODE: &str = "affine";

/// The quantization block from ``config.json`` (``quantization`` /
/// ``quantization_config``). Strict — an unknown field or a non-locked value is
/// a hard error, never a silent default.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QuantConfig {
    pub group_size: u32,
    pub bits: u32,
    pub mode: String,
}

impl QuantConfig {
    /// Validate against the M1-locked Gemma 4 Q4 (g64/b4/affine).
    ///
    /// # Errors
    /// Returns a [`WeightsError::QuantDrift`] if any field diverges from the lock.
    pub fn validate(&self) -> Result<(), WeightsError> {
        if self.group_size != LOCKED_GROUP_SIZE
            || self.bits != LOCKED_BITS
            || self.mode != LOCKED_MODE
        {
            return Err(WeightsError::QuantDrift {
                group_size: self.group_size,
                bits: self.bits,
                mode: self.mode.clone(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A failure reading or validating the weight manifest.
#[derive(Debug)]
pub enum WeightsError {
    /// A required manifest file could not be read.
    Io { which: &'static str, source: String },
    /// A manifest file was malformed JSON.
    Parse { which: &'static str, source: String },
    /// A required field was absent from the manifest.
    Missing { which: &'static str },
    /// The quantization block diverged from the M1 lock.
    QuantDrift {
        group_size: u32,
        bits: u32,
        mode: String,
    },
    /// The model_type is not the v1-accepted Gemma 4 unified text variant.
    WrongModelType { found: String },
    /// The weight schema disagreed with the config (e.g. layer count mismatch).
    SchemaDisagreement { detail: String },
}

impl fmt::Display for WeightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { which, source } => write!(f, "could not read {which}: {source}"),
            Self::Parse { which, source } => write!(f, "could not parse {which}: {source}"),
            Self::Missing { which } => write!(f, "manifest is missing required field: {which}"),
            Self::QuantDrift {
                group_size,
                bits,
                mode,
            } => write!(
                f,
                "quantization diverged from the M1 lock (g{LOCKED_GROUP_SIZE}/b{LOCKED_BITS}/{LOCKED_MODE}): \
                 found g{group_size}/b{bits}/{mode}"
            ),
            Self::WrongModelType { found } => {
                write!(f, "model_type is not gemma4_unified: found {found}")
            }
            Self::SchemaDisagreement { detail } => {
                write!(f, "weight schema disagrees with config: {detail}")
            }
        }
    }
}

impl std::error::Error for WeightsError {}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// The accepted v1 model_type (the dense unified 12B text stack).
pub const ACCEPTED_MODEL_TYPE: &str = "gemma4_unified";

/// The tensor-name prefix for the text backbone layers.
const LAYER_PREFIX: &str = "language_model.model.layers.";
/// The embedding tensor (tied with the LM head on the 12B).
const EMBED_TENSOR: &str = "language_model.model.embed_tokens.weight";
/// The final RMSNorm weight.
const FINAL_NORM_TENSOR: &str = "language_model.model.norm.weight";

/// The safetensors index — ``metadata`` + ``weight_map`` (tensor → shard file).
#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    metadata: SafetensorsMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsMetadata {
    total_size: u64,
    // MLX-converted indexes carry total_parameters; older HF indexes omit it.
    #[serde(default)]
    total_parameters: Option<u64>,
}

/// The validated weight manifest for one converted artifact directory.
#[derive(Clone, Debug)]
pub struct WeightManifest {
    model_type: String,
    quant: QuantConfig,
    num_hidden_layers: u32,
    total_size: u64,
    total_parameters: Option<u64>,
    weight_map: BTreeMap<String, String>,
}

impl WeightManifest {
    /// Read a converted artifact directory (``config.json`` +
    /// ``model.safetensors.index.json``) into a manifest. Does NOT load the
    /// weight tensors — this is the metadata view the native loader consults to
    /// decide which shard to mmap for a given tensor.
    ///
    /// # Errors
    /// See [`WeightsError`].
    pub fn from_directory(dir: &Path) -> Result<Self, WeightsError> {
        let config = read_config(dir)?;
        let index = read_index(dir)?;
        let num_hidden_layers = config.text_config.num_hidden_layers;

        // Derive the layer count from the weight schema as a parity check
        // against the config's num_hidden_layers.
        let schema_layers = count_layers(&index.weight_map);
        if schema_layers != num_hidden_layers {
            return Err(WeightsError::SchemaDisagreement {
                detail: format!(
                    "config declares {num_hidden_layers} layers but the weight schema has {schema_layers}"
                ),
            });
        }

        let quant = config
            .quantization
            .expect("read_config normalizes a quantization block into place");
        Ok(Self {
            model_type: config.model_type,
            quant,
            num_hidden_layers,
            total_size: index.metadata.total_size,
            total_parameters: index.metadata.total_parameters,
            weight_map: index.weight_map,
        })
    }

    /// Validate the manifest end-to-end: locked quant, accepted model_type,
    /// and the presence of the shared embedding + final norm.
    ///
    /// # Errors
    /// See [`WeightsError`].
    pub fn validate(&self) -> Result<(), WeightsError> {
        self.quant.validate()?;
        if self.model_type != ACCEPTED_MODEL_TYPE {
            return Err(WeightsError::WrongModelType {
                found: self.model_type.clone(),
            });
        }
        if !self.weight_map.contains_key(EMBED_TENSOR) {
            return Err(WeightsError::Missing {
                which: "embed_tokens.weight",
            });
        }
        if !self.weight_map.contains_key(FINAL_NORM_TENSOR) {
            return Err(WeightsError::Missing {
                which: "model.norm.weight",
            });
        }
        Ok(())
    }

    #[must_use]
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    #[must_use]
    pub fn quant(&self) -> &QuantConfig {
        &self.quant
    }

    #[must_use]
    pub fn num_hidden_layers(&self) -> u32 {
        self.num_hidden_layers
    }

    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    #[must_use]
    pub fn total_parameters(&self) -> Option<u64> {
        self.total_parameters
    }

    /// The shard file holding ``tensor``, or ``None`` if the tensor is absent.
    #[must_use]
    pub fn shard_for(&self, tensor: &str) -> Option<&str> {
        self.weight_map.get(tensor).map(String::as_str)
    }

    /// Whether a named tensor exists in the manifest.
    #[must_use]
    pub fn has_tensor(&self, tensor: &str) -> bool {
        self.weight_map.contains_key(tensor)
    }

    /// The attention kind of layer ``layer`` derived from the weight schema:
    /// a sliding layer stores a ``v_proj``; a full (global) layer does not
    /// (``attention_k_eq_v`` — K IS V). ``None`` if no tensors exist for the layer.
    #[must_use]
    pub fn layer_kind(&self, layer: u32) -> Option<LayerType> {
        let v_proj = format!("{LAYER_PREFIX}{layer}.self_attn.v_proj.weight");
        match self.weight_map.contains_key(&v_proj) {
            true => Some(LayerType::Sliding),
            false => {
                // Confirm the layer exists at all (k_proj is universal).
                let k_proj = format!("{LAYER_PREFIX}{layer}.self_attn.k_proj.weight");
                self.weight_map
                    .contains_key(&k_proj)
                    .then_some(LayerType::Full)
            }
        }
    }

    /// The per-layer attention kinds in layer order, derived from the weights.
    /// This must match the ``layer_types`` ``geometry`` parses from
    /// ``text_config`` — a disagreement is a bug.
    #[must_use]
    pub fn layer_kinds(&self) -> Vec<LayerType> {
        (0..self.num_hidden_layers)
            .map(|i| self.layer_kind(i))
            .map_while(|k| k)
            .collect()
    }

    /// All shard files referenced by the manifest, sorted.
    #[must_use]
    pub fn shards(&self) -> Vec<&str> {
        let mut shards: Vec<&str> = self.weight_map.values().map(String::as_str).collect();
        shards.sort_unstable();
        shards.dedup();
        shards
    }
}

// ---------------------------------------------------------------------------
// config.json (loose top-level, strict quant block)
// ---------------------------------------------------------------------------

/// The fields Hyperion reads from ``config.json``. The top-level is parsed
/// loosely (the 12B config carries vision/audio fields v1 sanitizes out), but the
/// ``quantization`` block is strict (``deny_unknown_fields``).
#[derive(Debug, Deserialize)]
struct ConfigJson {
    model_type: String,
    /// MLX-converted artifacts carry ``quantization``; HF-style ones carry
    /// ``quantization_config``. Accept either; prefer ``quantization``.
    #[serde(default)]
    quantization: Option<QuantConfig>,
    #[serde(default)]
    quantization_config: Option<QuantConfig>,
    text_config: TextConfig,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    num_hidden_layers: u32,
}

fn read_config(dir: &Path) -> Result<ConfigJson, WeightsError> {
    let path = dir.join("config.json");
    let bytes = std::fs::read(&path).map_err(|e| WeightsError::Io {
        which: "config.json",
        source: e.to_string(),
    })?;
    let mut value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| WeightsError::Parse {
            which: "config.json",
            source: e.to_string(),
        })?;
    // Normalize quantization_config -> quantization so the typed parse sees one
    // block (MLX-converted: only quantization; HF-style: only quantization_config).
    let obj = value.as_object_mut().ok_or(WeightsError::Missing {
        which: "config.json root object",
    })?;
    if !obj.contains_key("quantization")
        && let Some(qc) = obj.get("quantization_config").cloned()
    {
        obj.insert("quantization".to_string(), qc);
    }
    serde_json::from_value::<ConfigJson>(value)
        .map_err(|e| WeightsError::Parse {
            which: "config.json",
            source: e.to_string(),
        })
        .and_then(|c| {
            let quant = c
                .quantization
                .or(c.quantization_config)
                .ok_or(WeightsError::Missing {
                    which: "quantization",
                })?;
            Ok(ConfigJson {
                model_type: c.model_type,
                quantization: Some(quant),
                quantization_config: None,
                text_config: c.text_config,
            })
        })
}

fn read_index(dir: &Path) -> Result<SafetensorsIndex, WeightsError> {
    let path = dir.join("model.safetensors.index.json");
    let bytes = std::fs::read(&path).map_err(|e| WeightsError::Io {
        which: "model.safetensors.index.json",
        source: e.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|e| WeightsError::Parse {
        which: "model.safetensors.index.json",
        source: e.to_string(),
    })
}

/// The number of distinct layer indices present in the weight map.
fn count_layers(weight_map: &BTreeMap<String, String>) -> u32 {
    let mut max_layer: Option<u32> = None;
    for name in weight_map.keys() {
        let Some(rest) = name.strip_prefix(LAYER_PREFIX) else {
            continue;
        };
        // rest = "<N>.self_attn..." — take the leading integer.
        let Some(num_str) = rest.split('.').next() else {
            continue;
        };
        if let Ok(n) = num_str.parse::<u32>() {
            max_layer = Some(max_layer.map_or(n, |m| m.max(n)));
        }
    }
    // layer indices are 0-based, so count = max + 1.
    max_layer.map_or(0, |m| m + 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::LayerType;
    use std::path::PathBuf;

    /// A committed synthetic 6-layer gemma4_unified artifact (correct schema,
    /// tiny) so the model-free tier can exercise the manifest without the
    /// ~6.7 GB real weights (git-ignored).
    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("gemma4-unified-tiny")
    }

    /// The real M1-locked 12B artifact on the dev/M5 machine (git-ignored; the
    /// test self-skips on CI where it is absent).
    fn real_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../artifacts/models/gemma4-12b-qat-mlx-g64-b4")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from("/nonexistent"))
    }

    #[test]
    fn parses_synthetic_fixture_and_validates() {
        let manifest =
            WeightManifest::from_directory(&fixture_dir()).expect("the synthetic fixture parses");
        manifest
            .validate()
            .expect("the synthetic fixture validates");

        assert_eq!(manifest.model_type(), ACCEPTED_MODEL_TYPE);
        assert_eq!(manifest.num_hidden_layers(), 6);
        assert_eq!(manifest.quant().group_size, LOCKED_GROUP_SIZE);
        assert_eq!(manifest.quant().bits, LOCKED_BITS);
        assert_eq!(manifest.quant().mode, LOCKED_MODE);
    }

    #[test]
    fn derives_layer_kinds_from_v_proj_presence() {
        // The synthetic fixture is a 5:1 layout (layers 0-4 sliding, 5 global).
        // The global layer omits v_proj (K=V); the manifest must see it as Full.
        let manifest = WeightManifest::from_directory(&fixture_dir()).unwrap();
        let kinds = manifest.layer_kinds();
        assert_eq!(kinds.len(), 6, "one kind per layer");
        assert_eq!(
            kinds[0],
            LayerType::Sliding,
            "layer 0 is sliding (has v_proj)"
        );
        assert_eq!(
            kinds[5],
            LayerType::Full,
            "layer 5 is global (no v_proj — K IS V)"
        );
        // The last layer is always global (Gemma 4 invariant).
        assert_eq!(*kinds.last().unwrap(), LayerType::Full);
    }

    #[test]
    fn shard_map_is_consistent() {
        let manifest = WeightManifest::from_directory(&fixture_dir()).unwrap();
        assert!(manifest.has_tensor("language_model.model.embed_tokens.weight"));
        assert!(manifest.has_tensor("language_model.model.norm.weight"));
        // Every shard_for result is one of the declared shards.
        let shards: Vec<&str> = manifest.shards();
        assert!(!shards.is_empty());
        for tensor in manifest.weight_map.keys().take(3) {
            let shard = manifest.shard_for(tensor).expect("mapped");
            assert!(shards.contains(&shard));
        }
    }

    #[test]
    fn rejects_quant_drift() {
        let manifest = WeightManifest::from_directory(&fixture_dir()).unwrap();
        let mut bad = manifest.quant.clone();
        bad.bits = 8; // not the locked Q4
        assert!(matches!(
            bad.validate(),
            Err(WeightsError::QuantDrift { .. })
        ));

        let mut bad_mode = manifest.quant.clone();
        bad_mode.mode = "int8".to_string();
        assert!(matches!(
            bad_mode.validate(),
            Err(WeightsError::QuantDrift { .. })
        ));
    }

    #[test]
    fn rejects_unknown_quant_field_via_strict_parse() {
        // A quant block with an extra field must fail to deserialize (deny_unknown_fields).
        let json = r#"{"group_size":64,"bits":4,"mode":"affine","phantom":7}"#;
        assert!(serde_json::from_str::<QuantConfig>(json).is_err());
    }

    // ---- M5-gated real-12B parity (self-skips where the artifact is absent) ----

    #[test]
    fn real_12b_manifest_validates_and_matches_geometry() {
        let dir = real_dir();
        if !dir.join("model.safetensors.index.json").exists() {
            eprintln!("weights: skipping real-12B manifest test (artifact absent on this host)");
            return;
        }
        let manifest = WeightManifest::from_directory(&dir).expect("the real 12B artifact parses");
        manifest
            .validate()
            .expect("the real 12B artifact validates the lock");

        // The 12B: 48 layers, 5:1, last global.
        assert_eq!(manifest.num_hidden_layers(), 48);
        let kinds = manifest.layer_kinds();
        assert_eq!(kinds.len(), 48);
        assert_eq!(kinds[5], LayerType::Full, "layer 5 is the first global");
        assert_eq!(kinds[0], LayerType::Sliding);
        assert_eq!(*kinds.last().unwrap(), LayerType::Full, "last layer global");

        // The total size is the M1-locked 12B Q4 weight footprint (~6.7 GB).
        assert_eq!(manifest.total_size(), 6_698_991_200);
        assert_eq!(
            manifest.total_parameters(),
            Some(11_907_350_272),
            "11.95B dense params"
        );
    }

    // Guard: the committed fixture must actually exist on disk (CI sanity).
    #[test]
    fn fixture_is_committed() {
        assert!(
            fixture_dir().join("config.json").exists(),
            "synthetic fixture config.json is missing from the repo"
        );
        assert!(
            fixture_dir().join("model.safetensors.index.json").exists(),
            "synthetic fixture index is missing from the repo"
        );
    }
}
