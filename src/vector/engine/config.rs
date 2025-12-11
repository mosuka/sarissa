//! VectorEngine configuration types.
//!
//! This module provides engine configuration, field configuration, and embedder settings.
//!
//! # Configuration with Embedder
//!
//! The recommended way to configure a VectorEngine is to provide an `Embedder` directly
//! in the configuration, similar to how `Analyzer` is used in `LexicalEngine`.
//!
//! ```no_run
//! # #[cfg(feature = "embeddings-candle")]
//! # {
//! use platypus::embedding::per_field::PerFieldEmbedder;
//! use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
//! use platypus::embedding::text_embedder::TextEmbedder;
//! use platypus::vector::engine::VectorEngineConfig;
//! use std::sync::Arc;
//!
//! # fn example() -> platypus::error::Result<()> {
//! let text_embedder: Arc<dyn TextEmbedder> = Arc::new(
//!     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
//! );
//!
//! let mut embedder = PerFieldEmbedder::with_default_text(text_embedder);
//!
//! let config = VectorEngineConfig::builder()
//!     .embedder(embedder)
//!     .build()?;
//! # Ok(())
//! # }
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::embedding::embedder::Embedder;
use crate::error::{PlatypusError, Result};
use crate::vector::DistanceMetric;
use crate::vector::core::document::VectorType;

/// Configuration for a single vector collection.
///
/// This configuration can be created in two ways:
/// 1. Using the builder pattern with an `Embedder` (recommended)
/// 2. Using the legacy HashMap-based configuration (for backward compatibility)
#[derive(Clone)]
pub struct VectorIndexConfig {
    /// Field configurations.
    pub fields: HashMap<String, VectorFieldConfig>,

    /// Legacy embedder configurations (for backward compatibility).
    ///
    /// Prefer using the `embedder` field instead.
    #[deprecated(since = "0.2.0", note = "Use the `embedder` field instead")]
    pub embedders: HashMap<String, VectorEmbedderConfig>,

    /// Default fields for search.
    pub default_fields: Vec<String>,

    /// Metadata for the collection.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Embedder for text and image fields.
    ///
    /// This is analogous to `analyzer` in `InvertedIndexConfig`.
    /// Use `PerFieldEmbedder` for field-specific embedders.
    pub embedder: Option<Arc<dyn Embedder>>,
}

impl std::fmt::Debug for VectorIndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorEngineConfig")
            .field("fields", &self.fields)
            .field("default_fields", &self.default_fields)
            .field("metadata", &self.metadata)
            .field("has_embedder", &self.embedder.is_some())
            .finish()
    }
}

impl VectorIndexConfig {
    /// Create a new builder for VectorEngineConfig.
    pub fn builder() -> VectorIndexConfigBuilder {
        VectorIndexConfigBuilder::new()
    }

    /// Validate the configuration.
    #[allow(deprecated)]
    pub fn validate(&self) -> Result<()> {
        for field in &self.default_fields {
            if !self.fields.contains_key(field) {
                return Err(PlatypusError::invalid_config(format!(
                    "default field '{field}' is not defined"
                )));
            }
        }

        // If using new embedder API, skip legacy validation
        if self.embedder.is_some() {
            return Ok(());
        }

        // Legacy validation for embedders HashMap
        for (field_name, config) in &self.fields {
            if let Some(embedder_id) = config.embedder.as_deref()
                && !self.embedders.contains_key(embedder_id)
            {
                return Err(PlatypusError::invalid_config(format!(
                    "vector field '{field_name}' references undefined embedder '{embedder_id}'"
                )));
            }
        }
        Ok(())
    }

    /// Get the embedder for this configuration.
    pub fn get_embedder(&self) -> Option<&Arc<dyn Embedder>> {
        self.embedder.as_ref()
    }
}

/// Builder for VectorEngineConfig.
///
/// Provides a fluent API for constructing VectorEngineConfig.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "embeddings-candle")]
/// # {
/// use platypus::embedding::per_field::PerFieldEmbedder;
/// use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
/// use platypus::embedding::text_embedder::TextEmbedder;
/// use platypus::vector::engine::{VectorEngineConfig, VectorFieldConfig, VectorIndexKind};
/// use platypus::vector::DistanceMetric;
/// use platypus::vector::core::document::VectorType;
/// use std::sync::Arc;
///
/// # fn example() -> platypus::error::Result<()> {
/// let text_embedder: Arc<dyn TextEmbedder> = Arc::new(
///     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
/// let dim = text_embedder.dimension();
///
/// let mut embedder = PerFieldEmbedder::with_default_text(text_embedder);
///
/// let config = VectorEngineConfig::builder()
///     .embedder(embedder)
///     .field("content_embedding", VectorFieldConfig {
///         dimension: dim,
///         distance: DistanceMetric::Cosine,
///         index: VectorIndexKind::Flat,
///         embedder_id: "default".to_string(),
///         vector_type: VectorType::Text,
///         embedder: None,
///         base_weight: 1.0,
///     })
///     .default_field("content_embedding")
///     .build()?;
/// # Ok(())
/// # }
/// # }
/// ```
#[derive(Default)]
pub struct VectorIndexConfigBuilder {
    fields: HashMap<String, VectorFieldConfig>,
    default_fields: Vec<String>,
    metadata: HashMap<String, serde_json::Value>,
    embedder: Option<Arc<dyn Embedder>>,
}

impl VectorIndexConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedder for all fields.
    ///
    /// Use `PerFieldEmbedder` for field-specific embedders.
    pub fn embedder(mut self, embedder: impl Embedder + 'static) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    /// Set the embedder from an Arc.
    pub fn embedder_arc(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Add a field configuration.
    pub fn field(mut self, name: impl Into<String>, config: VectorFieldConfig) -> Self {
        let name = name.into();
        if !self.default_fields.contains(&name) {
            self.default_fields.push(name.clone());
        }
        self.fields.insert(name, config);
        self
    }

    /// Add a text field with automatic configuration from the embedder.
    ///
    /// The dimension will be inferred from the embedder if available.
    pub fn text_field(mut self, name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let dimension = self
            .embedder
            .as_ref()
            .and_then(|e| e.text_dimension(&name))
            .ok_or_else(|| {
                PlatypusError::invalid_config(format!(
                    "cannot infer dimension for text field '{}': no embedder configured or field not found",
                    name
                ))
            })?;

        let config = VectorFieldConfig {
            dimension,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "default".to_string(),
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        };

        if !self.default_fields.contains(&name) {
            self.default_fields.push(name.clone());
        }
        self.fields.insert(name, config);
        Ok(self)
    }

    /// Add an image field with automatic configuration from the embedder.
    ///
    /// The dimension will be inferred from the embedder if available.
    pub fn image_field(mut self, name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let dimension = self
            .embedder
            .as_ref()
            .and_then(|e| e.image_dimension(&name))
            .ok_or_else(|| {
                PlatypusError::invalid_config(format!(
                    "cannot infer dimension for image field '{}': no embedder configured or field not found",
                    name
                ))
            })?;

        let config = VectorFieldConfig {
            dimension,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "default".to_string(),
            vector_type: VectorType::Image,
            embedder: None,
            base_weight: 1.0,
        };

        if !self.default_fields.contains(&name) {
            self.default_fields.push(name.clone());
        }
        self.fields.insert(name, config);
        Ok(self)
    }

    /// Add a default field for search.
    pub fn default_field(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        if !self.default_fields.contains(&name) {
            self.default_fields.push(name);
        }
        self
    }

    /// Set the default fields for search.
    pub fn default_fields(mut self, fields: Vec<String>) -> Self {
        self.default_fields = fields;
        self
    }

    /// Add metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Build the configuration.
    #[allow(deprecated)]
    pub fn build(self) -> Result<VectorIndexConfig> {
        let config = VectorIndexConfig {
            fields: self.fields,
            embedders: HashMap::new(),
            default_fields: self.default_fields,
            metadata: self.metadata,
            embedder: self.embedder,
        };
        config.validate()?;
        Ok(config)
    }
}

// Implement Serialize manually to skip the embedder field
impl Serialize for VectorIndexConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("VectorEngineConfig", 4)?;
        state.serialize_field("fields", &self.fields)?;
        #[allow(deprecated)]
        state.serialize_field("embedders", &self.embedders)?;
        state.serialize_field("default_fields", &self.default_fields)?;
        state.serialize_field("metadata", &self.metadata)?;
        state.end()
    }
}

// Implement Deserialize manually to handle the embedder field
impl<'de> Deserialize<'de> for VectorIndexConfig {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct VectorEngineConfigHelper {
            fields: HashMap<String, VectorFieldConfig>,
            #[serde(default)]
            embedders: HashMap<String, VectorEmbedderConfig>,
            default_fields: Vec<String>,
            #[serde(default)]
            metadata: HashMap<String, serde_json::Value>,
        }

        let helper = VectorEngineConfigHelper::deserialize(deserializer)?;
        Ok(VectorIndexConfig {
            fields: helper.fields,
            embedders: helper.embedders,
            default_fields: helper.default_fields,
            metadata: helper.metadata,
            embedder: None, // Embedder must be set programmatically
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldConfig {
    pub dimension: usize,
    pub distance: DistanceMetric,
    pub index: VectorIndexKind,
    pub embedder_id: String,
    pub vector_type: VectorType,
    #[serde(default)]
    pub embedder: Option<String>,
    #[serde(default = "VectorFieldConfig::default_weight")]
    pub base_weight: f32,
}

impl VectorFieldConfig {
    fn default_weight() -> f32 {
        1.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndexKind {
    Flat,
    Hnsw,
    Ivf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEmbedderConfig {
    pub provider: VectorEmbedderProvider,
    pub model: String,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum VectorEmbedderProvider {
    CandleText,
    CandleMultimodal,
    OpenAiText,
    External,
}
