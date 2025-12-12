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
//! use platypus::embedding::embedder::Embedder;
//! use platypus::vector::engine::VectorIndexConfig;
//! use std::sync::Arc;
//!
//! # fn example() -> platypus::error::Result<()> {
//! let text_embedder: Arc<dyn Embedder> = Arc::new(
//!     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
//! );
//!
//! let embedder = PerFieldEmbedder::new(text_embedder);
//!
//! let config = VectorIndexConfig::builder()
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
use crate::embedding::noop::NoOpEmbedder;
use crate::embedding::per_field::PerFieldEmbedder;
use crate::error::{PlatypusError, Result};
use crate::vector::DistanceMetric;
use crate::vector::core::document::VectorType;

/// Configuration for a single vector collection.
///
/// This configuration should be created using the builder pattern with an `Embedder`.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "embeddings-candle")]
/// # {
/// use platypus::embedding::per_field::PerFieldEmbedder;
/// use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
/// use platypus::embedding::embedder::Embedder;
/// use platypus::vector::engine::{VectorIndexConfig, VectorFieldConfig, VectorIndexKind};
/// use platypus::vector::DistanceMetric;
/// use platypus::vector::core::document::VectorType;
/// use std::sync::Arc;
///
/// # fn example() -> platypus::error::Result<()> {
/// let text_embedder: Arc<dyn Embedder> = Arc::new(
///     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
/// let dim = text_embedder.dimension();
///
/// let embedder = PerFieldEmbedder::new(text_embedder);
///
/// let config = VectorIndexConfig::builder()
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
#[derive(Clone)]
pub struct VectorIndexConfig {
    /// Field configurations.
    pub fields: HashMap<String, VectorFieldConfig>,

    /// Default fields for search.
    pub default_fields: Vec<String>,

    /// Metadata for the collection.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Embedder for text and image fields.
    ///
    /// This is analogous to `analyzer` in `InvertedIndexConfig`.
    /// Use `PerFieldEmbedder` for field-specific embedders.
    /// Use `NoOpEmbedder` when using pre-computed vectors.
    pub embedder: Arc<dyn Embedder>,
}

impl std::fmt::Debug for VectorIndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndexConfig")
            .field("fields", &self.fields)
            .field("default_fields", &self.default_fields)
            .field("metadata", &self.metadata)
            .field("embedder", &format_args!("{:?}", self.embedder))
            .finish()
    }
}

impl VectorIndexConfig {
    /// Create a new builder for VectorIndexConfig.
    pub fn builder() -> VectorIndexConfigBuilder {
        VectorIndexConfigBuilder::new()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        for field in &self.default_fields {
            if !self.fields.contains_key(field) {
                return Err(PlatypusError::invalid_config(format!(
                    "default field '{field}' is not defined"
                )));
            }
        }
        Ok(())
    }

    /// Get the embedder for this configuration.
    pub fn get_embedder(&self) -> &Arc<dyn Embedder> {
        &self.embedder
    }
}

/// Builder for VectorIndexConfig.
///
/// Provides a fluent API for constructing VectorIndexConfig.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "embeddings-candle")]
/// # {
/// use platypus::embedding::per_field::PerFieldEmbedder;
/// use platypus::embedding::candle_text_embedder::CandleTextEmbedder;
/// use platypus::embedding::embedder::Embedder;
/// use platypus::vector::engine::{VectorIndexConfig, VectorFieldConfig, VectorIndexKind};
/// use platypus::vector::DistanceMetric;
/// use platypus::vector::core::document::VectorType;
/// use std::sync::Arc;
///
/// # fn example() -> platypus::error::Result<()> {
/// let text_embedder: Arc<dyn Embedder> = Arc::new(
///     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
/// let dim = text_embedder.dimension();
///
/// let embedder = PerFieldEmbedder::new(text_embedder);
///
/// let config = VectorIndexConfig::builder()
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
    /// For PerFieldEmbedder, the field-specific embedder will be used.
    pub fn text_field(mut self, name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let dimension = self.get_field_dimension(&name)?;

        let config = VectorFieldConfig {
            dimension,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "default".to_string(),
            vector_type: VectorType::Text,
            embedder: Some(name.clone()),
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
    /// For PerFieldEmbedder, the field-specific embedder will be used.
    pub fn image_field(mut self, name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let dimension = self.get_field_dimension(&name)?;

        let config = VectorFieldConfig {
            dimension,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "default".to_string(),
            vector_type: VectorType::Image,
            embedder: Some(name.clone()),
            base_weight: 1.0,
        };

        if !self.default_fields.contains(&name) {
            self.default_fields.push(name.clone());
        }
        self.fields.insert(name, config);
        Ok(self)
    }

    /// Get the dimension for a specific field from the embedder.
    ///
    /// If the embedder is a PerFieldEmbedder, it will try to get the
    /// field-specific embedder's dimension. Otherwise, it uses the
    /// default embedder's dimension.
    fn get_field_dimension(&self, field_name: &str) -> Result<usize> {
        let embedder = self.embedder.as_ref().ok_or_else(|| {
            PlatypusError::invalid_config(format!(
                "cannot infer dimension for field '{}': no embedder configured",
                field_name
            ))
        })?;

        // Try to downcast to PerFieldEmbedder for field-specific dimensions
        if let Some(per_field) = embedder.as_any().downcast_ref::<PerFieldEmbedder>() {
            let field_embedder = per_field.get_embedder(field_name);
            let dim = field_embedder.dimension();
            if dim == 0 {
                return Err(PlatypusError::invalid_config(format!(
                    "cannot infer dimension for field '{}': embedder returned dimension 0",
                    field_name
                )));
            }
            Ok(dim)
        } else {
            // Use the embedder's dimension directly
            let dim = embedder.dimension();
            if dim == 0 {
                return Err(PlatypusError::invalid_config(format!(
                    "cannot infer dimension for field '{}': embedder returned dimension 0",
                    field_name
                )));
            }
            Ok(dim)
        }
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
    ///
    /// If no embedder is set, defaults to `NoOpEmbedder` for pre-computed vectors.
    pub fn build(self) -> Result<VectorIndexConfig> {
        let embedder = self
            .embedder
            .unwrap_or_else(|| Arc::new(NoOpEmbedder::new()));

        let config = VectorIndexConfig {
            fields: self.fields,
            default_fields: self.default_fields,
            metadata: self.metadata,
            embedder,
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

        let mut state = serializer.serialize_struct("VectorIndexConfig", 3)?;
        state.serialize_field("fields", &self.fields)?;
        state.serialize_field("default_fields", &self.default_fields)?;
        state.serialize_field("metadata", &self.metadata)?;
        state.end()
    }
}

// Implement Deserialize manually to handle the embedder field
impl<'de> Deserialize<'de> for VectorIndexConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct VectorIndexConfigHelper {
            fields: HashMap<String, VectorFieldConfig>,
            default_fields: Vec<String>,
            #[serde(default)]
            metadata: HashMap<String, serde_json::Value>,
        }

        let helper = VectorIndexConfigHelper::deserialize(deserializer)?;
        Ok(VectorIndexConfig {
            fields: helper.fields,
            default_fields: helper.default_fields,
            metadata: helper.metadata,
            // Default to NoOpEmbedder; can be replaced programmatically
            embedder: Arc::new(NoOpEmbedder::new()),
        })
    }
}

/// Configuration for a single vector field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldConfig {
    /// The dimension of vectors in this field.
    pub dimension: usize,
    /// The distance metric used for similarity calculations.
    pub distance: DistanceMetric,
    /// The type of index to use (Flat, HNSW, IVF).
    pub index: VectorIndexKind,
    /// The ID of the embedder to use for this field.
    pub embedder_id: String,
    /// The type of vectors in this field (Text or Image).
    pub vector_type: VectorType,
    /// Optional embedder key for PerFieldEmbedder lookup.
    #[serde(default)]
    pub embedder: Option<String>,
    /// Base weight for scoring (default: 1.0).
    #[serde(default = "VectorFieldConfig::default_weight")]
    pub base_weight: f32,
}

impl VectorFieldConfig {
    fn default_weight() -> f32 {
        1.0
    }
}

/// The type of vector index to use.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndexKind {
    /// Flat (brute-force) index - exact but slower for large datasets.
    Flat,
    /// HNSW (Hierarchical Navigable Small World) - approximate but fast.
    Hnsw,
    /// IVF (Inverted File Index) - approximate with clustering.
    Ivf,
}
