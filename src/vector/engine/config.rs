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
//! use sarissa::embedding::per_field::PerFieldEmbedder;
//! use sarissa::embedding::candle_text_embedder::CandleTextEmbedder;
//! use sarissa::embedding::embedder::Embedder;
//! use sarissa::vector::engine::VectorIndexConfig;
//! use std::sync::Arc;
//!
//! # fn example() -> sarissa::error::Result<()> {
//! let text_embedder: Arc<dyn Embedder> = Arc::new(
//!     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
//! );
//!
//! let embedder = PerFieldEmbedder::new(text_embedder);
//!
//! let config = VectorIndexConfig::builder()
//!     .embedder(embedder)
//!     .text_field("title", 384)?
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
use crate::error::{Result, SarissaError};
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
/// use sarissa::embedding::per_field::PerFieldEmbedder;
/// use sarissa::embedding::candle_text_embedder::CandleTextEmbedder;
/// use sarissa::embedding::embedder::Embedder;
/// use sarissa::vector::engine::{VectorIndexConfig, VectorFieldConfig, VectorIndexKind};
/// use sarissa::vector::DistanceMetric;
/// use sarissa::vector::core::document::VectorType;
/// use std::sync::Arc;
///
/// # fn example() -> sarissa::error::Result<()> {
/// let text_embedder: Arc<dyn Embedder> = Arc::new(
///     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
///
/// let embedder = PerFieldEmbedder::new(text_embedder);
///
/// let config = VectorIndexConfig::builder()
///     .embedder(embedder)
///     .text_field("title", 384)?
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

    /// Default distance metric when auto-generating fields.
    pub default_distance: DistanceMetric,

    /// Default dimension when auto-generating fields (implicit schema).
    /// Must be set when `implicit_schema` is true.
    pub default_dimension: Option<usize>,

    /// Default index kind when auto-generating fields.
    pub default_index_kind: VectorIndexKind,

    /// Default vector type when auto-generating fields.
    pub default_vector_type: VectorType,

    /// Default base weight when auto-generating fields.
    pub default_base_weight: f32,

    /// Whether to allow implicit schema generation for unseen fields.
    pub implicit_schema: bool,

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
            .field("default_distance", &self.default_distance)
            .field("default_dimension", &self.default_dimension)
            .field("default_index_kind", &self.default_index_kind)
            .field("default_vector_type", &self.default_vector_type)
            .field("default_base_weight", &self.default_base_weight)
            .field("implicit_schema", &self.implicit_schema)
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
                return Err(SarissaError::invalid_config(format!(
                    "default field '{field}' is not defined"
                )));
            }
        }

        if self.implicit_schema {
            let dim = self.default_dimension.ok_or_else(|| {
                SarissaError::invalid_config("implicit_schema requires default_dimension")
            })?;
            if dim == 0 {
                return Err(SarissaError::invalid_config(
                    "default_dimension must be greater than zero when implicit_schema is enabled",
                ));
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
/// use sarissa::embedding::per_field::PerFieldEmbedder;
/// use sarissa::embedding::candle_text_embedder::CandleTextEmbedder;
/// use sarissa::embedding::embedder::Embedder;
/// use sarissa::vector::engine::{VectorIndexConfig, VectorFieldConfig, VectorIndexKind};
/// use sarissa::vector::DistanceMetric;
/// use sarissa::vector::core::document::VectorType;
/// use std::sync::Arc;
///
/// # fn example() -> sarissa::error::Result<()> {
/// let text_embedder: Arc<dyn Embedder> = Arc::new(
///     CandleTextEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
///
/// let embedder = PerFieldEmbedder::new(text_embedder);
///
/// let config = VectorIndexConfig::builder()
///     .embedder(embedder)
///     .field("content_embedding", VectorFieldConfig {
///         dimension: 384,
///         distance: DistanceMetric::Cosine,
///         index: VectorIndexKind::Flat,
///         source_tag: "content_embedding".to_string(),
///         vector_type: VectorType::Text,
///         base_weight: 1.0,
///     })
///     .default_field("content_embedding")
///     .build()?;
/// # Ok(())
/// # }
/// # }
/// ```
pub struct VectorIndexConfigBuilder {
    fields: HashMap<String, VectorFieldConfig>,
    default_fields: Vec<String>,
    metadata: HashMap<String, serde_json::Value>,
    embedder: Option<Arc<dyn Embedder>>,
    default_distance: DistanceMetric,
    default_dimension: Option<usize>,
    default_index_kind: VectorIndexKind,
    default_vector_type: VectorType,
    default_base_weight: f32,
    implicit_schema: bool,
}

impl VectorIndexConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            default_fields: Vec::new(),
            metadata: HashMap::new(),
            embedder: None,
            default_distance: DistanceMetric::Cosine,
            default_dimension: None,
            default_index_kind: VectorIndexKind::Flat,
            default_vector_type: VectorType::Text,
            default_base_weight: VectorFieldConfig::default_weight(),
            implicit_schema: false,
        }
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
    pub fn text_field(mut self, name: impl Into<String>, dimension: usize) -> Result<Self> {
        let name = name.into();

        let config = VectorFieldConfig {
            dimension,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            vector_type: VectorType::Text,
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
    pub fn image_field(mut self, name: impl Into<String>, dimension: usize) -> Result<Self> {
        let name = name.into();

        let config = VectorFieldConfig {
            dimension,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            vector_type: VectorType::Image,
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

    /// Set default distance for implicit field generation.
    pub fn default_distance(mut self, distance: DistanceMetric) -> Self {
        self.default_distance = distance;
        self
    }

    /// Set default dimension for implicit field generation.
    pub fn default_dimension(mut self, dimension: usize) -> Self {
        self.default_dimension = Some(dimension);
        self
    }

    /// Set default index kind for implicit field generation.
    pub fn default_index_kind(mut self, kind: VectorIndexKind) -> Self {
        self.default_index_kind = kind;
        self
    }

    /// Set default vector type for implicit field generation.
    pub fn default_vector_type(mut self, vector_type: VectorType) -> Self {
        self.default_vector_type = vector_type;
        self
    }

    /// Set default base weight for implicit field generation.
    pub fn default_base_weight(mut self, base_weight: f32) -> Self {
        self.default_base_weight = base_weight;
        self
    }

    /// Enable or disable implicit schema generation for unseen fields.
    pub fn implicit_schema(mut self, enabled: bool) -> Self {
        self.implicit_schema = enabled;
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
            default_distance: self.default_distance,
            default_dimension: self.default_dimension,
            default_index_kind: self.default_index_kind,
            default_vector_type: self.default_vector_type,
            default_base_weight: self.default_base_weight,
            implicit_schema: self.implicit_schema,
            embedder,
        };
        config.validate()?;
        Ok(config)
    }
}

impl Default for VectorIndexConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Implement Serialize manually to skip the embedder field
impl Serialize for VectorIndexConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("VectorIndexConfig", 9)?;
        state.serialize_field("fields", &self.fields)?;
        state.serialize_field("default_fields", &self.default_fields)?;
        state.serialize_field("metadata", &self.metadata)?;
        state.serialize_field("default_distance", &self.default_distance)?;
        state.serialize_field("default_dimension", &self.default_dimension)?;
        state.serialize_field("default_index_kind", &self.default_index_kind)?;
        state.serialize_field("default_vector_type", &self.default_vector_type)?;
        state.serialize_field("default_base_weight", &self.default_base_weight)?;
        state.serialize_field("implicit_schema", &self.implicit_schema)?;
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
            #[serde(default = "default_distance_metric")]
            default_distance: DistanceMetric,
            #[serde(default)]
            default_dimension: Option<usize>,
            #[serde(default = "default_index_kind")]
            default_index_kind: VectorIndexKind,
            #[serde(default = "default_vector_type")]
            default_vector_type: VectorType,
            #[serde(default = "VectorFieldConfig::default_weight")]
            default_base_weight: f32,
            #[serde(default)]
            implicit_schema: bool,
        }

        let helper = VectorIndexConfigHelper::deserialize(deserializer)?;
        Ok(VectorIndexConfig {
            fields: helper.fields,
            default_fields: helper.default_fields,
            metadata: helper.metadata,
            default_distance: helper.default_distance,
            default_dimension: helper.default_dimension,
            default_index_kind: helper.default_index_kind,
            default_vector_type: helper.default_vector_type,
            default_base_weight: helper.default_base_weight,
            implicit_schema: helper.implicit_schema,
            // Default to NoOpEmbedder; can be replaced programmatically
            embedder: Arc::new(NoOpEmbedder::new()),
        })
    }
}

fn default_distance_metric() -> DistanceMetric {
    DistanceMetric::Cosine
}

fn default_index_kind() -> VectorIndexKind {
    VectorIndexKind::Flat
}

fn default_vector_type() -> VectorType {
    VectorType::Text
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
    /// The type of vectors in this field (Text or Image).
    pub vector_type: VectorType,
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
