//! VectorStore configuration types.
//!
//! This module provides engine configuration, field configuration, and embedder settings.
//!
//! # Configuration with Embedder
//!
//! The recommended way to configure a VectorStore is to provide an `Embedder` directly
//! in the configuration, similar to how `Analyzer` is used in `LexicalStore`.
//!
//! ```no_run
//! # #[cfg(feature = "embeddings-candle")]
//! # {
//! use laurus::embedding::per_field::PerFieldEmbedder;
//! use laurus::embedding::candle_bert_embedder::CandleBertEmbedder;
//! use laurus::embedding::embedder::Embedder;
//! use laurus::vector::store::config::VectorIndexConfig;
//! use laurus::vector::core::field::FlatOption;
//! use std::sync::Arc;
//!
//! # fn example() -> laurus::Result<()> {
//! let text_embedder: Arc<dyn Embedder> = Arc::new(
//!     CandleBertEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
//! );
//!
//! let embedder = Arc::new(PerFieldEmbedder::new(text_embedder));
//!
//! let config = VectorIndexConfig::builder()
//!     .embedder(embedder)
//!     .add_field("title", FlatOption::new(384))?
//!     .build()?;
//! # Ok(())
//! # }
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
use crate::embedding::precomputed::PrecomputedEmbedder;
use crate::error::{LaurusError, Result};
use crate::lexical::store::config::LexicalIndexConfig;
use crate::maintenance::deletion::DeletionConfig;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::field::FieldOption;
use crate::vector::core::quantization;
use crate::vector::core::vector::Vector;

/// Configuration for a single vector collection.
///
/// This configuration should be created using the builder pattern with an `Embedder`.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "embeddings-candle")]
/// # {
/// use laurus::embedding::per_field::PerFieldEmbedder;
/// use laurus::embedding::candle_bert_embedder::CandleBertEmbedder;
/// use laurus::embedding::embedder::Embedder;
/// use laurus::vector::store::config::{VectorIndexConfig, VectorFieldConfig};
/// use laurus::vector::core::field::{VectorIndexKind, FlatOption};
/// use laurus::vector::core::distance::DistanceMetric;
/// use std::sync::Arc;
///
/// # fn example() -> laurus::Result<()> {
/// let text_embedder: Arc<dyn Embedder> = Arc::new(
///     CandleBertEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
///
/// let embedder = Arc::new(PerFieldEmbedder::new(text_embedder));
///
/// let config = VectorIndexConfig::builder()
///     .embedder(embedder)
///     .add_field("title", FlatOption::new(384))?
///     .build()?;
/// # Ok(())
/// # }
/// # }
/// ```
/// Mode of index loading.
///
/// Controls how the index data is loaded from storage.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum IndexLoadingMode {
    /// Load the entire index into memory (RAM).
    ///
    /// This provides the fastest search speed but requires memory
    /// proportional to the index size.
    #[default]
    InMemory,
    /// Use memory-mapped files (mmap) to access the index.
    ///
    /// This allows accessing the index without loading the entire
    /// data into RAM, relying on the OS page cache. This is ideal
    /// for large datasets that exceed available RAM.
    Mmap,
}

/// Vector index configuration enum that specifies which index type to use.
///
/// This enum provides a unified way to configure different vector index types.
/// Each variant contains the type-specific configuration.
///
/// # Example
///
/// ```rust
/// use laurus::vector::index::config::{VectorIndexTypeConfig, HnswIndexConfig};
/// use laurus::vector::core::distance::DistanceMetric;
///
/// let hnsw_config = HnswIndexConfig {
///     dimension: 384,
///     distance_metric: DistanceMetric::Cosine,
///     m: 16,
///     ef_construction: 200,
///     ..Default::default()
/// };
/// let config = VectorIndexTypeConfig::HNSW(hnsw_config);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VectorIndexTypeConfig {
    /// Flat index configuration
    Flat(FlatIndexConfig),
    /// HNSW index configuration
    HNSW(HnswIndexConfig),
    /// IVF index configuration
    IVF(IvfIndexConfig),
}

impl Default for VectorIndexTypeConfig {
    fn default() -> Self {
        VectorIndexTypeConfig::HNSW(HnswIndexConfig::default())
    }
}

impl VectorIndexTypeConfig {
    /// Get the index type as a string.
    pub fn index_type_name(&self) -> &'static str {
        match self {
            VectorIndexTypeConfig::Flat(_) => "Flat",
            VectorIndexTypeConfig::HNSW(_) => "HNSW",
            VectorIndexTypeConfig::IVF(_) => "IVF",
        }
    }

    /// Get the dimension from the config.
    pub fn dimension(&self) -> usize {
        match self {
            VectorIndexTypeConfig::Flat(config) => config.dimension,
            VectorIndexTypeConfig::HNSW(config) => config.dimension,
            VectorIndexTypeConfig::IVF(config) => config.dimension,
        }
    }

    /// Get the distance metric from the config.
    pub fn distance_metric(&self) -> DistanceMetric {
        match self {
            VectorIndexTypeConfig::Flat(config) => config.distance_metric,
            VectorIndexTypeConfig::HNSW(config) => config.distance_metric,
            VectorIndexTypeConfig::IVF(config) => config.distance_metric,
        }
    }

    /// Get the max vectors per segment from the config.
    pub fn max_vectors_per_segment(&self) -> u64 {
        match self {
            VectorIndexTypeConfig::Flat(config) => config.max_vectors_per_segment,
            VectorIndexTypeConfig::HNSW(config) => config.max_vectors_per_segment,
            VectorIndexTypeConfig::IVF(config) => config.max_vectors_per_segment,
        }
    }

    /// Get the merge factor from the config.
    pub fn merge_factor(&self) -> u32 {
        match self {
            VectorIndexTypeConfig::Flat(config) => config.merge_factor,
            VectorIndexTypeConfig::HNSW(config) => config.merge_factor,
            VectorIndexTypeConfig::IVF(config) => config.merge_factor,
        }
    }
}

/// Configuration specific to Flat index.
///
/// These settings control the behavior of the flat index implementation,
/// including segment management, buffering, and storage options.
#[derive(Clone, Serialize, Deserialize)]
pub struct FlatIndexConfig {
    /// Vector dimension.
    pub dimension: usize,

    /// Index loading mode.
    #[serde(default)]
    pub loading_mode: IndexLoadingMode,

    /// Distance metric to use.
    pub distance_metric: DistanceMetric,

    /// Whether to normalize vectors.
    pub normalize_vectors: bool,

    /// Maximum number of vectors per segment.
    ///
    /// When a segment reaches this size, it will be considered for merging.
    /// Larger values reduce merge overhead but increase memory usage.
    pub max_vectors_per_segment: u64,

    /// Buffer size for writing operations (in bytes).
    ///
    /// Controls how much data is buffered in memory before being flushed to disk.
    /// Larger buffers improve write performance but use more memory.
    pub write_buffer_size: usize,

    /// Whether to use quantization.
    pub use_quantization: bool,

    /// Quantization method.
    pub quantization_method: quantization::QuantizationMethod,

    /// Merge factor for segment merging.
    ///
    /// Controls how many segments are merged at once. Higher values reduce
    /// the number of merge operations but create larger temporary segments.
    pub merge_factor: u32,

    /// Maximum number of segments before merging.
    ///
    /// When the number of segments exceeds this threshold, a merge operation
    /// will be triggered to consolidate them.
    pub max_segments: u32,

    /// Embedder for converting text/images to vectors.
    ///
    /// This embedder is used when documents contain text or image fields that need to be
    /// converted to vector representations. For field-specific embedders, use
    /// `PerFieldEmbedder`.
    #[serde(skip)]
    #[serde(default = "default_embedder")]
    pub embedder: Arc<dyn Embedder>,
}

/// Default embedder for index configurations.
///
/// This is a mock embedder that returns zero vectors. In production use,
/// you should provide a real embedder implementation.
fn default_embedder() -> Arc<dyn Embedder> {
    use async_trait::async_trait;

    #[derive(Debug)]
    struct MockEmbedder;

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
            Ok(Vector::new(vec![0.0; 384]))
        }

        fn supported_input_types(&self) -> Vec<EmbedInputType> {
            vec![EmbedInputType::Text]
        }

        fn name(&self) -> &str {
            "MockEmbedder"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    Arc::new(MockEmbedder)
}

impl Default for FlatIndexConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            loading_mode: IndexLoadingMode::default(),
            distance_metric: DistanceMetric::Cosine,

            normalize_vectors: true,
            max_vectors_per_segment: 1000000,
            write_buffer_size: 1024 * 1024, // 1MB
            use_quantization: false,
            quantization_method: quantization::QuantizationMethod::Scalar8Bit,
            merge_factor: 10,
            max_segments: 100,
            embedder: default_embedder(),
        }
    }
}

impl std::fmt::Debug for FlatIndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlatIndexConfig")
            .field("dimension", &self.dimension)
            .field("dimension", &self.dimension)
            .field("loading_mode", &self.loading_mode)
            .field("distance_metric", &self.distance_metric)
            .field("distance_metric", &self.distance_metric)
            .field("normalize_vectors", &self.normalize_vectors)
            .field("max_vectors_per_segment", &self.max_vectors_per_segment)
            .field("write_buffer_size", &self.write_buffer_size)
            .field("use_quantization", &self.use_quantization)
            .field("quantization_method", &self.quantization_method)
            .field("merge_factor", &self.merge_factor)
            .field("max_segments", &self.max_segments)
            .field("embedder", &self.embedder.name())
            .finish()
    }
}

/// Configuration specific to HNSW index.
///
/// These settings control the behavior of the HNSW (Hierarchical Navigable Small World)
/// index implementation, including graph construction parameters and storage options.
#[derive(Clone, Serialize, Deserialize)]
pub struct HnswIndexConfig {
    /// Vector dimension.
    pub dimension: usize,

    /// Index loading mode.
    #[serde(default)]
    pub loading_mode: IndexLoadingMode,

    /// Distance metric to use.
    pub distance_metric: DistanceMetric,

    /// Whether to normalize vectors.
    pub normalize_vectors: bool,

    /// Number of bi-directional links created for every new element during construction.
    ///
    /// Higher values improve recall but increase memory usage and construction time.
    pub m: usize,

    /// Size of the dynamic candidate list during construction.
    ///
    /// Higher values improve index quality but increase construction time.
    pub ef_construction: usize,

    /// Maximum number of vectors per segment.
    pub max_vectors_per_segment: u64,

    /// Buffer size for writing operations (in bytes).
    pub write_buffer_size: usize,

    /// Whether to use quantization.
    pub use_quantization: bool,

    /// Quantization method.
    pub quantization_method: quantization::QuantizationMethod,

    /// Merge factor for segment merging.
    pub merge_factor: u32,

    /// Maximum number of segments before merging.
    pub max_segments: u32,

    /// Embedder for converting text/images to vectors.
    ///
    /// This embedder is used when documents contain text or image fields that need to be
    /// converted to vector representations. For field-specific embedders, use
    /// `PerFieldEmbedder`.
    #[serde(skip)]
    #[serde(default = "default_embedder")]
    pub embedder: Arc<dyn Embedder>,
}

impl Default for HnswIndexConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            loading_mode: IndexLoadingMode::default(),
            distance_metric: DistanceMetric::Cosine,

            normalize_vectors: true,
            m: 16,
            ef_construction: 200,
            max_vectors_per_segment: 1000000,
            write_buffer_size: 1024 * 1024, // 1MB
            use_quantization: false,
            quantization_method: quantization::QuantizationMethod::Scalar8Bit,
            merge_factor: 10,
            max_segments: 100,
            embedder: default_embedder(),
        }
    }
}

impl std::fmt::Debug for HnswIndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndexConfig")
            .field("dimension", &self.dimension)
            .field("dimension", &self.dimension)
            .field("loading_mode", &self.loading_mode)
            .field("distance_metric", &self.distance_metric)
            .field("distance_metric", &self.distance_metric)
            .field("normalize_vectors", &self.normalize_vectors)
            .field("m", &self.m)
            .field("ef_construction", &self.ef_construction)
            .field("max_vectors_per_segment", &self.max_vectors_per_segment)
            .field("write_buffer_size", &self.write_buffer_size)
            .field("use_quantization", &self.use_quantization)
            .field("quantization_method", &self.quantization_method)
            .field("merge_factor", &self.merge_factor)
            .field("max_segments", &self.max_segments)
            .field("embedder", &self.embedder.name())
            .finish()
    }
}

/// Configuration specific to IVF index.
///
/// These settings control the behavior of the IVF (Inverted File)
/// index implementation, including clustering parameters and storage options.
#[derive(Clone, Serialize, Deserialize)]
pub struct IvfIndexConfig {
    /// Vector dimension.
    pub dimension: usize,

    /// Index loading mode.
    #[serde(default)]
    pub loading_mode: IndexLoadingMode,

    /// Distance metric to use.
    pub distance_metric: DistanceMetric,

    /// Whether to normalize vectors.
    pub normalize_vectors: bool,

    /// Number of clusters for IVF.
    ///
    /// Higher values improve search quality but increase memory usage
    /// and construction time.
    pub n_clusters: usize,

    /// Number of clusters to probe during search.
    ///
    /// Higher values improve recall but increase search time.
    pub n_probe: usize,

    /// Maximum number of vectors per segment.
    pub max_vectors_per_segment: u64,

    /// Buffer size for writing operations (in bytes).
    pub write_buffer_size: usize,

    /// Whether to use quantization.
    pub use_quantization: bool,

    /// Quantization method.
    pub quantization_method: quantization::QuantizationMethod,

    /// Merge factor for segment merging.
    pub merge_factor: u32,

    /// Maximum number of segments before merging.
    pub max_segments: u32,

    /// Embedder for converting text/images to vectors.
    ///
    /// This embedder is used when documents contain text or image fields that need to be
    /// converted to vector representations. For field-specific embedders, use
    /// `PerFieldEmbedder`.
    #[serde(skip)]
    #[serde(default = "default_embedder")]
    pub embedder: Arc<dyn Embedder>,
}

impl Default for IvfIndexConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            loading_mode: IndexLoadingMode::default(),
            distance_metric: DistanceMetric::Cosine,

            normalize_vectors: true,
            n_clusters: 100,
            n_probe: 1,
            max_vectors_per_segment: 1000000,
            write_buffer_size: 1024 * 1024, // 1MB
            use_quantization: false,
            quantization_method: quantization::QuantizationMethod::Scalar8Bit,
            merge_factor: 10,
            max_segments: 100,
            embedder: default_embedder(),
        }
    }
}

impl std::fmt::Debug for IvfIndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IvfIndexConfig")
            .field("dimension", &self.dimension)
            .field("loading_mode", &self.loading_mode)
            .field("distance_metric", &self.distance_metric)
            .field("normalize_vectors", &self.normalize_vectors)
            .field("n_clusters", &self.n_clusters)
            .field("n_probe", &self.n_probe)
            .field("max_vectors_per_segment", &self.max_vectors_per_segment)
            .field("write_buffer_size", &self.write_buffer_size)
            .field("use_quantization", &self.use_quantization)
            .field("quantization_method", &self.quantization_method)
            .field("merge_factor", &self.merge_factor)
            .field("max_segments", &self.max_segments)
            .field("embedder", &self.embedder.name())
            .finish()
    }
}
#[derive(Clone)]
pub struct VectorIndexConfig {
    /// Field configurations.
    pub fields: HashMap<String, VectorFieldConfig>,

    /// Default fields for search.
    pub default_fields: Vec<String>,

    /// Metadata for the collection.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Embedder for text and image fields.
    pub embedder: Arc<dyn Embedder>,

    /// Deletion maintenance configuration.
    pub deletion_config: DeletionConfig,

    /// Shard ID for the collection.
    pub shard_id: u16,

    /// Metadata index configuration (LexicalStore).
    pub metadata_config: LexicalIndexConfig,
}

impl std::fmt::Debug for VectorIndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndexConfig")
            .field("fields", &self.fields)
            .field("default_fields", &self.default_fields)
            .field("metadata", &self.metadata)
            .field("embedder", &format_args!("{:?}", self.embedder))
            .field("deletion_config", &self.deletion_config)
            .field("shard_id", &self.shard_id)
            .field("metadata_config", &self.metadata_config)
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
                return Err(LaurusError::invalid_config(format!(
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

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self::builder()
            .build()
            .expect("Default config should be valid")
    }
}

/// Builder for VectorIndexConfig.
pub struct VectorIndexConfigBuilder {
    fields: HashMap<String, VectorFieldConfig>,
    default_fields: Vec<String>,
    metadata: HashMap<String, serde_json::Value>,
    embedder: Option<Arc<dyn Embedder>>,
    deletion_config: Option<DeletionConfig>,
    shard_id: Option<u16>,
    metadata_config: Option<LexicalIndexConfig>,
}

impl VectorIndexConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            default_fields: Vec::new(),
            metadata: HashMap::new(),
            embedder: None,
            deletion_config: None,
            shard_id: None,
            metadata_config: None,
        }
    }

    /// Set the embedder for all fields.
    ///
    /// Use `PerFieldEmbedder` for field-specific embedders.
    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
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

    /// Add a vector field with explicit options.
    ///
    /// The option can be a `VectorOption` or any type that converts into it
    /// (e.g. `FlatOption`, `HnswOption`).
    ///
    /// # Example
    /// ```no_run
    /// # use laurus::vector::store::config::VectorIndexConfig;
    /// # use laurus::vector::core::field::FlatOption;
    /// # fn example() {
    /// let _ = VectorIndexConfig::builder()
    ///     .add_field("title", FlatOption::default().dimension(384));
    /// # }
    /// ```
    pub fn add_field(
        mut self,
        name: impl Into<String>,
        option: impl Into<FieldOption>,
    ) -> Result<Self> {
        let name = name.into();
        let config = VectorFieldConfig {
            vector: Some(option.into()),
            lexical: None,
        };

        if !self.default_fields.contains(&name) {
            self.default_fields.push(name.clone());
        }
        self.fields.insert(name, config);
        Ok(self)
    }

    /// Add an image field.
    ///
    /// This is an alias for `add_field` but intended for image vectors.
    pub fn image_field(
        self,
        name: impl Into<String>,
        option: impl Into<FieldOption>,
    ) -> Result<Self> {
        self.add_field(name, option)
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

    /// Set deletion configuration.
    pub fn deletion_config(mut self, config: DeletionConfig) -> Self {
        self.deletion_config = Some(config);
        self
    }

    /// Set shard ID.
    pub fn shard_id(mut self, shard_id: u16) -> Self {
        self.shard_id = Some(shard_id);
        self
    }

    /// Set metadata index configuration.
    pub fn metadata_config(mut self, config: LexicalIndexConfig) -> Self {
        self.metadata_config = Some(config);
        self
    }

    /// Build the configuration.
    ///
    /// If no embedder is set, defaults to `PrecomputedEmbedder` for pre-computed vectors.
    pub fn build(self) -> Result<VectorIndexConfig> {
        let embedder = self
            .embedder
            .unwrap_or_else(|| Arc::new(PrecomputedEmbedder::new()));

        let config = VectorIndexConfig {
            fields: self.fields,
            default_fields: self.default_fields,
            metadata: self.metadata,
            embedder,
            deletion_config: self.deletion_config.unwrap_or_default(),
            shard_id: self.shard_id.unwrap_or(0),
            metadata_config: self.metadata_config.unwrap_or_default(),
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

        let mut state = serializer.serialize_struct("VectorIndexConfig", 5)?;
        state.serialize_field("fields", &self.fields)?;
        state.serialize_field("default_fields", &self.default_fields)?;
        state.serialize_field("metadata", &self.metadata)?;
        state.serialize_field("deletion_config", &self.deletion_config)?;
        state.serialize_field("shard_id", &self.shard_id)?;
        state.serialize_field("metadata_config", &self.metadata_config)?;
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
            #[serde(default)]
            deletion_config: DeletionConfig,
            #[serde(default)]
            shard_id: u16,
            #[serde(default)]
            metadata_config: LexicalIndexConfig,
        }

        let helper = VectorIndexConfigHelper::deserialize(deserializer)?;
        Ok(VectorIndexConfig {
            fields: helper.fields,
            default_fields: helper.default_fields,
            metadata: helper.metadata,
            deletion_config: helper.deletion_config,
            shard_id: helper.shard_id,
            metadata_config: helper.metadata_config,
            // Default to PrecomputedEmbedder; can be replaced programmatically
            embedder: Arc::new(PrecomputedEmbedder::new()),
        })
    }
}

/// Configuration for a single vector field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldConfig {
    /// Configuration options for the vector field (index type, dimension, distance metric, etc.).
    ///
    /// When `None`, the field has no vector index.
    #[serde(default)]
    pub vector: Option<FieldOption>,
    /// Configuration options for the lexical field.
    pub lexical: Option<crate::lexical::core::field::FieldOption>,
}

impl Default for VectorFieldConfig {
    fn default() -> Self {
        Self {
            vector: Some(FieldOption::default()),
            lexical: Some(crate::lexical::core::field::FieldOption::default()),
        }
    }
}

impl VectorFieldConfig {
    pub fn default_weight() -> f32 {
        1.0
    }
}

// Moved to crate::vector::core::field
// use crate::vector::core::field::{VectorOption, FlatOption, HnswOption, IvfOption, VectorIndexKind};
