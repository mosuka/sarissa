//! Configuration types for vector indexes.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::embedding::text_embedder::TextEmbedder;
use crate::error::Result;
use crate::vector::core::quantization;
use crate::vector::core::{DistanceMetric, Vector};

/// Vector normalization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorNormalization {
    /// No normalization.
    None,
    /// L2 normalization (unit length).
    L2,
    /// L1 normalization.
    L1,
    /// Min-max normalization.
    MinMax,
}

/// Vector validation error types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorValidationError {
    /// Vector dimension mismatch.
    DimensionMismatch { expected: usize, actual: usize },
    /// Vector contains invalid values (NaN, infinity).
    InvalidValues,
    /// Vector is empty.
    Empty,
    /// Custom validation error.
    Custom(String),
}

impl std::fmt::Display for VectorValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorValidationError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Vector dimension mismatch: expected {expected}, got {actual}"
                )
            }
            VectorValidationError::InvalidValues => {
                write!(f, "Vector contains invalid values (NaN or infinity)")
            }
            VectorValidationError::Empty => {
                write!(f, "Vector is empty")
            }
            VectorValidationError::Custom(msg) => write!(f, "Custom validation error: {msg}"),
        }
    }
}

impl std::error::Error for VectorValidationError {}

/// Helper functions for vector operations.
pub mod utils {
    use super::*;
    use crate::vector::core::DistanceMetric;

    /// Validate a vector against requirements.
    pub fn validate_vector(vector: &Vector, expected_dimension: Option<usize>) -> Result<()> {
        if vector.data.is_empty() {
            return Err(crate::error::YatagarasuError::InvalidOperation(
                VectorValidationError::Empty.to_string(),
            ));
        }

        if let Some(expected_dim) = expected_dimension
            && vector.data.len() != expected_dim
        {
            return Err(crate::error::YatagarasuError::InvalidOperation(
                VectorValidationError::DimensionMismatch {
                    expected: expected_dim,
                    actual: vector.data.len(),
                }
                .to_string(),
            ));
        }

        if !vector.is_valid() {
            return Err(crate::error::YatagarasuError::InvalidOperation(
                VectorValidationError::InvalidValues.to_string(),
            ));
        }

        Ok(())
    }

    /// Normalize a batch of vectors in parallel.
    pub fn normalize_vectors_parallel(vectors: &mut [Vector], method: VectorNormalization) {
        use rayon::prelude::*;

        match method {
            VectorNormalization::None => {
                // No normalization needed
            }
            VectorNormalization::L2 => {
                vectors.par_iter_mut().for_each(|vector| {
                    vector.normalize();
                });
            }
            VectorNormalization::L1 => {
                vectors.par_iter_mut().for_each(|vector| {
                    let l1_norm: f32 = vector.data.iter().map(|x| x.abs()).sum();
                    if l1_norm > 0.0 {
                        for value in &mut vector.data {
                            *value /= l1_norm;
                        }
                    }
                });
            }
            VectorNormalization::MinMax => {
                vectors.par_iter_mut().for_each(|vector| {
                    if let (Some(&min_val), Some(&max_val)) = (
                        vector.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
                        vector.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
                    ) {
                        let range = max_val - min_val;
                        if range > 0.0 {
                            for value in &mut vector.data {
                                *value = (*value - min_val) / range;
                            }
                        }
                    }
                });
            }
        }
    }

    /// Calculate batch similarities between a query and multiple vectors.
    pub fn batch_similarities(
        query: &Vector,
        vectors: &[Vector],
        metric: DistanceMetric,
    ) -> Result<Vec<f32>> {
        vectors
            .iter()
            .map(|vector| metric.similarity(&query.data, &vector.data))
            .collect()
    }

    /// Calculate batch distances between a query and multiple vectors.
    pub fn batch_distances(
        query: &Vector,
        vectors: &[Vector],
        metric: DistanceMetric,
    ) -> Result<Vec<f32>> {
        vectors
            .iter()
            .map(|vector| metric.distance(&query.data, &vector.data))
            .collect()
    }
}

/// Vector index configuration enum that specifies which index type to use.
///
/// This enum provides a unified way to configure different vector index types.
/// Each variant contains the type-specific configuration.
///
/// # Example
///
/// ```rust
/// use yatagarasu::vector::index::config::{VectorIndexConfig, HnswIndexConfig};
/// use yatagarasu::vector::core::DistanceMetric;
///
/// let hnsw_config = HnswIndexConfig {
///     dimension: 384,
///     distance_metric: DistanceMetric::Cosine,
///     m: 16,
///     ef_construction: 200,
///     ..Default::default()
/// };
/// let config = VectorIndexConfig::HNSW(hnsw_config);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VectorIndexConfig {
    /// Flat index configuration
    Flat(FlatIndexConfig),
    /// HNSW index configuration
    HNSW(HnswIndexConfig),
    /// IVF index configuration
    IVF(IvfIndexConfig),
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        VectorIndexConfig::Flat(FlatIndexConfig::default())
    }
}

impl VectorIndexConfig {
    /// Get the index type as a string.
    pub fn index_type_name(&self) -> &'static str {
        match self {
            VectorIndexConfig::Flat(_) => "Flat",
            VectorIndexConfig::HNSW(_) => "HNSW",
            VectorIndexConfig::IVF(_) => "IVF",
        }
    }

    /// Get the dimension from the config.
    pub fn dimension(&self) -> usize {
        match self {
            VectorIndexConfig::Flat(config) => config.dimension,
            VectorIndexConfig::HNSW(config) => config.dimension,
            VectorIndexConfig::IVF(config) => config.dimension,
        }
    }

    /// Get the distance metric from the config.
    pub fn distance_metric(&self) -> DistanceMetric {
        match self {
            VectorIndexConfig::Flat(config) => config.distance_metric,
            VectorIndexConfig::HNSW(config) => config.distance_metric,
            VectorIndexConfig::IVF(config) => config.distance_metric,
        }
    }

    /// Get the max vectors per segment from the config.
    pub fn max_vectors_per_segment(&self) -> u64 {
        match self {
            VectorIndexConfig::Flat(config) => config.max_vectors_per_segment,
            VectorIndexConfig::HNSW(config) => config.max_vectors_per_segment,
            VectorIndexConfig::IVF(config) => config.max_vectors_per_segment,
        }
    }

    /// Get the merge factor from the config.
    pub fn merge_factor(&self) -> u32 {
        match self {
            VectorIndexConfig::Flat(config) => config.merge_factor,
            VectorIndexConfig::HNSW(config) => config.merge_factor,
            VectorIndexConfig::IVF(config) => config.merge_factor,
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

    /// Text embedder for converting text to vectors.
    ///
    /// This embedder is used when documents contain text fields that need to be
    /// converted to vector representations. For field-specific embedders, use
    /// `PerFieldEmbedder`.
    #[serde(skip)]
    #[serde(default = "default_embedder")]
    pub embedder: Arc<dyn crate::embedding::text_embedder::TextEmbedder>,
}

/// Default embedder for index configurations.
///
/// This is a mock embedder that returns zero vectors. In production use,
/// you should provide a real embedder implementation.
fn default_embedder() -> Arc<dyn TextEmbedder> {
    use async_trait::async_trait;

    #[derive(Debug)]
    struct MockEmbedder;

    #[async_trait]
    impl TextEmbedder for MockEmbedder {
        async fn embed(&self, _text: &str) -> Result<Vector> {
            Ok(Vector::new(vec![0.0; 384]))
        }

        fn dimension(&self) -> usize {
            384
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
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: true,
            max_vectors_per_segment: 1000000,
            write_buffer_size: 1024 * 1024, // 1MB
            use_quantization: false,
            quantization_method: quantization::QuantizationMethod::None,
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

    /// Text embedder for converting text to vectors.
    ///
    /// This embedder is used when documents contain text fields that need to be
    /// converted to vector representations. For field-specific embedders, use
    /// `PerFieldEmbedder`.
    #[serde(skip)]
    #[serde(default = "default_embedder")]
    pub embedder: Arc<dyn crate::embedding::text_embedder::TextEmbedder>,
}

impl Default for HnswIndexConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: true,
            m: 16,
            ef_construction: 200,
            max_vectors_per_segment: 1000000,
            write_buffer_size: 1024 * 1024, // 1MB
            use_quantization: false,
            quantization_method: quantization::QuantizationMethod::None,
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

    /// Text embedder for converting text to vectors.
    ///
    /// This embedder is used when documents contain text fields that need to be
    /// converted to vector representations. For field-specific embedders, use
    /// `PerFieldEmbedder`.
    #[serde(skip)]
    #[serde(default = "default_embedder")]
    pub embedder: Arc<dyn crate::embedding::text_embedder::TextEmbedder>,
}

impl Default for IvfIndexConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: true,
            n_clusters: 100,
            n_probe: 1,
            max_vectors_per_segment: 1000000,
            write_buffer_size: 1024 * 1024, // 1MB
            use_quantization: false,
            quantization_method: quantization::QuantizationMethod::None,
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
