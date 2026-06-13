//! Configuration types for vector indexes.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
use crate::error::Result;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization;
use crate::vector::core::rerank::RerankStorageKind;
use crate::vector::core::vector::Vector;

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
    use crate::vector::core::distance::DistanceMetric;

    /// Validate a vector against requirements.
    pub fn validate_vector(vector: &Vector, expected_dimension: Option<usize>) -> Result<()> {
        if vector.data.is_empty() {
            return Err(crate::error::LaurusError::InvalidOperation(
                VectorValidationError::Empty.to_string(),
            ));
        }

        if let Some(expected_dim) = expected_dimension
            && vector.data.len() != expected_dim
        {
            return Err(crate::error::LaurusError::InvalidOperation(
                VectorValidationError::DimensionMismatch {
                    expected: expected_dim,
                    actual: vector.data.len(),
                }
                .to_string(),
            ));
        }

        if !vector.is_valid() {
            return Err(crate::error::LaurusError::InvalidOperation(
                VectorValidationError::InvalidValues.to_string(),
            ));
        }

        Ok(())
    }

    /// Normalize a batch of vectors in parallel.
    pub fn normalize_vectors_parallel(vectors: &mut [Vector], method: VectorNormalization) {
        match method {
            VectorNormalization::None => {
                // No normalization needed
            }
            VectorNormalization::L2 => {
                for vector in vectors.iter_mut() {
                    vector.normalize();
                }
            }
            VectorNormalization::L1 => {
                for vector in vectors.iter_mut() {
                    let l1_norm: f32 = vector.data.iter().map(|x| x.abs()).sum();
                    if l1_norm > 0.0 {
                        for value in Arc::make_mut(&mut vector.data) {
                            *value /= l1_norm;
                        }
                    }
                }
            }
            VectorNormalization::MinMax => {
                for vector in vectors.iter_mut() {
                    if let (Some(&min_val), Some(&max_val)) = (
                        vector.data.iter().min_by(|a, b| a.total_cmp(b)),
                        vector.data.iter().max_by(|a, b| a.total_cmp(b)),
                    ) {
                        let range = max_val - min_val;
                        if range > 0.0 {
                            for value in Arc::make_mut(&mut vector.data) {
                                *value = (*value - min_val) / range;
                            }
                        }
                    }
                }
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
        VectorIndexTypeConfig::Flat(FlatIndexConfig::default())
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

    /// Optional Stage 2 rerank sidecar storage (Issue #481).
    ///
    /// When `Some(_)`, the writer emits an extra LRS1 sidecar file
    /// alongside the main int8 segment so the searcher can do a wide
    /// candidate fetch over int8 and rescore against the original
    /// full-precision vectors. `None` keeps Stage 1 behavior
    /// (int8-only). Stage 2 sidecar is currently consumed only by
    /// the HNSW searcher; Flat / IVF accept the field for schema
    /// symmetry but do not yet emit or consume the sidecar.
    #[serde(default)]
    pub rerank_storage: Option<RerankStorageKind>,

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
            rerank_storage: None,
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
            .field("loading_mode", &self.loading_mode)
            .field("distance_metric", &self.distance_metric)
            .field("normalize_vectors", &self.normalize_vectors)
            .field("max_vectors_per_segment", &self.max_vectors_per_segment)
            .field("write_buffer_size", &self.write_buffer_size)
            .field("use_quantization", &self.use_quantization)
            .field("quantization_method", &self.quantization_method)
            .field("rerank_storage", &self.rerank_storage)
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

    /// Default size of the dynamic candidate list during search
    /// (`ef_search`).
    ///
    /// When `None` (the default), the searcher uses an internal
    /// fallback of `50` so existing schemas keep their previous
    /// behaviour. Per-query
    /// [`crate::vector::search::searcher::VectorIndexQueryParams::ef_search`]
    /// always takes precedence over this schema-level default.
    ///
    /// Issue [#644](https://github.com/mosuka/laurus/issues/644).
    #[serde(default)]
    pub default_ef_search: Option<usize>,

    /// Maximum number of vectors per segment.
    pub max_vectors_per_segment: u64,

    /// Buffer size for writing operations (in bytes).
    pub write_buffer_size: usize,

    /// Whether to use quantization.
    pub use_quantization: bool,

    /// Quantization method.
    pub quantization_method: quantization::QuantizationMethod,

    /// Optional Stage 2 rerank sidecar storage (Issue #481).
    ///
    /// When `Some(_)`, the writer emits an extra LRS1 sidecar file
    /// (`<path>.hnsw.f32`) alongside the main int8 segment so the
    /// HNSW searcher can do a wide candidate fetch over int8 and
    /// rescore against the original full-precision vectors. `None`
    /// keeps Stage 1 behavior (int8-only).
    #[serde(default)]
    pub rerank_storage: Option<RerankStorageKind>,

    /// Merge factor for segment merging.
    pub merge_factor: u32,

    /// Maximum number of segments before merging.
    pub max_segments: u32,

    /// Automatically compact (purge logically deleted vectors) on commit when
    /// the deletion ratio crosses [`Self::compaction_threshold`] (Issue #782).
    ///
    /// Deletions are logical (Issue #624): the node stays in the HNSW graph and
    /// is filtered at search time until compaction rebuilds the graph without
    /// it. When `true`, `commit()` triggers that rebuild automatically once
    /// enough of the index is deleted, so tombstones do not accumulate
    /// unboundedly. Defaults to `false` here; the `VectorStore` populates it
    /// from [`crate::maintenance::deletion::DeletionConfig::auto_compaction`].
    #[serde(default)]
    pub auto_compaction: bool,

    /// Deletion ratio (deleted / total committed nodes, `0.0`–`1.0`) at or above
    /// which [`Self::auto_compaction`] triggers a compaction on commit (Issue
    /// #782). Ignored when `auto_compaction` is `false`.
    #[serde(default = "default_compaction_threshold")]
    pub compaction_threshold: f64,

    /// Embedder for converting text/images to vectors.
    ///
    /// This embedder is used when documents contain text or image fields that need to be
    /// converted to vector representations. For field-specific embedders, use
    /// `PerFieldEmbedder`.
    #[serde(skip)]
    #[serde(default = "default_embedder")]
    pub embedder: Arc<dyn Embedder>,
}

/// Default deletion ratio threshold for auto-compaction (Issue #782).
///
/// Mirrors [`crate::maintenance::deletion::DeletionConfig`]'s default so a
/// `HnswIndexConfig` deserialized from an older on-disk layout (without the
/// field) behaves like a freshly configured one.
fn default_compaction_threshold() -> f64 {
    0.3
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
            default_ef_search: None,
            max_vectors_per_segment: 1000000,
            write_buffer_size: 1024 * 1024, // 1MB
            use_quantization: false,
            quantization_method: quantization::QuantizationMethod::Scalar8Bit,
            rerank_storage: None,
            merge_factor: 10,
            max_segments: 100,
            auto_compaction: false,
            compaction_threshold: default_compaction_threshold(),
            embedder: default_embedder(),
        }
    }
}

impl std::fmt::Debug for HnswIndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndexConfig")
            .field("dimension", &self.dimension)
            .field("loading_mode", &self.loading_mode)
            .field("distance_metric", &self.distance_metric)
            .field("normalize_vectors", &self.normalize_vectors)
            .field("m", &self.m)
            .field("ef_construction", &self.ef_construction)
            .field("default_ef_search", &self.default_ef_search)
            .field("max_vectors_per_segment", &self.max_vectors_per_segment)
            .field("write_buffer_size", &self.write_buffer_size)
            .field("use_quantization", &self.use_quantization)
            .field("quantization_method", &self.quantization_method)
            .field("rerank_storage", &self.rerank_storage)
            .field("merge_factor", &self.merge_factor)
            .field("max_segments", &self.max_segments)
            .field("auto_compaction", &self.auto_compaction)
            .field("compaction_threshold", &self.compaction_threshold)
            .field("embedder", &self.embedder.name())
            .finish()
    }
}

impl HnswIndexConfig {
    /// Build an index config from the schema-level
    /// [`HnswOption`](crate::vector::core::field::HnswOption)
    /// (Issue #790).
    ///
    /// This is the single place that maps the option-derived fields, so
    /// a field added to `HnswOption` is propagated (or consciously
    /// excluded) here instead of being silently dropped at every
    /// conversion site — the bug class behind Issue #790, where
    /// `rerank_storage` and `quantizer` never reached the writers.
    ///
    /// Mapped fields: `dimension`, `distance` → `distance_metric`, `m`,
    /// `ef_construction`, `default_ef_search`, `quantizer` →
    /// `quantization_method`, `rerank_storage`, and `normalize_vectors`
    /// (derived from the metric — see below). All other fields keep
    /// [`HnswIndexConfig::default`] values; the following are
    /// *intentionally* not mapped:
    ///
    /// * `embedder` — `HnswOption::embedder` is an embedder *name*
    ///   (`Option<String>`) while the config holds an
    ///   `Arc<dyn Embedder>`; resolution happens at the store level.
    /// * `base_weight` — a scoring-level knob with no config
    ///   counterpart.
    ///
    /// `normalize_vectors` is set to `distance == Cosine` (Issue #794):
    /// L2-normalizing the stored vectors only makes sense for
    /// magnitude-invariant metrics. Normalizing a Euclidean /
    /// DotProduct / Manhattan field would change its distances, so
    /// those keep their original vectors; Cosine is magnitude-invariant
    /// and normalization additionally tightens the int8 quantization
    /// range. Folding the rule here (rather than overlaying it at each
    /// call site) keeps every HNSW config-construction path consistent
    /// — the bug class behind Issue #794, where the store path left
    /// `normalize_vectors` at the always-on default.
    ///
    /// # Arguments
    ///
    /// * `opt` - The schema-level HNSW field option to convert.
    ///
    /// # Returns
    ///
    /// A config carrying the option-derived fields above and defaults
    /// for everything else; combine with struct-update syntax to set
    /// site-specific fields.
    pub fn from_hnsw_option(opt: &crate::vector::core::field::HnswOption) -> Self {
        Self {
            dimension: opt.dimension,
            distance_metric: opt.distance,
            m: opt.m,
            ef_construction: opt.ef_construction,
            default_ef_search: opt.default_ef_search,
            quantization_method: opt.quantizer,
            rerank_storage: opt.rerank_storage,
            normalize_vectors: opt.distance == DistanceMetric::Cosine,
            ..Self::default()
        }
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

    /// Optional Stage 2 rerank sidecar storage (Issue #481).
    ///
    /// IVF accepts the field for schema symmetry with the HNSW
    /// configuration but does not currently emit or consume the
    /// sidecar — Stage 2 lands HNSW only. See [`HnswIndexConfig::rerank_storage`].
    #[serde(default)]
    pub rerank_storage: Option<RerankStorageKind>,

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
            rerank_storage: None,
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
            .field("rerank_storage", &self.rerank_storage)
            .field("merge_factor", &self.merge_factor)
            .field("max_segments", &self.max_segments)
            .field("embedder", &self.embedder.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::field::HnswOption;
    use crate::vector::core::rerank::RerankStorageKind;

    /// Issue #790: every option-derived field must be mapped by
    /// `from_hnsw_option`, and the intentionally unmapped fields must
    /// keep their `HnswIndexConfig::default()` values.
    #[test]
    fn from_hnsw_option_maps_every_option_derived_field() {
        let opt = HnswOption {
            dimension: 8,
            distance: DistanceMetric::Euclidean,
            m: 5,
            ef_construction: 33,
            default_ef_search: Some(77),
            base_weight: 2.5,
            quantizer: quantization::QuantizationMethod::ProductQuantization { subvector_count: 4 },
            rerank_storage: Some(RerankStorageKind::F32),
            embedder: Some("my-embedder".to_string()),
        };

        let config = HnswIndexConfig::from_hnsw_option(&opt);

        // Mapped fields (all chosen to differ from the defaults).
        assert_eq!(config.dimension, 8);
        assert_eq!(config.distance_metric, DistanceMetric::Euclidean);
        assert_eq!(config.m, 5);
        assert_eq!(config.ef_construction, 33);
        assert_eq!(config.default_ef_search, Some(77));
        assert_eq!(
            config.quantization_method,
            quantization::QuantizationMethod::ProductQuantization { subvector_count: 4 }
        );
        assert_eq!(config.rerank_storage, Some(RerankStorageKind::F32));
        // Issue #794: normalize_vectors is derived from the metric — this
        // option is Euclidean, so it must NOT be normalized.
        assert!(!config.normalize_vectors);

        // Intentionally unmapped fields stay at their defaults.
        let defaults = HnswIndexConfig::default();
        assert_eq!(config.use_quantization, defaults.use_quantization);
        assert_eq!(config.auto_compaction, defaults.auto_compaction);
        assert_eq!(config.max_segments, defaults.max_segments);
        assert_eq!(config.embedder.name(), defaults.embedder.name());
    }

    /// Issue #794: `from_hnsw_option` sets `normalize_vectors` from the
    /// distance metric — only the magnitude-invariant Cosine metric is
    /// L2-normalized; Euclidean/DotProduct/Manhattan keep their original
    /// vectors so their distances are not corrupted.
    #[test]
    fn from_hnsw_option_normalizes_only_for_cosine() {
        let normalize_for = |distance: DistanceMetric| {
            HnswIndexConfig::from_hnsw_option(&HnswOption {
                distance,
                ..Default::default()
            })
            .normalize_vectors
        };

        assert!(normalize_for(DistanceMetric::Cosine));
        assert!(!normalize_for(DistanceMetric::Euclidean));
        assert!(!normalize_for(DistanceMetric::DotProduct));
        assert!(!normalize_for(DistanceMetric::Manhattan));
        assert!(!normalize_for(DistanceMetric::Angular));
    }
}
