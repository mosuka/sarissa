//! Vector indexing module for building and maintaining vector indexes.
//!
//! This module handles all vector index construction, maintenance, and optimization:
//! - Building HNSW, Flat, and IVF indexes
//! - Text embedding generation
//! - Vector quantization and compression
//! - Index optimization and maintenance

pub mod config;
pub mod factory;
pub mod flat;
pub mod hnsw;
pub mod ivf;
pub mod maintenance;
pub mod quantization;

use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

use crate::error::{Result, YatagarasuError};
use crate::storage::Storage;
use crate::vector::reader::VectorIndexReader;
use crate::vector::writer::VectorIndexWriter;
use crate::vector::{DistanceMetric, Vector};

/// Trait for vector index implementations.
///
/// This trait defines the high-level interface for vector indexes.
/// Different index types (Flat, HNSW, IVF, etc.) implement this trait
/// to provide their specific functionality while maintaining a common interface.
pub trait VectorIndex: Send + Sync + std::fmt::Debug {
    /// Get a reader for this index.
    ///
    /// Returns a reader that can be used to query the index.
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>>;

    /// Get a writer for this index.
    ///
    /// Returns a writer that can be used to add or update vectors.
    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>>;

    /// Get the storage backend for this index.
    ///
    /// Returns a reference to the underlying storage.
    fn storage(&self) -> &Arc<dyn Storage>;

    /// Close the index and release resources.
    ///
    /// This should flush any pending writes and release all resources.
    fn close(&mut self) -> Result<()>;

    /// Check if the index is closed.
    ///
    /// Returns true if the index has been closed.
    fn is_closed(&self) -> bool;

    /// Get index statistics.
    ///
    /// Returns statistics about the index such as vector count, dimension, etc.
    fn stats(&self) -> Result<VectorIndexStats>;

    /// Optimize the index.
    ///
    /// Performs index optimization to improve query performance.
    fn optimize(&mut self) -> Result<()>;
}

/// Statistics about a vector index.
#[derive(Debug, Clone)]
pub struct VectorIndexStats {
    /// Number of vectors in the index.
    pub vector_count: u64,

    /// Dimension of vectors.
    pub dimension: usize,

    /// Total size of the index in bytes.
    pub total_size: u64,

    /// Number of deleted vectors.
    pub deleted_count: u64,

    /// Last modified time (seconds since epoch).
    pub last_modified: u64,
}

/// Configuration for vector index types.
///
/// This enum provides type-safe configuration for different index implementations.
/// Each variant contains the configuration specific to that index type.
///
/// # Design Pattern
///
/// This follows an enum-based configuration pattern where:
/// - Each index type has its own dedicated config struct
/// - Pattern matching ensures exhaustive handling of all index types
/// - New index types can be added without breaking existing code
///
/// # Index Types
///
/// - **Flat**: Brute-force exact search (default)
///   - Best for small datasets (< 100K vectors)
///   - Guaranteed 100% recall
///   - Linear search complexity O(n)
///
/// - **HNSW**: Hierarchical Navigable Small World graph
///   - Best for medium to large datasets
///   - Fast approximate search
///   - Good balance between speed and accuracy
///
/// - **IVF**: Inverted File with clustering
///   - Best for very large datasets
///   - Memory-efficient
///   - Tunable speed/accuracy tradeoff
///
/// # Example
///
/// ```no_run
/// use yatagarasu::vector::index::{VectorIndexConfig, FlatIndexConfig, HnswIndexConfig};
/// use yatagarasu::vector::DistanceMetric;
///
/// // Use default flat index
/// let config = VectorIndexConfig::default();
///
/// // Custom flat index configuration
/// let flat_config = FlatIndexConfig {
///     dimension: 384,
///     distance_metric: DistanceMetric::Euclidean,
///     max_vectors_per_segment: 500_000,
///     ..Default::default()
/// };
/// let config = VectorIndexConfig::Flat(flat_config);
///
/// // HNSW configuration for approximate search
/// let hnsw_config = HnswIndexConfig {
///     dimension: 768,
///     m: 32,
///     ef_construction: 400,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        }
    }
}

/// Configuration specific to HNSW index.
///
/// These settings control the behavior of the HNSW (Hierarchical Navigable Small World)
/// index implementation, including graph construction parameters and storage options.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        }
    }
}

/// Configuration specific to IVF index.
///
/// These settings control the behavior of the IVF (Inverted File)
/// index implementation, including clustering parameters and storage options.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        }
    }
}

/// Internal implementation for managing vector index lifecycle.
///
/// This structure wraps a vector index writer and manages its state.
/// For most use cases, prefer using `VectorEngine` which provides a higher-level interface.
///
/// # Note
///
/// This is an internal implementation detail. The public API for vector indexes
/// is defined by the `VectorIndex` trait and `VectorIndexFactory`.
pub struct ManagedVectorIndex {
    config: VectorIndexConfig,
    builder: Arc<RwLock<Box<dyn VectorIndexWriter>>>,
    is_finalized: Arc<RwLock<bool>>,
    storage: Option<Arc<dyn Storage>>,
}

impl ManagedVectorIndex {
    /// Create a new vector index with the given configuration and storage.
    ///
    /// # Arguments
    ///
    /// * `config` - Vector index configuration including index type
    /// * `storage` - Storage backend (MemoryStorage, FileStorage, etc.)
    pub fn new(config: VectorIndexConfig, storage: Arc<dyn Storage>) -> Result<Self> {
        // Create builder based on config type
        let builder: Box<dyn VectorIndexWriter> = match &config {
            VectorIndexConfig::Flat(flat_config) => {
                let writer_config = Self::default_writer_config();
                Box::new(flat::writer::FlatIndexWriter::with_storage(
                    flat_config.clone(),
                    writer_config,
                    storage.clone(),
                )?)
            }
            VectorIndexConfig::HNSW(hnsw_config) => {
                let writer_config = Self::default_writer_config();
                Box::new(hnsw::writer::HnswIndexWriter::with_storage(
                    hnsw_config.clone(),
                    writer_config,
                    storage.clone(),
                )?)
            }
            VectorIndexConfig::IVF(ivf_config) => {
                let writer_config = Self::default_writer_config();
                Box::new(ivf::writer::IvfIndexWriter::with_storage(
                    ivf_config.clone(),
                    writer_config,
                    storage.clone(),
                )?)
            }
        };

        Ok(Self {
            config,
            builder: Arc::new(RwLock::new(builder)),
            is_finalized: Arc::new(RwLock::new(false)),
            storage: Some(storage),
        })
    }

    /// Helper to create a default writer config.
    fn default_writer_config() -> crate::vector::writer::VectorIndexWriterConfig {
        crate::vector::writer::VectorIndexWriterConfig::default()
    }

    /// Add vectors to the index.
    pub fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        let finalized = *self.is_finalized.read().unwrap();
        if finalized {
            return Err(YatagarasuError::InvalidOperation(
                "Cannot add vectors to finalized index".to_string(),
            ));
        }

        let mut builder = self.builder.write().unwrap();
        builder.add_vectors(vectors)?;
        Ok(())
    }

    /// Finalize the index construction.
    pub fn finalize(&mut self) -> Result<()> {
        let mut builder = self.builder.write().unwrap();
        builder.finalize()?;
        *self.is_finalized.write().unwrap() = true;
        Ok(())
    }

    /// Optimize the index.
    pub fn optimize(&mut self) -> Result<()> {
        let mut builder = self.builder.write().unwrap();
        builder.optimize()?;
        Ok(())
    }

    /// Get the configuration.
    pub fn config(&self) -> &VectorIndexConfig {
        &self.config
    }

    /// Get build progress (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        let builder = self.builder.read().unwrap();
        builder.progress()
    }

    /// Get estimated memory usage.
    pub fn estimated_memory_usage(&self) -> usize {
        let builder = self.builder.read().unwrap();
        builder.estimated_memory_usage()
    }

    /// Check if the index is finalized.
    pub fn is_finalized(&self) -> bool {
        *self.is_finalized.read().unwrap()
    }

    /// Get vectors from this index.
    /// Returns a copy of all vectors stored in the index.
    pub fn vectors(&self) -> Result<Vec<(u64, String, Vector)>> {
        let finalized = *self.is_finalized.read().unwrap();
        if !finalized {
            return Err(YatagarasuError::InvalidOperation(
                "Index must be finalized before accessing vectors".to_string(),
            ));
        }

        let builder = self.builder.read().unwrap();
        Ok(builder.vectors().to_vec())
    }

    /// Write the index to storage.
    /// The index must be finalized before calling this method.
    pub fn write(&self, path: &str) -> Result<()> {
        let finalized = *self.is_finalized.read().unwrap();
        if !finalized {
            return Err(YatagarasuError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let builder = self.builder.read().unwrap();
        if !builder.has_storage() {
            return Err(YatagarasuError::InvalidOperation(
                "Index was not created with storage support".to_string(),
            ));
        }

        builder.write(path)
    }

    /// Check if this index has storage configured.
    pub fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    /// Create a reader for this index.
    /// Returns a boxed VectorIndexReader that can be used for searching.
    pub fn reader(&self) -> Result<Arc<dyn crate::vector::reader::VectorIndexReader>> {
        let finalized = *self.is_finalized.read().unwrap();
        if !finalized {
            return Err(YatagarasuError::InvalidOperation(
                "Index must be finalized before creating a reader".to_string(),
            ));
        }

        // If storage is available, load from storage
        if self.storage.is_some() {
            // We need a path to load from - for now, we'll use a default
            // In a real implementation, this would be stored in the VectorIndex
            return Err(YatagarasuError::InvalidOperation(
                "Reader creation from storage not yet implemented. Use load() instead.".to_string(),
            ));
        }

        // Otherwise, create from in-memory vectors
        let vectors = self.vectors()?;
        let reader = crate::vector::reader::SimpleVectorReader::new(
            vectors,
            self.config.dimension(),
            self.config.distance_metric(),
        )?;
        Ok(Arc::new(reader))
    }
}
