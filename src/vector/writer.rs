//! Vector index writer trait and common types.
//!
//! This module defines the `VectorIndexWriter` trait which all vector index writer
//! implementations must follow. The primary implementations are:
//! - `HnswIndexWriter` in the `hnsw` module for approximate nearest neighbor search
//! - `FlatIndexWriter` in the `flat` module for exact search
//! - `IvfIndexWriter` in the `ivf` module for memory-efficient search

use crate::error::Result;
use crate::vector::Vector;

/// Trait for vector index writers.
///
/// This trait defines the common interface that all vector index writer implementations
/// must follow. Writers are responsible for:
/// - Building vector indexes from collections of vectors
/// - Adding vectors incrementally during construction
/// - Finalizing index construction
/// - Writing indexes to persistent storage
///
/// # Example
///
/// ```rust,no_run
/// use sage::vector::index::hnsw::writer::HnswIndexWriter;
/// use sage::vector::index::VectorIndexWriterConfig;
/// use sage::vector::writer::VectorIndexWriter;
/// use sage::storage::memory::MemoryStorage;
/// use sage::storage::traits::StorageConfig;
/// use std::sync::Arc;
///
/// let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
/// let config = VectorIndexWriterConfig::default();
/// let mut writer = HnswIndexWriter::with_storage(config, storage).unwrap();
///
/// // Use VectorIndexWriter trait methods
/// // writer.add_vectors(vectors).unwrap();
/// // writer.finalize().unwrap();
/// // writer.write("my_index").unwrap();
/// ```
pub trait VectorIndexWriter: Send + Sync {
    /// Build an index from a collection of vectors.
    fn build(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()>;

    /// Add vectors incrementally during construction.
    fn add_vectors(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()>;

    /// Finalize the index construction.
    fn finalize(&mut self) -> Result<()>;

    /// Get build progress (0.0 to 1.0).
    fn progress(&self) -> f32;

    /// Get estimated memory usage.
    fn estimated_memory_usage(&self) -> usize;

    /// Optimize the built index.
    fn optimize(&mut self) -> Result<()>;

    /// Get access to the stored vectors.
    /// Returns a reference to the vectors stored in the writer.
    fn vectors(&self) -> &[(u64, Vector)];

    /// Write the index to storage.
    /// This method must be called after finalize() to persist the index.
    fn write(&self, path: &str) -> Result<()>;

    /// Check if this writer has storage configured.
    fn has_storage(&self) -> bool;
}
