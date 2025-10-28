//! Vector index writer trait and common types.
//!
//! This module defines the `VectorIndexWriter` trait which all vector index writer
//! implementations must follow. The primary implementations are:
//! - `HnswIndexWriter` in the `hnsw` module for approximate nearest neighbor search
//! - `FlatIndexWriter` in the `flat` module for exact search
//! - `IvfIndexWriter` in the `ivf` module for memory-efficient search

use crate::error::Result;
use crate::vector::Vector;
use serde::{Deserialize, Serialize};

/// Configuration for vector index writers common to all index types.
///
/// This configuration contains settings that are common across all vector index writer
/// implementations (Flat, HNSW, IVF), similar to `InvertedIndexWriterConfig` in the lexical module.
///
/// Type-specific settings (dimension, distance metric, HNSW parameters, etc.) are defined
/// in the respective index configs: `FlatIndexConfig`, `HnswIndexConfig`, `IvfIndexConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexWriterConfig {
    /// Maximum number of vectors to buffer before flushing to storage.
    pub max_buffered_vectors: usize,

    /// Maximum memory usage for buffering (in bytes).
    pub max_buffer_memory: usize,

    /// Segment name prefix.
    pub segment_prefix: String,

    /// Build index in parallel (when supported by the index type).
    pub parallel_build: bool,

    /// Memory limit for index construction (in bytes).
    /// If None, no explicit limit is enforced.
    pub memory_limit: Option<usize>,

    /// Auto-flush threshold: flush when buffer reaches this percentage (0.0-1.0).
    pub auto_flush_threshold: f32,
}

impl Default for VectorIndexWriterConfig {
    fn default() -> Self {
        Self {
            max_buffered_vectors: 10000,
            max_buffer_memory: 512 * 1024 * 1024, // 512 MB
            segment_prefix: "segment".to_string(),
            parallel_build: true,
            memory_limit: None,
            auto_flush_threshold: 0.9,
        }
    }
}

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
/// use sage::vector::index::writer::hnsw::HnswIndexWriter;
/// use sage::vector::index::HnswIndexConfig;
/// use sage::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
/// use sage::storage::memory::{MemoryStorage, MemoryStorageConfig};
/// use sage::storage::StorageConfig;
/// use std::sync::Arc;
///
/// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
/// let index_config = HnswIndexConfig::default();
/// let writer_config = VectorIndexWriterConfig::default();
/// let mut writer = HnswIndexWriter::with_storage(index_config, writer_config, storage).unwrap();
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
