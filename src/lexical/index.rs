//! Lexical indexing module for building and maintaining lexical indexes.
//!
//! This module handles all lexical index construction, maintenance, and optimization:
//! - Building Inverted indexes
//! - Document indexing and analysis
//! - Segment management and merging
//! - Index optimization and maintenance

pub mod background_tasks;
pub mod deletion;
pub mod merge_engine;
pub mod merge_policy;
pub mod optimization;
pub mod reader;
pub mod segment_manager;
pub mod transaction;
pub mod writer;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::reader::IndexReader;
use crate::lexical::writer::IndexWriter;
use crate::storage::Storage;

/// Information about a segment in the index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// Segment identifier.
    pub segment_id: String,

    /// Number of documents in this segment.
    pub doc_count: u64,

    /// Document ID offset for this segment.
    pub doc_offset: u64,

    /// Generation number of this segment.
    pub generation: u64,

    /// Whether this segment has deletions.
    pub has_deletions: bool,
}

/// Trait for lexical index implementations.
///
/// This trait defines the high-level interface for lexical indexes.
/// Different index types (Inverted, ColumnStore, LSMTree, etc.) implement this trait
/// to provide their specific functionality while maintaining a common interface.
pub trait LexicalIndex: Send + Sync + std::fmt::Debug {
    /// Get a reader for this index.
    ///
    /// Returns a reader that can be used to query the index.
    fn reader(&self) -> Result<Box<dyn IndexReader>>;

    /// Get a writer for this index.
    ///
    /// Returns a writer that can be used to add or update documents.
    fn writer(&self) -> Result<Box<dyn IndexWriter>>;

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
    /// Returns statistics about the index such as document count, term count, etc.
    fn stats(&self) -> Result<IndexStats>;

    /// Optimize the index (merge segments, etc.).
    ///
    /// Performs index optimization such as merging segments to improve query performance.
    fn optimize(&mut self) -> Result<()>;
}

/// Statistics about an index.
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Number of documents in the index.
    pub doc_count: u64,

    /// Number of unique terms in the index.
    pub term_count: u64,

    /// Number of segments in the index.
    pub segment_count: u32,

    /// Total size of the index in bytes.
    pub total_size: u64,

    /// Number of deleted documents.
    pub deleted_count: u64,

    /// Last modified time (seconds since epoch).
    pub last_modified: u64,
}

/// Configuration for lexical index types.
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
/// - **Inverted**: Traditional inverted index (default)
///   - Fast full-text search
///   - Good for keyword queries
///   - Supports boolean operations
///
/// Future index types that could be added:
/// - **ColumnStore**: Column-oriented index for aggregations
/// - **LSMTree**: Log-structured merge-tree for write-heavy workloads
///
/// # Example
///
/// ```ignore
/// use sage::lexical::index::{LexicalIndexConfig, InvertedIndexConfig};
///
/// // Use default inverted index
/// let config = LexicalIndexConfig::default();
///
/// // Custom inverted index configuration
/// let mut inverted_config = InvertedIndexConfig::default();
/// inverted_config.max_docs_per_segment = 500_000;
/// inverted_config.compress_stored_fields = true;
/// let config = LexicalIndexConfig::Inverted(inverted_config);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LexicalIndexConfig {
    /// Inverted index configuration
    Inverted(InvertedIndexConfig),
    // Future index types can be added here:
    // ColumnStore(ColumnStoreConfig),
    // LSMTree(LSMTreeConfig),
}


impl Default for LexicalIndexConfig {
    fn default() -> Self {
        LexicalIndexConfig::Inverted(InvertedIndexConfig::default())
    }
}

/// Configuration specific to inverted index.
///
/// These settings control the behavior of the inverted index implementation,
/// including segment management, buffering, compression, and term storage options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedIndexConfig {
    /// Maximum number of documents per segment.
    ///
    /// When a segment reaches this size, it will be considered for merging.
    /// Larger values reduce merge overhead but increase memory usage.
    pub max_docs_per_segment: u64,

    /// Buffer size for writing operations (in bytes).
    ///
    /// Controls how much data is buffered in memory before being flushed to disk.
    /// Larger buffers improve write performance but use more memory.
    pub write_buffer_size: usize,

    /// Whether to use compression for stored fields.
    ///
    /// Enabling compression reduces disk usage but increases CPU overhead
    /// for indexing and retrieval operations.
    pub compress_stored_fields: bool,

    /// Whether to store term vectors.
    ///
    /// Term vectors enable advanced features like highlighting and more-like-this
    /// queries, but increase index size and indexing time.
    pub store_term_vectors: bool,

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

impl Default for InvertedIndexConfig {
    fn default() -> Self {
        InvertedIndexConfig {
            max_docs_per_segment: 1000000,
            write_buffer_size: 1024 * 1024, // 1MB
            compress_stored_fields: false,
            store_term_vectors: false,
            merge_factor: 10,
            max_segments: 100,
        }
    }
}


/// Factory for creating lexical index instances.
///
/// This factory follows the Factory design pattern to create appropriate
/// index implementations based on the provided configuration.
///
/// # Design Benefits
///
/// - **Decoupling**: Client code doesn't need to know about concrete index types
/// - **Extensibility**: New index types can be added by extending the enum
/// - **Type safety**: Pattern matching ensures all cases are handled
///
/// # Example with StorageFactory
///
/// ```ignore
/// use sage::lexical::index::{LexicalIndexFactory, LexicalIndexConfig};
/// use sage::storage::{StorageFactory, StorageConfig, MemoryStorageConfig};
/// use std::sync::Arc;
///
/// // Create storage using factory
/// let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
/// let storage = Arc::new(storage);
///
/// // Create index using factory
/// let config = LexicalIndexConfig::default();
/// let index = LexicalIndexFactory::create(storage, config)?;
/// ```
pub struct LexicalIndexFactory;

impl LexicalIndexFactory {
    /// Create a new lexical index with the given storage and configuration.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend (created using `StorageFactory`)
    /// * `config` - Index configuration enum containing type-specific settings
    ///
    /// # Returns
    ///
    /// A boxed trait object implementing `LexicalIndex` trait.
    /// The concrete type is determined by the config variant.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use sage::lexical::index::{LexicalIndexFactory, LexicalIndexConfig, InvertedIndexConfig};
    /// use sage::storage::{StorageFactory, StorageConfig, FileStorageConfig};
    /// use std::sync::Arc;
    ///
    /// // Create file storage
    /// let storage_config = StorageConfig::File(FileStorageConfig::new("/tmp/index"));
    /// let storage = Arc::new(StorageFactory::create(storage_config)?);
    ///
    /// // Create inverted index
    /// let index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig::default());
    /// let index = LexicalIndexFactory::create(storage, index_config)?;
    /// ```
    pub fn create(storage: Arc<dyn Storage>, config: LexicalIndexConfig) -> Result<Box<dyn LexicalIndex>> {
        match config {
            LexicalIndexConfig::Inverted(inverted_config) => {
                let index = writer::inverted::InvertedIndex::create(storage, inverted_config)?;
                Ok(Box::new(index))
            } // Future implementations will be added here
        }
    }

    /// Open an existing lexical index with the given storage and configuration.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend containing the existing index
    /// * `config` - Index configuration (must match the stored index type)
    ///
    /// # Returns
    ///
    /// A boxed index implementation based on the configured index type.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use sage::lexical::index::{LexicalIndexFactory, LexicalIndexConfig, InvertedIndexConfig};
    /// use sage::storage::file::FileStorage;
    /// use sage::storage::StorageConfig;
    /// use std::sync::Arc;
    ///
    /// let storage = Arc::new(FileStorage::new("./index", FileStorageConfig::new("./index"))?);
    /// let config = LexicalIndexConfig::Inverted(InvertedIndexConfig::default());
    /// let index = LexicalIndexFactory::open(storage, config)?;
    /// ```
    pub fn open(storage: Arc<dyn Storage>, config: LexicalIndexConfig) -> Result<Box<dyn LexicalIndex>> {
        match config {
            LexicalIndexConfig::Inverted(inverted_config) => {
                let index = writer::inverted::InvertedIndex::open(storage, inverted_config)?;
                Ok(Box::new(index))
            } // Future implementations will be added here
        }
    }
}

// Type aliases for backward compatibility
/// Type alias for backward compatibility. Use `writer::inverted::InvertedIndex` for new code.
pub type InvertedIndex = writer::inverted::InvertedIndex;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryStorage;
    use crate::storage::{FileStorageConfig, MemoryStorageConfig};

    #[test]
    fn test_lexical_index_creation() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = LexicalIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_lexical_index_open() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Create index
        let mut index = LexicalIndexFactory::create(storage.clone(), config.clone()).unwrap();
        index.close().unwrap();

        // Open index
        let index = LexicalIndexFactory::open(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_lexical_index_stats() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let stats = index.stats().unwrap();

        assert_eq!(stats.doc_count, 0);
        assert_eq!(stats.term_count, 0);
        assert_eq!(stats.segment_count, 0);
        assert_eq!(stats.deleted_count, 0);
        assert!(stats.last_modified > 0);
    }

    #[test]
    fn test_lexical_index_close() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut index = LexicalIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());

        index.close().unwrap();

        assert!(index.is_closed());

        // Operations should fail after close
        let result = index.stats();
        assert!(result.is_err());
    }

    #[test]
    fn test_lexical_index_config() {
        let config = LexicalIndexConfig::default();

        // Test that default is Inverted and check its configuration
        match config {
            LexicalIndexConfig::Inverted(inverted) => {
                assert_eq!(inverted.max_docs_per_segment, 1000000);
                assert_eq!(inverted.write_buffer_size, 1024 * 1024);
                assert!(!inverted.compress_stored_fields);
                assert!(!inverted.store_term_vectors);
                assert_eq!(inverted.merge_factor, 10);
                assert_eq!(inverted.max_segments, 100);
            }
        }
    }

    #[test]
    fn test_factory_create() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = LexicalIndexConfig::default();

        let index = LexicalIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());
    }
}
