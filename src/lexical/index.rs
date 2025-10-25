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

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::reader::IndexReader;
use crate::lexical::writer::IndexWriter;
use crate::storage::traits::Storage;

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

/// Trait for index implementations.
pub trait Index: Send + Sync + std::fmt::Debug {
    /// Get a reader for this index.
    fn reader(&self) -> Result<Box<dyn IndexReader>>;

    /// Get a writer for this index.
    fn writer(&self) -> Result<Box<dyn IndexWriter>>;

    /// Get the storage backend for this index.
    fn storage(&self) -> &Arc<dyn Storage>;

    /// Close the index and release resources.
    fn close(&mut self) -> Result<()>;

    /// Check if the index is closed.
    fn is_closed(&self) -> bool;

    /// Get index statistics.
    fn stats(&self) -> Result<IndexStats>;

    /// Optimize the index (merge segments, etc.).
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

/// Configuration for lexical index creation and management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalIndexConfig {
    /// Index type to build.
    pub index_type: LexicalIndexType,

    /// Maximum number of documents per segment.
    pub max_docs_per_segment: u64,

    /// Buffer size for writing operations.
    pub write_buffer_size: usize,

    /// Whether to use compression for stored fields.
    pub compress_stored_fields: bool,

    /// Whether to store term vectors.
    pub store_term_vectors: bool,

    /// Merge factor for segment merging.
    pub merge_factor: u32,

    /// Maximum number of segments before merging.
    pub max_segments: u32,

    /// Whether to use memory mapping for reading.
    pub use_mmap: bool,
}

impl Default for LexicalIndexConfig {
    fn default() -> Self {
        LexicalIndexConfig {
            index_type: LexicalIndexType::Inverted,
            max_docs_per_segment: 1000000,
            write_buffer_size: 1024 * 1024, // 1MB
            compress_stored_fields: false,
            store_term_vectors: false,
            merge_factor: 10,
            max_segments: 100,
            use_mmap: false,
        }
    }
}

/// Types of lexical indexes that can be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LexicalIndexType {
    /// Inverted index for full-text search.
    Inverted,
    // Future additions:
    // ColumnStore,
    // LSMTree,
    // SuffixArray,
}

/// Factory for creating lexical index instances.
pub struct LexicalIndexFactory;

impl LexicalIndexFactory {
    /// Create a new lexical index based on configuration.
    pub fn create(
        storage: Arc<dyn Storage>,
        config: LexicalIndexConfig,
    ) -> Result<Box<dyn Index>> {
        match config.index_type {
            LexicalIndexType::Inverted => {
                let index = writer::inverted::InvertedIndex::create(storage, config)?;
                Ok(Box::new(index))
            }
            // Future implementations will be added here
        }
    }

    /// Open an existing lexical index based on configuration.
    pub fn open(storage: Arc<dyn Storage>, config: LexicalIndexConfig) -> Result<Box<dyn Index>> {
        match config.index_type {
            LexicalIndexType::Inverted => {
                let index = writer::inverted::InvertedIndex::open(storage, config)?;
                Ok(Box::new(index))
            }
            // Future implementations will be added here
        }
    }

    /// Create a new lexical index in a directory.
    pub fn create_in_dir<P: AsRef<Path>>(
        dir: P,
        config: LexicalIndexConfig,
    ) -> Result<Box<dyn Index>> {
        use crate::storage::file::FileStorage;
        use crate::storage::traits::StorageConfig;

        let storage_config = StorageConfig {
            use_mmap: config.use_mmap,
            ..Default::default()
        };

        let storage = Arc::new(FileStorage::new(dir, storage_config)?);
        Self::create(storage, config)
    }

    /// Open an existing lexical index from a directory.
    pub fn open_dir<P: AsRef<Path>>(
        dir: P,
        config: LexicalIndexConfig,
    ) -> Result<Box<dyn Index>> {
        use crate::storage::file::FileStorage;
        use crate::storage::traits::StorageConfig;

        let storage_config = StorageConfig {
            use_mmap: config.use_mmap,
            ..Default::default()
        };

        let storage = Arc::new(FileStorage::new(dir, storage_config)?);
        Self::open(storage, config)
    }
}

/// A high-level lexical index that manages the lifecycle of readers and writers.
/// This is similar to VectorIndex in the vector module.
pub struct LexicalIndex {
    /// The underlying index implementation.
    index: Box<dyn Index>,

    /// Index configuration.
    config: LexicalIndexConfig,
}

impl LexicalIndex {
    /// Create a new lexical index with the given configuration.
    pub fn create(storage: Arc<dyn Storage>, config: LexicalIndexConfig) -> Result<Self> {
        let index = LexicalIndexFactory::create(storage, config.clone())?;
        Ok(Self { index, config })
    }

    /// Open an existing lexical index.
    pub fn open(storage: Arc<dyn Storage>, config: LexicalIndexConfig) -> Result<Self> {
        let index = LexicalIndexFactory::open(storage, config.clone())?;
        Ok(Self { index, config })
    }

    /// Create a new lexical index in a directory.
    pub fn create_in_dir<P: AsRef<Path>>(dir: P, config: LexicalIndexConfig) -> Result<Self> {
        use crate::storage::file::FileStorage;
        use crate::storage::traits::StorageConfig;

        let storage_config = StorageConfig {
            use_mmap: config.use_mmap,
            ..Default::default()
        };

        let storage = Arc::new(FileStorage::new(dir, storage_config)?);
        Self::create(storage, config)
    }

    /// Open an existing lexical index from a directory.
    pub fn open_dir<P: AsRef<Path>>(dir: P, config: LexicalIndexConfig) -> Result<Self> {
        use crate::storage::file::FileStorage;
        use crate::storage::traits::StorageConfig;

        let storage_config = StorageConfig {
            use_mmap: config.use_mmap,
            ..Default::default()
        };

        let storage = Arc::new(FileStorage::new(dir, storage_config)?);
        Self::open(storage, config)
    }

    /// Get a reader for this index.
    pub fn reader(&self) -> Result<Box<dyn IndexReader>> {
        self.index.reader()
    }

    /// Get a writer for this index.
    pub fn writer(&self) -> Result<Box<dyn IndexWriter>> {
        self.index.writer()
    }

    /// Get the storage backend for this index.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        self.index.storage()
    }

    /// Close the index and release resources.
    pub fn close(&mut self) -> Result<()> {
        self.index.close()
    }

    /// Check if the index is closed.
    pub fn is_closed(&self) -> bool {
        self.index.is_closed()
    }

    /// Get index statistics.
    pub fn stats(&self) -> Result<IndexStats> {
        self.index.stats()
    }

    /// Optimize the index (merge segments, etc.).
    pub fn optimize(&mut self) -> Result<()> {
        self.index.optimize()
    }

    /// Get the configuration.
    pub fn config(&self) -> &LexicalIndexConfig {
        &self.config
    }
}

impl std::fmt::Debug for LexicalIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LexicalIndex")
            .field("config", &self.config)
            .field("is_closed", &self.is_closed())
            .finish()
    }
}

// Type aliases for backward compatibility
/// Type alias for backward compatibility. Use `LexicalIndexConfig` for new code.
pub type IndexConfig = LexicalIndexConfig;

/// Type alias for backward compatibility. Use `LexicalIndex` for new code.
pub type InvertedIndex = LexicalIndex;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryStorage;
    use crate::storage::traits::StorageConfig;

    #[test]
    fn test_lexical_index_creation() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = LexicalIndexConfig::default();

        let index = LexicalIndex::create(storage, config).unwrap();

        assert!(!index.is_closed());
        assert_eq!(index.config().index_type, LexicalIndexType::Inverted);
    }

    #[test]
    fn test_lexical_index_open() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = LexicalIndexConfig::default();

        // Create index
        let mut index = LexicalIndex::create(storage.clone(), config.clone()).unwrap();
        index.close().unwrap();

        // Open index
        let index = LexicalIndex::open(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_lexical_index_stats() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = LexicalIndexConfig::default();

        let index = LexicalIndex::create(storage, config).unwrap();
        let stats = index.stats().unwrap();

        assert_eq!(stats.doc_count, 0);
        assert_eq!(stats.term_count, 0);
        assert_eq!(stats.segment_count, 0);
        assert_eq!(stats.deleted_count, 0);
        assert!(stats.last_modified > 0);
    }

    #[test]
    fn test_lexical_index_close() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = LexicalIndexConfig::default();

        let mut index = LexicalIndex::create(storage, config).unwrap();

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

        assert_eq!(config.index_type, LexicalIndexType::Inverted);
        assert_eq!(config.max_docs_per_segment, 1000000);
        assert_eq!(config.write_buffer_size, 1024 * 1024);
        assert!(!config.compress_stored_fields);
        assert!(!config.store_term_vectors);
        assert_eq!(config.merge_factor, 10);
        assert_eq!(config.max_segments, 100);
        assert!(!config.use_mmap);
    }

    #[test]
    fn test_factory_create() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = LexicalIndexConfig::default();

        let index = LexicalIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());
    }
}
