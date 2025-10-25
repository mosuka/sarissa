//! Index management and coordination.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SageError};
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

/// Configuration for index creation and management.
#[derive(Debug, Clone)]
pub struct IndexConfig {
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

impl Default for IndexConfig {
    fn default() -> Self {
        IndexConfig {
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

/// A concrete inverted index implementation for schema-less lexical indexing.
///
/// This index can use any storage backend (file, memory, mmap, etc.) via the Storage trait.
#[derive(Debug)]
pub struct InvertedIndex {
    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// Index configuration.
    #[allow(dead_code)]
    config: IndexConfig,

    /// Whether the index is closed.
    closed: bool,

    /// Index metadata.
    metadata: IndexMetadata,
}

/// Metadata about an index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Version of the index format.
    pub version: u32,

    /// Creation time (seconds since epoch).
    pub created: u64,

    /// Last modified time (seconds since epoch).
    pub modified: u64,

    /// Number of documents indexed.
    pub doc_count: u64,

    /// Generation number for updates.
    pub generation: u64,
}

impl Default for IndexMetadata {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        IndexMetadata {
            version: 1,
            created: now,
            modified: now,
            doc_count: 0,
            generation: 0,
        }
    }
}

impl InvertedIndex {
    /// Create a new index in the given storage (schema-less mode).
    pub fn create(storage: Arc<dyn Storage>, config: IndexConfig) -> Result<Self> {
        // Create metadata
        let metadata = IndexMetadata::default();

        let index = InvertedIndex {
            storage,
            config,
            closed: false,
            metadata,
        };

        // Write initial metadata
        let index = index;
        index.write_metadata()?;

        Ok(index)
    }

    /// Open an existing index from storage (schema-less mode).
    pub fn open(storage: Arc<dyn Storage>, config: IndexConfig) -> Result<Self> {
        // Check if index exists
        if !storage.file_exists("metadata.json") {
            return Err(SageError::index("Index does not exist"));
        }

        // Read metadata
        let metadata = Self::read_metadata(storage.as_ref())?;

        Ok(InvertedIndex {
            storage,
            config,
            closed: false,
            metadata,
        })
    }

    /// Create an index in a directory (schema-less mode).
    pub fn create_in_dir<P: AsRef<Path>>(dir: P, config: IndexConfig) -> Result<Self> {
        use crate::storage::file::FileStorage;
        use crate::storage::traits::StorageConfig;

        let storage_config = StorageConfig {
            use_mmap: config.use_mmap,
            ..Default::default()
        };

        let storage = Arc::new(FileStorage::new(dir, storage_config)?);
        Self::create(storage, config)
    }

    /// Open an index from a directory.
    pub fn open_dir<P: AsRef<Path>>(dir: P, config: IndexConfig) -> Result<Self> {
        use crate::storage::file::FileStorage;
        use crate::storage::traits::StorageConfig;

        let storage_config = StorageConfig {
            use_mmap: config.use_mmap,
            ..Default::default()
        };

        let storage = Arc::new(FileStorage::new(dir, storage_config)?);
        Self::open(storage, config)
    }

    /// Write metadata to storage.
    fn write_metadata(&self) -> Result<()> {
        let metadata_json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| SageError::index(format!("Failed to serialize metadata: {e}")))?;

        let mut output = self.storage.create_output("metadata.json")?;
        std::io::Write::write_all(&mut output, metadata_json.as_bytes())?;
        output.close()?;

        // Schema-less mode: no schema file needed

        Ok(())
    }

    /// Read metadata from storage.
    fn read_metadata(storage: &dyn Storage) -> Result<IndexMetadata> {
        let mut input = storage.open_input("metadata.json")?;
        let mut metadata_json = String::new();
        std::io::Read::read_to_string(&mut input, &mut metadata_json)?;

        let metadata: IndexMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| SageError::index(format!("Failed to deserialize metadata: {e}")))?;

        Ok(metadata)
    }

    /// Update metadata and write to storage.
    fn update_metadata(&mut self) -> Result<()> {
        self.metadata.modified = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.write_metadata()
    }

    /// Update the document count in the index metadata.
    pub fn update_doc_count(&mut self, additional_docs: u64) -> Result<()> {
        self.check_closed()?;
        self.metadata.doc_count += additional_docs;
        self.update_metadata()
    }

    /// Check if the index is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            Err(SageError::index("Index is closed"))
        } else {
            Ok(())
        }
    }

    /// Load segment information from storage.
    fn load_segments(&self) -> Result<Vec<SegmentInfo>> {
        let files = self.storage.list_files()?;
        let mut segments = Vec::new();

        // Find all segment metadata files
        for file in &files {
            if file.starts_with("segment_") && file.ends_with(".meta") {
                let mut input = self.storage.open_input(file)?;
                let mut data = Vec::new();
                std::io::Read::read_to_end(&mut input, &mut data)?;

                let segment_info: SegmentInfo = serde_json::from_slice(&data).map_err(|e| {
                    SageError::index(format!("Failed to parse segment metadata: {e}"))
                })?;

                segments.push(segment_info);
            }
        }

        // Sort by generation for consistent ordering
        segments.sort_by_key(|s| s.generation);

        Ok(segments)
    }
}

impl Index for InvertedIndex {
    fn reader(&self) -> Result<Box<dyn IndexReader>> {
        self.check_closed()?;

        use crate::lexical::index::reader::inverted_index::{
            InvertedIndexReader, InvertedIndexReaderConfig,
        };

        // Load segment information
        let segments = self.load_segments()?;

        let reader = InvertedIndexReader::new(
            segments,
            self.storage.clone(),
            InvertedIndexReaderConfig::default(),
        )?;
        Ok(Box::new(reader))
    }

    fn writer(&self) -> Result<Box<dyn IndexWriter>> {
        self.check_closed()?;

        use crate::lexical::index::writer::inverted_index::{
            InvertedIndexWriter, InvertedIndexWriterConfig,
        };

        let writer =
            InvertedIndexWriter::new(self.storage.clone(), InvertedIndexWriterConfig::default())?;
        Ok(Box::new(writer))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    fn close(&mut self) -> Result<()> {
        if !self.closed {
            self.closed = true;
            // We don't close the storage here as it might be shared
        }
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed
    }

    fn stats(&self) -> Result<IndexStats> {
        self.check_closed()?;

        Ok(IndexStats {
            doc_count: self.metadata.doc_count,
            term_count: 0,    // TODO: Calculate from segments
            segment_count: 0, // TODO: Count segments
            total_size: 0,    // TODO: Calculate from storage
            deleted_count: 0, // TODO: Track deletions
            last_modified: self.metadata.modified,
        })
    }

    fn optimize(&mut self) -> Result<()> {
        self.check_closed()?;

        // TODO: Implement segment merging
        self.update_metadata()?;
        Ok(())
    }
}

/// Helper functions for index operations.
impl InvertedIndex {
    /// Check if an index exists in the given directory.
    pub fn exists_in_dir<P: AsRef<Path>>(dir: P) -> bool {
        let metadata_path = dir.as_ref().join("metadata.json");
        metadata_path.exists()
    }

    /// Delete an index from the given directory.
    pub fn delete_in_dir<P: AsRef<Path>>(dir: P) -> Result<()> {
        use crate::storage::file::FileStorage;
        use crate::storage::traits::StorageConfig;

        let storage = FileStorage::new(dir, StorageConfig::default())?;

        // Delete all index files
        for file in storage.list_files()? {
            storage.delete_file(&file)?;
        }

        Ok(())
    }

    /// List all files in the index.
    pub fn list_files(&self) -> Result<Vec<String>> {
        self.check_closed()?;
        self.storage.list_files()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryStorage;
    use crate::storage::traits::StorageConfig;
    use std::sync::Arc;

    #[allow(dead_code)]
    #[test]
    fn test_index_creation() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = IndexConfig::default();

        let index = InvertedIndex::create(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_index_open() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = IndexConfig::default();

        // Create index
        let mut index = InvertedIndex::create(storage.clone(), config.clone()).unwrap();
        index.close().unwrap();

        // Open index
        let index = InvertedIndex::open(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_index_metadata() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = IndexConfig::default();

        let index = InvertedIndex::create(storage, config).unwrap();
        let stats = index.stats().unwrap();

        assert_eq!(stats.doc_count, 0);
        assert_eq!(stats.term_count, 0);
        assert_eq!(stats.segment_count, 0);
        assert_eq!(stats.deleted_count, 0);
        assert!(stats.last_modified > 0);
    }

    #[test]
    fn test_index_close() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = IndexConfig::default();

        let mut index = InvertedIndex::create(storage, config).unwrap();

        assert!(!index.is_closed());

        index.close().unwrap();

        assert!(index.is_closed());

        // Operations should fail after close
        let result = index.stats();
        assert!(result.is_err());
    }

    #[test]
    fn test_index_config() {
        let config = IndexConfig::default();

        assert_eq!(config.max_docs_per_segment, 1000000);
        assert_eq!(config.write_buffer_size, 1024 * 1024);
        assert!(!config.compress_stored_fields);
        assert!(!config.store_term_vectors);
        assert_eq!(config.merge_factor, 10);
        assert_eq!(config.max_segments, 100);
        assert!(!config.use_mmap);
    }

    #[test]
    fn test_index_files() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let config = IndexConfig::default();

        let index = InvertedIndex::create(storage, config).unwrap();
        let files = index.list_files().unwrap();

        // Should have metadata and schema files
        assert!(files.contains(&"metadata.json".to_string()));
    }
}
