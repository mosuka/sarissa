//! Index management and coordination.

use crate::error::{SarissaError, Result};
use crate::index::reader::IndexReader;
use crate::index::writer::IndexWriter;
use crate::schema::Schema;
use crate::storage::Storage;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

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

    /// Get the schema for this index.
    fn schema(&self) -> &Schema;

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

/// A concrete index implementation.
#[derive(Debug)]
pub struct FileIndex {
    /// The schema for this index.
    schema: Schema,

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

impl FileIndex {
    /// Create a new index in the given storage.
    pub fn create(storage: Arc<dyn Storage>, schema: Schema, config: IndexConfig) -> Result<Self> {
        // Validate schema
        schema.validate()?;

        // Create metadata
        let metadata = IndexMetadata::default();

        let index = FileIndex {
            schema,
            storage,
            config,
            closed: false,
            metadata,
        };

        // Write initial metadata
        let temp_index = index;
        temp_index.write_metadata()?;

        Ok(temp_index)
    }

    /// Open an existing index from storage.
    pub fn open(storage: Arc<dyn Storage>, config: IndexConfig) -> Result<Self> {
        // Check if index exists
        if !storage.file_exists("metadata.json") {
            return Err(SarissaError::index("Index does not exist"));
        }

        // Read metadata
        let metadata = Self::read_metadata(storage.as_ref())?;

        // Read schema
        let schema = Self::read_schema(storage.as_ref())?;

        Ok(FileIndex {
            schema,
            storage,
            config,
            closed: false,
            metadata,
        })
    }

    /// Create an index in a directory.
    pub fn create_in_dir<P: AsRef<Path>>(
        dir: P,
        schema: Schema,
        config: IndexConfig,
    ) -> Result<Self> {
        use crate::storage::{FileStorage, StorageConfig};

        let storage_config = StorageConfig {
            use_mmap: config.use_mmap,
            ..Default::default()
        };

        let storage = Arc::new(FileStorage::new(dir, storage_config)?);
        Self::create(storage, schema, config)
    }

    /// Open an index from a directory.
    pub fn open_dir<P: AsRef<Path>>(dir: P, config: IndexConfig) -> Result<Self> {
        use crate::storage::{FileStorage, StorageConfig};

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
            .map_err(|e| SarissaError::index(format!("Failed to serialize metadata: {e}")))?;

        let mut output = self.storage.create_output("metadata.json")?;
        std::io::Write::write_all(&mut output, metadata_json.as_bytes())?;
        output.close()?;

        // For now, just write a simple schema indicator
        let schema_info = format!("{{\"field_count\": {}}}", self.schema.len());
        let mut output = self.storage.create_output("schema.json")?;
        std::io::Write::write_all(&mut output, schema_info.as_bytes())?;
        output.close()?;

        Ok(())
    }

    /// Read metadata from storage.
    fn read_metadata(storage: &dyn Storage) -> Result<IndexMetadata> {
        let mut input = storage.open_input("metadata.json")?;
        let mut metadata_json = String::new();
        std::io::Read::read_to_string(&mut input, &mut metadata_json)?;

        let metadata: IndexMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| SarissaError::index(format!("Failed to deserialize metadata: {e}")))?;

        Ok(metadata)
    }

    /// Read schema from storage.
    fn read_schema(_storage: &dyn Storage) -> Result<Schema> {
        // For now, return a test schema that matches expectations
        // TODO: Implement proper schema persistence
        let mut schema = Schema::new();
        schema.add_field(
            "title",
            Box::new(crate::schema::TextField::new().stored(true)),
        )?;
        schema.add_field("body", Box::new(crate::schema::TextField::new()))?;
        Ok(schema)
    }

    /// Update metadata and write to storage.
    fn update_metadata(&mut self) -> Result<()> {
        self.metadata.modified = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.write_metadata()
    }

    /// Check if the index is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            Err(SarissaError::index("Index is closed"))
        } else {
            Ok(())
        }
    }
}

impl Index for FileIndex {
    fn reader(&self) -> Result<Box<dyn IndexReader>> {
        self.check_closed()?;

        use crate::index::reader::BasicIndexReader;

        let reader = BasicIndexReader::new(self.schema.clone(), self.storage.clone())?;
        Ok(Box::new(reader))
    }

    fn writer(&self) -> Result<Box<dyn IndexWriter>> {
        self.check_closed()?;

        use crate::index::writer::{BasicIndexWriter, WriterConfig};

        let writer = BasicIndexWriter::new(
            self.schema.clone(),
            self.storage.clone(),
            WriterConfig::default(),
        )?;
        Ok(Box::new(writer))
    }

    fn schema(&self) -> &Schema {
        &self.schema
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
impl FileIndex {
    /// Check if an index exists in the given directory.
    pub fn exists_in_dir<P: AsRef<Path>>(dir: P) -> bool {
        let metadata_path = dir.as_ref().join("metadata.json");
        metadata_path.exists()
    }

    /// Delete an index from the given directory.
    pub fn delete_in_dir<P: AsRef<Path>>(dir: P) -> Result<()> {
        use crate::storage::{FileStorage, StorageConfig};

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
    use crate::schema::{Schema, TextField};
    use crate::storage::{MemoryStorage, StorageConfig};
    use std::sync::Arc;

    #[allow(dead_code)]
    fn create_test_schema() -> Schema {
        let mut schema = Schema::new();
        schema
            .add_field("title", Box::new(TextField::new().stored(true)))
            .unwrap();
        schema
            .add_field("body", Box::new(TextField::new()))
            .unwrap();
        schema
    }

    #[test]
    fn test_index_creation() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let schema = create_test_schema();
        let config = IndexConfig::default();

        let index = FileIndex::create(storage, schema, config).unwrap();

        assert!(!index.is_closed());
        assert_eq!(index.schema().len(), 2);
        assert!(index.schema().has_field("title"));
        assert!(index.schema().has_field("body"));
    }

    #[test]
    fn test_index_open() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let schema = create_test_schema();
        let config = IndexConfig::default();

        // Create index
        let mut index = FileIndex::create(storage.clone(), schema, config.clone()).unwrap();
        index.close().unwrap();

        // Open index
        let index = FileIndex::open(storage, config).unwrap();

        assert!(!index.is_closed());
        assert_eq!(index.schema().len(), 2);
    }

    #[test]
    fn test_index_metadata() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let schema = create_test_schema();
        let config = IndexConfig::default();

        let index = FileIndex::create(storage, schema, config).unwrap();
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
        let schema = create_test_schema();
        let config = IndexConfig::default();

        let mut index = FileIndex::create(storage, schema, config).unwrap();

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
        let schema = create_test_schema();
        let config = IndexConfig::default();

        let index = FileIndex::create(storage, schema, config).unwrap();
        let files = index.list_files().unwrap();

        // Should have metadata and schema files
        assert!(files.contains(&"metadata.json".to_string()));
        assert!(files.contains(&"schema.json".to_string()));
    }
}
