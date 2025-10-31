//! Inverted index implementation for full-text search.
//!
//! This module provides the core inverted index implementation:
//! - Core data structures (posting lists, term enumeration)
//! - Index creation and management
//! - Writer for building the index
//! - Reader for querying the index
//! - Searcher for executing searches
//! - Segment management and merging
//! - Index maintenance operations
//! - Query types for searching

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{Result, SageError};
use crate::lexical::index::LexicalIndex;
use crate::lexical::index::config::InvertedIndexConfig;
use crate::lexical::reader::IndexReader;
use crate::lexical::writer::IndexWriter;
use crate::storage::Storage;
use crate::storage::file::{FileStorage, FileStorageConfig};

pub mod core;
pub mod maintenance;
pub mod query;
pub mod reader;
pub mod searcher;
pub mod segment;
pub mod writer;

use self::reader::{InvertedIndexReader, InvertedIndexReaderConfig};
use self::segment::SegmentInfo;
use self::writer::{InvertedIndexWriter, InvertedIndexWriterConfig};

/// Metadata about an inverted index.
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

/// Statistics about an inverted index.
#[derive(Debug, Clone)]
pub struct InvertedIndexStats {
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

/// A concrete inverted index implementation for schema-less lexical indexing.
#[derive(Debug)]
pub struct InvertedIndex {
    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// Inverted index specific configuration.
    config: InvertedIndexConfig,

    /// Whether the index is closed.
    closed: bool,

    /// Index metadata.
    metadata: IndexMetadata,
}

impl InvertedIndex {
    /// Create a new index in the given storage.
    pub fn create(storage: Arc<dyn Storage>, config: InvertedIndexConfig) -> Result<Self> {
        let metadata = IndexMetadata::default();

        let index = InvertedIndex {
            storage,
            config,
            closed: false,
            metadata,
        };

        index.write_metadata()?;
        Ok(index)
    }

    /// Open an existing index from storage.
    pub fn open(storage: Arc<dyn Storage>, config: InvertedIndexConfig) -> Result<Self> {
        if !storage.file_exists("metadata.json") {
            return Err(SageError::index("Index does not exist"));
        }

        let metadata = Self::read_metadata(storage.as_ref())?;

        Ok(InvertedIndex {
            storage,
            config,
            closed: false,
            metadata,
        })
    }

    /// Create an index in a directory.
    pub fn create_in_dir<P: AsRef<Path>>(dir: P, config: InvertedIndexConfig) -> Result<Self> {
        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::create(storage, config)
    }

    /// Open an index from a directory.
    pub fn open_dir<P: AsRef<Path>>(dir: P, config: InvertedIndexConfig) -> Result<Self> {
        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::open(storage, config)
    }

    /// Write metadata to storage.
    fn write_metadata(&self) -> Result<()> {
        let metadata_json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| SageError::index(format!("Failed to serialize metadata: {e}")))?;

        let mut output = self.storage.create_output("metadata.json")?;
        std::io::Write::write_all(&mut output, metadata_json.as_bytes())?;
        output.close()?;

        Ok(())
    }

    /// Read metadata from storage.
    fn read_metadata(storage: &dyn Storage) -> Result<IndexMetadata> {
        let mut input = storage.open_input("metadata.json")?;
        let mut metadata_json = String::new();
        Read::read_to_string(&mut input, &mut metadata_json)?;

        let metadata: IndexMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| SageError::index(format!("Failed to deserialize metadata: {e}")))?;

        Ok(metadata)
    }

    /// Update metadata and write to storage.
    fn update_metadata(&mut self) -> Result<()> {
        self.metadata.modified = SystemTime::now()
            .duration_since(UNIX_EPOCH)
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

        for file in &files {
            if file.starts_with("segment_") && file.ends_with(".meta") {
                let mut input = self.storage.open_input(file)?;
                let mut data = Vec::new();
                Read::read_to_end(&mut input, &mut data)?;

                let segment_info: SegmentInfo = serde_json::from_slice(&data).map_err(|e| {
                    SageError::index(format!("Failed to parse segment metadata: {e}"))
                })?;

                segments.push(segment_info);
            }
        }

        segments.sort_by_key(|s| s.generation);
        Ok(segments)
    }

    /// Check if an index exists in the given directory.
    pub fn exists_in_dir<P: AsRef<Path>>(dir: P) -> bool {
        let metadata_path = dir.as_ref().join("metadata.json");
        metadata_path.exists()
    }

    /// Delete an index from the given directory.
    pub fn delete_in_dir<P: AsRef<Path>>(dir: P) -> Result<()> {
        let storage_config = FileStorageConfig::new(&dir);
        let storage = FileStorage::new(&dir, storage_config)?;

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

impl LexicalIndex for InvertedIndex {
    fn reader(&self) -> Result<Box<dyn IndexReader>> {
        self.check_closed()?;

        let segments = self.load_segments()?;

        // Use analyzer from index config
        let reader_config = InvertedIndexReaderConfig {
            analyzer: self.config.analyzer.clone(),
            ..InvertedIndexReaderConfig::default()
        };

        let reader = InvertedIndexReader::new(segments, self.storage.clone(), reader_config)?;
        Ok(Box::new(reader))
    }

    fn writer(&self) -> Result<Box<dyn IndexWriter>> {
        self.check_closed()?;

        // Use analyzer from index config
        let writer_config = InvertedIndexWriterConfig {
            analyzer: self.config.analyzer.clone(),
            ..Default::default()
        };
        let writer = InvertedIndexWriter::new(self.storage.clone(), writer_config)?;
        Ok(Box::new(writer))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    fn close(&mut self) -> Result<()> {
        if !self.closed {
            self.closed = true;
        }
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed
    }

    fn stats(&self) -> Result<InvertedIndexStats> {
        self.check_closed()?;

        Ok(InvertedIndexStats {
            doc_count: self.metadata.doc_count,
            term_count: 0,
            segment_count: 0,
            total_size: 0,
            deleted_count: 0,
            last_modified: self.metadata.modified,
        })
    }

    fn optimize(&mut self) -> Result<()> {
        self.check_closed()?;
        self.update_metadata()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::document::Document;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use std::sync::Arc;

    #[allow(dead_code)]
    fn create_test_document(title: &str, body: &str) -> Document {
        Document::builder()
            .add_text("title", title)
            .add_text("body", body)
            .build()
    }

    #[test]
    fn test_inverted_index_writer_creation() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let writer = InvertedIndexWriter::new(storage, config).unwrap();

        assert_eq!(writer.pending_docs(), 0);
        assert_eq!(writer.stats().docs_added, 0);
    }

    #[test]
    fn test_add_document() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();
        let doc = create_test_document("Test Title", "This is test content");

        writer.add_document(doc).unwrap();

        assert_eq!(writer.pending_docs(), 1);
        assert_eq!(writer.stats().docs_added, 1);
        assert!(writer.stats().unique_terms > 0);
    }

    #[test]
    fn test_auto_flush() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig {
            max_buffered_docs: 2,
            ..Default::default()
        };

        let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();

        // Add first document
        writer
            .add_document(create_test_document("Doc 1", "Content 1"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 1);

        // Add second document - should trigger flush
        writer
            .add_document(create_test_document("Doc 2", "Content 2"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 0); // Flushed
        assert_eq!(writer.stats().segments_created, 1);

        // Check that files were created
        let files = storage.list_files().unwrap();
        assert!(files.iter().any(|f| f.contains("segment_000000")));
    }

    #[test]
    fn test_commit() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();

        writer
            .add_document(create_test_document("Test", "Content"))
            .unwrap();
        writer.commit().unwrap();

        assert_eq!(writer.pending_docs(), 0);

        // Check that files were created
        let files = storage.list_files().unwrap();
        assert!(files.contains(&"index.meta".to_string()));
        assert!(files.iter().any(|f| f.starts_with("segment_")));
    }

    #[test]
    fn test_rollback() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();

        writer
            .add_document(create_test_document("Test", "Content"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 1);

        writer.rollback().unwrap();
        assert_eq!(writer.pending_docs(), 0);
        assert_eq!(writer.stats().docs_added, 1); // Stats don't rollback
    }

    #[test]
    fn test_multiple_field_types() {
        // Schema-less mode: fields are inferred from document
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();

        let doc = Document::builder()
            .add_text("title", "Test Document")
            .add_text("id", "doc1")
            .add_numeric("count", 42.0)
            .build();

        writer.add_document(doc).unwrap();
        writer.commit().unwrap();

        assert_eq!(writer.stats().docs_added, 1);
        assert!(writer.stats().unique_terms >= 3); // At least title, id, count fields
    }
}
