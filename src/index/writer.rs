//! Index writer trait and common types.
//!
//! This module defines the `IndexWriter` trait which all index writer
//! implementations must follow. The primary implementation is `AdvancedIndexWriter`
//! in the `advanced_writer` module.

use crate::document::Document;
use crate::error::Result;
use crate::index::advanced_writer::AnalyzedDocument;

/// Trait for index writers.
///
/// This trait defines the common interface that all index writer implementations
/// must follow. The primary implementation is `AdvancedIndexWriter`.
///
/// # Example
///
/// ```rust,no_run
/// use sarissa::index::advanced_writer::{AdvancedIndexWriter, AdvancedWriterConfig};
/// use sarissa::index::writer::IndexWriter;
/// use sarissa::storage::{MemoryStorage, StorageConfig};
/// use std::sync::Arc;
///
/// let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
/// let config = AdvancedWriterConfig::default();
/// let mut writer = AdvancedIndexWriter::new(storage, config).unwrap();
///
/// // Use IndexWriter trait methods
/// // writer.add_document(doc).unwrap();
/// // writer.commit().unwrap();
/// ```
pub trait IndexWriter: Send + std::fmt::Debug {
    /// Add a document to the index.
    fn add_document(&mut self, doc: Document) -> Result<()>;

    /// Add an already analyzed document to the index.
    ///
    /// This allows adding pre-analyzed documents that were processed
    /// using DocumentParser or from external tokenization systems.
    fn add_analyzed_document(&mut self, doc: AnalyzedDocument) -> Result<()>;

    /// Delete documents matching the given term.
    fn delete_documents(&mut self, field: &str, value: &str) -> Result<u64>;

    /// Update a document (delete old, add new).
    fn update_document(&mut self, field: &str, value: &str, doc: Document) -> Result<()>;

    /// Commit all pending changes to the index.
    fn commit(&mut self) -> Result<()>;

    /// Rollback all pending changes.
    fn rollback(&mut self) -> Result<()>;

    /// Get the number of documents added since the last commit.
    fn pending_docs(&self) -> u64;

    /// Close the writer and release resources.
    fn close(&mut self) -> Result<()>;

    /// Check if the writer is closed.
    fn is_closed(&self) -> bool;
}
