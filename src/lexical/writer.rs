//! Lexical index writer trait and common types.
//!
//! This module defines the `LexicalIndexWriter` trait which all lexical index writer
//! implementations must follow. The primary implementation is `InvertedIndexWriter`.

use crate::error::Result;
use crate::lexical::core::analyzed::AnalyzedDocument;
use crate::lexical::core::document::Document;

/// Trait for lexical index writers.
///
/// This trait defines the common interface that all lexical index writer implementations
/// must follow. The primary implementation is `InvertedIndexWriter`.
///
/// # Example
///
/// ```rust,no_run
/// use sarissa::lexical::index::inverted::writer::{InvertedIndexWriter, InvertedIndexWriterConfig};
/// use sarissa::lexical::writer::LexicalIndexWriter;
/// use sarissa::storage::memory::{MemoryStorage, MemoryStorageConfig};
/// use sarissa::storage::StorageConfig;
/// use std::sync::Arc;
///
/// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
/// let config = InvertedIndexWriterConfig::default();
/// let mut writer = InvertedIndexWriter::new(storage, config).unwrap();
///
/// // Use LexicalIndexWriter trait methods
/// // writer.add_document(doc).unwrap();
/// // writer.commit().unwrap();
/// ```
pub trait LexicalIndexWriter: Send + Sync + std::fmt::Debug {
    /// Add a document to the index with automatic ID assignment.
    /// Returns the assigned document ID.
    fn add_document(&mut self, doc: Document) -> Result<u64>;

    /// Upsert a document to the index with a specific document ID.
    fn upsert_document(&mut self, doc_id: u64, doc: Document) -> Result<()>;

    /// Add an already analyzed document to the index with automatic ID assignment.
    /// Returns the assigned document ID.
    ///
    /// This allows adding pre-analyzed documents that were processed
    /// using DocumentParser or from external tokenization systems.
    fn add_analyzed_document(&mut self, doc: AnalyzedDocument) -> Result<u64>;

    /// Upsert an already analyzed document to the index with a specific document ID.
    fn upsert_analyzed_document(&mut self, doc_id: u64, doc: AnalyzedDocument) -> Result<()>;

    /// Delete a document by ID.
    fn delete_document(&mut self, doc_id: u64) -> Result<()>;

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
