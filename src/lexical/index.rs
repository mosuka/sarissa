//! Lexical indexing module for building and maintaining lexical indexes.
//!
//! This module handles all lexical index construction:
//! - Building inverted indexes
//! - Document indexing and analysis
//!
//! # Module Structure
//!
//! - `config`: Index configuration
//! - `factory`: Index factory for creating and opening indexes
//! - `inverted`: Inverted index implementation (including segments and maintenance)

use std::sync::Arc;

use crate::error::Result;
use crate::lexical::index::inverted::InvertedIndexStats;
use crate::lexical::reader::LexicalIndexReader;
use crate::lexical::search::searcher::LexicalSearcher;
use crate::lexical::writer::LexicalIndexWriter;
use crate::storage::Storage;

/// Trait for lexical index implementations.
///
/// This trait defines the high-level interface for lexical indexes.
/// Different index types (Inverted, ColumnStore, LSMTree, etc.) implement this trait
/// to provide their specific functionality while maintaining a common interface.
pub trait LexicalIndex: Send + Sync + std::fmt::Debug {
    /// Get a reader for this index.
    ///
    /// Returns a reader that can be used to query the index.
    fn reader(&self) -> Result<Arc<dyn LexicalIndexReader>>;

    /// Get a writer for this index.
    ///
    /// Returns a writer that can be used to add documents to the index.
    fn writer(&self) -> Result<Box<dyn LexicalIndexWriter>>;

    /// Get the storage backend for this index.
    ///
    /// Returns a reference to the underlying storage.
    fn storage(&self) -> &Arc<dyn Storage>;

    /// Close the index and release resources.
    ///
    /// This should flush any pending writes and release all resources.
    /// Uses interior mutability for thread-safe access.
    fn close(&self) -> Result<()>;

    /// Check if the index is closed.
    ///
    /// Returns true if the index has been closed.
    fn is_closed(&self) -> bool;

    /// Get index statistics.
    ///
    /// Returns statistics about the index such as document count, term count, etc.
    fn stats(&self) -> Result<InvertedIndexStats>;

    /// Optimize the index (merge segments, etc.).
    ///
    /// Performs index optimization such as merging segments to improve query performance.
    /// Uses interior mutability for thread-safe access.
    fn optimize(&self) -> Result<()>;

    /// Create a searcher tailored for this index implementation.
    ///
    /// Returns a boxed [`LexicalSearcher`] capable of executing search/count operations.
    fn searcher(&self) -> Result<Box<dyn LexicalSearcher>>;

    /// Get the default fields configured for this index.
    fn default_fields(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    // =========================================================================
    // Cached access methods (for LexicalEngine delegation)
    // =========================================================================

    /// Get or create a cached writer and add a document.
    ///
    /// This method lazily creates a writer on first use and caches it for subsequent calls.
    fn add_document(&self, doc: crate::lexical::core::document::Document) -> Result<u64>;

    /// Get or create a cached writer and upsert a document.
    fn upsert_document(
        &self,
        doc_id: u64,
        doc: crate::lexical::core::document::Document,
    ) -> Result<()>;

    /// Get or create a cached writer and delete a document.
    fn delete_document(&self, doc_id: u64) -> Result<()>;

    /// Get or create a cached writer and add multiple documents.
    ///
    /// Returns a vector of assigned document IDs.
    fn add_documents(
        &self,
        docs: Vec<crate::lexical::core::document::Document>,
    ) -> Result<Vec<u64>>;

    /// Get or create a cached searcher and execute a search.
    fn search(
        &self,
        request: crate::lexical::search::searcher::LexicalSearchRequest,
    ) -> Result<crate::lexical::index::inverted::query::LexicalSearchResults>;

    /// Get or create a cached searcher and count matching documents.
    fn count(&self, request: crate::lexical::search::searcher::LexicalSearchRequest)
    -> Result<u64>;

    /// Commit pending writes and invalidate caches.
    ///
    /// This method commits any pending write operations from the cached writer,
    /// then invalidates the searcher cache to ensure subsequent searches see the new data.
    fn commit(&self) -> Result<()>;

    /// Invalidate searcher cache to see latest changes.
    ///
    /// This method clears the cached searcher so that the next search operation
    /// will create a fresh searcher with the latest index state.
    fn refresh(&self) -> Result<()>;
}

pub mod config;
pub mod factory;

pub mod inverted;
pub mod structures;
