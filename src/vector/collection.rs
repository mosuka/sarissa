//! Vector collection trait and related types.
//!
//! This module defines the high-level interface for vector collections,
//! analogous to `LexicalIndex` for lexical search.
//!
//! # Module Structure
//!
//! - [`VectorCollection`] - Core trait for vector collection implementations
//! - [`factory`] - Factory for creating vector collections
//! - [`multifield`] - Multi-field vector collection implementation

pub mod factory;
pub mod multifield;

use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::core::document::{DocumentPayload, DocumentVector, FieldPayload};
use crate::vector::engine::config::VectorIndexConfig;
use crate::vector::engine::request::QueryVector;
use crate::vector::engine::response::VectorStats;
use crate::vector::field::{VectorField, VectorFieldReader, VectorFieldStats};
use crate::vector::search::searcher::VectorSearcher;

/// Trait for vector collection implementations.
///
/// This trait defines the high-level interface for vector collections,
/// analogous to [`crate::lexical::index::LexicalIndex`] for lexical search.
/// Different collection types implement this trait to provide their
/// specific functionality while maintaining a common interface.
///
/// # Example
///
/// ```ignore
/// use platypus::vector::collection::VectorCollection;
/// use platypus::vector::collection::factory::VectorCollectionFactory;
///
/// let collection = VectorCollectionFactory::create(config, storage, None)?;
/// collection.add_document(doc)?;
/// collection.commit()?;
/// let results = collection.search(&request)?;
/// ```
pub trait VectorIndex: Send + Sync + std::fmt::Debug {
    // =========================================================================
    // Configuration
    // =========================================================================

    /// Get the collection configuration.
    fn config(&self) -> &VectorIndexConfig;

    // =========================================================================
    // Document Operations
    // =========================================================================

    /// Add a document to the collection.
    ///
    /// Returns the assigned document ID.
    fn add_document(&self, doc: DocumentVector) -> Result<u64>;

    /// Add a document from payload (will be embedded if configured).
    ///
    /// Returns the assigned document ID.
    fn add_document_payload(&self, payload: DocumentPayload) -> Result<u64>;

    /// Upsert a document with a specific document ID.
    fn upsert_document(&self, doc_id: u64, doc: DocumentVector) -> Result<()>;

    /// Upsert a document from payload (will be embedded if configured).
    fn upsert_document_payload(&self, doc_id: u64, payload: DocumentPayload) -> Result<()>;

    /// Delete a document by ID.
    fn delete_document(&self, doc_id: u64) -> Result<()>;

    // =========================================================================
    // Embedding Operations
    // =========================================================================

    /// Embed a document payload into vectors.
    fn embed_document_payload(&self, payload: DocumentPayload) -> Result<DocumentVector>;

    /// Embed a field payload for query.
    fn embed_query_field_payload(
        &self,
        field_name: &str,
        payload: FieldPayload,
    ) -> Result<Vec<QueryVector>>;

    // =========================================================================
    // Field Operations
    // =========================================================================

    /// Register an external field implementation.
    fn register_field(&self, name: String, field: Box<dyn VectorField>) -> Result<()>;

    /// Get statistics for a specific field.
    fn field_stats(&self, field_name: &str) -> Result<VectorFieldStats>;

    /// Replace the reader for a specific field.
    fn replace_field_reader(
        &self,
        field_name: &str,
        reader: Box<dyn VectorFieldReader>,
    ) -> Result<()>;

    /// Reset the reader for a specific field to default.
    fn reset_field_reader(&self, field_name: &str) -> Result<()>;

    /// Materialize the delegate reader for a field (build persistent index).
    fn materialize_delegate_reader(&self, field_name: &str) -> Result<()>;

    // =========================================================================
    // Search Operations
    // =========================================================================

    /// Create a searcher for this index.
    ///
    /// Returns a boxed [`VectorSearcher`] capable of executing search and count operations.
    /// This method is analogous to [`LexicalIndex::searcher()`](crate::lexical::index::LexicalIndex::searcher)
    /// in the lexical search module.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let searcher = index.searcher()?;
    /// let results = searcher.search(&request)?;
    /// let count = searcher.count(&request)?;
    /// ```
    fn searcher(&self) -> Result<Box<dyn VectorSearcher>>;

    // =========================================================================
    // Persistence Operations
    // =========================================================================

    /// Commit pending changes (persist state).
    fn commit(&self) -> Result<()>;

    // =========================================================================
    // Statistics and Lifecycle
    // =========================================================================

    /// Get collection statistics.
    fn stats(&self) -> Result<VectorStats>;

    /// Get the storage backend.
    fn storage(&self) -> &Arc<dyn Storage>;

    /// Close the collection and release resources.
    fn close(&mut self) -> Result<()>;

    /// Check if the collection is closed.
    fn is_closed(&self) -> bool;
}
