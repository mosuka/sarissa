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
use crate::vector::engine::request::{QueryVector, VectorEngineSearchRequest};
use crate::vector::engine::response::{VectorEngineSearchResults, VectorEngineStats};
use crate::vector::field::{VectorField, VectorFieldReader, VectorFieldStats};

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

    /// Execute a search query.
    fn search(&self, request: &VectorEngineSearchRequest) -> Result<VectorEngineSearchResults>;

    /// Count documents matching the search criteria.
    ///
    /// Returns the number of documents that would match the given search request.
    /// This is equivalent to performing a search and counting the results,
    /// but may be more efficient for certain implementations.
    ///
    /// # Arguments
    ///
    /// * `request` - Search request defining the query vectors, filters, and min_score threshold.
    ///               If `query_vectors` is empty, returns total document count.
    ///
    /// # Returns
    ///
    /// The count of matching documents.
    fn count(&self, request: &VectorEngineSearchRequest) -> Result<usize>;

    // =========================================================================
    // Persistence Operations
    // =========================================================================

    /// Commit pending changes (persist state).
    fn commit(&self) -> Result<()>;

    // =========================================================================
    // Statistics and Lifecycle
    // =========================================================================

    /// Get collection statistics.
    fn stats(&self) -> Result<VectorEngineStats>;

    /// Get the storage backend.
    fn storage(&self) -> &Arc<dyn Storage>;

    /// Close the collection and release resources.
    fn close(&mut self) -> Result<()>;

    /// Check if the collection is closed.
    fn is_closed(&self) -> bool;
}
