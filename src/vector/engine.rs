//! VectorEngine: High-level vector search engine.
//!
//! This module provides a unified interface for vector indexing and search,
//! analogous to `LexicalEngine` for lexical search.
//!
//! # Module Structure
//!
//! - [`config`] - Configuration types (VectorIndexConfig, VectorFieldConfig, VectorIndexKind)
//! - [`embedder`] - Embedding utilities
//! - [`filter`] - Metadata filtering
//! - [`memory`] - In-memory field implementation
//! - [`registry`] - Document vector registry
//! - [`request`] - Search request types
//! - [`response`] - Search response types
//! - [`snapshot`] - Snapshot persistence
//! - [`wal`] - Write-Ahead Logging
//!
//! # Example
//!
//! ```ignore
//! use platypus::vector::engine::VectorEngine;
//! use platypus::vector::engine::config::VectorIndexConfig;
//! use platypus::storage::memory::MemoryStorage;
//! use std::sync::Arc;
//!
//! let storage = Arc::new(MemoryStorage::new(Default::default()));
//! let config = VectorIndexConfig::builder()
//!     .field("body", field_config)
//!     .build()?;
//! let engine = VectorEngine::new(storage, config)?;
//! engine.add_document(doc)?;
//! engine.commit()?;
//! ```

pub mod config;
pub mod embedder;
pub mod filter;
pub mod memory;
pub mod registry;
pub mod request;
pub mod response;
pub mod snapshot;
pub mod wal;

use std::sync::Arc;

use crate::embedding::embedder::Embedder;
use crate::error::Result;
use crate::storage::Storage;
use crate::vector::collection::VectorCollection;
use crate::vector::core::document::{DocumentPayload, DocumentVector, FieldPayload};
use crate::vector::field::{VectorField, VectorFieldReader, VectorFieldStats};

pub use config::{VectorFieldConfig, VectorIndexConfig, VectorIndexKind};
pub use filter::{MetadataFilter, VectorFilter};
pub use registry::{DocumentVectorRegistry, RegistryVersion};
pub use request::{FieldSelector, QueryVector, VectorScoreMode, VectorSearchRequest};
pub use response::{VectorHit, VectorSearchResults, VectorStats};

/// A high-level vector search engine that provides both indexing and searching.
///
/// The `VectorEngine` wraps a `VectorCollection` and provides
/// a simplified, unified interface for all vector operations.
///
/// This design mirrors `LexicalEngine` which wraps a `LexicalIndex` implementation.
///
/// # Example
///
/// ```ignore
/// use platypus::vector::engine::VectorEngine;
/// use platypus::vector::engine::config::VectorIndexConfig;
/// use platypus::storage::memory::MemoryStorage;
/// use std::sync::Arc;
///
/// let storage = Arc::new(MemoryStorage::new(Default::default()));
/// let config = VectorIndexConfig::builder()
///     .field("body", field_config)
///     .build()?;
/// let engine = VectorEngine::new(storage, config)?;
///
/// // Add documents
/// engine.add_document(doc)?;
/// engine.commit()?;
///
/// // Search
/// let results = engine.search(request)?;
/// ```
pub struct VectorEngine {
    collection: VectorCollection,
}

impl std::fmt::Debug for VectorEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorEngine")
            .field("collection", &self.collection)
            .finish()
    }
}

impl VectorEngine {
    // =========================================================================
    // Constructor
    // =========================================================================

    /// Create a new vector engine with the given storage and configuration.
    ///
    /// This is the primary constructor for `VectorEngine`. It creates the
    /// underlying `VectorCollection` internally, providing a simple
    /// and consistent API that mirrors `LexicalEngine::new(storage, config)`.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend for persistence
    /// * `config` - The vector index configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// use platypus::vector::engine::VectorEngine;
    /// use platypus::vector::engine::config::VectorIndexConfig;
    /// use platypus::storage::memory::MemoryStorage;
    /// use std::sync::Arc;
    ///
    /// let storage = Arc::new(MemoryStorage::new(Default::default()));
    /// let config = VectorIndexConfig::builder()
    ///     .field("body", field_config)
    ///     .build()?;
    /// let engine = VectorEngine::new(storage, config)?;
    /// ```
    pub fn new(storage: Arc<dyn Storage>, config: VectorIndexConfig) -> Result<Self> {
        let collection = VectorCollection::new(config, storage, None)?;
        Ok(Self { collection })
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Get the collection configuration.
    pub fn config(&self) -> &VectorIndexConfig {
        self.collection.config()
    }

    /// Get the embedder for this engine.
    ///
    /// This provides direct access to the embedder configured for this engine,
    /// similar to `LexicalEngine::analyzer()`.
    ///
    /// # Returns
    ///
    /// Returns `Arc<dyn Embedder>` containing the embedder.
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(self.collection.config().get_embedder())
    }

    // =========================================================================
    // Document Operations
    // =========================================================================

    /// Add a document with automatically assigned doc_id.
    ///
    /// Returns the assigned document ID.
    pub fn add_document(&self, doc: DocumentVector) -> Result<u64> {
        self.collection.add_document(doc)
    }

    /// Add a document from payload (will be embedded if configured).
    ///
    /// Returns the assigned document ID.
    pub fn add_document_payload(&self, payload: DocumentPayload) -> Result<u64> {
        self.collection.add_document_payload(payload)
    }

    /// Add multiple documents with automatically assigned doc_ids.
    ///
    /// Returns a vector of assigned document IDs in the same order as input.
    ///
    /// # Arguments
    ///
    /// * `docs` - Iterator of documents to add.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let docs = vec![doc1, doc2, doc3];
    /// let doc_ids = engine.add_documents(docs)?;
    /// assert_eq!(doc_ids.len(), 3);
    /// ```
    pub fn add_documents(
        &self,
        docs: impl IntoIterator<Item = DocumentVector>,
    ) -> Result<Vec<u64>> {
        docs.into_iter().map(|doc| self.add_document(doc)).collect()
    }

    /// Add multiple documents from payloads (will be embedded if configured).
    ///
    /// Returns a vector of assigned document IDs in the same order as input.
    ///
    /// # Arguments
    ///
    /// * `payloads` - Iterator of document payloads to add.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let payloads = vec![payload1, payload2, payload3];
    /// let doc_ids = engine.add_documents_payload(payloads)?;
    /// assert_eq!(doc_ids.len(), 3);
    /// ```
    pub fn add_documents_payload(
        &self,
        payloads: impl IntoIterator<Item = DocumentPayload>,
    ) -> Result<Vec<u64>> {
        payloads
            .into_iter()
            .map(|payload| self.add_document_payload(payload))
            .collect()
    }

    /// Upsert a document with a specific document ID.
    pub fn upsert_document(&self, doc_id: u64, doc: DocumentVector) -> Result<()> {
        self.collection.upsert_document(doc_id, doc)
    }

    /// Upsert a document from payload (will be embedded if configured).
    pub fn upsert_document_payload(&self, doc_id: u64, payload: DocumentPayload) -> Result<()> {
        self.collection.upsert_document_payload(doc_id, payload)
    }

    /// Delete a document by ID.
    pub fn delete_document(&self, doc_id: u64) -> Result<()> {
        self.collection.delete_document(doc_id)
    }

    // =========================================================================
    // Embedding Operations
    // =========================================================================

    /// Embed a document payload into vectors.
    pub fn embed_document_payload(&self, payload: DocumentPayload) -> Result<DocumentVector> {
        self.collection.embed_document_payload(payload)
    }

    /// Embed a field payload for query.
    pub fn embed_query_field_payload(
        &self,
        field_name: &str,
        payload: FieldPayload,
    ) -> Result<Vec<QueryVector>> {
        self.collection
            .embed_query_field_payload(field_name, payload)
    }

    // =========================================================================
    // Field Operations
    // =========================================================================

    /// Register an external field implementation.
    pub fn register_field(&self, name: String, field: Box<dyn VectorField>) -> Result<()> {
        self.collection.register_field(name, field)
    }

    /// Get statistics for a specific field.
    pub fn field_stats(&self, field_name: &str) -> Result<VectorFieldStats> {
        self.collection.field_stats(field_name)
    }

    /// Replace the reader for a specific field.
    pub fn replace_field_reader(
        &self,
        field_name: &str,
        reader: Box<dyn VectorFieldReader>,
    ) -> Result<()> {
        self.collection.replace_field_reader(field_name, reader)
    }

    /// Reset the reader for a specific field to default.
    pub fn reset_field_reader(&self, field_name: &str) -> Result<()> {
        self.collection.reset_field_reader(field_name)
    }

    /// Materialize the delegate reader for a field (build persistent index).
    pub fn materialize_delegate_reader(&self, field_name: &str) -> Result<()> {
        self.collection.materialize_delegate_reader(field_name)
    }

    // =========================================================================
    // Search Operations
    // =========================================================================

    /// Execute a search query.
    ///
    /// This method creates a [`VectorSearcher`](crate::vector::search::searcher::VectorSearcher)
    /// from the underlying collection and delegates the search operation to it.
    pub fn search(&self, request: VectorSearchRequest) -> Result<VectorSearchResults> {
        let searcher = self.collection.searcher()?;
        searcher.search(&request)
    }

    // =========================================================================
    // Persistence Operations
    // =========================================================================

    /// Count documents matching the search criteria.
    ///
    /// Returns the number of documents that would match the given search request.
    /// If the request has no query vectors, returns the total document count.
    ///
    /// This method creates a [`VectorSearcher`](crate::vector::search::searcher::VectorSearcher)
    /// from the underlying collection and delegates the count operation to it.
    ///
    /// # Arguments
    ///
    /// * `request` - Search request defining query vectors, filters, and min_score threshold.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Count all documents
    /// let total = engine.count(VectorSearchRequest::default())?;
    ///
    /// // Count documents matching a query with min_score threshold
    /// let mut request = VectorSearchRequest::default();
    /// request.query_vectors = vec![query_vector];
    /// request.min_score = 0.8;
    /// let matching = engine.count(request)?;
    /// ```
    pub fn count(&self, request: VectorSearchRequest) -> Result<u64> {
        let searcher = self.collection.searcher()?;
        searcher.count(&request)
    }

    /// Commit pending changes (persist state).
    pub fn commit(&self) -> Result<()> {
        self.collection.commit()
    }

    // =========================================================================
    // Statistics and Lifecycle
    // =========================================================================

    /// Get collection statistics.
    pub fn stats(&self) -> Result<VectorStats> {
        self.collection.stats()
    }

    /// Get the storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        self.collection.storage()
    }

    /// Close the engine and release resources.
    pub fn close(&self) -> Result<()> {
        self.collection.close()
    }

    /// Check if the engine is closed.
    pub fn is_closed(&self) -> bool {
        self.collection.is_closed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::DistanceMetric;
    use crate::vector::core::document::{FieldVectors, StoredVector, VectorType};
    use std::collections::HashMap;

    fn sample_config() -> VectorIndexConfig {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        };
        use crate::embedding::noop::NoOpEmbedder;

        VectorIndexConfig {
            fields: HashMap::from([("body".into(), field_config)]),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
            embedder: Arc::new(NoOpEmbedder::new()),
        }
    }

    fn sample_query(limit: usize) -> VectorSearchRequest {
        let mut query = VectorSearchRequest::default();
        query.limit = limit;
        query.query_vectors.push(QueryVector {
            vector: StoredVector::new(
                Arc::<[f32]>::from([1.0, 0.0, 0.0]),
                "mock".into(),
                VectorType::Text,
            ),
            weight: 1.0,
        });
        query
    }

    fn create_engine(config: VectorIndexConfig, storage: Arc<dyn Storage>) -> VectorEngine {
        VectorEngine::new(storage, config).expect("engine")
    }

    #[test]
    fn engine_creation_works() {
        let config = sample_config();
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = create_engine(config, storage);

        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 0);
        assert!(stats.fields.contains_key("body"));
    }

    #[test]
    fn engine_add_and_search() {
        let config = sample_config();
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = create_engine(config, storage);

        let mut doc = DocumentVector::new();
        let mut vectors = FieldVectors::default();
        vectors.vectors.push(StoredVector::new(
            Arc::<[f32]>::from([1.0, 0.0, 0.0]),
            "mock".into(),
            VectorType::Text,
        ));
        doc.add_field("body", vectors);

        let doc_id = engine.add_document(doc).expect("add document");

        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 1);

        let results = engine.search(sample_query(5)).expect("search");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, doc_id);
    }

    #[test]
    fn engine_upsert_and_delete() {
        let config = sample_config();
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = create_engine(config, storage);

        let mut doc = DocumentVector::new();
        let mut vectors = FieldVectors::default();
        vectors.vectors.push(StoredVector::new(
            Arc::<[f32]>::from([0.5, 0.5, 0.0]),
            "mock".into(),
            VectorType::Text,
        ));
        doc.add_field("body", vectors);

        engine.upsert_document(42, doc).expect("upsert");
        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 1);

        engine.delete_document(42).expect("delete");
        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 0);
    }

    #[test]
    fn engine_persistence_across_instances() {
        let config = sample_config();
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let engine = create_engine(config.clone(), storage.clone());
            let mut doc = DocumentVector::new();
            let mut vectors = FieldVectors::default();
            vectors.vectors.push(StoredVector::new(
                Arc::<[f32]>::from([1.0, 0.0, 0.0]),
                "mock".into(),
                VectorType::Text,
            ));
            doc.add_field("body", vectors);
            engine.upsert_document(10, doc).expect("upsert");
        }

        let engine = create_engine(config, storage);
        let stats = engine.stats().expect("stats");
        assert_eq!(stats.document_count, 1);

        let results = engine.search(sample_query(5)).expect("search");
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.hits[0].doc_id, 10);
    }
}
