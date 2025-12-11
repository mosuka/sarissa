//! VectorEngine: High-level vector search engine.
//!
//! This module provides a unified interface for vector indexing and search,
//! analogous to `LexicalEngine` for lexical search.
//!
//! # Module Structure
//!
//! - [`config`] - Configuration types (VectorEngineConfig, VectorFieldConfig, VectorIndexKind)
//! - [`embedder`] - Embedding registry and executor
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
//! use platypus::vector::collection::factory::VectorCollectionFactory;
//! use platypus::vector::engine::config::VectorEngineConfig;
//!
//! let config = VectorEngineConfig::default();
//! let collection = VectorCollectionFactory::create(config, storage, None)?;
//! let engine = VectorEngine::new(collection)?;
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

use crate::embedding::image_embedder::ImageEmbedder;
use crate::embedding::text_embedder::TextEmbedder;
use crate::error::Result;
use crate::storage::Storage;
use crate::vector::collection::VectorCollection;
use crate::vector::core::document::{DocumentPayload, DocumentVector, FieldPayload};
use crate::vector::field::{VectorField, VectorFieldReader, VectorFieldStats};

pub use config::{
    VectorEmbedderConfig, VectorEmbedderProvider, VectorEngineConfig, VectorFieldConfig,
    VectorIndexKind,
};
pub use filter::{MetadataFilter, VectorEngineFilter};
pub use registry::{DocumentVectorRegistry, RegistryVersion};
pub use request::{FieldSelector, QueryVector, VectorEngineSearchRequest, VectorScoreMode};
pub use response::{VectorEngineHit, VectorEngineSearchResults, VectorEngineStats};

/// A high-level vector search engine that provides both indexing and searching.
///
/// The `VectorEngine` wraps a `VectorCollection` trait object and provides
/// a simplified, unified interface for all vector operations.
///
/// This design mirrors `LexicalEngine` which wraps `LexicalIndex` trait object.
///
/// # Example
///
/// ```ignore
/// use platypus::vector::engine::VectorEngine;
/// use platypus::vector::collection::factory::VectorCollectionFactory;
/// use platypus::vector::engine::config::VectorEngineConfig;
///
/// // Create collection via factory
/// let collection = VectorCollectionFactory::create(config, storage, None)?;
/// let engine = VectorEngine::new(collection)?;
///
/// // Add documents
/// engine.add_document(doc)?;
/// engine.commit()?;
///
/// // Search
/// let results = engine.search(request)?;
/// ```
pub struct VectorEngine {
    collection: Box<dyn VectorCollection>,
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

    /// Create a new vector engine with the given collection.
    ///
    /// # Arguments
    ///
    /// * `collection` - A boxed `VectorCollection` trait object
    ///
    /// # Example
    ///
    /// ```ignore
    /// let collection = VectorCollectionFactory::create(config, storage, None)?;
    /// let engine = VectorEngine::new(collection)?;
    /// ```
    pub fn new(collection: Box<dyn VectorCollection>) -> Result<Self> {
        Ok(Self { collection })
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Get the collection configuration.
    pub fn config(&self) -> &VectorEngineConfig {
        self.collection.config()
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
    // Embedder Operations
    // =========================================================================

    /// Register an external text embedder instance.
    pub fn register_embedder_instance(
        &self,
        embedder_id: String,
        embedder: Arc<dyn TextEmbedder>,
    ) -> Result<()> {
        self.collection
            .register_embedder_instance(embedder_id, embedder)
    }

    /// Register an external multimodal embedder instance.
    pub fn register_multimodal_embedder_instance(
        &self,
        embedder_id: String,
        text_embedder: Arc<dyn TextEmbedder>,
        image_embedder: Arc<dyn ImageEmbedder>,
    ) -> Result<()> {
        self.collection.register_multimodal_embedder_instance(
            embedder_id,
            text_embedder,
            image_embedder,
        )
    }

    // =========================================================================
    // Search Operations
    // =========================================================================

    /// Execute a search query.
    pub fn search(&self, request: VectorEngineSearchRequest) -> Result<VectorEngineSearchResults> {
        self.collection.search(&request)
    }

    // =========================================================================
    // Persistence Operations
    // =========================================================================

    /// Commit pending changes (persist state).
    pub fn commit(&self) -> Result<()> {
        self.collection.commit()
    }

    // =========================================================================
    // Statistics and Lifecycle
    // =========================================================================

    /// Get collection statistics.
    pub fn stats(&self) -> Result<VectorEngineStats> {
        self.collection.stats()
    }

    /// Get the storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        self.collection.storage()
    }

    /// Close the engine and release resources.
    pub fn close(&mut self) -> Result<()> {
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
    use crate::vector::collection::factory::VectorCollectionFactory;
    use crate::vector::core::document::{FieldVectors, StoredVector, VectorType};
    use std::collections::HashMap;

    fn sample_config() -> VectorEngineConfig {
        let field_config = VectorFieldConfig {
            dimension: 3,
            distance: DistanceMetric::Cosine,
            index: VectorIndexKind::Flat,
            embedder_id: "mock".into(),
            vector_type: VectorType::Text,
            embedder: None,
            base_weight: 1.0,
        };
        #[allow(deprecated)]
        VectorEngineConfig {
            fields: HashMap::from([("body".into(), field_config)]),
            embedders: HashMap::new(),
            default_fields: vec!["body".into()],
            metadata: HashMap::new(),
            embedder: None,
        }
    }

    fn sample_query(limit: usize) -> VectorEngineSearchRequest {
        let mut query = VectorEngineSearchRequest::default();
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

    fn create_engine(config: VectorEngineConfig, storage: Arc<dyn Storage>) -> VectorEngine {
        let collection =
            VectorCollectionFactory::create(config, storage, None).expect("collection");
        VectorEngine::new(collection).expect("engine")
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
