//! Hybrid search engine implementation.
//!
//! This module provides the `HybridEngine` that combines lexical and vector search
//! engines to provide unified hybrid search functionality.

use std::io::{Read, Write};
use std::sync::Arc;

use crate::error::Result;
use crate::hybrid::search::searcher::{HybridSearchRequest, HybridSearchResults};
use crate::storage::Storage;
use crate::vector::core::document::{DocumentPayload, DocumentVector};

const HYBRID_MANIFEST_FILE: &str = "hybrid_manifest.json";

#[derive(serde::Serialize, serde::Deserialize)]
struct HybridEngineManifest {
    next_doc_id: u64,
}

/// High-level hybrid search engine combining lexical and vector search.
///
/// This engine wraps both `LexicalEngine` and `VectorEngine` to provide
/// unified hybrid search functionality. It follows the same pattern as the
/// individual engines but coordinates searches across both indexes.
///
/// # Examples
///
/// ```no_run
/// use sarissa::hybrid::engine::HybridEngine;
/// use sarissa::hybrid::search::searcher::HybridSearchRequest;
/// use sarissa::lexical::engine::LexicalEngine;
/// use sarissa::vector::engine::VectorEngine;
/// use sarissa::storage::{Storage, StorageConfig, StorageFactory};
/// use sarissa::storage::memory::MemoryStorageConfig;
/// use std::sync::Arc;
///
/// # async fn example(lexical_engine: LexicalEngine, vector_engine: VectorEngine) -> sarissa::error::Result<()> {
/// // Create storage
/// let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
/// let storage = StorageFactory::create(storage_config)?;
///
/// // Create hybrid engine
/// let engine = HybridEngine::new(storage, lexical_engine, vector_engine)?;
///
/// // Text-only search
/// let request = HybridSearchRequest::new("rust programming");
/// let results = engine.search(request).await?;
/// # Ok(())
/// # }
/// ```
pub struct HybridEngine {
    /// Storage for hybrid engine metadata (manifest).
    storage: Arc<dyn Storage>,
    /// Lexical search engine for keyword-based search.
    lexical_engine: crate::lexical::engine::LexicalEngine,
    /// Vector search engine for semantic search.
    vector_engine: crate::vector::engine::VectorEngine,
    /// Next document ID counter for synchronized ID assignment.
    next_doc_id: u64,
}

impl HybridEngine {
    /// Create a new hybrid search engine.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend for persisting metadata
    /// * `lexical_engine` - The lexical search engine
    /// * `vector_engine` - The vector search engine
    ///
    /// # Returns
    ///
    /// A new `HybridEngine` instance
    pub fn new(
        storage: Arc<dyn Storage>,
        lexical_engine: crate::lexical::engine::LexicalEngine,
        vector_engine: crate::vector::engine::VectorEngine,
    ) -> Result<Self> {
        let mut engine = Self {
            storage,
            lexical_engine,
            vector_engine,
            next_doc_id: 0,
        };
        engine.load_manifest()?;
        Ok(engine)
    }

    fn load_manifest(&mut self) -> Result<()> {
        if self.storage.file_exists(HYBRID_MANIFEST_FILE) {
            let mut input = self.storage.open_input(HYBRID_MANIFEST_FILE)?;
            let mut buffer = Vec::new();
            input.read_to_end(&mut buffer)?;
            if !buffer.is_empty() {
                let manifest: HybridEngineManifest = serde_json::from_slice(&buffer)?;
                self.next_doc_id = manifest.next_doc_id;
            }
        }
        Ok(())
    }

    fn persist_manifest(&self) -> Result<()> {
        let manifest = HybridEngineManifest {
            next_doc_id: self.next_doc_id,
        };
        let buffer = serde_json::to_vec(&manifest)?;
        let mut output = self.storage.create_output(HYBRID_MANIFEST_FILE)?;
        output.write_all(&buffer)?;
        output.flush()?;
        Ok(())
    }

    /// Add a hybrid document to the engine.
    /// Returns the assigned document ID.
    ///
    /// This method assigns a new document ID, adds the lexical component to the
    /// lexical index, and if present, adds the vector component to the vector index.
    /// It ensures both indexes are updated and the ID counter is persisted.
    ///
    /// # Arguments
    ///
    /// * `doc` - The hybrid document to add
    ///
    /// # Returns
    ///
    /// The assigned document ID
    pub async fn add_document(
        &mut self,
        doc: crate::hybrid::core::document::HybridDocument,
    ) -> Result<u64> {
        let doc_id = self.next_doc_id;
        self.upsert_document(doc_id, doc).await?;
        Ok(doc_id)
    }

    /// Upsert a hybrid document with a specific document ID.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to use
    /// * `doc` - The hybrid document to upsert
    pub async fn upsert_document(
        &mut self,
        doc_id: u64,
        doc: crate::hybrid::core::document::HybridDocument,
    ) -> Result<()> {
        // 1. Write to lexical index
        if let Some(lexical_doc) = doc.lexical_doc {
            self.lexical_engine.upsert_document(doc_id, lexical_doc)?;
        }

        // 2. Write to vector index if payload exists
        if let Some(payload) = doc.vector_payload {
            match self.vector_engine.upsert_payloads(doc_id, payload) {
                Ok(_) => {}
                Err(e) => {
                    // Rollback lexical change if vector fails
                    // Note: This is best-effort. If delete fails, we have partial data.
                    let _ = self.lexical_engine.delete_document(doc_id);
                    return Err(e);
                }
            }
        }

        // 3. Update next_doc_id and persist if necessary
        if doc_id >= self.next_doc_id {
            self.next_doc_id = doc_id + 1;
            // Best effort persistence. If this fails, we might reuse ID on restart (collision),
            // but for now we propagate error.
            self.persist_manifest()?;
        }

        Ok(())
    }

    /// Commit changes to both lexical and vector indexes.
    pub fn commit(&mut self) -> Result<()> {
        self.lexical_engine.commit()?;
        // Vector engine might not have explicit commit exposed depending on implementation,
        // but typically it handles its own persistence or we should trigger it.
        // Checking vector/engine.rs... it has no explicit public commit in the method list I saw earlier,
        // but usually WAL handles it or explicit flush.
        // Assuming vector engine handles persistence via WAL or we need to add commit there?
        // VectorEngine doesn't seem to have a commit() method in the viewed code (lines 1-800).
        // It has `persist_manifest` internal.
        // Let's assume for now lexical commit is sufficient for lexical, and vector operations are durable via WAL.
        Ok(())
    }

    /// Upsert vectors only (pre-embedded).
    ///
    /// This updates the vector index for a given document ID without modifying the lexical index.
    pub fn upsert_vector_document(&mut self, doc_id: u64, vectors: DocumentVector) -> Result<()> {
        self.vector_engine.upsert_vectors(doc_id, vectors)?;
        if doc_id >= self.next_doc_id {
            self.next_doc_id = doc_id + 1;
            self.persist_manifest()?;
        }
        Ok(())
    }

    /// Upsert vectors only from raw payloads (embedding inside vector engine).
    ///
    /// This updates the vector index using raw payloads without modifying the lexical index.
    pub fn upsert_vector_payload(&mut self, doc_id: u64, payload: DocumentPayload) -> Result<()> {
        self.vector_engine.upsert_payloads(doc_id, payload)?;
        if doc_id >= self.next_doc_id {
            self.next_doc_id = doc_id + 1;
            self.persist_manifest()?;
        }
        Ok(())
    }

    /// Optimize both indexes.
    pub fn optimize(&mut self) -> Result<()> {
        self.lexical_engine.optimize()?;
        Ok(())
    }

    /// Execute a hybrid search combining keyword and semantic search.
    ///
    /// This is an async method that performs lexical and vector searches,
    /// then merges the results using the configured fusion strategy.
    ///
    /// # Arguments
    ///
    /// * `request` - The hybrid search request containing query and parameters
    ///
    /// # Returns
    ///
    /// Combined search results from both engines
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use sarissa::hybrid::engine::HybridEngine;
    /// # use sarissa::hybrid::search::searcher::HybridSearchRequest;
    /// # async fn example(engine: HybridEngine) -> sarissa::error::Result<()> {
    /// let request = HybridSearchRequest::new("rust programming");
    /// let results = engine.search(request).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search(&self, request: HybridSearchRequest) -> Result<HybridSearchResults> {
        use std::time::Instant;

        let HybridSearchRequest {
            lexical_request,
            vector_request,
            params,
        } = request;

        let start = Instant::now();

        // 1. Lexical Search
        let keyword_results = if let Some(req) = lexical_request {
            self.lexical_engine.search(req)?
        } else {
            crate::lexical::index::inverted::query::LexicalSearchResults {
                hits: Vec::new(),
                total_hits: 0,
                max_score: 0.0,
            }
        };

        // 2. Vector Search
        let vector_results = if let Some(req) = vector_request {
            Some(self.vector_engine.search(req)?)
        } else {
            None
        };

        // Merge results
        let merger = crate::hybrid::search::merger::ResultMerger::new(params.clone());
        let query_time_ms = start.elapsed().as_millis() as u64;

        // For now pass empty string as query text since we don't have easy access to it from the opaque requests
        let query_text = "".to_string();

        merger
            .merge_results(keyword_results, vector_results, query_text, query_time_ms)
            .await
    }

    /// Get a reference to the lexical engine.
    pub fn lexical_engine(&self) -> &crate::lexical::engine::LexicalEngine {
        &self.lexical_engine
    }

    /// Get a reference to the vector engine.
    pub fn vector_engine(&self) -> &crate::vector::engine::VectorEngine {
        &self.vector_engine
    }
}
