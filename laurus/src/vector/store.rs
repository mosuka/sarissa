//! VectorStore: Simplified vector storage following LexicalStore pattern.
//!
//! This module provides a vector storage component with a simple 3-member structure:
//! - `index`: The underlying vector index
//! - `writer_cache`: Cached writer for write operations (`tokio::sync::Mutex`)
//! - `searcher_cache`: Cached searcher for search operations (`parking_lot::RwLock`)
//!
//! # Concurrency Strategy
//!
//! - **Searcher cache** uses double-checked locking with `RwLockWriteGuard::downgrade()`
//!   so that only searcher *creation* (on cache miss) holds an exclusive lock; the actual
//!   search executes under a shared read lock, allowing concurrent queries.
//! - **Writer cache** is protected by a `tokio::sync::Mutex`. Embedding (potentially slow
//!   network I/O) is performed *outside* the lock; only the final `delete + add_vectors`
//!   step runs while the lock is held, keeping the critical section short.
//!
//! # Module Structure
//!
//! - [`config`] - Configuration types (VectorIndexConfig, VectorFieldConfig)
//! - [`embedding_writer`] - Embedding writer wrapper
//! - [`request`] - Search request types
//! - [`response`] - Search response types

pub mod config;
pub mod embedding_writer;
pub mod memory;
pub mod request;
pub mod response;

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::data::{DataValue, Document};
use crate::embedding::embedder::{EmbedInput, Embedder};
use crate::embedding::per_field::PerFieldEmbedder;
use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use crate::vector::index::VectorIndex;
use crate::vector::index::config::VectorIndexTypeConfig;
use crate::vector::index::factory::VectorIndexFactory;
use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
use crate::vector::writer::VectorIndexWriter;

use self::config::VectorIndexConfig;
use self::request::{VectorScoreMode, VectorSearchRequest};
use self::response::{VectorHit, VectorSearchResults, VectorStats};

/// A simplified vector storage component following the LexicalStore pattern.
///
/// This structure mirrors `LexicalStore` with only 3 members:
/// - `index`: The underlying vector index
/// - `writer_cache`: Cached writer for write operations
/// - `searcher_cache`: Cached searcher for search operations
pub struct VectorStore {
    /// The underlying vector index.
    index: Box<dyn VectorIndex>,
    /// Cached writer (created on-demand).
    writer_cache: Mutex<Option<Box<dyn VectorIndexWriter>>>,
    /// Cached searcher (invalidated after commit/optimize).
    searcher_cache: parking_lot::RwLock<Option<Box<dyn VectorIndexSearcher>>>,
}

impl std::fmt::Debug for VectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStore")
            .field("index", &self.index)
            .finish()
    }
}

impl VectorStore {
    /// Create a new vector store with the given storage and high-level configuration.
    ///
    /// This constructor is compatible with Engine and accepts VectorIndexConfig.
    /// It extracts the index type configuration from the first field.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend for persisting index data
    /// * `config` - High-level configuration (compatible with Engine)
    ///
    /// # Returns
    ///
    /// Returns a new `VectorStore` instance.
    pub fn new(storage: Arc<dyn Storage>, config: VectorIndexConfig) -> Result<Self> {
        // Extract index type config from the first field, or use default
        let index_type_config = Self::extract_index_type_config(&config);
        Self::with_index_type_config(storage, index_type_config)
    }

    /// Create a new vector store with explicit index type configuration.
    ///
    /// This is a lower-level constructor for when you have a specific
    /// VectorIndexTypeConfig.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend for persisting index data
    /// * `config` - Configuration for the vector index (Flat, HNSW, or IVF)
    ///
    /// # Returns
    ///
    /// Returns a new `VectorStore` instance.
    pub fn with_index_type_config(
        storage: Arc<dyn Storage>,
        config: VectorIndexTypeConfig,
    ) -> Result<Self> {
        let index = VectorIndexFactory::open_or_create(storage, "vector_index", config)?;
        Ok(Self {
            index,
            writer_cache: Mutex::new(None),
            searcher_cache: parking_lot::RwLock::new(None),
        })
    }

    /// Extract VectorIndexTypeConfig from VectorIndexConfig.
    ///
    /// Uses the first field's configuration if available, otherwise returns default.
    fn extract_index_type_config(config: &VectorIndexConfig) -> VectorIndexTypeConfig {
        use crate::vector::core::field::FieldOption;
        use crate::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};

        // Try to get config from the first field with vector configuration
        for field_config in config.fields.values() {
            if let Some(ref vector_opt) = field_config.vector {
                return match vector_opt {
                    FieldOption::Flat(opt) => VectorIndexTypeConfig::Flat(FlatIndexConfig {
                        dimension: opt.dimension,
                        distance_metric: opt.distance,
                        embedder: config.embedder.clone(),
                        ..Default::default()
                    }),
                    FieldOption::Hnsw(opt) => VectorIndexTypeConfig::HNSW(HnswIndexConfig {
                        dimension: opt.dimension,
                        distance_metric: opt.distance,
                        m: opt.m,
                        ef_construction: opt.ef_construction,
                        default_ef_search: opt.default_ef_search,
                        embedder: config.embedder.clone(),
                        ..Default::default()
                    }),
                    FieldOption::Ivf(opt) => VectorIndexTypeConfig::IVF(IvfIndexConfig {
                        dimension: opt.dimension,
                        distance_metric: opt.distance,
                        n_clusters: opt.n_clusters,
                        n_probe: opt.n_probe,
                        embedder: config.embedder.clone(),
                        ..Default::default()
                    }),
                };
            }
        }

        // Default to HNSW with config's embedder
        VectorIndexTypeConfig::HNSW(HnswIndexConfig {
            embedder: config.embedder.clone(),
            ..Default::default()
        })
    }

    /// Upsert a document by its internal ID.
    ///
    /// This method first deletes any existing vectors for the given `doc_id`,
    /// then iterates over all fields in the document and passes each field value
    /// to the writer's [`add_value()`](crate::vector::writer::VectorIndexWriter::add_value)
    /// method, which handles embedding automatically when the writer is wrapped
    /// in an `EmbeddingVectorIndexWriter`.
    ///
    /// It is primarily used during WAL recovery where the internal ID
    /// is already known.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID.
    /// * `doc` - The document whose fields will be indexed as vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining/creating the writer fails, if deleting the
    /// existing document fails, or if adding any field value fails.
    pub async fn upsert_document_by_internal_id(&self, doc_id: u64, doc: Document) -> Result<()> {
        // Phase 1: Embed all fields OUTSIDE the lock.
        // This allows multiple concurrent upserts to perform embedding in parallel
        // rather than being serialized by the writer Mutex.
        let embedder = self.index.embedder();
        let mut embedded_vectors: Vec<(u64, String, Vector)> = Vec::new();

        for (field_name, value) in &doc.fields {
            let vector = match value {
                DataValue::Vector(v) => Vector::new(v.clone()),
                DataValue::Text(_) | DataValue::Bytes(_, _) => {
                    Self::embed_value(&*embedder, field_name, value).await?
                }
                _ => continue,
            };
            embedded_vectors.push((doc_id, field_name.clone(), vector));
        }

        // Phase 2: Acquire lock and write pre-computed vectors (fast, sync-only).
        let mut guard = self.writer_cache.lock().await;
        if guard.is_none() {
            *guard = Some(self.index.writer()?);
        }
        let writer = guard.as_mut().unwrap();
        writer.delete_document(doc_id)?;
        writer.add_vectors(embedded_vectors)?;

        Ok(())
    }

    /// Validate input and embed a single field value into a vector.
    ///
    /// This is a helper extracted from `EmbeddingVectorIndexWriter::add_value()`
    /// to allow embedding to happen outside the writer lock.
    ///
    /// # Arguments
    ///
    /// * `embedder` - The embedder to use for converting content to vectors.
    /// * `field_name` - The name of the field being embedded.
    /// * `value` - The data value to embed (must be `Text` or `Bytes`).
    ///
    /// # Errors
    ///
    /// Returns an error if the embedder does not support the input type or if
    /// the embedding operation fails.
    async fn embed_value(
        embedder: &dyn Embedder,
        field_name: &str,
        value: &DataValue,
    ) -> Result<Vector> {
        // Validate input type compatibility
        match value {
            DataValue::Text(_) if !embedder.supports_text() => {
                return Err(LaurusError::invalid_argument(format!(
                    "Embedder '{}' does not support text input",
                    embedder.name()
                )));
            }
            DataValue::Bytes(_, mime)
                if !embedder.supports_image()
                    && mime.as_ref().is_some_and(|m| m.starts_with("image/")) =>
            {
                return Err(LaurusError::invalid_argument(format!(
                    "Embedder '{}' does not support image input",
                    embedder.name()
                )));
            }
            _ => {}
        }

        // Prepare owned data for the embed call
        let (text_owned, bytes_owned, mime_owned) = match value {
            DataValue::Text(t) => (Some(t.clone()), None, None),
            DataValue::Bytes(b, m) => (None, Some(b.clone()), m.clone()),
            _ => {
                return Err(LaurusError::invalid_argument(
                    "Unsupported data type for embedding",
                ));
            }
        };

        let input = if let Some(ref text) = text_owned {
            EmbedInput::Text(text)
        } else if let Some(ref bytes) = bytes_owned {
            EmbedInput::Bytes(bytes, mime_owned.as_deref())
        } else {
            return Err(LaurusError::internal("Unreachable state in embed_value"));
        };

        // Use field-specific embedder if PerFieldEmbedder, otherwise default.
        if let Some(per_field) = embedder.as_any().downcast_ref::<PerFieldEmbedder>() {
            per_field.embed_field(field_name, &input).await
        } else {
            embedder.embed(&input).await
        }
    }

    /// Delete a document by its internal ID.
    ///
    /// Obtains (or creates) the cached writer and removes all vectors
    /// associated with the given `doc_id` from the index buffer.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID to delete.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining/creating the writer fails or if the
    /// underlying delete operation fails.
    pub async fn delete_document_by_internal_id(&self, doc_id: u64) -> Result<()> {
        let mut guard = self.writer_cache.lock().await;
        if guard.is_none() {
            *guard = Some(self.index.writer()?);
        }
        let writer = guard.as_mut().unwrap();

        writer.delete_document(doc_id)?;

        Ok(())
    }

    /// Commit any pending changes to the index.
    ///
    /// If a cached writer exists, this method takes it and calls
    /// [`commit()`](crate::vector::writer::VectorIndexWriter::commit) (which
    /// finalizes the index and writes it to storage). It then syncs the
    /// underlying storage to ensure all file metadata is flushed to disk,
    /// refreshes the index metadata, and invalidates the searcher cache so
    /// that subsequent searches see the committed data.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer commit, storage sync, or index refresh
    /// fails.
    pub async fn commit(&self) -> Result<()> {
        if let Some(mut writer) = self.writer_cache.lock().await.take() {
            // commit() calls finalize() then write() to persist to storage
            writer.commit()?;
        }
        // Sync storage to ensure all file metadata (creation, rename, size) is
        // flushed to disk. This is critical on Windows where directory listings
        // and file visibility may be cached until the directory is synced.
        self.index.storage().sync()?;
        self.index.refresh()?;
        *self.searcher_cache.write() = None;
        Ok(())
    }

    /// Optimize the index for improved query performance.
    ///
    /// Delegates to the underlying [`VectorIndex::optimize()`] implementation
    /// and then invalidates the searcher cache so the next search creates a
    /// fresh searcher reflecting the optimized state.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying index optimization fails.
    pub fn optimize(&self) -> Result<()> {
        self.index.optimize()?;
        *self.searcher_cache.write() = None;
        Ok(())
    }

    /// Invalidate the searcher cache.
    ///
    /// Clears the cached searcher so that the next search operation creates a
    /// new one. This is useful after external writes that bypass this store's
    /// commit path.
    pub fn refresh(&self) -> Result<()> {
        *self.searcher_cache.write() = None;
        Ok(())
    }

    /// Acquire a read lock on the cached searcher, populating the cache on miss.
    ///
    /// Uses double-checked locking: first tries a shared read lock (fast path),
    /// then falls back to an exclusive write lock to create the searcher and
    /// atomically downgrades it to a read lock so concurrent searches are not
    /// blocked while the actual query executes.
    fn acquire_searcher_guard(
        &self,
    ) -> Result<parking_lot::RwLockReadGuard<'_, Option<Box<dyn VectorIndexSearcher>>>> {
        // Fast path: cache hit under read lock.
        {
            let guard = self.searcher_cache.read();
            if guard.is_some() {
                return Ok(guard);
            }
        }

        // Slow path: populate under write lock, then downgrade.
        let mut guard = self.searcher_cache.write();
        if guard.is_none() {
            *guard = Some(self.index.searcher()?);
        }
        Ok(parking_lot::RwLockWriteGuard::downgrade(guard))
    }

    /// Execute a low-level vector similarity search.
    pub fn search_index(
        &self,
        request: &VectorIndexQuery,
    ) -> Result<crate::vector::search::searcher::VectorIndexQueryResults> {
        let guard = self.acquire_searcher_guard()?;
        guard.as_ref().unwrap().search(request)
    }

    /// Execute a high-level vector search (compatible with Engine).
    ///
    /// This method extracts query vectors from the
    /// [`VectorSearchQuery`](crate::vector::search::searcher::VectorSearchQuery)
    /// inside the request, performs a similarity search against the index, and
    /// aggregates the per-vector scores according to the requested
    /// [`score_mode`](crate::vector::search::searcher::VectorSearchParams::score_mode).
    /// Results are filtered by
    /// [`allowed_ids`](crate::vector::search::searcher::VectorSearchParams::allowed_ids)
    /// and
    /// [`min_score`](crate::vector::search::searcher::VectorSearchParams::min_score),
    /// sorted by descending score, and truncated to
    /// [`limit`](crate::vector::search::searcher::VectorSearchParams::limit).
    ///
    /// **Note:** The following request fields are currently **ignored** by this
    /// implementation:
    /// - `VectorSearchQuery::Payloads` -- callers must embed payloads into
    ///   vectors before calling this method.
    /// - [`fields`](crate::vector::search::searcher::VectorSearchParams::fields)
    ///   -- field-level filtering is not yet implemented; all indexed vectors
    ///   are searched.
    /// - [`overfetch`](crate::vector::search::searcher::VectorSearchParams::overfetch)
    ///   -- a hardcoded 2x overfetch (`limit * 2`) is used instead.
    ///
    /// # Arguments
    ///
    /// * `request` - The search request containing query vectors, filters, and
    ///   scoring options.
    ///
    /// # Returns
    ///
    /// A [`VectorSearchResults`] containing hits sorted by descending score.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining the searcher or executing the underlying
    /// index search fails, or if the query contains unresolved payloads.
    pub fn search(&self, request: VectorSearchRequest) -> Result<VectorSearchResults> {
        self.search_impl(request, None)
    }

    /// Test-only variant of [`Self::search`] that lets the caller pin the
    /// multi-vector parallelisation threshold.
    ///
    /// When `parallel_threshold == 0` the multi-vector path always runs in
    /// parallel (when the `native` feature is on); when it is `usize::MAX`
    /// the path always runs serially. Production code goes through
    /// [`Self::search`], which uses the searcher's
    /// [`VectorIndexSearcher::parallel_threshold`] (default `4`).
    ///
    /// Issue [#710](https://github.com/mosuka/laurus/issues/710) Phase 1 of
    /// [#648](https://github.com/mosuka/laurus/issues/648); refactored in
    /// Phase 2 ([#712](https://github.com/mosuka/laurus/issues/712)) to
    /// dispatch through the trait method.
    #[doc(hidden)]
    pub fn search_with_threshold(
        &self,
        request: VectorSearchRequest,
        parallel_threshold: usize,
    ) -> Result<VectorSearchResults> {
        self.search_impl(request, Some(parallel_threshold))
    }

    /// Common implementation for [`Self::search`] and
    /// [`Self::search_with_threshold`].
    ///
    /// `threshold_override == None` uses the searcher's own
    /// [`VectorIndexSearcher::parallel_threshold`]; `Some(t)` pins the
    /// threshold for tests.
    fn search_impl(
        &self,
        request: VectorSearchRequest,
        threshold_override: Option<usize>,
    ) -> Result<VectorSearchResults> {
        use crate::vector::search::searcher::VectorSearchQuery;

        let query_vectors = match &request.query {
            VectorSearchQuery::Vectors(vecs) => vecs,
            VectorSearchQuery::Payloads(_) => {
                return Err(crate::error::LaurusError::invalid_argument(
                    "VectorStore::search requires pre-embedded vectors; \
                     Payloads must be embedded before calling this method",
                ));
            }
        };

        if query_vectors.is_empty() {
            return Ok(VectorSearchResults::default());
        }

        let searcher_guard = self.acquire_searcher_guard()?;
        let searcher = searcher_guard.as_ref().unwrap();

        // Fast path: single query vector, skip HashMap aggregation.
        if query_vectors.len() == 1 {
            let qv = &query_vectors[0];
            let mut index_request = VectorIndexQuery::new(qv.vector.clone())
                .top_k(request.params.limit.saturating_mul(2));
            if let Some(factor) = request.params.rerank_factor {
                index_request = index_request.rerank_factor(factor);
            }
            if let Some(ef) = request.params.ef_search {
                index_request = index_request.ef_search(ef);
            }
            let results = searcher.search(&index_request)?;

            let mut hits: Vec<VectorHit> = results
                .results
                .into_iter()
                .filter(|r| {
                    if let Some(ref allowed) = request.params.allowed_ids
                        && !allowed.contains(&r.doc_id)
                    {
                        return false;
                    }
                    r.similarity >= request.params.min_score
                })
                .map(|r| VectorHit {
                    doc_id: r.doc_id,
                    score: r.similarity * qv.weight,
                    field_hits: vec![],
                })
                .collect();

            // Use partial sort for top-K selection when the result set is larger
            // than the requested limit.
            let limit = request.params.limit.min(hits.len());
            if limit > 0 && limit < hits.len() {
                hits.select_nth_unstable_by(limit - 1, |a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                hits.truncate(limit);
                hits.sort_unstable_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            } else if !hits.is_empty() {
                hits.sort_unstable_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            return Ok(VectorSearchResults { hits });
        }

        // Phase 2 of #648 (issue #712): build all B index queries, dispatch
        // via the `VectorIndexSearcher::search_batch_with_threshold` trait
        // method, then merge serially. Parallelisation across the B queries
        // happens inside the trait method's default impl (or any override).
        let index_queries: Vec<VectorIndexQuery> = query_vectors
            .iter()
            .map(|qv| {
                let mut q = VectorIndexQuery::new(qv.vector.clone())
                    .top_k(request.params.limit.saturating_mul(2));
                if let Some(factor) = request.params.rerank_factor {
                    q = q.rerank_factor(factor);
                }
                if let Some(ef) = request.params.ef_search {
                    q = q.ef_search(ef);
                }
                q
            })
            .collect();
        let per_query_results = match threshold_override {
            Some(t) => searcher.search_batch_with_threshold(&index_queries, t)?,
            None => searcher.search_batch(&index_queries)?,
        };

        // Serial merge by score_mode (applies allowed_ids / min_score filter
        // and the per-query weight that the trait method intentionally does
        // not know about).
        let mut all_hits: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();
        for (qv, results) in query_vectors.iter().zip(per_query_results) {
            for result in results.results {
                if let Some(ref allowed) = request.params.allowed_ids
                    && !allowed.contains(&result.doc_id)
                {
                    continue;
                }
                if result.similarity < request.params.min_score {
                    continue;
                }
                let weighted_score = result.similarity * qv.weight;
                let entry = all_hits.entry(result.doc_id).or_insert(0.0);
                match request.params.score_mode {
                    VectorScoreMode::WeightedSum | VectorScoreMode::LateInteraction => {
                        // WeightedSum: sum of similarity * weight across all query vectors.
                        // LateInteraction: for each query vector, find the max similarity
                        // across document vectors, then sum. In the current single-vector-
                        // per-field architecture, this is equivalent to WeightedSum since
                        // each query vector already gets a single best match per document.
                        *entry += weighted_score;
                    }
                    VectorScoreMode::MaxSim => {
                        // MaxSim: take the maximum weighted similarity across query vectors.
                        if weighted_score > *entry {
                            *entry = weighted_score;
                        }
                    }
                }
            }
        }

        // Convert to VectorHit and sort by score with doc_id tiebreak for
        // parallel-deterministic ordering (issue #710 Phase 1 of #648).
        let mut hits: Vec<VectorHit> = all_hits
            .into_iter()
            .map(|(doc_id, score)| VectorHit {
                doc_id,
                score,
                field_hits: vec![],
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        // Apply limit
        if hits.len() > request.params.limit {
            hits.truncate(request.params.limit);
        }

        Ok(VectorSearchResults { hits })
    }

    /// Count the number of vectors matching the given search request.
    ///
    /// Delegates to the searcher's
    /// [`count()`](crate::vector::search::searcher::VectorIndexSearcher::count)
    /// method, which returns the total number of vectors that match the query
    /// criteria.
    ///
    /// # Arguments
    ///
    /// * `request` - A low-level vector index search request specifying the
    ///   query vector and parameters.
    ///
    /// # Returns
    ///
    /// The number of matching vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining the searcher or executing the count fails.
    pub fn count(&self, request: VectorIndexQuery) -> Result<u64> {
        let guard = self.acquire_searcher_guard()?;
        guard.as_ref().unwrap().count(request)
    }

    /// Get index statistics including per-field vector counts.
    ///
    /// Returns a [`VectorStats`] containing the total document count and
    /// per-field statistics (vector count and dimension) for each vector
    /// field in the index. The dimension is derived from the actual vectors
    /// stored for each field, falling back to the index-level dimension when
    /// no vectors are present.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining the reader fails.
    pub fn stats(&self) -> Result<VectorStats> {
        let reader = self.index.reader()?;
        let doc_count = reader.vector_count();
        let index_dimension = reader.dimension();

        let mut fields = std::collections::HashMap::new();
        if let Ok(field_names) = reader.field_names() {
            for name in field_names {
                let vectors = reader.get_vectors_by_field(&name).unwrap_or_default();
                let vector_count = vectors.len();
                // Derive dimension from actual vectors; fall back to
                // index-level dimension when no vectors exist for this field.
                let dimension = vectors
                    .first()
                    .map(|(_, v)| v.data.len())
                    .unwrap_or(index_dimension);
                fields.insert(
                    name,
                    crate::vector::index::field::VectorFieldStats {
                        vector_count,
                        dimension,
                    },
                );
            }
        }

        Ok(VectorStats {
            document_count: doc_count,
            fields,
        })
    }

    /// Get the storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        self.index.storage()
    }

    /// Close the store.
    pub async fn close(&self) -> Result<()> {
        *self.writer_cache.lock().await = None;
        *self.searcher_cache.write() = None;
        self.index.close()
    }

    /// Check if the store is closed.
    pub fn is_closed(&self) -> bool {
        self.index.is_closed()
    }

    /// Get the embedder.
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        self.index.embedder()
    }

    /// Get the last processed WAL sequence number.
    pub fn last_wal_seq(&self) -> u64 {
        self.index.last_wal_seq()
    }

    /// Set the last processed WAL sequence number.
    ///
    /// Note: This method doesn't return Result for Engine compatibility.
    /// Errors are silently ignored.
    pub fn set_last_wal_seq(&self, seq: u64) {
        let _ = self.index.set_last_wal_seq(seq);
    }

    /// Register a field-specific embedder for a dynamically added vector field.
    ///
    /// If the underlying index's embedder is a
    /// [`PerFieldEmbedder`](crate::embedding::per_field::PerFieldEmbedder),
    /// this method registers the given embedder for the specified field.
    /// The writer and searcher caches are invalidated afterwards.
    ///
    /// # Arguments
    ///
    /// * `name` - The vector field name
    /// * `embedder` - Optional field-specific embedder to register
    pub async fn add_field(
        &self,
        name: &str,
        embedder: Option<Arc<dyn crate::embedding::embedder::Embedder>>,
    ) {
        if let Some(field_embedder) = embedder {
            let index_embedder = self.index.embedder();
            if let Some(pfe) = index_embedder
                .as_any()
                .downcast_ref::<crate::embedding::per_field::PerFieldEmbedder>()
            {
                pfe.add_embedder(name, field_embedder);
            }
        }

        // Invalidate caches so the next writer/searcher uses updated config.
        *self.writer_cache.lock().await = None;
        *self.searcher_cache.write() = None;
    }

    /// Remove a field from the vector store.
    ///
    /// Unregisters any field-specific embedder from the `PerFieldEmbedder` and
    /// invalidates writer/searcher caches. Existing vector data in the index is
    /// not deleted.
    ///
    /// # Arguments
    ///
    /// * `name` - The vector field name to remove
    pub async fn delete_field(&self, name: &str) {
        // Remove the field-specific embedder from the PerFieldEmbedder if present.
        let index_embedder = self.index.embedder();
        if let Some(pfe) = index_embedder
            .as_any()
            .downcast_ref::<crate::embedding::per_field::PerFieldEmbedder>()
        {
            pfe.remove_embedder(name);
        }

        // Invalidate caches so the next writer/searcher uses updated config.
        *self.writer_cache.lock().await = None;
        *self.searcher_cache.write() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    #[test]
    fn test_vectorstore_creation() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let config = VectorIndexTypeConfig::default();
        let store = VectorStore::with_index_type_config(storage, config).unwrap();

        assert!(!store.is_closed());
    }

    #[tokio::test]
    async fn test_vectorstore_close() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let config = VectorIndexTypeConfig::default();
        let store = VectorStore::with_index_type_config(storage, config).unwrap();

        assert!(!store.is_closed());
        store.close().await.unwrap();
        assert!(store.is_closed());
    }
}
