//! High-level unified vector engine that combines indexing and searching.
//!
//! This module provides a unified interface for vector indexing and search,
//! similar to the lexical SearchEngine.

use std::cell::{RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::error::{PlatypusError, Result};
use crate::storage::Storage;
use crate::vector::collection::{
    FieldSelector, VectorCollectionHit, VectorCollectionQuery, VectorCollectionSearchResults,
    VectorScoreMode,
};
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::document::{
    METADATA_EMBEDDER_ID, METADATA_ROLE, METADATA_WEIGHT, VectorRole,
};
use crate::vector::core::vector::Vector;
use crate::vector::field::FieldHit;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorSearcher;
use crate::vector::search::searcher::{VectorSearchRequest, VectorSearchResults};

/// A high-level unified vector engine that provides both indexing and searching capabilities.
/// This is similar to the lexical SearchEngine but for vector search.
///
/// # Example
///
/// ```no_run
/// use platypus::vector::engine::VectorEngine;
/// use platypus::vector::collection::{QueryVector, VectorCollectionQuery};
/// use platypus::vector::core::distance::DistanceMetric;
/// use platypus::vector::core::document::StoredVector;
/// use platypus::vector::core::vector::Vector;
/// use platypus::vector::index::config::{VectorIndexConfig, FlatIndexConfig};
/// use platypus::vector::index::factory::VectorIndexFactory;
/// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
/// use platypus::document::document::Document;
/// use platypus::embedding::text_embedder::TextEmbedder;
/// use std::sync::Arc;
///
/// # async fn example(embedder: Arc<dyn TextEmbedder>) -> platypus::error::Result<()> {
/// // Create engine with flat index
/// let config = VectorIndexConfig::Flat(FlatIndexConfig {
///     dimension: 3,
///     distance_metric: DistanceMetric::Cosine,
///     embedder: embedder.clone(),
///     ..Default::default()
/// });
/// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
/// let index = VectorIndexFactory::create(storage, config)?;
/// let mut engine = VectorEngine::new(index)?;
///
/// // Add documents with vector fields
/// use platypus::document::field::VectorOption;
/// let doc1 = Document::builder()
///     .add_vector("embedding", "Machine Learning", VectorOption::default())
///     .build();
/// let doc2 = Document::builder()
///     .add_vector("embedding", "Deep Learning", VectorOption::default())
///     .build();
///
/// engine.add_document(doc1).await?;
/// engine.add_document(doc2).await?;
/// engine.commit()?;
///
/// // Search
/// let query_vector = Vector::new(vec![1.0, 0.1, 0.0]);
/// let mut query = VectorCollectionQuery::default();
/// query.limit = 2;
/// query.query_vectors.push(QueryVector {
///     vector: StoredVector::from(query_vector),
///     weight: 1.0,
/// });
/// let results = engine.search(query)?;
/// # Ok(())
/// # }
/// ```
pub struct VectorEngine {
    /// The underlying index.
    index: Box<dyn VectorIndex>,
    /// The reader for executing queries (cached for efficiency).
    reader: RefCell<Option<Arc<dyn crate::vector::reader::VectorIndexReader>>>,
    /// The writer for adding/updating vectors (cached for efficiency).
    writer: RefCell<Option<Box<dyn crate::vector::writer::VectorIndexWriter>>>,
    /// The searcher for executing searches (cached for efficiency).
    searcher: RefCell<Option<Box<dyn VectorSearcher>>>,
}

impl VectorEngine {
    /// Create a new vector engine with the given vector index.
    ///
    /// This constructor wraps a `VectorIndex` and initializes empty caches for
    /// the reader and writer. The reader and writer will be created on-demand
    /// when needed.
    ///
    /// # Arguments
    ///
    /// * `index` - A vector index trait object (contains configuration and storage)
    ///
    /// # Returns
    ///
    /// Returns a new `VectorEngine` instance.
    ///
    /// # Example with Memory Storage
    ///
    /// ```rust,no_run
    /// use platypus::vector::engine::VectorEngine;
    /// use platypus::vector::index::config::VectorIndexConfig;
    /// use platypus::vector::index::factory::VectorIndexFactory;
    /// use platypus::storage::{StorageConfig, StorageFactory};
    /// use platypus::storage::memory::MemoryStorageConfig;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// let storage = StorageFactory::create(storage_config)?;
    /// let index = VectorIndexFactory::create(storage, VectorIndexConfig::default())?;
    /// let engine = VectorEngine::new(index)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Example with File Storage
    ///
    /// ```rust,no_run
    /// use platypus::vector::engine::VectorEngine;
    /// use platypus::vector::index::config::VectorIndexConfig;
    /// use platypus::vector::index::factory::VectorIndexFactory;
    /// use platypus::storage::{StorageConfig, StorageFactory};
    /// use platypus::storage::file::FileStorageConfig;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let storage_config = StorageConfig::File(FileStorageConfig::new("/tmp/vector_index"));
    /// let storage = StorageFactory::create(storage_config)?;
    /// let index = VectorIndexFactory::create(storage, VectorIndexConfig::default())?;
    /// let engine = VectorEngine::new(index)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(index: Box<dyn VectorIndex>) -> Result<Self> {
        Ok(Self {
            index,
            reader: RefCell::new(None),
            writer: RefCell::new(None),
            searcher: RefCell::new(None),
        })
    }

    /// Get or create a reader for this engine.
    fn get_or_create_reader(&self) -> Result<std::cell::Ref<'_, Arc<dyn VectorIndexReader>>> {
        {
            let mut reader_ref = self.reader.borrow_mut();
            if reader_ref.is_none() {
                *reader_ref = Some(self.index.reader()?);
            }
        }

        // Return a reference to the reader
        Ok(std::cell::Ref::map(self.reader.borrow(), |opt| {
            opt.as_ref().unwrap()
        }))
    }

    /// Get or create a writer for this engine.
    fn get_or_create_writer(
        &self,
    ) -> Result<RefMut<'_, Box<dyn crate::vector::writer::VectorIndexWriter>>> {
        {
            let mut writer_ref = self.writer.borrow_mut();
            if writer_ref.is_none() {
                *writer_ref = Some(self.index.writer()?);
            }
        }

        // Return a mutable reference to the writer
        Ok(RefMut::map(self.writer.borrow_mut(), |opt| {
            opt.as_mut().unwrap()
        }))
    }

    /// Refresh the reader to see latest changes.
    pub fn refresh(&mut self) -> Result<()> {
        *self.reader.borrow_mut() = None;
        *self.searcher.borrow_mut() = None;
        Ok(())
    }

    /// Get or create a searcher for this engine.
    ///
    /// The searcher is created by the underlying index implementation.
    fn get_or_create_searcher(&self) -> Result<RefMut<'_, Box<dyn VectorSearcher>>> {
        {
            let mut searcher_ref = self.searcher.borrow_mut();
            if searcher_ref.is_none() {
                *searcher_ref = Some(self.index.searcher()?);
            }
        }

        // Return a mutable reference to the searcher
        Ok(RefMut::map(self.searcher.borrow_mut(), |opt| {
            opt.as_mut().unwrap()
        }))
    }

    /// Add multiple vectors to the index (with user-specified IDs and field names).
    /// This is an internal helper method. Use `add_document()` instead.
    fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        let mut writer = self.get_or_create_writer()?;
        writer.add_vectors(vectors)?;
        Ok(())
    }

    /// Add a single vector with a specific ID and field name (internal use only).
    /// This is exposed for HybridEngine. Regular users should use `add_document()`.
    #[doc(hidden)]
    pub fn add_vector_with_id(
        &mut self,
        vec_id: u64,
        field_name: String,
        vector: Vector,
    ) -> Result<()> {
        self.add_vectors(vec![(vec_id, field_name, vector)])
    }

    /// Add a document to the index, converting vector fields to embeddings.
    ///
    /// This method processes all `FieldValue::Vector` fields in the document,
    /// converts their text to embeddings using the provided embedder, and adds
    /// the resulting vectors to the index.
    ///
    /// # Arguments
    ///
    /// * `document` - The document containing vector fields
    /// * `embedder` - The embedder to use for converting text to vectors
    ///
    /// # Returns
    ///
    /// Returns the assigned document ID.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use platypus::vector::engine::VectorEngine;
    /// use platypus::vector::index::config::VectorIndexConfig;
    /// use platypus::vector::index::factory::VectorIndexFactory;
    /// use platypus::document::document::Document;
    /// use platypus::embedding::text_embedder::TextEmbedder;
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # async fn example(embedder: Arc<dyn TextEmbedder>) -> platypus::error::Result<()> {
    /// let mut config = VectorIndexConfig::default();
    /// // Set embedder in config (assuming Flat variant)
    /// if let VectorIndexConfig::Flat(ref mut flat_config) = config {
    ///     flat_config.embedder = embedder;
    /// }
    /// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// let index = VectorIndexFactory::create(storage, config)?;
    /// let mut engine = VectorEngine::new(index)?;
    ///
    /// use platypus::document::field::{TextOption, VectorOption};
    /// let doc = Document::builder()
    ///     .add_text("title", "Machine Learning", TextOption::default())
    ///     .add_vector("title_embedding", "Machine Learning", VectorOption::default())
    ///     .build();
    ///
    /// let doc_id = engine.add_document(doc).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn add_document(
        &mut self,
        document: crate::document::document::Document,
    ) -> Result<u64> {
        let mut writer = self.get_or_create_writer()?;
        writer.add_document(document).await
    }

    /// Add a document with a specific ID, converting vector fields to embeddings.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to use
    /// * `document` - The document containing vector fields
    pub async fn add_document_with_id(
        &mut self,
        doc_id: u64,
        document: crate::document::document::Document,
    ) -> Result<()> {
        let mut writer = self.get_or_create_writer()?;
        writer.add_document_with_id(doc_id, document).await
    }

    /// Add multiple documents to the index.
    ///
    /// Returns a vector of assigned document IDs.
    /// Note: You must call `commit()` to persist the changes.
    pub async fn add_documents(
        &mut self,
        docs: Vec<crate::document::document::Document>,
    ) -> Result<Vec<u64>> {
        let mut doc_ids = Vec::new();
        for doc in docs {
            let doc_id = self.add_document(doc).await?;
            doc_ids.push(doc_id);
        }
        Ok(doc_ids)
    }

    /// Delete documents matching the given field and value.
    ///
    /// Returns the number of documents deleted.
    /// Note: You must call `commit()` to persist the changes.
    pub fn delete_documents(&mut self, field: &str, value: &str) -> Result<u64> {
        let mut writer = self.get_or_create_writer()?;
        writer.delete_documents(field, value)
    }

    /// Update a document (delete old, add new).
    ///
    /// This is a convenience method that deletes documents matching the field/value
    /// and adds the new document.
    /// Note: You must call `commit()` to persist the changes.
    pub async fn update_document(
        &mut self,
        field: &str,
        value: &str,
        doc: crate::document::document::Document,
    ) -> Result<()> {
        let mut writer = self.get_or_create_writer()?;
        writer.update_document(field, value, doc).await
    }

    /// Commit any pending changes to the index.
    ///
    /// This method finalizes the index and makes all changes visible to subsequent searches.
    /// The searcher cache is invalidated to ensure fresh data on the next search.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the commit fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use platypus::vector::engine::VectorEngine;
    /// use platypus::vector::index::config::VectorIndexConfig;
    /// use platypus::vector::index::factory::VectorIndexFactory;
    /// use platypus::document::document::Document;
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # async fn example() -> platypus::error::Result<()> {
    /// let config = VectorIndexConfig::default();
    /// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// let index = VectorIndexFactory::create(storage, config)?;
    /// let mut engine = VectorEngine::new(index)?;
    ///
    /// // Add documents with vector fields
    /// use platypus::document::field::VectorOption;
    /// let doc = Document::builder()
    ///     .add_vector("embedding", "Sample text", VectorOption::default())
    ///     .build();
    /// engine.add_document(doc).await?;
    ///
    /// // Commit changes
    /// engine.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn commit(&mut self) -> Result<()> {
        // Commit the writer if it exists (unified with lexical)
        if let Some(mut writer) = self.writer.borrow_mut().take() {
            writer.commit()?;
        }

        // Invalidate reader and searcher caches to reflect the new changes
        *self.reader.borrow_mut() = None;
        *self.searcher.borrow_mut() = None;

        Ok(())
    }

    /// Optimize the index.
    pub fn optimize(&mut self) -> Result<()> {
        self.index.optimize()?;
        // Invalidate reader and searcher caches
        *self.reader.borrow_mut() = None;
        *self.searcher.borrow_mut() = None;
        Ok(())
    }

    /// Get index statistics.
    ///
    /// Returns basic statistics about the vector index.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use platypus::vector::engine::VectorEngine;
    /// use platypus::vector::index::config::VectorIndexConfig;
    /// use platypus::vector::index::factory::VectorIndexFactory;
    /// use platypus::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> platypus::error::Result<()> {
    /// let config = VectorIndexConfig::default();
    /// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// let index = VectorIndexFactory::create(storage, config)?;
    /// let engine = VectorEngine::new(index)?;
    ///
    /// let stats = engine.stats()?;
    /// println!("Vector count: {}", stats.vector_count);
    /// # Ok(())
    /// # }
    /// ```
    pub fn stats(&self) -> Result<VectorIndexStats> {
        self.index.stats()
    }

    /// Get the storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        self.index.storage()
    }

    /// Search for similar vectors using the doc-centric query model.
    ///
    /// This method automatically selects the appropriate searcher implementation
    /// based on the underlying index type (Flat, HNSW, or IVF).
    /// The searcher is cached for efficiency.
    pub fn search(&self, query: VectorCollectionQuery) -> Result<VectorCollectionSearchResults> {
        Self::validate_query(&query)?;

        let field_selection = self.resolve_field_selection(&query)?;
        let field_limit = Self::scaled_field_limit(query.limit, query.overfetch);
        let search_runs = self.execute_legacy_searches(&query, &field_selection, field_limit)?;

        Ok(Self::map_legacy_results(
            &query,
            &field_selection,
            search_runs,
        ))
    }

    fn validate_query(query: &VectorCollectionQuery) -> Result<()> {
        if query.query_vectors.is_empty() {
            return Err(PlatypusError::invalid_argument(
                "VectorCollectionQuery requires at least one query vector",
            ));
        }

        if query.limit == 0 {
            return Err(PlatypusError::invalid_argument(
                "VectorCollectionQuery limit must be greater than zero",
            ));
        }

        if matches!(query.score_mode, VectorScoreMode::LateInteraction) {
            return Err(PlatypusError::invalid_argument(
                "VectorScoreMode::LateInteraction is not supported by VectorEngine",
            ));
        }

        Ok(())
    }

    fn resolve_field_selection(&self, query: &VectorCollectionQuery) -> Result<FieldSelection> {
        let reader_ref = self.get_or_create_reader()?;
        let available_fields = reader_ref.field_names()?;
        drop(reader_ref);

        let mut resolved_fields = Vec::new();
        let mut required_roles = Vec::new();
        let mut has_explicit_field_selector = false;

        if let Some(selectors) = &query.fields {
            if selectors.is_empty() {
                return Err(PlatypusError::invalid_argument(
                    "VectorCollectionQuery field selector list cannot be empty",
                ));
            }

            let available_lookup: HashSet<String> = available_fields.iter().cloned().collect();

            for selector in selectors {
                match selector {
                    FieldSelector::Exact(name) => {
                        has_explicit_field_selector = true;
                        if !available_lookup.is_empty() && !available_lookup.contains(name) {
                            return Err(PlatypusError::not_found(format!(
                                "vector field '{name}' is not available in the legacy index"
                            )));
                        }
                        resolved_fields.push(name.clone());
                    }
                    FieldSelector::Prefix(prefix) => {
                        has_explicit_field_selector = true;
                        let mut matched = false;
                        for field in &available_fields {
                            if field.starts_with(prefix) {
                                resolved_fields.push(field.clone());
                                matched = true;
                            }
                        }
                        if !matched {
                            return Err(PlatypusError::not_found(format!(
                                "no vector fields match prefix '{prefix}'"
                            )));
                        }
                    }
                    FieldSelector::Role(role) => {
                        required_roles.push(role.clone());
                    }
                }
            }
        }

        let scope = if has_explicit_field_selector {
            let mut seen = HashSet::new();
            resolved_fields.retain(|field| seen.insert(field.clone()));
            FieldSearchScope::Named(resolved_fields)
        } else {
            FieldSearchScope::All
        };

        Ok(FieldSelection {
            scope,
            required_roles,
        })
    }

    fn scaled_field_limit(limit: usize, overfetch: f32) -> usize {
        if limit == 0 {
            return 0;
        }
        let factor = if overfetch.is_finite() && overfetch > 0.0 {
            overfetch
        } else {
            1.0
        };
        let scaled = (limit as f32 * factor).ceil() as usize;
        scaled.max(limit).max(1)
    }

    fn execute_legacy_searches(
        &self,
        query: &VectorCollectionQuery,
        selection: &FieldSelection,
        field_limit: usize,
    ) -> Result<Vec<LegacySearchRun>> {
        let searcher = self.get_or_create_searcher()?;
        let mut runs = Vec::new();

        for target in selection.targets() {
            for query_vector in &query.query_vectors {
                let mut request = VectorSearchRequest::new(query_vector.vector.to_vector());
                request.params.top_k = field_limit.max(1);
                if let Some(field_name) = target.clone() {
                    request.field_name = Some(field_name);
                }

                let results = searcher.search(&request)?;
                runs.push(LegacySearchRun {
                    query_weight: query_vector.weight * query_vector.vector.weight,
                    query_role: query_vector.vector.role.clone(),
                    query_embedder_id: query_vector.vector.embedder_id.clone(),
                    results,
                });
            }
        }

        Ok(runs)
    }

    fn map_legacy_results(
        query: &VectorCollectionQuery,
        selection: &FieldSelection,
        runs: Vec<LegacySearchRun>,
    ) -> VectorCollectionSearchResults {
        if runs.is_empty() {
            return VectorCollectionSearchResults::default();
        }

        let role_filters: HashSet<String> = selection
            .required_roles
            .iter()
            .map(|role| role.to_string())
            .collect();
        let field_filter = query
            .filter
            .as_ref()
            .map(|filter| filter.field.equals.clone());
        let document_filter = query
            .filter
            .as_ref()
            .map(|filter| filter.document.equals.clone())
            .filter(|filter| !filter.is_empty());

        let mut hits_by_doc: HashMap<u64, VectorCollectionHit> = HashMap::new();

        for run in runs {
            for result in run.results.results {
                let crate::vector::search::searcher::VectorSearchResult {
                    doc_id,
                    field_name,
                    similarity,
                    distance,
                    metadata,
                    ..
                } = result;

                if !role_filters.is_empty()
                    && !Self::metadata_role_matches(&role_filters, metadata.get(METADATA_ROLE))
                {
                    continue;
                }

                if !Self::role_matches_query(&run.query_role, metadata.get(METADATA_ROLE)) {
                    continue;
                }

                if !Self::embedder_matches(
                    &run.query_embedder_id,
                    metadata.get(METADATA_EMBEDDER_ID),
                ) {
                    continue;
                }

                if let Some(ref filter) = field_filter {
                    if !Self::metadata_matches(filter, &metadata) {
                        continue;
                    }
                }

                if let Some(ref filter) = document_filter {
                    if !Self::metadata_matches(filter, &metadata) {
                        continue;
                    }
                }

                let weighted_score =
                    similarity * run.query_weight * Self::metadata_weight(&metadata);
                if weighted_score == 0.0 {
                    continue;
                }

                let field_hit = FieldHit {
                    doc_id,
                    field: field_name,
                    score: weighted_score,
                    distance,
                    metadata,
                };

                let entry = hits_by_doc
                    .entry(doc_id)
                    .or_insert_with(|| VectorCollectionHit {
                        doc_id,
                        score: if matches!(query.score_mode, VectorScoreMode::MaxSim) {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        },
                        field_hits: Vec::new(),
                    });

                match query.score_mode {
                    VectorScoreMode::WeightedSum => {
                        entry.score += field_hit.score;
                    }
                    VectorScoreMode::MaxSim => {
                        entry.score = entry.score.max(field_hit.score);
                    }
                    VectorScoreMode::LateInteraction => unreachable!(),
                }

                entry.field_hits.push(field_hit);
            }
        }

        let mut hits: Vec<VectorCollectionHit> = hits_by_doc.into_values().collect();
        for hit in &mut hits {
            if hit.score == f32::NEG_INFINITY {
                hit.score = 0.0;
            }
        }
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        if hits.len() > query.limit {
            hits.truncate(query.limit);
        }

        VectorCollectionSearchResults { hits }
    }

    fn metadata_role_matches(required: &HashSet<String>, candidate: Option<&String>) -> bool {
        match candidate {
            Some(role) => required.contains(role),
            None => false,
        }
    }

    fn role_matches_query(query_role: &VectorRole, candidate: Option<&String>) -> bool {
        match query_role {
            VectorRole::Generic => true,
            _ => candidate
                .map(|role| role == &query_role.to_string())
                .unwrap_or(false),
        }
    }

    fn embedder_matches(query_embedder: &str, candidate: Option<&String>) -> bool {
        if query_embedder.is_empty() {
            return true;
        }

        match candidate {
            Some(embedder) => embedder == query_embedder,
            None => true,
        }
    }

    fn metadata_matches(
        expected: &HashMap<String, String>,
        metadata: &HashMap<String, String>,
    ) -> bool {
        expected.iter().all(|(key, value)| {
            metadata
                .get(key)
                .map(|candidate| candidate == value)
                .unwrap_or(false)
        })
    }

    fn metadata_weight(metadata: &HashMap<String, String>) -> f32 {
        metadata
            .get(METADATA_WEIGHT)
            .and_then(|raw| raw.parse::<f32>().ok())
            .filter(|value| value.is_normal() || *value > 0.0)
            .unwrap_or(1.0)
    }

    /// Count the number of vectors matching the request.
    ///
    /// This method counts vectors that match the field filter in the request.
    /// It's more efficient than searching when you only need the count.
    pub fn count(&self, query: VectorCollectionQuery) -> Result<u64> {
        let results = self.search(query)?;
        Ok(results.hits.len() as u64)
    }

    /// Get build progress (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        self.writer
            .borrow()
            .as_ref()
            .map(|w| w.progress())
            .unwrap_or(1.0)
    }

    /// Get estimated memory usage.
    pub fn estimated_memory_usage(&self) -> usize {
        self.writer
            .borrow()
            .as_ref()
            .map(|w| w.estimated_memory_usage())
            .unwrap_or(0)
    }

    /// Check if the index is finalized.
    pub fn is_finalized(&self) -> bool {
        // If writer is None, it means finalize() was already called
        self.writer.borrow().is_none()
    }

    /// Get the dimension.
    pub fn dimension(&self) -> Result<usize> {
        let reader = self.get_or_create_reader()?;
        Ok(reader.dimension())
    }

    /// Get the distance metric.
    pub fn distance_metric(&self) -> Result<DistanceMetric> {
        let reader = self.get_or_create_reader()?;
        Ok(reader.distance_metric())
    }

    /// Close the vector engine.
    ///
    /// This method releases all cached resources (reader and writer) and closes the
    /// underlying index. After calling this method, the engine should not be used.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the close operation fails.
    pub fn close(&mut self) -> Result<()> {
        // Drop the cached writer
        *self.writer.borrow_mut() = None;
        // Drop the cached reader
        *self.reader.borrow_mut() = None;
        // Drop the cached searcher
        *self.searcher.borrow_mut() = None;
        self.index.close()
    }

    /// Check if the engine is closed.
    ///
    /// # Returns
    ///
    /// Returns `true` if the engine has been closed, `false` otherwise.
    pub fn is_closed(&self) -> bool {
        self.index.is_closed()
    }
}

#[derive(Debug, Clone)]
struct LegacySearchRun {
    query_weight: f32,
    query_role: VectorRole,
    query_embedder_id: String,
    results: VectorSearchResults,
}

#[derive(Debug, Clone)]
struct FieldSelection {
    scope: FieldSearchScope,
    required_roles: Vec<VectorRole>,
}

#[derive(Debug, Clone)]
enum FieldSearchScope {
    All,
    Named(Vec<String>),
}

impl FieldSelection {
    fn targets(&self) -> Vec<Option<String>> {
        match &self.scope {
            FieldSearchScope::All => vec![None],
            FieldSearchScope::Named(fields) => fields.iter().cloned().map(Some).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::field::{TextOption, VectorOption};
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::collection::{
        MetadataFilter, QueryVector, VectorCollectionFilter, VectorCollectionQuery,
    };
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::core::document::{METADATA_WEIGHT, StoredVector};
    use crate::vector::core::vector::ORIGINAL_TEXT_METADATA_KEY;
    use crate::vector::index::config::{FlatIndexConfig, VectorIndexConfig};
    use crate::vector::index::factory::VectorIndexFactory;
    use crate::vector::search::searcher::{VectorSearchResult, VectorSearchResults};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_vector_engine_basic() -> Result<()> {
        use crate::document::document::Document;
        use crate::embedding::text_embedder::TextEmbedder;
        use async_trait::async_trait;

        // Mock embedder for testing
        #[derive(Debug)]
        struct MockEmbedder {
            dimension: usize,
        }

        #[async_trait]
        impl TextEmbedder for MockEmbedder {
            async fn embed(&self, text: &str) -> Result<Vector> {
                // Simple mock: convert text to a deterministic vector
                let bytes = text.as_bytes();
                let mut values = vec![0.0; self.dimension];
                for (i, &byte) in bytes.iter().enumerate() {
                    if i >= self.dimension {
                        break;
                    }
                    values[i] = (byte as f32) / 255.0;
                }
                Ok(Vector::new(values))
            }

            fn dimension(&self) -> usize {
                self.dimension
            }

            fn name(&self) -> &str {
                "mock"
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        let embedder: Arc<dyn TextEmbedder> = Arc::new(MockEmbedder { dimension: 3 });

        let config = VectorIndexConfig::Flat(FlatIndexConfig {
            dimension: 3,
            distance_metric: DistanceMetric::Cosine,
            embedder: embedder.clone(),
            ..Default::default()
        });
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = VectorIndexFactory::create(storage, config)?;
        let mut engine = VectorEngine::new(index)?;

        // Add documents with vector fields
        let doc1 = Document::builder()
            .add_vector("embedding", "Machine Learning", VectorOption::default())
            .build();
        let doc2 = Document::builder()
            .add_vector("embedding", "Deep Learning", VectorOption::default())
            .build();
        let doc3 = Document::builder()
            .add_vector("embedding", "Neural Network", VectorOption::default())
            .build();

        engine.add_document(doc1).await?;
        engine.add_document(doc2).await?;
        engine.add_document(doc3).await?;
        engine.commit()?;

        // Search for similar vectors
        let query_vector = Vector::new(vec![0.5, 0.5, 0.5]);
        let mut query = VectorCollectionQuery::default();
        query.limit = 2;
        query.query_vectors.push(QueryVector {
            vector: StoredVector::from(query_vector),
            weight: 1.0,
        });

        let results = engine.search(query)?;
        assert_eq!(results.hits.len(), 2);

        let expected_texts = ["Machine Learning", "Deep Learning", "Neural Network"];
        for hit in &results.hits {
            for field_hit in &hit.field_hits {
                let stored = field_hit
                    .metadata
                    .get(ORIGINAL_TEXT_METADATA_KEY)
                    .expect("stored text missing");
                assert!(expected_texts.contains(&stored.as_str()));
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_vector_engine_with_document() -> Result<()> {
        use crate::document::document::Document;
        use crate::embedding::text_embedder::TextEmbedder;
        use async_trait::async_trait;

        // Mock embedder for testing
        #[derive(Debug)]
        struct MockEmbedder {
            dimension: usize,
        }

        #[async_trait]
        impl TextEmbedder for MockEmbedder {
            async fn embed(&self, text: &str) -> Result<Vector> {
                // Simple mock: convert text length to a normalized vector
                let len = text.len() as f32;
                let val = (len % 100.0) / 100.0;
                Ok(Vector::new(vec![val; self.dimension]))
            }

            fn dimension(&self) -> usize {
                self.dimension
            }

            fn name(&self) -> &str {
                "mock"
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        let embedder: Arc<dyn TextEmbedder> = Arc::new(MockEmbedder { dimension: 3 });

        let config = VectorIndexConfig::Flat(FlatIndexConfig {
            dimension: 3,
            distance_metric: DistanceMetric::Cosine,
            embedder: embedder.clone(),
            ..Default::default()
        });
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = VectorIndexFactory::create(storage, config)?;
        let mut engine = VectorEngine::new(index)?;

        // Create document with vector field
        let doc = Document::builder()
            .add_text("title", "Machine Learning", TextOption::default())
            .add_vector(
                "title_embedding",
                "Machine Learning",
                VectorOption::default(),
            )
            .build();

        // Add document
        let _doc_id = engine.add_document(doc).await?;
        engine.commit()?;

        // The important thing is that the document was accepted and committed without errors
        // Stats can be queried without error
        let _stats = engine.stats()?;

        Ok(())
    }

    #[test]
    fn map_legacy_results_honor_document_filters() {
        let mut query = VectorCollectionQuery::default();
        query.limit = 5;
        query.score_mode = VectorScoreMode::WeightedSum;
        let mut doc_filter = MetadataFilter::default();
        doc_filter
            .equals
            .insert("lang".to_string(), "ja".to_string());
        query.filter = Some(VectorCollectionFilter {
            document: doc_filter,
            field: MetadataFilter::default(),
        });

        let selection = FieldSelection {
            scope: FieldSearchScope::All,
            required_roles: Vec::new(),
        };

        let mut english = HashMap::new();
        english.insert("lang".to_string(), "en".to_string());
        english.insert(METADATA_WEIGHT.to_string(), "1.0".to_string());

        let mut japanese = HashMap::new();
        japanese.insert("lang".to_string(), "ja".to_string());
        japanese.insert(METADATA_WEIGHT.to_string(), "1.0".to_string());

        let mut results = VectorSearchResults::new();
        results.results = vec![
            VectorSearchResult {
                doc_id: 1,
                field_name: "body".into(),
                similarity: 0.9,
                distance: 0.1,
                vector: None,
                metadata: english,
            },
            VectorSearchResult {
                doc_id: 2,
                field_name: "body".into(),
                similarity: 0.8,
                distance: 0.2,
                vector: None,
                metadata: japanese,
            },
        ];

        let run = LegacySearchRun {
            query_weight: 1.0,
            query_role: VectorRole::Generic,
            query_embedder_id: String::new(),
            results,
        };

        let merged = VectorEngine::map_legacy_results(&query, &selection, vec![run]);
        assert_eq!(merged.hits.len(), 1);
        assert_eq!(merged.hits[0].doc_id, 2);
    }
}
