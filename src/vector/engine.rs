//! High-level unified vector engine that combines indexing and searching.
//!
//! This module provides a unified interface for vector indexing and search,
//! similar to the lexical SearchEngine.

use std::cell::{RefCell, RefMut};
use std::sync::Arc;

use crate::error::{Result, YatagarasuError};
use crate::storage::Storage;
use crate::vector::index::flat::reader::FlatVectorIndexReader;
use crate::vector::index::flat::searcher::FlatVectorSearcher;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::searcher::HnswSearcher;
use crate::vector::index::ivf::reader::IvfIndexReader;
use crate::vector::index::ivf::searcher::IvfSearcher;
use crate::vector::index::reader::VectorIndexReader;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::search::searcher::VectorSearcher;
use crate::vector::search::searcher::{VectorSearchRequest, VectorSearchResults};
use crate::vector::{DistanceMetric, Vector};

/// A high-level unified vector engine that provides both indexing and searching capabilities.
/// This is similar to the lexical SearchEngine but for vector search.
///
/// # Example
///
/// ```no_run
/// use yatagarasu::vector::engine::VectorEngine;
/// use yatagarasu::vector::index::config::{VectorIndexConfig, FlatIndexConfig};
/// use yatagarasu::vector::index::factory::VectorIndexFactory;
/// use yatagarasu::vector::{Vector, DistanceMetric};
/// use yatagarasu::vector::search::searcher::VectorSearchRequest;
/// use yatagarasu::storage::memory::{MemoryStorage, MemoryStorageConfig};
/// use yatagarasu::document::document::Document;
/// use yatagarasu::embedding::text_embedder::TextEmbedder;
/// use std::sync::Arc;
///
/// # async fn example(embedder: Arc<dyn TextEmbedder>) -> yatagarasu::error::Result<()> {
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
/// use yatagarasu::document::field::VectorOption;
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
/// let request = VectorSearchRequest::new(query_vector).top_k(2);
/// let results = engine.search(request)?;
/// # Ok(())
/// # }
/// ```
pub struct VectorEngine {
    /// The underlying index.
    index: Box<dyn VectorIndex>,
    /// The reader for executing queries (cached for efficiency).
    reader: RefCell<Option<Arc<dyn crate::vector::index::reader::VectorIndexReader>>>,
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
    /// use yatagarasu::vector::engine::VectorEngine;
    /// use yatagarasu::vector::index::config::VectorIndexConfig;
    /// use yatagarasu::vector::index::factory::VectorIndexFactory;
    /// use yatagarasu::storage::{StorageConfig, StorageFactory};
    /// use yatagarasu::storage::memory::MemoryStorageConfig;
    ///
    /// # fn main() -> yatagarasu::error::Result<()> {
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
    /// use yatagarasu::vector::engine::VectorEngine;
    /// use yatagarasu::vector::index::config::VectorIndexConfig;
    /// use yatagarasu::vector::index::factory::VectorIndexFactory;
    /// use yatagarasu::storage::{StorageConfig, StorageFactory};
    /// use yatagarasu::storage::file::FileStorageConfig;
    ///
    /// # fn main() -> yatagarasu::error::Result<()> {
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
    /// The searcher is created based on the reader type (Flat, HNSW, or IVF)
    /// and cached for efficiency.
    fn get_or_create_searcher(&self) -> Result<RefMut<'_, Box<dyn VectorSearcher>>> {
        {
            let mut searcher_ref = self.searcher.borrow_mut();
            if searcher_ref.is_none() {
                let reader = self.get_or_create_reader()?;

                // Try to downcast to specific reader types and create appropriate searcher
                let searcher: Box<dyn VectorSearcher> =
                    if reader.as_any().is::<FlatVectorIndexReader>() {
                        Box::new(FlatVectorSearcher::new(reader.clone())?)
                    } else if reader.as_any().is::<HnswIndexReader>() {
                        Box::new(HnswSearcher::new(reader.clone())?)
                    } else if reader.as_any().is::<IvfIndexReader>() {
                        Box::new(IvfSearcher::new(reader.clone())?)
                    } else {
                        return Err(YatagarasuError::InvalidOperation(
                            "Unknown vector index reader type".to_string(),
                        ));
                    };

                *searcher_ref = Some(searcher);
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
    /// use yatagarasu::vector::engine::VectorEngine;
    /// use yatagarasu::vector::index::config::VectorIndexConfig;
    /// use yatagarasu::vector::index::factory::VectorIndexFactory;
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::embedding::text_embedder::TextEmbedder;
    /// use yatagarasu::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # async fn example(embedder: Arc<dyn TextEmbedder>) -> yatagarasu::error::Result<()> {
    /// let mut config = VectorIndexConfig::default();
    /// // Set embedder in config (assuming Flat variant)
    /// if let VectorIndexConfig::Flat(ref mut flat_config) = config {
    ///     flat_config.embedder = embedder;
    /// }
    /// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// let index = VectorIndexFactory::create(storage, config)?;
    /// let mut engine = VectorEngine::new(index)?;
    ///
    /// use yatagarasu::document::field::{TextOption, VectorOption};
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

    /// Get the next available vector ID.
    fn next_vector_id(&self) -> u64 {
        let writer = self.get_or_create_writer().unwrap();
        writer.next_vector_id()
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
    /// use yatagarasu::vector::engine::VectorEngine;
    /// use yatagarasu::vector::index::config::VectorIndexConfig;
    /// use yatagarasu::vector::index::factory::VectorIndexFactory;
    /// use yatagarasu::document::document::Document;
    /// use yatagarasu::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # async fn example() -> yatagarasu::error::Result<()> {
    /// let config = VectorIndexConfig::default();
    /// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// let index = VectorIndexFactory::create(storage, config)?;
    /// let mut engine = VectorEngine::new(index)?;
    ///
    /// // Add documents with vector fields
    /// use yatagarasu::document::field::VectorOption;
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
    /// use yatagarasu::vector::engine::VectorEngine;
    /// use yatagarasu::vector::index::config::VectorIndexConfig;
    /// use yatagarasu::vector::index::factory::VectorIndexFactory;
    /// use yatagarasu::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> yatagarasu::error::Result<()> {
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

    /// Search for similar vectors.
    ///
    /// This method automatically selects the appropriate searcher implementation
    /// based on the underlying index type (Flat, HNSW, or IVF).
    /// The searcher is cached for efficiency.
    pub fn search(&self, request: VectorSearchRequest) -> Result<VectorSearchResults> {
        let searcher = self.get_or_create_searcher()?;
        searcher.search(&request)
    }

    /// Count the number of vectors matching the request.
    ///
    /// This method counts vectors that match the field filter in the request.
    /// It's more efficient than searching when you only need the count.
    pub fn count(&self, request: VectorSearchRequest) -> Result<u64> {
        let searcher = self.get_or_create_searcher()?;
        searcher.count(request)
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
        let reader = self.index.reader()?;
        Ok(reader.dimension())
    }

    /// Get the distance metric.
    pub fn distance_metric(&self) -> Result<DistanceMetric> {
        let reader = self.index.reader()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::field::{TextOption, VectorOption};
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::core::DistanceMetric;
    use crate::vector::index::config::{FlatIndexConfig, VectorIndexConfig};
    use crate::vector::index::factory::VectorIndexFactory;
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
        let query = Vector::new(vec![0.5, 0.5, 0.5]);
        let request = VectorSearchRequest::new(query).top_k(2);

        let results = engine.search(request)?;
        assert_eq!(results.results.len(), 2);

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
}
