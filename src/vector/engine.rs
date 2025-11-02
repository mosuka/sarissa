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
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorSearcher;
use crate::vector::search::searcher::{VectorSearchRequest, VectorSearchResults};
use crate::vector::{DistanceMetric, Vector};

/// A high-level unified vector engine that provides both indexing and searching capabilities.
/// This is similar to the lexical SearchEngine but for vector search.
///
/// # Example
///
/// ```
/// use yatagarasu::vector::engine::VectorEngine;
/// use yatagarasu::vector::index::{VectorIndexConfig, FlatIndexConfig};
/// use yatagarasu::vector::index::factory::VectorIndexFactory;
/// use yatagarasu::vector::{Vector, DistanceMetric};
/// use yatagarasu::vector::search::searcher::VectorSearchRequest;
/// use yatagarasu::storage::memory::{MemoryStorage, MemoryStorageConfig};
/// use yatagarasu::storage::StorageConfig;
/// use std::sync::Arc;
///
/// # fn main() -> yatagarasu::error::Result<()> {
/// // Create engine with flat index
/// let config = VectorIndexConfig::Flat(FlatIndexConfig {
///     dimension: 3,
///     distance_metric: DistanceMetric::Cosine,
///     ..Default::default()
/// });
/// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
/// let index = VectorIndexFactory::create(storage, config)?;
/// let mut engine = VectorEngine::new(index)?;
///
/// // Add vectors
/// let vectors = vec![
///     (1, Vector::new(vec![1.0, 0.0, 0.0])),
///     (2, Vector::new(vec![0.0, 1.0, 0.0])),
/// ];
/// engine.add_vectors(vectors)?;
/// engine.commit()?;
///
/// // Search
/// let query_vector = Vector::new(vec![1.0, 0.1, 0.0]);
/// let request = VectorSearchRequest::new(query_vector).top_k(2);
/// let results = engine.search(request)?;
/// assert_eq!(results.results.len(), 2);
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
    /// use yatagarasu::vector::engine::VectorEngine;
    /// use yatagarasu::vector::index::VectorIndexConfig;
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
    /// use yatagarasu::vector::index::VectorIndexConfig;
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

    /// Add vectors to the index.
    pub fn add_vectors(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()> {
        let mut writer = self.get_or_create_writer()?;
        writer.add_vectors(vectors)?;
        Ok(())
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
    /// use yatagarasu::vector::index::VectorIndexConfig;
    /// use yatagarasu::vector::index::factory::VectorIndexFactory;
    /// use yatagarasu::vector::Vector;
    /// use yatagarasu::storage::memory::{MemoryStorage, MemoryStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> yatagarasu::error::Result<()> {
    /// let config = VectorIndexConfig::default();
    /// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// let index = VectorIndexFactory::create(storage, config)?;
    /// let mut engine = VectorEngine::new(index)?;
    ///
    /// // Add vectors
    /// engine.add_vectors(vec![(1, Vector::new(vec![1.0, 0.0, 0.0]))])?;
    ///
    /// // Commit changes
    /// engine.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn commit(&mut self) -> Result<()> {
        // Finalize the writer if it exists
        if let Some(mut writer) = self.writer.borrow_mut().take() {
            writer.finalize()?;
            // Write the index to storage
            writer.write("default_index")?;
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
    /// use yatagarasu::vector::index::VectorIndexConfig;
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
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::DistanceMetric;
    use crate::vector::index::factory::VectorIndexFactory;
    use crate::vector::index::{FlatIndexConfig, VectorIndexConfig};
    use std::sync::Arc;

    #[test]
    fn test_vector_engine_basic() -> Result<()> {
        let config = VectorIndexConfig::Flat(FlatIndexConfig {
            dimension: 3,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        });
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = VectorIndexFactory::create(storage, config)?;
        let mut engine = VectorEngine::new(index)?;

        // Add some vectors
        let vectors = vec![
            (1, Vector::new(vec![1.0, 0.0, 0.0])),
            (2, Vector::new(vec![0.0, 1.0, 0.0])),
            (3, Vector::new(vec![0.0, 0.0, 1.0])),
        ];

        engine.add_vectors(vectors)?;
        engine.commit()?;

        // Search for similar vectors
        let query = Vector::new(vec![1.0, 0.1, 0.0]);
        let request = VectorSearchRequest::new(query).top_k(2);

        let results = engine.search(request)?;
        assert_eq!(results.results.len(), 2);

        Ok(())
    }
}
