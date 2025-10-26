//! High-level unified vector engine that combines indexing and searching.
//!
//! This module provides a unified interface for vector indexing and search,
//! similar to the lexical SearchEngine.

use std::cell::RefCell;
use std::sync::Arc;

use crate::error::Result;
use crate::vector::Vector;
use crate::vector::index::{VectorIndex, VectorIndexWriterConfig};
use crate::vector::search::VectorSearcher;
use crate::vector::search::searcher::flat::FlatVectorSearcher;
use crate::vector::types::{VectorSearchRequest, VectorSearchResults};

/// A high-level unified vector engine that provides both indexing and searching capabilities.
/// This is similar to the lexical SearchEngine but for vector search.
///
/// # Example
///
/// ```
/// use sage::vector::engine::VectorEngine;
/// use sage::vector::index::{VectorIndexWriterConfig, VectorIndexType};
/// use sage::vector::{Vector, DistanceMetric};
/// use sage::vector::types::VectorSearchRequest;
/// use sage::storage::memory::MemoryStorage;
/// use sage::storage::StorageConfig;
/// use std::sync::Arc;
///
/// # fn main() -> sage::error::Result<()> {
/// // Create engine
/// let config = VectorIndexWriterConfig {
///     dimension: 3,
///     distance_metric: DistanceMetric::Cosine,
///     index_type: VectorIndexType::Flat,
///     ..Default::default()
/// };
/// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
/// let mut engine = VectorEngine::new(config, storage)?;
///
/// // Add vectors
/// let vectors = vec![
///     (1, Vector::new(vec![1.0, 0.0, 0.0])),
///     (2, Vector::new(vec![0.0, 1.0, 0.0])),
/// ];
/// engine.add_vectors(vectors)?;
/// engine.finalize()?;
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
    index: VectorIndex,
    /// The searcher for executing queries (lazily created).
    searcher: RefCell<Option<Box<dyn VectorSearcher>>>,
}

impl VectorEngine {
    /// Create a new vector engine with the given configuration and storage.
    ///
    /// # Arguments
    ///
    /// * `config` - Vector index configuration including index type
    /// * `storage` - Storage backend (MemoryStorage, FileStorage, etc.)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use sage::vector::engine::VectorEngine;
    /// use sage::vector::index::{VectorIndexWriterConfig, VectorIndexType};
    /// use sage::vector::DistanceMetric;
    /// use sage::storage::memory::MemoryStorage;
    /// use sage::storage::StorageConfig;
    /// use std::sync::Arc;
    ///
    /// let config = VectorIndexWriterConfig {
    ///     dimension: 128,
    ///     distance_metric: DistanceMetric::Cosine,
    ///     index_type: VectorIndexType::Flat,
    ///     ..Default::default()
    /// };
    /// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// let engine = VectorEngine::new(config, storage).unwrap();
    /// ```
    pub fn new(
        config: VectorIndexWriterConfig,
        storage: Arc<dyn crate::storage::Storage>,
    ) -> Result<Self> {
        let index = VectorIndex::new(config, storage)?;
        Ok(Self {
            index,
            searcher: RefCell::new(None),
        })
    }

    /// Add vectors to the index.
    pub fn add_vectors(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()> {
        self.index.add_vectors(vectors)?;
        // Invalidate searcher cache
        *self.searcher.borrow_mut() = None;
        Ok(())
    }

    /// Finalize the index construction.
    /// This must be called before searching.
    pub fn finalize(&mut self) -> Result<()> {
        self.index.finalize()?;
        // Invalidate searcher cache so it will be recreated with finalized index
        *self.searcher.borrow_mut() = None;
        Ok(())
    }

    /// Optimize the index.
    pub fn optimize(&mut self) -> Result<()> {
        self.index.optimize()?;
        // Invalidate searcher cache
        *self.searcher.borrow_mut() = None;
        Ok(())
    }

    /// Get or create a searcher for this engine.
    fn get_searcher(&self) -> Result<std::cell::Ref<'_, Box<dyn VectorSearcher>>> {
        {
            let mut searcher_ref = self.searcher.borrow_mut();
            if searcher_ref.is_none() {
                // Get vectors directly from the index
                let vectors = self.index.vectors()?;
                let config = self.index.config();

                // Create SimpleVectorReader
                let reader = crate::vector::reader::SimpleVectorReader::new(
                    vectors,
                    config.dimension,
                    config.distance_metric,
                )?;

                let searcher: Box<dyn VectorSearcher> =
                    Box::new(FlatVectorSearcher::new(Arc::new(reader))?);
                *searcher_ref = Some(searcher);
            }
        }

        // Return a reference to the searcher
        Ok(std::cell::Ref::map(self.searcher.borrow(), |opt| {
            opt.as_ref().unwrap()
        }))
    }

    /// Refresh the searcher to see latest changes.
    pub fn refresh(&mut self) -> Result<()> {
        *self.searcher.borrow_mut() = None;
        Ok(())
    }

    /// Search for similar vectors.
    pub fn search(&self, request: VectorSearchRequest) -> Result<VectorSearchResults> {
        let searcher = self.get_searcher()?;
        searcher.search(&request.query, &request.config)
    }

    /// Get build progress (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        self.index.progress()
    }

    /// Get estimated memory usage.
    pub fn estimated_memory_usage(&self) -> usize {
        self.index.estimated_memory_usage()
    }

    /// Check if the index is finalized.
    pub fn is_finalized(&self) -> bool {
        self.index.is_finalized()
    }

    /// Get the configuration.
    pub fn config(&self) -> &VectorIndexWriterConfig {
        self.index.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryStorage;
    use crate::storage::{FileStorageConfig, MemoryStorageConfig};
    use crate::vector::DistanceMetric;
    use crate::vector::index::VectorIndexType;
    use std::sync::Arc;

    #[test]
    fn test_vector_engine_basic() -> Result<()> {
        let config = VectorIndexWriterConfig {
            dimension: 3,
            distance_metric: DistanceMetric::Cosine,
            index_type: VectorIndexType::Flat,
            ..Default::default()
        };
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut engine = VectorEngine::new(config, storage)?;

        // Add some vectors
        let vectors = vec![
            (1, Vector::new(vec![1.0, 0.0, 0.0])),
            (2, Vector::new(vec![0.0, 1.0, 0.0])),
            (3, Vector::new(vec![0.0, 0.0, 1.0])),
        ];

        engine.add_vectors(vectors)?;
        engine.finalize()?;

        // Search for similar vectors
        let query = Vector::new(vec![1.0, 0.1, 0.0]);
        let request = VectorSearchRequest::new(query).top_k(2);

        let results = engine.search(request)?;
        assert_eq!(results.results.len(), 2);

        Ok(())
    }
}
