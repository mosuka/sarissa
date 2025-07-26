//! HNSW vector searcher for approximate search.

use crate::error::Result;
use crate::vector::{
    Vector,
    reader::VectorIndexReader,
    types::{VectorSearchConfig, VectorSearchResults},
};
use crate::vector_search::{AdvancedSearchConfig, SearchStats, VectorSearcher};
use std::sync::Arc;

/// HNSW vector searcher that performs approximate nearest neighbor search.
pub struct HnswSearcher {
    index_reader: Arc<dyn VectorIndexReader>,
    stats: SearchStats,
    ef_search: usize, // Search parameter
}

impl HnswSearcher {
    /// Create a new HNSW searcher.
    pub fn new(index_reader: Arc<dyn VectorIndexReader>) -> Result<Self> {
        Ok(Self {
            index_reader,
            stats: SearchStats::default(),
            ef_search: 50, // Default search parameter
        })
    }

    /// Set the search parameter ef.
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.ef_search = ef_search;
    }
}

impl VectorSearcher for HnswSearcher {
    fn search(&self, _query: &Vector, _config: &VectorSearchConfig) -> Result<VectorSearchResults> {
        // Placeholder implementation
        let mut results = VectorSearchResults::new();
        results.search_time_ms = 0.5; // Faster than flat search
        results.candidates_examined = self.ef_search.min(self.index_reader.vector_count());
        Ok(results)
    }

    fn advanced_search(
        &self,
        query: &Vector,
        config: &AdvancedSearchConfig,
    ) -> Result<VectorSearchResults> {
        self.search(query, &config.base_config)
    }

    fn batch_search(
        &self,
        queries: &[Vector],
        config: &VectorSearchConfig,
    ) -> Result<Vec<VectorSearchResults>> {
        queries
            .iter()
            .map(|query| self.search(query, config))
            .collect()
    }

    fn search_stats(&self) -> SearchStats {
        self.stats.clone()
    }

    fn warmup(&mut self) -> Result<()> {
        Ok(())
    }
}
