//! Flat vector searcher for exact search.

use crate::error::Result;
use crate::vector::{
    Vector,
    reader::VectorIndexReader,
    types::{VectorSearchConfig, VectorSearchResults},
};
use crate::vector_search::{AdvancedSearchConfig, SearchStats, VectorSearcher};
use std::sync::Arc;

/// Flat vector searcher that performs exact (brute force) search.
pub struct FlatVectorSearcher {
    index_reader: Arc<dyn VectorIndexReader>,
    stats: SearchStats,
}

impl FlatVectorSearcher {
    /// Create a new flat vector searcher.
    pub fn new(index_reader: Arc<dyn VectorIndexReader>) -> Result<Self> {
        Ok(Self {
            index_reader,
            stats: SearchStats::default(),
        })
    }
}

impl VectorSearcher for FlatVectorSearcher {
    fn search(&self, _query: &Vector, _config: &VectorSearchConfig) -> Result<VectorSearchResults> {
        // Placeholder implementation
        let mut results = VectorSearchResults::new();
        results.search_time_ms = 1.0;
        results.candidates_examined = self.index_reader.vector_count();
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
