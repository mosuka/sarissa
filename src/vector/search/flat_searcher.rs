//! Flat vector searcher for exact search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::Vector;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::{AdvancedSearchConfig, SearchStats, VectorSearcher};
use crate::vector::types::{VectorSearchConfig, VectorSearchResults};

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
    fn search(&self, query: &Vector, config: &VectorSearchConfig) -> Result<VectorSearchResults> {
        use std::time::Instant;

        let start = Instant::now();
        let mut results = VectorSearchResults::new();

        // Get all vectors from the index
        let vector_count = self.index_reader.vector_count();
        results.candidates_examined = vector_count;

        // Get all vector IDs
        let vector_ids = self.index_reader.vector_ids()?;

        // Calculate similarities for all vectors
        let mut candidates: Vec<(u64, f32, f32, crate::vector::Vector)> =
            Vec::with_capacity(vector_count);
        for doc_id in vector_ids {
            if let Ok(Some(vector)) = self.index_reader.get_vector(doc_id) {
                let similarity = self
                    .index_reader
                    .distance_metric()
                    .similarity(&query.data, &vector.data)?;
                let distance = self
                    .index_reader
                    .distance_metric()
                    .distance(&query.data, &vector.data)?;
                candidates.push((doc_id, similarity, distance, vector));
            }
        }

        // Sort by similarity (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k results
        let top_k = config.top_k.min(candidates.len());
        for (doc_id, similarity, distance, vector) in candidates.into_iter().take(top_k) {
            results
                .results
                .push(crate::vector::types::VectorSearchResult {
                    doc_id,
                    similarity,
                    distance,
                    vector: Some(vector),
                    metadata: std::collections::HashMap::new(),
                });
        }

        results.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
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
