//! HNSW vector searcher for approximate search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorSearcher;
use crate::vector::search::searcher::{VectorSearchRequest, VectorSearchResults};

/// HNSW vector searcher that performs approximate nearest neighbor search.
pub struct HnswSearcher {
    index_reader: Arc<dyn VectorIndexReader>,
    ef_search: usize,
}

impl HnswSearcher {
    /// Create a new HNSW searcher.
    pub fn new(index_reader: Arc<dyn VectorIndexReader>) -> Result<Self> {
        // Default ef_search value
        let ef_search = 50;
        Ok(Self {
            index_reader,
            ef_search,
        })
    }

    /// Set the search parameter ef.
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.ef_search = ef_search;
    }
}

impl VectorSearcher for HnswSearcher {
    fn search(&self, request: &VectorSearchRequest) -> Result<VectorSearchResults> {
        use std::time::Instant;

        let start = Instant::now();
        let mut results = VectorSearchResults::new();

        let ef_search = self.ef_search;

        let vector_ids = self.index_reader.vector_ids()?;
        let max_candidates = ef_search.min(vector_ids.len());
        results.candidates_examined = max_candidates;

        let mut candidates: Vec<(u64, f32, f32, crate::vector::Vector)> =
            Vec::with_capacity(max_candidates);

        for doc_id in vector_ids.iter().take(max_candidates) {
            if let Ok(Some(vector)) = self.index_reader.get_vector(*doc_id) {
                let similarity = self
                    .index_reader
                    .distance_metric()
                    .similarity(&request.query.data, &vector.data)?;
                let distance = self
                    .index_reader
                    .distance_metric()
                    .distance(&request.query.data, &vector.data)?;
                candidates.push((*doc_id, similarity, distance, vector));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = request.params.top_k.min(candidates.len());
        for (doc_id, similarity, distance, vector) in candidates.into_iter().take(top_k) {
            // Apply minimum similarity threshold
            if similarity < request.params.min_similarity {
                break;
            }

            results
                .results
                .push(crate::vector::search::searcher::VectorSearchResult {
                    doc_id,
                    similarity,
                    distance,
                    vector: if request.params.include_vectors {
                        Some(vector)
                    } else {
                        None
                    },
                    metadata: std::collections::HashMap::new(),
                });
        }

        results.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(results)
    }
}
