//! IVF vector searcher for memory-efficient approximate search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::Vector;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::{SearchStats, VectorSearcher};
use crate::vector::types::{VectorSearchConfig, VectorSearchResults};

/// IVF (Inverted File) vector searcher that performs memory-efficient approximate search.
pub struct IvfSearcher {
    index_reader: Arc<dyn VectorIndexReader>,
    stats: SearchStats,
    n_probe: usize, // Number of clusters to search
}

impl IvfSearcher {
    /// Create a new IVF searcher.
    pub fn new(index_reader: Arc<dyn VectorIndexReader>) -> Result<Self> {
        // Default n_probe value
        let n_probe = 1;
        Ok(Self {
            index_reader,
            stats: SearchStats::default(),
            n_probe,
        })
    }

    /// Set the number of clusters to probe during search.
    pub fn set_n_probe(&mut self, n_probe: usize) {
        self.n_probe = n_probe;
    }

    /// Find the nearest centroids to the query vector.
    fn find_nearest_centroids(&self, _query: &Vector, n_probe: usize) -> Result<Vec<usize>> {
        // In a full implementation, we would:
        // 1. Get centroids from the index
        // 2. Calculate distances to all centroids
        // 3. Return indices of nearest n_probe centroids
        // For now, return a placeholder
        Ok((0..n_probe).collect())
    }
}

impl VectorSearcher for IvfSearcher {
    fn search(&self, query: &Vector, config: &VectorSearchConfig) -> Result<VectorSearchResults> {
        use std::time::Instant;

        let start = Instant::now();
        let mut results = VectorSearchResults::new();

        // Find nearest centroids to probe
        let n_probe = self.n_probe.min(10); // Default max clusters
        let _nearest_centroid_indices = self.find_nearest_centroids(query, n_probe)?;

        // In a full implementation, we would search only vectors in the nearest clusters
        // For now, we'll search all vectors but track which clusters we examined
        results.candidates_examined = n_probe;

        // Get all vector IDs (in a real IVF implementation, we'd only get IDs from selected clusters)
        let vector_ids = self.index_reader.vector_ids()?;

        // Calculate similarities for all vectors
        let mut candidates: Vec<(u64, f32, f32, crate::vector::Vector)> =
            Vec::with_capacity(vector_ids.len());

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

        // Apply filters if specified
        if !config.filters.is_empty() {
            candidates.retain(|(doc_id, similarity, _, _)| {
                self.matches_filters(*doc_id, *similarity, &config.filters)
            });
        }

        // Take top_k results
        let candidates_len = candidates.len();
        let top_k = config.top_k.min(candidates_len);
        for (doc_id, similarity, distance, vector) in candidates.into_iter().take(top_k) {
            // Apply minimum similarity threshold
            if similarity < config.min_similarity {
                break;
            }

            results
                .results
                .push(crate::vector::types::VectorSearchResult {
                    doc_id,
                    similarity,
                    distance,
                    vector: if config.include_vectors {
                        Some(vector)
                    } else {
                        None
                    },
                    metadata: std::collections::HashMap::new(),
                });
        }

        results.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        results.candidates_examined = candidates_len;
        Ok(results)
    }

    fn search_stats(&self) -> SearchStats {
        self.stats.clone()
    }
}

impl IvfSearcher {
    /// フィルタにマッチするかチェック
    fn matches_filters(
        &self,
        doc_id: u64,
        similarity: f32,
        filters: &[crate::vector::search::SearchFilter],
    ) -> bool {
        use crate::vector::search::SearchFilter;

        filters.iter().all(|filter| match filter {
            SearchFilter::SimilarityThreshold(threshold) => similarity >= *threshold,
            SearchFilter::DocIdRange {
                min_doc_id,
                max_doc_id,
            } => {
                let min_ok = min_doc_id.map(|min| doc_id >= min).unwrap_or(true);
                let max_ok = max_doc_id.map(|max| doc_id <= max).unwrap_or(true);
                min_ok && max_ok
            }
            SearchFilter::MetadataFilter { .. } => true, // TODO: メタデータフィルタ実装
            SearchFilter::Custom(_) => true,             // TODO: カスタムフィルタ実装
        })
    }
}
