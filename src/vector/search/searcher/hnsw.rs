//! HNSW vector searcher for approximate search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::Vector;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::{SearchStats, VectorSearcher};
use crate::vector::types::{VectorSearchConfig, VectorSearchResults};

/// HNSW vector searcher that performs approximate nearest neighbor search.
pub struct HnswSearcher {
    index_reader: Arc<dyn VectorIndexReader>,
    stats: SearchStats,
    ef_search: usize,
}

impl HnswSearcher {
    /// Create a new HNSW searcher.
    pub fn new(index_reader: Arc<dyn VectorIndexReader>) -> Result<Self> {
        // Default ef_search value
        let ef_search = 50;
        Ok(Self {
            index_reader,
            stats: SearchStats::default(),
            ef_search,
        })
    }

    /// Set the search parameter ef.
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.ef_search = ef_search;
    }
}

impl VectorSearcher for HnswSearcher {
    fn search(&self, query: &Vector, config: &VectorSearchConfig) -> Result<VectorSearchResults> {
        use std::time::Instant;

        let start = Instant::now();
        let mut results = VectorSearchResults::new();

        // 検索戦略に基づいてef_searchを調整
        let ef_search = if let Some(ref strategy) = config.strategy {
            use crate::vector::search::SearchStrategy;
            match strategy {
                SearchStrategy::Approximate { quality } => {
                    // qualityに基づいて動的調整 (0.0-1.0 -> ef_search)
                    ((quality * 200.0).max(self.ef_search as f32)) as usize
                }
                SearchStrategy::Exact => self.index_reader.vector_count(), // 全探索
                _ => self.ef_search,
            }
        } else {
            self.ef_search
        };

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
                    .similarity(&query.data, &vector.data)?;
                let distance = self
                    .index_reader
                    .distance_metric()
                    .distance(&query.data, &vector.data)?;
                candidates.push((*doc_id, similarity, distance, vector));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply filters if specified
        if !config.filters.is_empty() {
            candidates.retain(|(doc_id, similarity, _, _)| {
                self.matches_filters(*doc_id, *similarity, &config.filters)
            });
        }

        let top_k = config.top_k.min(candidates.len());
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
        Ok(results)
    }

    fn search_stats(&self) -> SearchStats {
        self.stats.clone()
    }
}

impl HnswSearcher {
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
