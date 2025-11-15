//! Flat vector searcher for exact search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::vector::Vector;
use crate::vector::index::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorSearcher;
use crate::vector::search::searcher::{VectorSearchRequest, VectorSearchResults};

/// Flat vector searcher that performs exact (brute force) search.
#[derive(Debug)]
pub struct FlatVectorSearcher {
    index_reader: Arc<dyn VectorIndexReader>,
}

impl FlatVectorSearcher {
    /// Create a new flat vector searcher.
    pub fn new(index_reader: Arc<dyn VectorIndexReader>) -> Result<Self> {
        Ok(Self { index_reader })
    }
}

impl VectorSearcher for FlatVectorSearcher {
    fn search(&self, request: &VectorSearchRequest) -> Result<VectorSearchResults> {
        use std::time::Instant;

        let start = Instant::now();
        let mut results = VectorSearchResults::new();

        // Get all vectors from the index
        let vector_count = self.index_reader.vector_count();
        results.candidates_examined = vector_count;

        // Get all vector IDs with field names
        let vector_ids = self.index_reader.vector_ids()?;

        // Filter by field_name if specified
        let filtered_vector_ids: Vec<(u64, String)> =
            if let Some(ref field_name) = request.field_name {
                vector_ids
                    .into_iter()
                    .filter(|(_, f)| f == field_name)
                    .collect()
            } else {
                vector_ids
            };

        // Calculate similarities for all vectors
        let mut candidates: Vec<(u64, String, f32, f32, Vector)> =
            Vec::with_capacity(filtered_vector_ids.len());
        for (doc_id, field_name) in filtered_vector_ids {
            if let Ok(Some(vector)) = self.index_reader.get_vector(doc_id, &field_name) {
                let similarity = self
                    .index_reader
                    .distance_metric()
                    .similarity(&request.query.data, &vector.data)?;
                let distance = self
                    .index_reader
                    .distance_metric()
                    .distance(&request.query.data, &vector.data)?;
                candidates.push((doc_id, field_name, similarity, distance, vector));
            }
        }

        // Sort by similarity (descending)
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k results
        let top_k = request.params.top_k.min(candidates.len());
        for (doc_id, field_name, similarity, distance, vector) in candidates.into_iter().take(top_k)
        {
            // Apply minimum similarity threshold
            if similarity < request.params.min_similarity {
                break;
            }

            results
                .results
                .push(crate::vector::search::searcher::VectorSearchResult {
                    doc_id,
                    field_name,
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

    fn count(&self, request: VectorSearchRequest) -> Result<u64> {
        // Get all vector IDs with field names
        let vector_ids = self.index_reader.vector_ids()?;

        // Filter by field_name if specified
        if let Some(ref field_name) = request.field_name {
            Ok(vector_ids.iter().filter(|(_, f)| f == field_name).count() as u64)
        } else {
            Ok(vector_ids.len() as u64)
        }
    }
}
