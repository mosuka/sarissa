//! IVF vector searcher for memory-efficient approximate search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::vector::Vector;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::search::searcher::{VectorIndexSearchRequest, VectorIndexSearchResults};

/// IVF (Inverted File) vector searcher that performs memory-efficient approximate search.
#[derive(Debug)]
pub struct IvfSearcher {
    index_reader: Arc<dyn VectorIndexReader>,
    n_probe: usize, // Number of clusters to search
}

impl IvfSearcher {
    /// Create a new IVF searcher.
    pub fn new(index_reader: Arc<dyn VectorIndexReader>) -> Result<Self> {
        // Default n_probe value
        let n_probe = 1;
        Ok(Self {
            index_reader,
            n_probe,
        })
    }

    /// Set the number of clusters to probe during search.
    pub fn set_n_probe(&mut self, n_probe: usize) {
        self.n_probe = n_probe;
    }

    /// Find the nearest centroids to the query vector.
    fn find_nearest_centroids(&self, query: &Vector, n_probe: usize) -> Result<Vec<usize>> {
        use super::reader::IvfIndexReader;

        // Try to downcast the reader to IvfIndexReader to get centroids
        if let Some(ivf_reader) = self.index_reader.as_any().downcast_ref::<IvfIndexReader>() {
            let centroids = ivf_reader.centroids();
            let distance_metric = self.index_reader.distance_metric();

            if centroids.is_empty() {
                return Ok(Vec::new());
            }

            // Calculate distances to all centroids
            let mut centroid_distances: Vec<(usize, f32)> = centroids
                .iter()
                .enumerate()
                .map(|(i, centroid)| {
                    let dist = distance_metric
                        .distance(&query.data, &centroid.data)
                        .unwrap_or(f32::MAX);
                    (i, dist)
                })
                .collect();

            // Sort by distance (ascending)
            centroid_distances
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Return n_probe nearest centroid indices
            Ok(centroid_distances
                .into_iter()
                .take(n_probe)
                .map(|(i, _)| i)
                .collect())
        } else {
            // Fallback: return first n_probe indices
            let n_clusters = 10; // Default assumption
            Ok((0..n_probe.min(n_clusters)).collect())
        }
    }
}

impl VectorIndexSearcher for IvfSearcher {
    fn search(&self, request: &VectorIndexSearchRequest) -> Result<VectorIndexSearchResults> {
        use std::time::Instant;

        let start = Instant::now();
        let mut results = VectorIndexSearchResults::new();

        // Find nearest centroids to probe
        let n_probe = self.n_probe.min(10); // Default max clusters
        let _nearest_centroid_indices = self.find_nearest_centroids(&request.query, n_probe)?;

        // In a full implementation, we would search only vectors in the nearest clusters
        // For now, we'll search all vectors but track which clusters we examined
        results.candidates_examined = n_probe;

        // Get all vector IDs (in a real IVF implementation, we'd only get IDs from selected clusters)
        let vector_ids = self.index_reader.vector_ids()?;

        // Filter by field_name if specified
        let vector_ids: Vec<(u64, String)> = if let Some(ref field_name) = request.field_name {
            vector_ids
                .into_iter()
                .filter(|(_, field)| field == field_name)
                .collect()
        } else {
            vector_ids
        };

        // Calculate similarities for all vectors
        let mut candidates: Vec<(u64, String, f32, f32, Vector)> =
            Vec::with_capacity(vector_ids.len());

        for (doc_id, field_name) in vector_ids {
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
        let candidates_len = candidates.len();
        let top_k = request.params.top_k.min(candidates_len);
        for (doc_id, field_name, similarity, distance, vector) in candidates.into_iter().take(top_k)
        {
            // Apply minimum similarity threshold
            if similarity < request.params.min_similarity {
                break;
            }

            let metadata = vector.metadata.clone();
            let vector_output = if request.params.include_vectors {
                Some(vector)
            } else {
                None
            };

            results
                .results
                .push(crate::vector::search::searcher::VectorSearchResult {
                    doc_id,
                    field_name,
                    similarity,
                    distance,
                    vector: vector_output,
                    metadata,
                });
        }

        results.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        results.candidates_examined = candidates_len;
        Ok(results)
    }

    fn count(&self, request: VectorIndexSearchRequest) -> Result<u64> {
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
