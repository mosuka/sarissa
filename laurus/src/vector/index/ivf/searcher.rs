//! IVF vector searcher for memory-efficient approximate search.

use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::vector::core::vector::Vector;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryResults};

/// IVF (Inverted File) vector searcher that performs approximate search by
/// restricting distance computations to vectors in the `n_probe` nearest
/// clusters.
#[derive(Debug)]
pub struct IvfSearcher {
    index_reader: Arc<dyn VectorIndexReader>,
    /// Number of clusters to probe during search.
    n_probe: usize,
}

impl IvfSearcher {
    /// Create a new IVF searcher with `n_probe = 1`.
    ///
    /// # Arguments
    ///
    /// * `index_reader` - The underlying vector index reader (must be an
    ///   [`IvfIndexReader`](super::reader::IvfIndexReader)).
    ///
    /// # Returns
    ///
    /// A new `IvfSearcher` instance.
    pub fn new(index_reader: Arc<dyn VectorIndexReader>) -> Result<Self> {
        let n_probe = 1;
        Ok(Self {
            index_reader,
            n_probe,
        })
    }

    /// Set the number of clusters to probe during search.
    ///
    /// # Arguments
    ///
    /// * `n_probe` - Number of nearest clusters to search.
    pub fn set_n_probe(&mut self, n_probe: usize) {
        self.n_probe = n_probe;
    }

    /// Find the `n_probe` nearest centroids to the query vector and return
    /// the vector IDs belonging to those clusters.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector.
    /// * `n_probe` - Number of nearest clusters to probe.
    /// * `field_name` - Optional field name filter.
    ///
    /// # Returns
    ///
    /// A `Vec` of `(doc_id, field_name)` pairs from the probed clusters,
    /// optionally filtered by `field_name`.
    fn probe_clusters(
        &self,
        query: &Vector,
        n_probe: usize,
        field_name: Option<&str>,
    ) -> Result<Vec<(u64, String)>> {
        use super::reader::IvfIndexReader;

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
            centroid_distances.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Collect vector IDs from the n_probe nearest clusters
            let mut result = Vec::new();
            for &(cluster_idx, _) in centroid_distances.iter().take(n_probe) {
                let cluster_vecs = ivf_reader.cluster_vectors(cluster_idx);
                if let Some(field) = field_name {
                    result.extend(cluster_vecs.iter().filter(|(_, f)| f == field).cloned());
                } else {
                    result.extend_from_slice(cluster_vecs);
                }
            }

            Ok(result)
        } else {
            Err(LaurusError::InvalidOperation(
                "IVF searcher requires an IvfIndexReader, but a different reader type was provided"
                    .to_string(),
            ))
        }
    }
}

impl VectorIndexSearcher for IvfSearcher {
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        use crate::util::time::Timer;

        let start = Timer::now();
        let mut results = VectorIndexQueryResults::new();

        // Probe only the n_probe nearest clusters
        let n_probe = self.n_probe.min(10);
        let vector_ids =
            self.probe_clusters(&request.query, n_probe, request.field_name.as_deref())?;

        // Calculate distances for vectors in the probed clusters
        let metric = self.index_reader.distance_metric();
        let mut candidates: Vec<(u64, String, f32, f32, Vector)> =
            Vec::with_capacity(vector_ids.len());

        for (doc_id, field_name) in &vector_ids {
            if let Ok(Some(vector)) = self.index_reader.get_vector(*doc_id, field_name) {
                let distance = metric.distance(&request.query.data, &vector.data)?;
                let similarity = metric.distance_to_similarity(distance);
                candidates.push((*doc_id, field_name.clone(), similarity, distance, vector));
            }
        }

        // Sort by similarity (descending)
        candidates
            .sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k results
        let candidates_len = candidates.len();
        let top_k = request.params.top_k.min(candidates_len);
        for (doc_id, field_name, similarity, distance, vector) in candidates.into_iter().take(top_k)
        {
            // Apply minimum similarity threshold
            if similarity < request.params.min_similarity {
                break;
            }

            let vector_output = if request.params.include_vectors {
                Some(vector)
            } else {
                None
            };

            results
                .results
                .push(crate::vector::search::searcher::VectorIndexQueryResult {
                    doc_id,
                    field_name,
                    similarity,
                    distance,
                    vector: vector_output,
                });
        }

        results.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        results.candidates_examined = candidates_len;
        Ok(results)
    }

    fn count(&self, request: VectorIndexQuery) -> Result<u64> {
        // Field-filtered counts use the pre-built per-field index (#405);
        // avoids allocating + linear-filtering the full `vector_ids`.
        if let Some(ref field_name) = request.field_name {
            Ok(self.index_reader.doc_ids_for_field(field_name).len() as u64)
        } else {
            Ok(self.index_reader.vector_ids()?.len() as u64)
        }
    }
}
