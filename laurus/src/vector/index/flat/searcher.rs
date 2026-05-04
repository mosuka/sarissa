//! Flat vector searcher for exact search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::vector::Vector;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryResults};

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

impl VectorIndexSearcher for FlatVectorSearcher {
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        use crate::util::time::Timer;

        let start = Timer::now();
        let mut results = VectorIndexQueryResults::new();
        let metric = self.index_reader.distance_metric();

        if let Some(ref field_name) = request.field_name {
            // Field-filtered path: fetch the per-field doc-id slice from the
            // reader's pre-built index (#405 — O(1) Arc clone, avoids the full
            // `Vec<(u64, String)>` clone and the linear filter scan). Since
            // every candidate shares the same `field_name`, do not store it
            // per-candidate; clone it only when constructing the top_k
            // results.
            let ids = self.index_reader.doc_ids_for_field(field_name);
            results.candidates_examined = ids.len();

            let mut candidates: Vec<(u64, f32, f32, Vector)> = Vec::with_capacity(ids.len());
            for &doc_id in ids.iter() {
                if let Ok(Some(vector)) = self.index_reader.get_vector(doc_id, field_name) {
                    let distance = metric.distance(&request.query.data, &vector.data)?;
                    let similarity = metric.distance_to_similarity(distance);
                    candidates.push((doc_id, similarity, distance, vector));
                }
            }

            candidates.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let top_k = request.params.top_k.min(candidates.len());
            for (doc_id, similarity, distance, vector) in candidates.into_iter().take(top_k) {
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
                        field_name: field_name.clone(),
                        similarity,
                        distance,
                        vector: vector_output,
                    });
            }
        } else {
            // Unfiltered path: each doc may belong to a different field, so the
            // field name must travel with each candidate.
            let candidates_list = self.index_reader.vector_ids()?;
            results.candidates_examined = self.index_reader.vector_count();

            let mut candidates: Vec<(u64, String, f32, f32, Vector)> =
                Vec::with_capacity(candidates_list.len());
            for (doc_id, field_name) in candidates_list {
                if let Ok(Some(vector)) = self.index_reader.get_vector(doc_id, &field_name) {
                    let distance = metric.distance(&request.query.data, &vector.data)?;
                    let similarity = metric.distance_to_similarity(distance);
                    candidates.push((doc_id, field_name, similarity, distance, vector));
                }
            }

            candidates.sort_unstable_by(|a, b| {
                b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
            });

            let top_k = request.params.top_k.min(candidates.len());
            for (doc_id, field_name, similarity, distance, vector) in
                candidates.into_iter().take(top_k)
            {
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
        }

        results.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(results)
    }

    fn count(&self, request: VectorIndexQuery) -> Result<u64> {
        // For a field-filtered count, use the pre-built per-field index
        // (#405); avoids allocating + iterating the full `vector_ids`.
        if let Some(ref field_name) = request.field_name {
            Ok(self.index_reader.doc_ids_for_field(field_name).len() as u64)
        } else {
            Ok(self.index_reader.vector_ids()?.len() as u64)
        }
    }
}
