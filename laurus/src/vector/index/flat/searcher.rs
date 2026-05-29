//! Flat vector searcher for exact search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::distance_quantized::{QuantizedQuery, distance_quantized};
use crate::vector::core::vector::Vector;
use crate::vector::index::flat::reader::FlatVectorIndexReader;
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

        // `request.filter` (Issue #645 filter-aware traversal) is intentionally
        // ignored here: the flat scan is exhaustive, so the store's post-filter
        // already discards non-matching docs without losing recall. Honouring
        // it inline is a follow-up optimisation, not a correctness need.

        // Issue #481 Stage 2 (rerank) -- API surface only in Stage 1.
        if request.params.rerank_factor.is_some() {
            return Err(crate::error::LaurusError::NotImplemented(
                "Two-stage rerank (Issue #481 Stage 2) is not yet implemented. \
                 Pass rerank_factor = None for the Stage 1 quantized search."
                    .to_string(),
            ));
        }

        let start = Timer::now();
        let mut results = VectorIndexQueryResults::new();
        let metric = self.index_reader.distance_metric();
        // Cache the query-side norm once per search (#414): for Cosine /
        // Angular this skips one `||query||²` accumulation per
        // candidate; for the other metrics the prepared variant
        // forwards to `distance` and the cached value is unused.
        let prepared_query = metric.prepare_query(&request.query.data);

        // Issue #481 Stage 1, Step 7: try the int8 hot path for the
        // field-filtered case if the reader holds an OwnedQuantized
        // pool. Build per-search QuantizedQuery + per-field position
        // index once before the candidate loop.
        let flat_reader = self
            .index_reader
            .as_any()
            .downcast_ref::<FlatVectorIndexReader>();
        let quant_pool = flat_reader.and_then(|r| r.vectors().quantized_pool().cloned());

        if let Some(ref field_name) = request.field_name {
            // Field-filtered path: fetch the per-field doc-id slice from the
            // reader's pre-built index (#405 — O(1) Arc clone, avoids the full
            // `Vec<(u64, String)>` clone and the linear filter scan). Since
            // every candidate shares the same `field_name`, do not store it
            // per-candidate; clone it only when constructing the top_k
            // results.
            let ids = self.index_reader.doc_ids_for_field(field_name);
            results.candidates_examined = ids.len();

            // Step-7 hot path: prepare quantized query once and look
            // up per-field doc_id -> position once per search.
            let quant_ctx = quant_pool.as_ref().and_then(|pool| {
                pool.field_position_index(field_name).map(|idx| {
                    let prepared = QuantizedQuery::prepare(&request.query.data, &pool.params);
                    (prepared, pool.clone(), idx)
                })
            });

            // Distance scan, parallelised across candidates above
            // PARALLEL_SCAN_THRESHOLD (#662). The quantized hot path and the
            // f32 fallback both run inside the per-candidate closure; a
            // missing vector yields `Ok(None)` (skipped) and a dimension
            // mismatch propagates as `Err`.
            let mut candidates: Vec<(u64, f32, f32, Vector)> =
                crate::vector::search::searcher::parallel_scan(&ids[..], |&doc_id| {
                    if let Some((prepared, pool, idx)) = &quant_ctx
                        && let Some(&pos) = idx.get(&doc_id)
                    {
                        let (int8, meta) = pool.record_at(pos);
                        let distance = distance_quantized(metric, prepared, int8, meta);
                        let similarity = metric.distance_to_similarity(distance);
                        // include_vectors path still needs the f32 vector;
                        // dequantize lazily only when requested.
                        let vector = if request.params.include_vectors {
                            pool.dequantize_to_vector(doc_id, field_name)
                                .unwrap_or_else(|| Vector::new(Vec::new()))
                        } else {
                            Vector::new(Vec::new())
                        };
                        return Ok(Some((doc_id, similarity, distance, vector)));
                    }
                    if let Ok(Some(vector)) = self.index_reader.get_vector(doc_id, field_name) {
                        let distance =
                            metric.distance_with_prepared(&prepared_query, &vector.data)?;
                        let similarity = metric.distance_to_similarity(distance);
                        return Ok(Some((doc_id, similarity, distance, vector)));
                    }
                    Ok(None)
                })?;

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

            // Distance scan, parallelised across candidates above
            // PARALLEL_SCAN_THRESHOLD (#662). Each candidate may belong to a
            // different field, so the field name travels with the result.
            let mut candidates: Vec<(u64, String, f32, f32, Vector)> =
                crate::vector::search::searcher::parallel_scan(
                    &candidates_list[..],
                    |(doc_id, field_name)| {
                        if let Ok(Some(vector)) = self.index_reader.get_vector(*doc_id, field_name)
                        {
                            let distance =
                                metric.distance_with_prepared(&prepared_query, &vector.data)?;
                            let similarity = metric.distance_to_similarity(distance);
                            return Ok(Some((
                                *doc_id,
                                field_name.clone(),
                                similarity,
                                distance,
                                vector,
                            )));
                        }
                        Ok(None)
                    },
                )?;

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
