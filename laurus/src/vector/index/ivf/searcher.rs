//! IVF vector searcher for memory-efficient approximate search.

use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::vector::core::vector::Vector;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryResults};

/// Fallback `n_probe` used when no schema-level value is supplied.
///
/// Matches the
/// [`IvfIndexConfig`](crate::vector::index::config::IvfIndexConfig) default,
/// so a searcher built via [`IvfSearcher::new`] probes only the single
/// nearest cluster. Callers that go through the index
/// [`searcher()`](crate::vector::index::VectorIndex::searcher) factory
/// instead receive the configured `n_probe` via [`IvfSearcher::with_n_probe`].
const IVF_DEFAULT_N_PROBE: usize = 1;

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
    /// Create a new IVF searcher with the built-in fallback `n_probe`
    /// ([`IVF_DEFAULT_N_PROBE`] = 1).
    ///
    /// Most callers should construct the searcher through the index
    /// [`searcher()`](crate::vector::index::VectorIndex::searcher) factory,
    /// which threads the schema-level
    /// [`IvfIndexConfig::n_probe`](crate::vector::index::config::IvfIndexConfig::n_probe)
    /// in via [`Self::with_n_probe`]. The number of probed clusters can still
    /// be adjusted afterwards with [`Self::set_n_probe`].
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
        Self::with_n_probe(index_reader, IVF_DEFAULT_N_PROBE)
    }

    /// Create a new IVF searcher that probes the configured number of
    /// nearest clusters.
    ///
    /// This is the constructor used by the index
    /// [`searcher()`](crate::vector::index::VectorIndex::searcher) factory so
    /// the schema-level
    /// [`IvfIndexConfig::n_probe`](crate::vector::index::config::IvfIndexConfig::n_probe)
    /// is honoured at search time (Issue
    /// [#741](https://github.com/mosuka/laurus/issues/741)). Before this, the
    /// searcher always probed a single cluster regardless of configuration.
    ///
    /// # Arguments
    ///
    /// * `index_reader` - The underlying vector index reader (must be an
    ///   [`IvfIndexReader`](super::reader::IvfIndexReader)).
    /// * `n_probe` - Number of nearest clusters to probe during search.
    ///   Higher values improve recall at the cost of query latency. The
    ///   effective count is capped at the number of available clusters.
    ///
    /// # Returns
    ///
    /// A new `IvfSearcher` instance.
    pub fn with_n_probe(index_reader: Arc<dyn VectorIndexReader>, n_probe: usize) -> Result<Self> {
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

        // `request.filter` (Issue #645 filter-aware traversal) is intentionally
        // ignored here: the IVF scan over the probed clusters is exhaustive, so
        // the store's post-filter discards non-matching docs without recall
        // loss. Honouring it inline is a follow-up optimisation.

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

        // Probe the configured number of nearest clusters (Issue #741).
        // `probe_clusters` already caps the effective count at the number of
        // available centroids via `take`, so no artificial upper clamp is
        // applied here.
        let vector_ids =
            self.probe_clusters(&request.query, self.n_probe, request.field_name.as_deref())?;

        // Calculate distances for vectors in the probed clusters.
        // Cache the query-side norm once per search (#414); for Cosine /
        // Angular this skips the per-candidate `||query||²` accumulation.
        let metric = self.index_reader.distance_metric();
        let prepared_query = metric.prepare_query(&request.query.data);

        // Issue #481 Stage 1, Step 7: try the int8 hot path when the
        // reader holds an OwnedQuantized pool. Build per-search
        // QuantizedQuery once before the candidate loop.
        let ivf_reader = self
            .index_reader
            .as_any()
            .downcast_ref::<crate::vector::index::ivf::reader::IvfIndexReader>();
        let quant_pool = ivf_reader.and_then(|r| r.vectors().quantized_pool().cloned());
        let prepared_quantized = quant_pool.as_ref().map(|pool| {
            crate::vector::core::distance_quantized::QuantizedQuery::prepare(
                &request.query.data,
                &pool.params,
            )
        });

        // Distance scan over the probed clusters, parallelised across
        // candidates above PARALLEL_SCAN_THRESHOLD (#662). The quantized hot
        // path and the f32 fallback both run inside the per-candidate closure.
        let mut candidates: Vec<(u64, String, f32, f32, Vector)> =
            crate::vector::search::searcher::parallel_scan(
                &vector_ids[..],
                |(doc_id, field_name)| {
                    if let (Some(pool), Some(prepared)) = (&quant_pool, &prepared_quantized)
                        && let Some((int8, meta)) = pool.get_record(*doc_id, field_name)
                    {
                        let distance = crate::vector::core::distance_quantized::distance_quantized(
                            metric, prepared, int8, meta,
                        );
                        let similarity = metric.distance_to_similarity(distance);
                        let vector = if request.params.include_vectors {
                            pool.dequantize_to_vector(*doc_id, field_name)
                                .unwrap_or_else(|| Vector::new(Vec::new()))
                        } else {
                            Vector::new(Vec::new())
                        };
                        return Ok(Some((
                            *doc_id,
                            field_name.clone(),
                            similarity,
                            distance,
                            vector,
                        )));
                    }
                    if let Ok(Some(vector)) = self.index_reader.get_vector(*doc_id, field_name) {
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
