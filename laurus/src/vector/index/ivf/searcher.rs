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
    /// A `Vec` of `(doc_id, field_id)` pairs from the probed clusters
    /// (ids index the reader's field dictionary, Issue #633 PR-B),
    /// optionally filtered by `field_name` — resolved once to an id, so
    /// the per-candidate filter is an integer compare.
    fn probe_clusters(
        &self,
        query: &Vector,
        n_probe: usize,
        field_name: Option<&str>,
    ) -> Result<Vec<(u64, u16)>> {
        use super::reader::IvfIndexReader;

        if let Some(ivf_reader) = self.index_reader.as_any().downcast_ref::<IvfIndexReader>() {
            let centroids = ivf_reader.centroids();
            let distance_metric = self.index_reader.distance_metric();

            if centroids.is_empty() {
                return Ok(Vec::new());
            }

            // Distance to every centroid. Kept serial: each item is a single
            // distance computation, so the candidate-scan rayon path (#662)
            // adds more per-job dispatch overhead than it saves here — at
            // K = 2048 parallelising this scan measured ~+9% slower (Issue
            // #668). The centroid count K is ~√N in practice, so the serial
            // scan stays well below where parallelism would pay off.
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

            // Select the `n_probe` nearest centroids. Their relative order is
            // irrelevant — `search()` re-sorts the collected candidates by
            // similarity afterwards — so a full sort (O(K log K)) is wasteful.
            // Partition around the `n`-th smallest distance in O(K) with
            // `select_nth_unstable_by` instead (Issue #668, suggested fix (1)).
            let n = n_probe.min(centroid_distances.len());
            if n == 0 {
                return Ok(Vec::new());
            }
            if n < centroid_distances.len() {
                centroid_distances.select_nth_unstable_by(n - 1, |a, b| a.1.total_cmp(&b.1));
            }

            // Resolve the requested field once; per-candidate filtering
            // below is then a u16 compare (Issue #633 PR-B).
            let target = match field_name {
                Some(field) => {
                    match crate::vector::index::format::resolve_field_id(
                        &ivf_reader.field_dict(),
                        field,
                    ) {
                        Some(fid) => Some(fid),
                        // Unknown field ⇒ no candidates.
                        None => return Ok(Vec::new()),
                    }
                }
                None => None,
            };

            // Collect vector IDs from the `n` nearest clusters.
            let mut result = Vec::new();
            for &(cluster_idx, _) in centroid_distances.iter().take(n) {
                let cluster_vecs = ivf_reader.cluster_vectors(cluster_idx);
                if let Some(target) = target {
                    result.extend(cluster_vecs.iter().filter(|&&(_, f)| f == target).copied());
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

        // `request.filter` (Issue #645 allow-set) is honoured inline (Issue
        // #740): candidates whose `doc_id` is not in the set are skipped before
        // the distance kernel, saving the distance computation for a selective
        // filter. The store's post-filter still runs but becomes a no-op for
        // the already-filtered results, so recall is unchanged.

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

        // The probe above already rejected non-IVF readers, so the
        // downcast below is guaranteed to succeed; the dictionary maps
        // the probe results' u16 ids back to names at emission.
        let field_dict = self
            .index_reader
            .as_any()
            .downcast_ref::<crate::vector::index::ivf::reader::IvfIndexReader>()
            .map(|r| r.field_dict())
            .unwrap_or_default();

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

        // Filter-aware allow-set honoured inline (Issue #740). Borrowed once
        // before the scan; `&AHashSet` is `Send + Sync`, so it composes with
        // the rayon-parallel candidate loop below.
        let filter = request.filter.as_deref();

        // Distance scan over the probed clusters, parallelised across
        // candidates above PARALLEL_SCAN_THRESHOLD (#662). The quantized hot
        // path and the f32 fallback both run inside the per-candidate closure.
        let mut candidates: Vec<(u64, u16, f32, f32, Vector)> =
            crate::vector::search::searcher::parallel_scan(&vector_ids[..], |(doc_id, fid)| {
                let field_name: &str = &field_dict[*fid as usize];
                // Skip non-matching candidates before the distance kernel.
                if let Some(allowed) = filter
                    && !allowed.contains(*doc_id)
                {
                    return Ok(None);
                }
                if let (Some(pool), Some(prepared)) = (&quant_pool, &prepared_quantized)
                    && let Some((int8, meta)) = pool.get_record(*doc_id, field_name)
                    && !ivf_reader.is_some_and(|r| r.is_deleted(*doc_id))
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
                    return Ok(Some((*doc_id, *fid, similarity, distance, vector)));
                }
                if let Ok(Some(vector)) = self.index_reader.get_vector(*doc_id, field_name) {
                    let distance = metric.distance_with_prepared(&prepared_query, &vector.data)?;
                    let similarity = metric.distance_to_similarity(distance);
                    return Ok(Some((*doc_id, *fid, similarity, distance, vector)));
                }
                Ok(None)
            })?;

        // Sort by similarity (descending)
        candidates.sort_unstable_by(|a, b| b.2.total_cmp(&a.2));

        // Take top_k results
        let candidates_len = candidates.len();
        let top_k = request.params.top_k.min(candidates_len);
        for (doc_id, fid, similarity, distance, vector) in candidates.into_iter().take(top_k) {
            // Rehydrate the field name only for emitted hits.
            let field_name = field_dict[fid as usize].to_string();
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

#[cfg(test)]
mod tests {
    //! Unit tests for `probe_clusters` partial-selection correctness (Issue
    //! #668). The child module can reach the private `probe_clusters`, so the
    //! selected cluster *set* is asserted directly — independent of `top_k` /
    //! `min_similarity` filtering in the public `search()` path.

    use std::collections::BTreeSet;
    use std::sync::Arc;

    use super::IvfSearcher;
    use crate::storage::memory::MemoryStorage;
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::core::vector::Vector;
    use crate::vector::index::config::IvfIndexConfig;
    use crate::vector::index::ivf::reader::IvfIndexReader;
    use crate::vector::index::ivf::writer::IvfIndexWriter;
    use crate::vector::reader::VectorIndexReader;
    use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

    /// Build 12 well-separated singleton clusters (one vector per centroid,
    /// spaced 1000 units apart) and return a loaded reader. Because each
    /// cluster holds exactly one vector whose `doc_id == position / 1000 - 1`,
    /// the set of vectors `probe_clusters` returns is a direct read-out of
    /// which centroids were selected.
    fn singleton_cluster_reader(name: &str) -> Arc<dyn VectorIndexReader> {
        let storage = Arc::new(MemoryStorage::default());
        let config = IvfIndexConfig {
            dimension: 2,
            distance_metric: DistanceMetric::Euclidean,
            n_clusters: 12,
            n_probe: 1,
            normalize_vectors: false,
            ..IvfIndexConfig::default()
        };
        let mut writer = IvfIndexWriter::with_storage(
            config,
            VectorIndexWriterConfig::default(),
            name,
            storage.clone(),
        )
        .unwrap();

        let vectors: Vec<(u64, String, Vector)> = (0..12)
            .map(|i| {
                (
                    i as u64,
                    "f".to_string(),
                    Vector::new(vec![(i as f32 + 1.0) * 1000.0, 0.0]),
                )
            })
            .collect();
        writer.build(vectors).unwrap();
        writer.finalize().unwrap();
        writer.write().unwrap();

        let reader = IvfIndexReader::load(storage, name, DistanceMetric::Euclidean).unwrap();
        assert_eq!(
            reader.centroids().len(),
            12,
            "k-means should produce one centroid per well-separated input"
        );
        Arc::new(reader)
    }

    fn probed_doc_ids(searcher: &IvfSearcher, query: &Vector, n_probe: usize) -> BTreeSet<u64> {
        searcher
            .probe_clusters(query, n_probe, None)
            .unwrap()
            .into_iter()
            .map(|(doc_id, _)| doc_id)
            .collect()
    }

    /// `select_nth_unstable_by` must keep the *nearest* `n_probe` centroids,
    /// not an arbitrary `n_probe` subset. Centroids sit at x = 1000..=12000;
    /// a query at x = 1000 has them in strictly increasing distance order, so
    /// the nearest three are docs {0, 1, 2}.
    #[test]
    fn probe_clusters_selects_nearest_set() {
        let reader = singleton_cluster_reader("test_probe_nearest_set");
        let searcher = IvfSearcher::with_n_probe(reader, 1).unwrap();

        let near = Vector::new(vec![1000.0, 0.0]);
        assert_eq!(
            probed_doc_ids(&searcher, &near, 3),
            BTreeSet::from([0, 1, 2])
        );

        // Querying from the far end selects the far centroids instead.
        let far = Vector::new(vec![12000.0, 0.0]);
        assert_eq!(probed_doc_ids(&searcher, &far, 2), BTreeSet::from([10, 11]));
    }

    /// `n_probe >= number of centroids` probes every cluster (the
    /// `select_nth_unstable_by` step is skipped). Covers both `n_probe == K`
    /// and `n_probe > K`.
    #[test]
    fn probe_clusters_n_probe_ge_k_probes_all() {
        let reader = singleton_cluster_reader("test_probe_ge_k");
        let searcher = IvfSearcher::with_n_probe(reader, 1).unwrap();
        let query = Vector::new(vec![1000.0, 0.0]);

        let all: BTreeSet<u64> = (0..12).collect();
        assert_eq!(probed_doc_ids(&searcher, &query, 12), all);
        assert_eq!(probed_doc_ids(&searcher, &query, 20), all);
    }

    /// `n_probe == 0` probes nothing (the early-return guard).
    #[test]
    fn probe_clusters_n_probe_zero_returns_empty() {
        let reader = singleton_cluster_reader("test_probe_zero");
        let searcher = IvfSearcher::with_n_probe(reader, 1).unwrap();
        let query = Vector::new(vec![1000.0, 0.0]);

        assert!(probed_doc_ids(&searcher, &query, 0).is_empty());
    }
}
