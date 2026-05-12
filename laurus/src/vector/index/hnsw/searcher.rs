//! HNSW vector searcher for approximate search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::distance_quantized::{QuantizedQuery, distance_quantized};
use crate::vector::core::vector::Vector;
use crate::vector::index::hnsw::graph::HnswGraph;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::quantized_storage::QuantizedVectorPool;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::search::searcher::{
    VectorIndexQuery, VectorIndexQueryResult, VectorIndexQueryResults,
};
use bit_vec::BitVec;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Per-search state for the int8 hot path (Issue #481 Stage 1, Step 6).
///
/// Built once at the start of `search_graph` when the reader is
/// [`crate::vector::index::storage::VectorStorage::OwnedQuantized`].
/// Threaded through `calc_dist` so each per-candidate call is one
/// O(1) `field_idx.get` plus a [`distance_quantized`] invocation —
/// no per-call allocation, no `String` clone.
struct QuantizedSearchCtx {
    /// Quantized query (int8 + cached norm + offset/scale), prepared
    /// once per search via [`QuantizedQuery::prepare`].
    prepared: QuantizedQuery,
    /// The reader's in-memory int8 storage. Cloned `Arc` so the pool
    /// stays alive even if the reader is dropped mid-search (it isn't,
    /// but the borrow checker needs the lifetime extension).
    pool: Arc<QuantizedVectorPool>,
    /// Per-field doc_id -> vector position in `pool.data`. None when
    /// the searched field is absent from this segment (impossible for
    /// the HNSW graph search path, but kept for type symmetry).
    field_idx: Arc<HashMap<u64, u32>>,
    /// Distance metric, cached so the hot loop skips the
    /// `reader.distance_metric()` indirection.
    metric: DistanceMetric,
}

impl QuantizedSearchCtx {
    /// Compute distance from the prepared query to the candidate at
    /// `doc_id`. Returns `f32::MAX` if the candidate is missing, which
    /// is what HNSW's calc_dist expects for absent neighbours.
    #[inline]
    fn distance(&self, doc_id: u64) -> f32 {
        match self.field_idx.get(&doc_id) {
            Some(&pos) => {
                let (int8, meta) = self.pool.record_at(pos);
                distance_quantized(self.metric, &self.prepared, int8, meta)
            }
            None => f32::MAX,
        }
    }
}

/// HNSW vector searcher that performs approximate nearest neighbor search.
#[derive(Debug)]
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

impl VectorIndexSearcher for HnswSearcher {
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        use crate::util::time::Timer;

        // Stage 2 (Issue #481): rerank_factor is honored on the HNSW
        // graph path when the reader has a rerank storage pool loaded
        // (`reader.rerank_storage().is_some()`). Otherwise the value
        // is silently ignored: there is no f32 information to recover
        // for Stage 1 segments or for the brute-force fallback below
        // (which already runs against the dequantized f32 vectors).

        let start = Timer::now();

        // correct approach: usage of downcast_ref to check if we can use graph search
        if let Some(reader) = self.index_reader.as_any().downcast_ref::<HnswIndexReader>()
            && let Some(graph) = &reader.graph
            && let Some(ref field_name) = request.field_name
        {
            // Perform Graph Search
            let mut results = self.search_graph(reader, graph, request, field_name)?;
            results.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
            return Ok(results);
        }

        // Fallback to Linear Scan (brute-force over all vectors)
        let mut results = VectorIndexQueryResults::new();
        let metric = self.index_reader.distance_metric();
        // Cache the query-side norm once per search (#414): for Cosine /
        // Angular this skips one `||query||²` accumulation per
        // candidate; other metrics fall back to the unprepared path.
        let prepared_query = metric.prepare_query(&request.query.data);

        if let Some(ref field_name) = request.field_name {
            // Field-filtered path: fetch the per-field doc-id slice from the
            // reader's pre-built index (#405 — O(1) Arc clone, avoids the full
            // `Vec<(u64, String)>` clone and the linear retain scan). Every
            // candidate shares the same `field_name`, so it is not stored
            // per-candidate; it is cloned only when emitting the top_k
            // results.
            let ids = self.index_reader.doc_ids_for_field(field_name);
            results.candidates_examined = ids.len();

            let mut candidates: Vec<(u64, f32, f32, Vector)> = Vec::with_capacity(ids.len());
            for &doc_id in ids.iter() {
                if let Ok(Some(vector)) = self.index_reader.get_vector(doc_id, field_name) {
                    // Compute distance once and derive similarity from it.
                    // `similarity()` would otherwise re-run the SIMD distance
                    // kernel a second time on the same pair.
                    let distance = metric.distance_with_prepared(&prepared_query, &vector.data)?;
                    let similarity = metric.distance_to_similarity(distance);
                    candidates.push((doc_id, similarity, distance, vector));
                }
            }

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

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
            // Unfiltered path: docs may belong to different fields, so the
            // field name must travel with each candidate.
            let candidates_list = self.index_reader.vector_ids()?;
            results.candidates_examined = candidates_list.len();

            let mut candidates: Vec<(u64, String, f32, f32, Vector)> =
                Vec::with_capacity(candidates_list.len());
            for (doc_id, field_name) in candidates_list.iter() {
                if let Ok(Some(vector)) = self.index_reader.get_vector(*doc_id, field_name) {
                    let distance = metric.distance_with_prepared(&prepared_query, &vector.data)?;
                    let similarity = metric.distance_to_similarity(distance);
                    candidates.push((*doc_id, field_name.clone(), similarity, distance, vector));
                }
            }

            candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

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
        // Field-filtered counts use the pre-built per-field index (#405);
        // avoids allocating + linear-filtering the full `vector_ids`.
        if let Some(ref field_name) = request.field_name {
            Ok(self.index_reader.doc_ids_for_field(field_name).len() as u64)
        } else {
            Ok(self.index_reader.vector_ids()?.len() as u64)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    id: u64,
    distance: f32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance > larger distance for Visitor (nearest first)
        // But for Result (Found), we might want Max-heap (furthest first) to keep ef smallest.
        // HNSW logic typically uses Min-heap for "candidates to visit" and Max-heap for "dynamic list of found nearest"
        // Here we define one Candidate struct. Let's assume standard PartialOrd (smaller < larger).
        // Then BinaryHeap is MaxHeap (largest at top).

        // This impl makes BinaryHeap a MIN-HEAP (smallest distance at top)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ResultCandidate {
    id: u64,
    distance: f32,
}

impl Eq for ResultCandidate {}
impl Ord for ResultCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance at top (to remove worst)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for ResultCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl HnswSearcher {
    fn search_graph(
        &self,
        reader: &HnswIndexReader,
        graph: &HnswGraph,
        request: &VectorIndexQuery,
        field_name: &str,
    ) -> Result<VectorIndexQueryResults> {
        let entry_point = match graph.entry_point {
            Some(ep) => ep,
            None => return Ok(VectorIndexQueryResults::new()),
        };

        let query = &request.query;
        let ef_search = self.ef_search;

        // Issue #481 Stage 1, Step 6: when the reader holds the
        // OwnedQuantized in-memory pool, prepare the int8 hot path
        // here so the per-candidate calc_dist call collapses into one
        // O(1) doc_id lookup + distance_quantized (int8 SIMD).
        let quant_ctx: Option<QuantizedSearchCtx> =
            reader.vectors().quantized_pool().and_then(|pool| {
                pool.field_position_index(field_name).map(|field_idx| {
                    let prepared = QuantizedQuery::prepare(&query.data, &pool.params);
                    QuantizedSearchCtx {
                        prepared,
                        pool: pool.clone(),
                        field_idx,
                        metric: reader.distance_metric(),
                    }
                })
            });

        // Retrieve the per-field prefetch index once per search call (O(1), no allocation).
        // `None` for on-demand (disk-backed) storage; the prefetch loop is skipped entirely.
        let field_prefetch = reader.field_prefetch_index(field_name);
        // Prefetch payload size: int8 record (dim + 8 bytes meta) for
        // the quantized hot path; legacy f32 size otherwise.
        let prefetch_n_bytes = if quant_ctx.is_some() {
            QuantizedVectorPool::record_size(reader.dimension())
        } else {
            reader.dimension() * std::mem::size_of::<f32>()
        };

        // 1. Start from entry point at max_level
        let mut curr_obj = entry_point;
        // Note: Assuming entry_point is in field_name. If not, we might fail to get vector.
        // If doc_id corresponds to field_name, we get vector.
        // Since HNSW here is single-graph for mixed IDs (potentially), we must hope entry point is valid for calc_dist with this field?
        // Ref discussion: assuming HnswIndex is single-field.

        let mut dist = self.calc_dist(reader, query, quant_ctx.as_ref(), curr_obj, field_name)?;

        // 2. Greedy descent
        for lc in (1..=graph.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(neighbors) = graph.get_neighbors(curr_obj, lc) {
                    // Pass 1: issue prefetch hints for all neighbors before computing
                    // distances.  For datasets larger than L3 cache this hides the
                    // memory latency of loading Vec<f32> data.
                    if let Some(idx) = field_prefetch {
                        for &neighbor_id in neighbors {
                            Self::prefetch_neighbor(idx, neighbor_id, prefetch_n_bytes);
                        }
                    }
                    // Pass 2: compute distances (data is being fetched in the background).
                    for &neighbor_id in neighbors {
                        let d = self.calc_dist(
                            reader,
                            query,
                            quant_ctx.as_ref(),
                            neighbor_id,
                            field_name,
                        )?;
                        if d < dist {
                            dist = d;
                            curr_obj = neighbor_id;
                            changed = true;
                        }
                    }
                }
            }
        }

        // 3. Search at layer 0 with ef_search
        let mut candidates = BinaryHeap::new(); // Min-heap (nearest first)
        let mut found = BinaryHeap::new(); // Max-heap (furthest first)

        candidates.push(Candidate {
            id: curr_obj,
            distance: dist,
        });
        found.push(ResultCandidate {
            id: curr_obj,
            distance: dist,
        });

        // Visited set as a dense bitmap. doc_ids in laurus are assigned
        // sequentially from 0, so the bitmap is sized to fit every node
        // currently in the graph (`max_doc_id + 1` bits ≈ N / 8 bytes for
        // N nodes). `BitVec::get` / `BitVec::set` are a single array
        // index + bit op, materially cheaper than `HashSet<u64>`'s hash
        // + bucket lookup that the audit (#406) flagged for ef_search
        // graph traversals which examine hundreds-to-thousands of nodes.
        let mut visited = BitVec::from_elem(graph.max_doc_id() as usize + 1, false);
        visited.set(curr_obj as usize, true);

        while let Some(curr) = candidates.pop() {
            if let Some(furthest) = found.peek()
                && curr.distance > furthest.distance
                && found.len() >= ef_search
            {
                break;
            }

            if let Some(neighbors) = graph.get_neighbors(curr.id, 0) {
                // Pass 1: issue prefetch hints for unvisited neighbors.
                // O(1) per neighbor (u64 HashMap lookup, no allocation).
                if let Some(idx) = field_prefetch {
                    for &neighbor_id in neighbors {
                        if !visited.get(neighbor_id as usize).unwrap_or(false) {
                            Self::prefetch_neighbor(idx, neighbor_id, prefetch_n_bytes);
                        }
                    }
                }

                // Pass 2: compute distances for unvisited neighbors (data loading
                // overlaps with the prefetch hints issued above).
                for &neighbor_id in neighbors {
                    let nbr_idx = neighbor_id as usize;
                    if visited.get(nbr_idx).unwrap_or(false) {
                        continue;
                    }
                    visited.set(nbr_idx, true);

                    let d =
                        self.calc_dist(reader, query, quant_ctx.as_ref(), neighbor_id, field_name)?;
                    let furthest_dist = found.peek().map(|c| c.distance).unwrap_or(f32::MAX);

                    if d < furthest_dist || found.len() < ef_search {
                        candidates.push(Candidate {
                            id: neighbor_id,
                            distance: d,
                        });
                        found.push(ResultCandidate {
                            id: neighbor_id,
                            distance: d,
                        });

                        if found.len() > ef_search {
                            found.pop();
                        }
                    }
                }
            }
        }

        // Stage 2 (Issue #481): if the query asks for rerank
        // (`rerank_factor`) and the reader has the LRS1 sidecar loaded
        // (`reader.rerank_storage()`), widen the int8 candidate set to
        // `top_k * rerank_factor`, recompute distances against the
        // original f32 vectors, and use the new distances for the
        // ranking that the existing convert-to-results pipeline below
        // operates on. When either prerequisite is missing we silently
        // fall through to Stage 1 ranking — Stage 1 segments cannot
        // recover the f32 information that was discarded at index
        // time, so there's nothing better to do.
        let candidates_for_results: Vec<ResultCandidate> =
            match (request.params.rerank_factor, reader.rerank_storage()) {
                (Some(factor), Some(pool)) => {
                    let widened = request.params.top_k.saturating_mul(factor);
                    let int8_sorted: Vec<ResultCandidate> = found.into_sorted_vec();
                    let metric = reader.distance_metric();
                    let prepared_query = metric.prepare_query(&query.data);
                    let field_idx = pool.field_position_index(field_name);
                    let mut rescored: Vec<ResultCandidate> = Vec::with_capacity(widened);
                    for c in int8_sorted.into_iter().take(widened) {
                        let pos = match field_idx.as_ref().and_then(|idx| idx.get(&c.id).copied()) {
                            Some(p) => p,
                            None => continue,
                        };
                        let f32_slice = pool.f32_slice_at(pos);
                        let distance = metric.distance_with_prepared(&prepared_query, f32_slice)?;
                        rescored.push(ResultCandidate { id: c.id, distance });
                    }
                    rescored
                }
                _ => found.into_iter().collect(),
            };

        // Convert candidate set to results.
        let field_name_owned = field_name.to_string();
        let mut final_results = Vec::new();
        for c in candidates_for_results {
            // Convert cached distance to similarity without re-reading vectors.
            let similarity = reader.distance_metric().distance_to_similarity(c.distance);

            // Apply min_score filter.
            if similarity < request.params.min_similarity {
                continue;
            }

            // Only load vector data if explicitly requested.
            let vector = if request.params.include_vectors {
                reader.get_vector(c.id, field_name)?
            } else {
                None
            };

            final_results.push(VectorIndexQueryResult {
                doc_id: c.id,
                field_name: field_name_owned.clone(),
                similarity,
                distance: c.distance,
                vector,
            });
        }

        // Sort results (similarity descending)
        final_results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });

        // Top K
        let top_k = request.params.top_k.min(final_results.len());
        final_results.truncate(top_k);

        Ok(VectorIndexQueryResults {
            results: final_results,
            candidates_examined: visited.len(),
            search_time_ms: 0.0, // Set by caller
            query_metadata: std::collections::HashMap::new(),
        })
    }

    fn calc_dist(
        &self,
        reader: &HnswIndexReader,
        query: &Vector,
        quant_ctx: Option<&QuantizedSearchCtx>,
        doc_id: u64,
        field_name: &str,
    ) -> Result<f32> {
        // Issue #481 Stage 1, Step 6: prefer the int8 hot path when
        // the reader exposes a QuantizedVectorPool. The fallback
        // remains f32 for backward compatibility (OnDemand mode and
        // legacy f32 Owned).
        if let Some(ctx) = quant_ctx {
            return Ok(ctx.distance(doc_id));
        }
        if let Some(target) = reader.get_vector(doc_id, field_name)? {
            reader.distance_metric().distance(&query.data, &target.data)
        } else {
            // Vector not found in this field?
            // Should return max distance or error?
            // Since graph contains doc_id, it should exist.
            // But if mixed fields, it might not exist in *this* field.
            Ok(f32::MAX)
        }
    }

    /// Issue software prefetch hints for the vector identified by `doc_id`.
    ///
    /// Performs an O(1) `u64` lookup in `idx` (no `String` allocation) to
    /// retrieve the base address of the vector's `f32` data, then emits one
    /// prefetch instruction per 64-byte cache line.  This lets the CPU start
    /// fetching the data from RAM before the distance computation begins,
    /// reducing memory-latency stalls on datasets larger than L3 cache.
    ///
    /// # Safety
    ///
    /// The addresses in `idx` were recorded from `Vec<f32>::as_ptr()` at reader
    /// construction time.  The backing `Arc<Vec<f32>>` is kept alive by
    /// `VectorStorage::Owned` inside the same `HnswIndexReader`, so every
    /// pointer is valid for the entire lifetime of the search.
    /// `_mm_prefetch` / `prfm` are pure hints that never dereference the pointer.
    #[inline]
    #[allow(unused_variables)]
    fn prefetch_neighbor(idx: &HashMap<u64, usize>, doc_id: u64, n_bytes: usize) {
        if let Some(&addr) = idx.get(&doc_id) {
            let base_ptr = addr as *const i8;
            let mut offset = 0;
            while offset < n_bytes {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: see method doc comment.
                unsafe {
                    use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                    _mm_prefetch::<_MM_HINT_T0>(base_ptr.add(offset));
                }
                #[cfg(target_arch = "aarch64")]
                // SAFETY: see method doc comment.
                unsafe {
                    std::arch::asm!(
                        "prfm pldl1keep, [{p}]",
                        p = in(reg) base_ptr.add(offset),
                        options(nostack, readonly),
                    );
                }
                offset += 64;
            }
        }
    }
}
