//! HNSW vector searcher for approximate search.

use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::vector::Vector;
use crate::vector::index::hnsw::graph::HnswGraph;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::search::searcher::{
    VectorIndexSearchRequest, VectorIndexSearchResults, VectorSearchResult,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

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
    fn search(&self, request: &VectorIndexSearchRequest) -> Result<VectorIndexSearchResults> {
        use std::time::Instant;

        let start = Instant::now();

        // correct approach: usage of downcast_ref to check if we can use graph search
        if let Some(reader) = self.index_reader.as_any().downcast_ref::<HnswIndexReader>() {
            if let Some(graph) = &reader.graph {
                if let Some(ref field_name) = request.field_name {
                    // Perform Graph Search
                    let mut results = self.search_graph(reader, graph, request, field_name)?;
                    results.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
                    return Ok(results);
                }
            }
        }

        // Fallback to Linear Scan
        let mut results = VectorIndexSearchResults::new();

        let ef_search = self.ef_search;

        let mut vector_ids = self.index_reader.vector_ids()?;

        // Filter by field_name if specified
        if let Some(ref field_name) = request.field_name {
            vector_ids.retain(|(_, fname)| fname == field_name);
        }

        let max_candidates = ef_search.min(vector_ids.len());
        results.candidates_examined = max_candidates;

        let mut candidates: Vec<(u64, String, f32, f32, Vector)> =
            Vec::with_capacity(max_candidates);

        for (doc_id, field_name) in vector_ids.iter().take(max_candidates) {
            if let Ok(Some(vector)) = self.index_reader.get_vector(*doc_id, field_name) {
                let similarity = self
                    .index_reader
                    .distance_metric()
                    .similarity(&request.query.data, &vector.data)?;
                let distance = self
                    .index_reader
                    .distance_metric()
                    .distance(&request.query.data, &vector.data)?;
                candidates.push((*doc_id, field_name.clone(), similarity, distance, vector));
            }
        }

        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = request.params.top_k.min(candidates.len());
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
        request: &VectorIndexSearchRequest,
        field_name: &str,
    ) -> Result<VectorIndexSearchResults> {
        let entry_point = match graph.entry_point {
            Some(ep) => ep,
            None => return Ok(VectorIndexSearchResults::new()),
        };

        let query = &request.query;
        let ef_search = self.ef_search;

        // 1. Start from entry point at max_level
        let mut curr_obj = entry_point;
        // Note: Assuming entry_point is in field_name. If not, we might fail to get vector.
        // If doc_id corresponds to field_name, we get vector.
        // Since HNSW here is single-graph for mixed IDs (potentially), we must hope entry point is valid for calc_dist with this field?
        // Ref discussion: assuming HnswIndex is single-field.

        let mut dist = self.calc_dist(reader, query, curr_obj, field_name)?;

        // 2. Greedy descent
        for lc in (1..=graph.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if let Some(neighbors) = graph.get_neighbors(curr_obj, lc) {
                    for &neighbor_id in neighbors {
                        let d = self.calc_dist(reader, query, neighbor_id, field_name)?;
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

        let mut visited = HashSet::new();
        visited.insert(curr_obj);

        while let Some(curr) = candidates.pop() {
            if let Some(furthest) = found.peek() {
                if curr.distance > furthest.distance && found.len() >= ef_search {
                    break;
                }
            }

            if let Some(neighbors) = graph.get_neighbors(curr.id, 0) {
                for &neighbor_id in neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);

                    let d = self.calc_dist(reader, query, neighbor_id, field_name)?;
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

        // Convert found heaps to results
        let mut final_results = Vec::new();
        for c in found {
            if let Ok(Some(vector)) = reader.get_vector(c.id, field_name) {
                // Recalculate similarity? Or deduce from distance?
                // DistanceMetric::similarity is not strictly inverse of distance for all metrics,
                // but Reader knows metric.
                let similarity = reader
                    .distance_metric()
                    .similarity(&query.data, &vector.data)?;

                // Apply min_score
                if similarity < request.params.min_similarity {
                    continue;
                }

                final_results.push(VectorSearchResult {
                    doc_id: c.id,
                    field_name: field_name.to_string(),
                    similarity,
                    distance: c.distance,
                    metadata: vector.metadata.clone(),
                    vector: if request.params.include_vectors {
                        Some(vector)
                    } else {
                        None
                    },
                });
            }
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

        Ok(VectorIndexSearchResults {
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
        doc_id: u64,
        field_name: &str,
    ) -> Result<f32> {
        // Optimization: HnswIndexReader *could* support getting raw bytes or avoiding clone,
        // but get_vector returns Option<Vector>.
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
}
