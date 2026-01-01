//! HNSW (Hierarchical Navigable Small World) index builder for approximate search.

use std::sync::Arc;

use crate::error::{Result, SarissaError};
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use crate::vector::index::HnswIndexConfig;
use crate::vector::index::field::LegacyVectorFieldWriter;
use crate::vector::index::hnsw::graph::HnswGraph;
use crate::vector::index::io::{read_metadata, write_metadata};
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
use parking_lot::RwLock;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Abstract trait to allow reading from both HnswGraph (serial) and ConcurrentHnswGraph (parallel)
trait GraphView {
    fn get_neighbors_view(&self, doc_id: u64, level: usize) -> Option<Vec<u64>>;
}

impl GraphView for HnswGraph {
    fn get_neighbors_view(&self, doc_id: u64, level: usize) -> Option<Vec<u64>> {
        self.get_neighbors(doc_id, level).cloned()
    }
}

/// A thread-safe view of the HNSW graph during construction
struct ConcurrentHnswGraph {
    max_level: usize,
    // Map from doc_id to layers. Each layer is a RwLock-protected list of neighbors
    nodes: HashMap<u64, Vec<RwLock<Vec<u64>>>>,
}

impl ConcurrentHnswGraph {
    fn new(nodes_with_levels: Vec<(u64, usize)>, max_level: usize) -> Self {
        let mut nodes = HashMap::new();
        for (doc_id, level) in nodes_with_levels {
            // Initialize levels 0 to level with empty neighbor lists wrapped in RwLock
            let mut layers = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                layers.push(RwLock::new(Vec::new()));
            }
            nodes.insert(doc_id, layers);
        }

        Self { max_level, nodes }
    }

    fn set_neighbors(&self, doc_id: u64, level: usize, new_neighbors: Vec<u64>) {
        if let Some(layers) = self.nodes.get(&doc_id) {
            if let Some(lock) = layers.get(level) {
                *lock.write() = new_neighbors;
            }
        }
    }

    fn add_neighbor_with_pruning(
        &self,
        doc_id: u64,
        level: usize,
        neighbor_id: u64,
        max_conn: usize,
        writer: &HnswIndexWriter,
    ) -> Result<()> {
        if let Some(layers) = self.nodes.get(&doc_id) {
            if let Some(lock) = layers.get(level) {
                let mut neighbors = lock.write();
                if !neighbors.contains(&neighbor_id) {
                    neighbors.push(neighbor_id);
                }

                if neighbors.len() > max_conn {
                    // Prune while holding lock
                    let pruned = writer.prune_neighbors(doc_id, neighbors.clone(), max_conn)?;
                    *neighbors = pruned;
                }
            }
        }
        Ok(())
    }

    fn get_neighbors_raw(&self, doc_id: u64, level: usize) -> Option<Vec<u64>> {
        self.nodes
            .get(&doc_id)
            .and_then(|layers| layers.get(level).map(|lock| lock.read().clone()))
    }
}

impl GraphView for ConcurrentHnswGraph {
    fn get_neighbors_view(&self, doc_id: u64, level: usize) -> Option<Vec<u64>> {
        self.get_neighbors_raw(doc_id, level)
    }
}

/// Builder for HNSW vector indexes (approximate search).
#[derive(Debug)]
pub struct HnswIndexWriter {
    index_config: HnswIndexConfig,
    writer_config: VectorIndexWriterConfig,
    storage: Option<Arc<dyn Storage>>,
    _ml: f64, // Level normalization factor
    vectors: Vec<(u64, String, Vector)>,
    // Map from doc_id to index in vectors for fast access
    doc_id_map: std::collections::HashMap<u64, usize>,
    graph: Option<HnswGraph>,
    is_finalized: bool,
    total_vectors_to_add: Option<usize>,
    next_vec_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    id: u64,
    distance: f32,
    similarity: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (nearest first) or max-heap depending on usage.
        // For keeping top-K nearest, we usually want max-heap to pop largest distance.
        // But let's define standard ordering: smaller distance = smaller.
        // Wait, for BinaryHeap in Rust, it's a max-heap.
        // If we want smallest distance at top, we need reverse.
        // If we want largest distance at top (to remove worst candidate), we use standard.
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl HnswIndexWriter {
    /// Create a new HNSW index builder.
    pub fn new(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
    ) -> Result<Self> {
        Ok(Self {
            index_config,
            writer_config,
            storage: None,
            _ml: 1.0 / (2.0_f64).ln(), // 1/ln(2)
            vectors: Vec::new(),
            doc_id_map: std::collections::HashMap::new(),
            graph: None,
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Create a new HNSW index builder with storage.
    pub fn with_storage(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            _ml: 1.0 / (2.0_f64).ln(),
            vectors: Vec::new(),
            doc_id_map: std::collections::HashMap::new(),
            graph: None,
            is_finalized: false,
            total_vectors_to_add: None,
            next_vec_id: 0,
        })
    }

    /// Convert this writer into a doc-centric field writer adapter.
    pub fn into_field_writer(self, field_name: impl Into<String>) -> LegacyVectorFieldWriter<Self> {
        LegacyVectorFieldWriter::new(field_name, self)
    }

    /// Load an existing HNSW index from storage.
    pub fn load(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
        storage: Arc<dyn Storage>,
        path: &str,
    ) -> Result<Self> {
        use std::io::Read;

        // Open the index file
        let file_name = format!("{}.hnsw", path);
        let mut input = storage.open_input(&file_name)?;

        // Read metadata
        let mut num_vectors_buf = [0u8; 4];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u32::from_le_bytes(num_vectors_buf) as usize;

        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        let mut m_buf = [0u8; 4];
        input.read_exact(&mut m_buf)?;
        let _m = u32::from_le_bytes(m_buf) as usize;

        let mut ef_construction_buf = [0u8; 4];
        input.read_exact(&mut ef_construction_buf)?;
        let _ef_construction = u32::from_le_bytes(ef_construction_buf) as usize;

        if dimension != index_config.dimension {
            return Err(SarissaError::InvalidOperation(format!(
                "Dimension mismatch: expected {}, found {}",
                index_config.dimension, dimension
            )));
        }

        // Read vectors
        let mut vectors = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let mut doc_id_buf = [0u8; 8];
            input.read_exact(&mut doc_id_buf)?;
            let doc_id = u64::from_le_bytes(doc_id_buf);

            // Read field name
            let mut field_name_len_buf = [0u8; 4];
            input.read_exact(&mut field_name_len_buf)?;
            let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;

            let mut field_name_buf = vec![0u8; field_name_len];
            input.read_exact(&mut field_name_buf)?;
            let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                SarissaError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
            })?;

            // Read metadata and vector data
            let metadata = read_metadata(&mut input)?;
            let mut values = vec![0.0f32; dimension];
            for value in &mut values {
                let mut value_buf = [0u8; 4];
                input.read_exact(&mut value_buf)?;
                *value = f32::from_le_bytes(value_buf);
            }

            vectors.push((doc_id, field_name, Vector::with_metadata(values, metadata)));
        }

        // Calculate next_vec_id from loaded vectors
        let max_id = vectors.iter().map(|(id, _, _)| *id).max().unwrap_or(0);
        let next_vec_id = if num_vectors > 0 { max_id + 1 } else { 0 };

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            _ml: 1.0 / (2.0_f64).ln(),
            vectors,
            is_finalized: true,
            total_vectors_to_add: Some(num_vectors),
            next_vec_id,
            doc_id_map: std::collections::HashMap::new(), // TODO: populate if needed for append
            graph: None,                                  // TODO: Load graph if needed for append
        })
    }

    /// Set HNSW-specific parameters.
    pub fn with_hnsw_params(mut self, m: usize, ef_construction: usize) -> Self {
        self.index_config.m = m;
        self.index_config.ef_construction = ef_construction;
        self
    }

    /// Set the expected total number of vectors (for progress tracking).
    pub fn set_expected_vector_count(&mut self, count: usize) {
        self.total_vectors_to_add = Some(count);
    }

    /// Calculate the layer for a new vector.
    fn select_layer(&self) -> usize {
        let mut layer = 0;
        let mut rng = rand::rng();

        while rand::Rng::random::<f64>(&mut rng) < 0.5 && layer < 16 {
            layer += 1;
        }

        layer
    }

    /// Validate vectors before adding them.
    fn validate_vectors(&self, vectors: &[(u64, String, Vector)]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        for (doc_id, _field_name, vector) in vectors {
            if vector.dimension() != self.index_config.dimension {
                return Err(SarissaError::InvalidOperation(format!(
                    "Vector {} has dimension {}, expected {}",
                    doc_id,
                    vector.dimension(),
                    self.index_config.dimension
                )));
            }

            if !vector.is_valid() {
                return Err(SarissaError::InvalidOperation(format!(
                    "Vector {doc_id} contains invalid values (NaN or infinity)"
                )));
            }
        }

        Ok(())
    }

    /// Normalize vectors if configured to do so.
    fn normalize_vectors(&self, vectors: &mut [(u64, String, Vector)]) {
        if !self.index_config.normalize_vectors {
            return;
        }

        use rayon::prelude::*;

        if self.writer_config.parallel_build && vectors.len() > 100 {
            vectors.par_iter_mut().for_each(|(_, _, vector)| {
                vector.normalize();
            });
        } else {
            for (_, _, vector) in vectors {
                vector.normalize();
            }
        }
    }

    /// Initialize lookups for fast vector access
    fn rebuild_doc_id_map(&mut self) {
        self.doc_id_map.clear();
        for (idx, (doc_id, _, _)) in self.vectors.iter().enumerate() {
            self.doc_id_map.insert(*doc_id, idx);
        }
    }

    /// Build the HNSW graph structure.
    fn build_hnsw_graph(&mut self) -> Result<()> {
        let count = self.vectors.len();
        if count == 0 {
            return Ok(());
        }

        println!("Building HNSW graph with {} vectors (parallel)", count);
        println!(
            "Parameters: M={}, efConstruction={}",
            self.index_config.m, self.index_config.ef_construction
        );

        // Ensure doc_id_map is up to date
        self.rebuild_doc_id_map();

        let m = self.index_config.m;
        let m_max = m;
        let m_max_0 = m * 2;
        let ef_construction = self.index_config.ef_construction;

        // 1. Assign levels to all vectors
        let mut node_levels = Vec::with_capacity(count);
        // We can't use par_iter easily with rng unless we use thread_rng or seeded per thread
        // Pre-calculating levels serially is fast enough.
        for i in 0..count {
            let (doc_id, _, _) = self.vectors[i];
            let level = self.select_layer();
            node_levels.push((doc_id, level));
        }

        let max_level = node_levels.iter().map(|(_, l)| *l).max().unwrap_or(0);
        let entry_point = node_levels
            .iter()
            .find(|(_, l)| *l == max_level)
            .map(|(id, _)| *id);

        // 2. Initialize ConcurrentHnswGraph
        let graph = ConcurrentHnswGraph::new(node_levels.clone(), max_level);

        // 3. Parallel Insertion
        // We iterate indices 0..count. doc_id is self.vectors[i].0.
        // We skip the entry point during insertion as it's already "in" the graph (implicitly, via new)

        let writer_ref = &*self; // Immutable reference for threads

        // Use par_iter on indices
        (0..count).into_par_iter().try_for_each(|i| -> Result<()> {
            let (doc_id, _, ref vector) = writer_ref.vectors[i];

            if let Some(ep) = entry_point {
                // If this is the entry point, it's already added.
                if ep == doc_id {
                    return Ok(());
                }

                // Get the level assigned to this node
                // We need to find it directly. node_levels is aligned with vectors if we iterate 0..count
                let level = node_levels[i].1;

                let max_level = graph.max_level;
                let mut curr_obj = ep;
                let mut dist = writer_ref.calc_dist(vector, curr_obj)?;

                // 4. Search from top layer down to level + 1
                for lc in (level + 1..=max_level).rev() {
                    let mut changed = true;
                    while changed {
                        changed = false;
                        if let Some(neighbors) = graph.get_neighbors_view(curr_obj, lc) {
                            for neighbor_id in neighbors {
                                let d = writer_ref.calc_dist(vector, neighbor_id)?;
                                if d < dist {
                                    dist = d;
                                    curr_obj = neighbor_id;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                // 5. Search & Connect from min(max_level, level) down to 0
                let top_level = usize::min(max_level, level);
                for lc in (0..=top_level).rev() {
                    // search_layer: greedy search to find ef_construction candidates
                    let candidates =
                        writer_ref.search_layer(&graph, curr_obj, vector, ef_construction, lc)?;

                    // Select greedy neighbor as the best candidate for next layer
                    // Note: candidates is a MAX-heap (furthest at top).
                    // We need the nearest one to proceed to the next layer.
                    // Since BinaryHeap doesn't support random access/min-element efficiently without draining,
                    // and select_neighbors needs the heap, we iterate.
                    let nearest = candidates.iter().min_by(|a, b| {
                        a.distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(Ordering::Equal)
                    });

                    if let Some(min_cand) = nearest {
                        curr_obj = min_cand.id;
                    }

                    // Heuristic: select neighbors
                    let neighbors = writer_ref.select_neighbors(&candidates, m, lc, m_max, m_max_0);

                    // Add connections bidirectionally
                    // 1. From doc_id to neighbors
                    graph.set_neighbors(doc_id, lc, neighbors.clone());

                    // 2. From neighbors to doc_id
                    for neighbor_id in neighbors {
                        // Add doc_id to neighbor's list and prune if necessary
                        let current_m_max = if lc == 0 { m_max_0 } else { m_max };
                        graph.add_neighbor_with_pruning(
                            neighbor_id,
                            lc,
                            doc_id,
                            current_m_max,
                            writer_ref,
                        )?;
                    }
                }
            }
            Ok(())
        })?;

        // 4. Convert ConcurrentGraph to HnswGraph
        // This consumes the ConcurrentGraph and creates HnswGraph
        let mut final_nodes = HashMap::new();
        for (doc_id, layers) in graph.nodes {
            let mut vec_layers = Vec::with_capacity(layers.len());
            for lock in layers {
                vec_layers.push(lock.into_inner()); // Consume RwLock
            }
            final_nodes.insert(doc_id, vec_layers);
        }

        self.graph = Some(HnswGraph {
            entry_point,
            max_level,
            nodes: final_nodes,
            m,
            m_max,
            m_max_0,
            ef_construction,
            level_mult: self._ml,
        });

        Ok(())
    }

    // Calculates distance between a query vector and a document in the index
    fn calc_dist(&self, query: &Vector, doc_id: u64) -> Result<f32> {
        let idx = *self
            .doc_id_map
            .get(&doc_id)
            .ok_or_else(|| SarissaError::internal(format!("Doc ID {} not found in map", doc_id)))?;
        let target = &self.vectors[idx].2;
        self.index_config
            .distance_metric
            .distance(&query.data, &target.data)
    }

    /// Search for nearest neighbors in a specific layer
    fn search_layer<G: GraphView>(
        &self,
        graph: &G,
        entry_point: u64,
        query: &Vector,
        ef: usize,
        level: usize,
    ) -> Result<BinaryHeap<Candidate>> {
        let mut visited = HashSet::new();

        let dist = self.calc_dist(query, entry_point)?;
        // We use min-heap for "results" to keep track of nearest found?
        // No, HNSW "v" list (candidates to visit) is min-heap (nearest first).
        // "C" list (found candidates) is max-heap (furthest first) to keep ef smallest.

        // Let's use two heaps:
        // 1. candidates_to_visit (min-heap by distance): nodes to explore
        // 2. found_candidates (max-heap by distance): keeps `ef` nearest nodes found so far

        #[derive(Debug, Clone, PartialEq)]
        struct VisitorCandidate {
            id: u64,
            distance: f32,
        }
        impl Eq for VisitorCandidate {}
        impl Ord for VisitorCandidate {
            fn cmp(&self, other: &Self) -> Ordering {
                // Min-heap: smaller distance > larger distance
                other
                    .distance
                    .partial_cmp(&self.distance)
                    .unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for VisitorCandidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut to_visit = BinaryHeap::new();
        let mut found = BinaryHeap::new(); // Max-heap (Candidate stores distance, PartialOrd is normal (larger > smaller))

        to_visit.push(VisitorCandidate {
            id: entry_point,
            distance: dist,
        });
        found.push(Candidate {
            id: entry_point,
            distance: dist,
            similarity: 0.0,
        });
        visited.insert(entry_point);

        while let Some(curr) = to_visit.pop() {
            // If closest candidate to visit is further than the furthest found candidate, and we found enough, stop
            if let Some(furthest_found) = found.peek() {
                if curr.distance > furthest_found.distance && found.len() >= ef {
                    break;
                }
            }

            if let Some(neighbors) = graph.get_neighbors_view(curr.id, level) {
                for neighbor_id in neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);

                    let neighbor_dist = self.calc_dist(query, neighbor_id)?;
                    let furthest_dist = found.peek().map(|c| c.distance).unwrap_or(f32::MAX);

                    if neighbor_dist < furthest_dist || found.len() < ef {
                        let c = Candidate {
                            id: neighbor_id,
                            distance: neighbor_dist,
                            similarity: 0.0,
                        };
                        let vc = VisitorCandidate {
                            id: neighbor_id,
                            distance: neighbor_dist,
                        };

                        found.push(c);
                        to_visit.push(vc);

                        if found.len() > ef {
                            found.pop();
                        }
                    }
                }
            }
        }

        Ok(found)
    }

    fn select_neighbors(
        &self,
        candidates: &BinaryHeap<Candidate>,
        m: usize,
        _level: usize,
        _m_max: usize,
        _m_max_0: usize,
    ) -> Vec<u64> {
        // Simple heuristic: take M nearest
        // Candidates are in a max-heap (furthest at top).
        // We want smallest distances.
        let mut sorted: Vec<_> = candidates.clone().into_sorted_vec();
        // into_sorted_vec returns ascending order [min ... max] for max-heap?
        // No, pop() returns max. sorted vec will be [small, ..., large].
        // Wait, doc says: "The elements are sorted in ascending order." for BinaryHeap::into_sorted_vec().

        // But BinaryHeap<T> is a MaxHeap.
        // pop() gives largest.
        // into_sorted_vec gives ascending order.
        // So if Candidate implies larger distance > smaller, then ascending is [small, ..., large].
        // We want the smallest distance ones (start of vec).

        sorted.truncate(m);
        sorted.into_iter().map(|c| c.id).collect()
    }

    fn prune_neighbors(
        &self,
        _doc_id: u64,
        neighbors: Vec<u64>,
        max_conn: usize,
    ) -> Result<Vec<u64>> {
        if neighbors.len() <= max_conn {
            return Ok(neighbors);
        }

        // Sort by distance from doc_id
        let doc_vec = &self.vectors[*self.doc_id_map.get(&_doc_id).unwrap()].2;

        let mut candidates = Vec::new();
        for nid in neighbors {
            let dist = self.calc_dist(doc_vec, nid)?;
            candidates.push(Candidate {
                id: nid,
                distance: dist,
                similarity: 0.0,
            });
        }

        // We want to keep nearest. Move to min-heap or just sort.
        candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        candidates.truncate(max_conn);

        Ok(candidates.into_iter().map(|c| c.id).collect())
    }

    /// Check for memory limits.
    fn check_memory_limit(&self) -> Result<()> {
        if let Some(limit) = self.writer_config.memory_limit {
            let current_usage = self.estimated_memory_usage();
            if current_usage > limit {
                return Err(SarissaError::ResourceExhausted(format!(
                    "Memory usage {current_usage} bytes exceeds limit {limit} bytes"
                )));
            }
        }
        Ok(())
    }

    /// Get the stored vectors (for testing/debugging).
    pub fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    /// Get HNSW parameters.
    pub fn hnsw_params(&self) -> (usize, usize) {
        (self.index_config.m, self.index_config.ef_construction)
    }
}

impl VectorIndexWriter for HnswIndexWriter {
    fn next_vector_id(&self) -> u64 {
        self.next_vec_id
    }

    fn build(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Cannot build on finalized index".to_string(),
            ));
        }

        self.validate_vectors(&vectors)?;
        self.normalize_vectors(&mut vectors);

        // Update next_vec_id
        if let Some(max_id) = vectors.iter().map(|(id, _, _)| *id).max()
            && max_id >= self.next_vec_id
        {
            self.next_vec_id = max_id + 1;
        }

        self.vectors = vectors;
        self.total_vectors_to_add = Some(self.vectors.len());

        self.check_memory_limit()?;
        Ok(())
    }

    fn add_vectors(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Cannot add vectors to finalized index".to_string(),
            ));
        }

        self.validate_vectors(&vectors)?;
        self.normalize_vectors(&mut vectors);

        // Update next_vec_id
        if let Some(max_id) = vectors.iter().map(|(id, _, _)| *id).max()
            && max_id >= self.next_vec_id
        {
            self.next_vec_id = max_id + 1;
        }

        self.vectors.extend(vectors);
        self.check_memory_limit()?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        if self.is_finalized {
            return Ok(());
        }

        // Build the actual HNSW graph structure
        self.build_hnsw_graph()?;

        self.is_finalized = true;
        Ok(())
    }

    fn progress(&self) -> f32 {
        if let Some(total) = self.total_vectors_to_add {
            if total == 0 {
                if self.is_finalized { 1.0 } else { 0.0 }
            } else {
                let current = self.vectors.len() as u64 as f32;
                let progress = current / total as f32;
                if self.is_finalized {
                    1.0
                } else {
                    progress.min(0.99) // Never report 100% until finalized
                }
            }
        } else if self.is_finalized {
            1.0
        } else {
            0.0
        }
    }

    fn estimated_memory_usage(&self) -> usize {
        let vector_memory = self.vectors.len()
            * (
                8 + // doc_id
            self.index_config.dimension * 4 + // f32 values
            std::mem::size_of::<Vector>()
                // Vector struct overhead
            );

        // HNSW graph overhead (rough estimate)
        // Each vector can have up to M connections per layer
        // Average layers per vector is approximately 1/(1-p) where p=0.5
        let avg_layers = 2.0;
        let graph_memory =
            self.vectors.len() * (self.index_config.m as f32 * avg_layers * 8.0) as usize;

        let metadata_memory = self.vectors.len() * 128; // Increased for graph structure

        vector_memory + graph_memory + metadata_memory
    }

    fn optimize(&mut self) -> Result<()> {
        if !self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Index must be finalized before optimization".to_string(),
            ));
        }

        // HNSW optimization could include:
        // 1. Graph pruning to remove low-quality connections
        // 2. Memory compaction
        // 3. Connection rebalancing

        println!("Optimizing HNSW index...");

        // For now, just compact memory
        self.vectors.shrink_to_fit();

        Ok(())
    }

    fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    fn write(&self, path: &str) -> Result<()> {
        use std::io::Write;

        if !self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| SarissaError::InvalidOperation("No storage configured".to_string()))?;

        // Create the index file
        let file_name = format!("{}.hnsw", path);
        let mut output = storage.create_output(&file_name)?;

        // Write metadata
        output.write_all(&(self.vectors.len() as u64 as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.dimension as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.m as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.ef_construction as u32).to_le_bytes())?;

        // Write vectors with field names and metadata
        for (doc_id, field_name, vector) in &self.vectors {
            output.write_all(&doc_id.to_le_bytes())?;

            // Write field name length and field name
            let field_name_bytes = field_name.as_bytes();
            output.write_all(&(field_name_bytes.len() as u32).to_le_bytes())?;
            output.write_all(field_name_bytes)?;

            write_metadata(&mut output, &vector.metadata)?;

            // Write vector data
            for value in &vector.data {
                output.write_all(&value.to_le_bytes())?;
            }
        }

        // Write Graph Data
        if let Some(graph) = &self.graph {
            // Write graph metadata
            let has_graph = 1u8;
            output.write_all(&[has_graph])?;

            // Serialize graph
            // Since HnswGraph is Serialize, we can just use serde_json or binary
            // For efficiency, let's use serde_json for now as it's easier and we have it
            // Or manual binary for speed. Let's use serde_json for simplicity of implementation given struct.
            // But strict binary is better for indexes.
            // Let's stick to simple manual binary as per other parts.

            // Entry point
            let entry_point = graph.entry_point.unwrap_or(u64::MAX);
            output.write_all(&entry_point.to_le_bytes())?;
            output.write_all(&(graph.max_level as u32).to_le_bytes())?;

            // Nodes
            let node_count = graph.nodes.len() as u32;
            output.write_all(&node_count.to_le_bytes())?;

            for (doc_id, layers) in &graph.nodes {
                output.write_all(&doc_id.to_le_bytes())?;

                let layer_count = layers.len() as u32;
                output.write_all(&layer_count.to_le_bytes())?;

                for neighbors in layers {
                    let neighbor_count = neighbors.len() as u32;
                    output.write_all(&neighbor_count.to_le_bytes())?;
                    for neighbor in neighbors {
                        output.write_all(&neighbor.to_le_bytes())?;
                    }
                }
            }
        } else {
            // No graph built
            let has_graph = 0u8;
            output.write_all(&[has_graph])?;
        }

        output.flush()?;
        Ok(())
    }

    fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        if self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Cannot delete documents from finalized index".to_string(),
            ));
        }

        // Logical deletion from buffer
        self.vectors.retain(|(id, _, _)| *id != doc_id);
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        self.vectors.clear();
        self.doc_id_map.clear();
        self.graph = None;
        self.is_finalized = false;
        self.next_vec_id = 0;
        Ok(())
    }

    fn pending_docs(&self) -> u64 {
        if self.is_finalized {
            0
        } else {
            self.vectors.len() as u64
        }
    }

    fn close(&mut self) -> Result<()> {
        self.vectors.clear();
        self.doc_id_map.clear();
        self.graph = None;
        self.is_finalized = true;
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.is_finalized && self.vectors.is_empty()
    }
}
