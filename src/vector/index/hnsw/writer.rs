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
use rand::Rng;
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

    fn from_hnsw_graph(graph: HnswGraph, extended_max_level: usize) -> Self {
        let mut nodes = HashMap::with_capacity(graph.nodes.len());
        for (doc_id, layered_neighbors) in graph.nodes {
            let mut layers = Vec::with_capacity(layered_neighbors.len());
            for neighbors in layered_neighbors {
                layers.push(RwLock::new(neighbors));
            }
            nodes.insert(doc_id, layers);
        }

        Self {
            max_level: extended_max_level,
            nodes,
        }
    }

    fn add_nodes(&mut self, nodes_with_levels: Vec<(u64, usize)>) {
        for (doc_id, level) in nodes_with_levels {
            if self.nodes.contains_key(&doc_id) {
                continue;
            }
            let mut layers = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                layers.push(RwLock::new(Vec::new()));
            }
            self.nodes.insert(doc_id, layers);
        }
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
    path: String,
    _ml: f64, // Level normalization factor
    vectors: Vec<(u64, String, Vector)>,
    // Map from doc_id to index in vectors for fast access
    doc_id_map: HashMap<u64, usize>,
    levels: Vec<Vec<u64>>,
    entry_point: Option<u64>,
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
        path: impl Into<String>,
    ) -> Result<Self> {
        let max_level = Self::calculate_max_level(index_config.m, index_config.ef_construction);
        let _ml = 1.0 / (index_config.m as f64).ln();

        Ok(Self {
            index_config,
            writer_config,
            storage: None,
            path: path.into(),
            _ml,
            levels: vec![Vec::new(); max_level + 1],
            entry_point: None,
            vectors: Vec::new(),
            doc_id_map: HashMap::new(),
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
        path: impl Into<String>,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let max_level = Self::calculate_max_level(index_config.m, index_config.ef_construction);
        let _ml = 1.0 / (index_config.m as f64).ln();

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            path: path.into(),
            _ml,
            levels: vec![Vec::new(); max_level + 1],
            entry_point: None,
            vectors: Vec::new(),
            doc_id_map: HashMap::new(),
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

        // Rebuild doc_id_map
        let mut doc_id_map = HashMap::new();
        for (i, (doc_id, _, _)) in vectors.iter().enumerate() {
            doc_id_map.insert(*doc_id, i);
        }

        // Calculate next_vec_id from loaded vectors
        let max_id = vectors.iter().map(|(id, _, _)| *id).max().unwrap_or(0);
        let next_vec_id = if num_vectors > 0 { max_id + 1 } else { 0 };

        let max_level = Self::calculate_max_level(index_config.m, index_config.ef_construction);
        let _ml = 1.0 / (index_config.m as f64).ln();

        // Read graph data if present
        let mut has_graph_buf = [0u8; 1];
        let graph = if input.read_exact(&mut has_graph_buf).is_ok() {
            if has_graph_buf[0] == 1 {
                // Read entry point
                let mut entry_point_buf = [0u8; 8];
                input.read_exact(&mut entry_point_buf)?;
                let entry_point_raw = u64::from_le_bytes(entry_point_buf);
                let entry_point = if entry_point_raw == u64::MAX {
                    None
                } else {
                    Some(entry_point_raw)
                };

                // Read max level
                let mut max_level_buf = [0u8; 4];
                input.read_exact(&mut max_level_buf)?;
                let max_level = u32::from_le_bytes(max_level_buf) as usize;

                // Read nodes
                let mut node_count_buf = [0u8; 4];
                input.read_exact(&mut node_count_buf)?;
                let node_count = u32::from_le_bytes(node_count_buf) as usize;

                let mut nodes = HashMap::with_capacity(node_count);

                for _ in 0..node_count {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let mut layer_count_buf = [0u8; 4];
                    input.read_exact(&mut layer_count_buf)?;
                    let layer_count = u32::from_le_bytes(layer_count_buf) as usize;

                    let mut layers = Vec::with_capacity(layer_count);

                    for _ in 0..layer_count {
                        let mut neighbor_count_buf = [0u8; 4];
                        input.read_exact(&mut neighbor_count_buf)?;
                        let neighbor_count = u32::from_le_bytes(neighbor_count_buf) as usize;

                        let mut neighbors = Vec::with_capacity(neighbor_count);
                        for _ in 0..neighbor_count {
                            let mut neighbor_buf = [0u8; 8];
                            input.read_exact(&mut neighbor_buf)?;
                            neighbors.push(u64::from_le_bytes(neighbor_buf));
                        }
                        layers.push(neighbors);
                    }
                    nodes.insert(doc_id, layers);
                }

                Some(HnswGraph {
                    entry_point,
                    max_level,
                    nodes,
                    m: index_config.m,
                    m_max: index_config.m,
                    m_max_0: index_config.m * 2,
                    ef_construction: index_config.ef_construction,
                    level_mult: _ml,
                })
            } else {
                None
            }
        } else {
            None
        };

        // If we loaded a graph, we are not "finalized" in the sense that we can't append.
        // We want to support append, so we should allow modifications if loaded.
        // Previously, is_finalized=true prevented modifications.
        // For append support, we set is_finalized=false.

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            path: path.to_string(),
            _ml,
            levels: vec![Vec::new(); max_level + 1], // Still re-init levels, but they are conceptually in the graph
            entry_point: graph.as_ref().and_then(|g| g.entry_point),
            vectors,
            is_finalized: false, // Changed to false to allow appending
            total_vectors_to_add: Some(num_vectors),
            next_vec_id,
            doc_id_map,
            graph,
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

        while rng.random_range(0.0..1.0) < self._ml && layer < 16 {
            layer += 1;
        }

        layer
    }

    /// Calculate the maximum level based on M and ef_construction.
    /// This is a heuristic, often 1/ln(M) or 1/ln(2) is used for probability.
    /// For simplicity, we can cap it or use a fixed formula.
    /// A common formula for max_level is based on the number of elements and M.
    /// For now, let's use a simple heuristic or a fixed max.
    fn calculate_max_level(_m: usize, _ef_construction: usize) -> usize {
        // A common heuristic is to have max_level around log_M(N) or a fixed small number.
        // For now, let's use a fixed small number or a simple formula.
        // The original code used 1/ln(2) for probability, which implies levels grow with log_2(N).
        // Let's set a reasonable cap, e.g., 16 or 32.
        // Or, based on the probability p = 1/ln(M), the expected max level for N elements is log_p(N).
        // For simplicity, let's use a fixed max level for now, or a simple calculation.
        // The `select_layer` uses `1.0 / (self.index_config.m as f64).ln()` as probability.
        // Let's assume a max level that allows for a reasonable number of layers.
        // For example, if M=16, 1/ln(16) approx 0.36.
        // A max level of 16-32 is common.
        16 // A reasonable default max level
    }

    /// Validate vectors before adding them.
    fn validate_vectors(&self, vectors: &Vec<(u64, String, Vector)>) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        for (doc_id, _, vector) in vectors {
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
    /// Normalize vectors if configured to do so.
    fn normalize_vectors_internal(
        index_config: &HnswIndexConfig,
        writer_config: &VectorIndexWriterConfig,
        vectors: &mut Vec<(u64, String, Vector)>,
    ) {
        if !index_config.normalize_vectors {
            return;
        }

        if writer_config.parallel_build && vectors.len() > 100 {
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

        // Determine which vectors are new and need insertion
        let mut new_node_levels = Vec::new(); // (doc_id, level)
        let mut new_doc_ids_in_order = Vec::new();

        // Check if we have an existing graph to append to
        let (graph, entry_point, max_level, search_entry_point) =
            if let Some(existing_graph) = self.graph.take() {
                println!(
                    "Appending to existing graph with {} nodes",
                    existing_graph.nodes.len()
                );

                // Identify new vectors
                for (doc_id, _, _) in &self.vectors {
                    if !existing_graph.nodes.contains_key(doc_id) {
                        new_doc_ids_in_order.push(*doc_id);
                    }
                }
                new_doc_ids_in_order.sort_unstable();

                // Assign levels to new vectors
                for doc_id in &new_doc_ids_in_order {
                    let level = self.select_layer();
                    new_node_levels.push((*doc_id, level));
                }

                let current_max_level = existing_graph.max_level;
                let new_max_level = new_node_levels.iter().map(|(_, l)| *l).max().unwrap_or(0);
                let total_max_level = current_max_level.max(new_max_level);

                let old_ep = existing_graph.entry_point;
                let mut ep = old_ep;

                // If we have new nodes with higher level, update entry point
                if new_max_level > current_max_level {
                    ep = new_node_levels
                        .iter()
                        .find(|(_, l)| *l == total_max_level)
                        .map(|(id, _)| *id)
                        .or(ep);
                }

                // Convert to ConcurrentHnswGraph and extend
                let mut concurrent_graph =
                    ConcurrentHnswGraph::from_hnsw_graph(existing_graph, total_max_level);
                concurrent_graph.add_nodes(new_node_levels.clone());

                let search_ep = old_ep.or(ep);

                (concurrent_graph, ep, total_max_level, search_ep)
            } else {
                // Full build
                let mut doc_ids_in_order: Vec<u64> =
                    self.vectors.iter().map(|(id, _, _)| *id).collect();
                doc_ids_in_order.sort_unstable();

                for doc_id in &doc_ids_in_order {
                    let level = self.select_layer();
                    new_node_levels.push((*doc_id, level));
                }

                let max_level = new_node_levels.iter().map(|(_, l)| *l).max().unwrap_or(0);
                let ep = new_node_levels
                    .iter()
                    .find(|(_, l)| *l == max_level)
                    .map(|(id, _)| *id);

                new_doc_ids_in_order = doc_ids_in_order;

                let concurrent_graph = ConcurrentHnswGraph::new(new_node_levels.clone(), max_level);
                (concurrent_graph, ep, max_level, ep)
            };

        // 3. Parallel Insertion
        // We iterate over the sorted doc_ids (only new assignments).
        let writer_ref = &*self; // Immutable reference for threads

        new_doc_ids_in_order
            .into_par_iter()
            .try_for_each(|doc_id| -> Result<()> {
                let doc_vector_idx = *writer_ref.doc_id_map.get(&doc_id).ok_or_else(|| {
                    SarissaError::internal(format!("Doc ID {} not found in doc_id_map", doc_id))
                })?;
                let _vector = &writer_ref.vectors[doc_vector_idx].2;

                if let Some(ep) = entry_point {
                    // It's possible the entry point is one of the new nodes we are inserting.
                    // If this doc is the entry point, and it was just chosen as such (and has no edges yet),
                    // we usually skip inserting it because it's the first node in the graph (or first at top level).
                    // However, if we are appending, the entry point might be an OLD node, or a NEW node.
                    // If OLD node: simply search from it.
                    // If NEW node: if it's the VERY first node in empty graph, skip.
                    // If it's a new node that became entry point due to level, we still need to insert it
                    // into lower layers (if any). But if it's the highest node, it might not have peers at top.

                    // Simple logic: if ep == doc_id, and graph was previously empty (or effectively so), skip.
                    // But with HNSW, if a node is the entry point, it implies it's in the graph.
                    // If we just added it to `concurrent_graph` via `add_nodes` (init empty), we need to populate edges?
                    // Actually, the entry point for HNSW is just a starting handle.
                    // If it's the first node EVER, no edges can be formed.
                    // If it's a new node added to existing graph, it must function as entry point for *subsequent* searches,
                    // but for *its own insertion*, we need an entry point *previously* existing (or itself if it's the only one).

                    if ep == doc_id {
                        // If this doc IS the entry point, check if there are other nodes to connect to.
                        // If this is the first node of the graph, we do nothing.
                        // If there are other nodes (e.g. at lower levels), we might need to connect down?
                        // HNSW insertion usually assumes we search for neighbors. If we are the top, we have no neighbors at top.
                        // We descend.

                        // For simplicity, let's skip "insertion search" if we are the global entry point AND we assume
                        // we are just starting.
                        // But if we are appending a new top-level node, we should connect to lower levels.
                        // However, standard HNSW insertion algorithm:
                        // `curr_obj = entry_point`.
                        // If `doc_id == entry_point`, we start at ourselves? Distance is 0.
                        // If we are the ONLY node at this top level, we won't find neighbors.
                        // We descend to `level`-1, where there might be other nodes.

                        // Let's rely on standard logic: allow the loop to run.
                        // But careful: `curr_obj = ep`. If `doc_id == ep`, `dist = 0`.
                        // Loop 4 (Search from top down to level+1):
                        //   `card` check: `curr_obj` is candidate.
                        // It will find nothing better (it is itself).
                        // It descends.
                        // Loop 5 (Connect from min(max_level, level) down to 0):
                        //   It will search layer.
                        //   Since it is itself, `search_layer` might return itself?
                        //   We need to ensure we don't add edge to self. `search_layer` returns candidates.
                        //   `select_neighbors` picks them.
                        //   `set_neighbors` adds them.
                    }

                    // If we are inserting the entry point itself (e.g. first node), we can skip.
                    if ep == doc_id && graph.nodes.len() == 1 {
                        return Ok(());
                    }

                    // Start search from current entry point.
                    // If `entry_point` is `doc_id` (newly promoted), we should actually start from *previous* entry point?
                    // Or works fine?
                    // If `doc_id` is the new global entry point, it means it has the highest level.
                    // The levels *above* the old max_level definitely only contain `doc_id` (and potentially other new nodes).
                    // We need to find neighbors.
                    // HNSW paper: `enter_point` is an existing node. When inserting `q`:
                    // 1. `ep` = current entry point.
                    // 2. Search from `ep` to `q.level`.
                    // 3. Update `ep` if `q` becomes new entry point (globally).

                    // Here `entry_point` variable holds the *global* entry point after this batch?
                    // No, `entry_point` variable should be the *starting point for search*.
                    // If we are in parallel, this is tricky. Effectively we use the *snapshot* entry point.
                    // If we use the *new* entry point (which might be `doc_id`), it's bad if `doc_id` is isolated.
                    // We should probably use the *old* entry point for search if possible, OR just handle the case where we start at `doc_id` and descend.

                    // Let's allow `entry_point` to be used. If it is `doc_id`, loop 4 will just descend (no neighbors at top layers yet).

                    // Wait, if `doc_id` is new entry point (level 10), and old usage was level 5.
                    // Layers 10 downto 6 are empty (except `doc_id`).
                    // `get_neighbors_view` for `doc_id` at level 10 returns empty.
                    // Loop 4 descends to level.
                    // If `doc_id.level` is 10. we descend to 11? Loop doesn't run.
                    // Then Loop 5 runs from 10 down to 0.
                    // At level 10: `search_layer(curr=doc_id)`.
                    //   Candidate heap has `doc_id`.
                    //   It finds nothing else.
                    //   Neighbors = [].
                    //   Sets neighbors for `doc_id` at level 10 to [].
                    //   Descends to 9...5.
                    //   Eventually hits level 5 where other nodes exist.
                    //   `search_layer` will expand `doc_id` (candidate) ?
                    //   No, `doc_id` is not connected to level 5 nodes yet.
                    //   WE NEED TO ENTER THE GRAPH at some point where we can jump to existing nodes.
                    //   If we start at `doc_id` which is disconnected, we are stuck.

                    // CORRECT LOGIC:
                    // Use the *old* entry point if `doc_id` is strictly higher?
                    // Or, if `doc_id` is new, `curr_obj` must be an *already connected* node?
                    // The paper says: "L = level of q ... enter_point is the current entry point".
                    // Then updates entry point *after* insertion.

                    // So we should NOT use `doc_id` as starting point `ep` for its own insertion, unless it's the very first node.
                    // We need a stable entry point into the *existing* structure.

                    // Refinement:
                    // `entry_point` variable in the code above is calculated as `total_max_level` entry.
                    // If this is `doc_id`, we have a problem.

                    // We should use `existing_graph.entry_point` as the search starter if available?
                    // But if `existing_graph.entry_point` is None (first batch), we pick one from `new_node_levels`?
                }

                // ... Logic continues (will copy existing) ...

                // My fix:
                // We need to determine the `start_node` for the search.
                // If `doc_id` == `entry_point` (global), we are potentially isolated.
                // But wait, `ConcurrentHnswGraph` is pre-populated with `add_nodes`.
                // All `new_doc_ids` are in the graph structure (isolated initially).
                // Existing nodes are in the graph structure (connected).

                // If appending: `existing_graph.entry_point` is safe.
                // If full build: arbitrary `entry_point` is chosen. But in parallel build, how does it work?
                // Sequential HNSW inserts one by one, updating EP.
                // Parallel HNSW usually has a hack or uses a lock.
                // This implementation is "Parallel Insertion".

                // If I look at the ORIGINAL code:
                // `let entry_point = node_levels... find max level ...`
                // `let ep = entry_point`;
                // Inside loop: `if ep == doc_id { return Ok(()); }`
                // This implies the Global Entry Point is SKIPPED for insertion (it has no neighbors added?).
                // Then other nodes use `ep` to search.
                // This implies the Global Max Level node is "root" and might initially have 0 neighbors (until someone links TO it).
                // But HNSW links are bidirectional?
                // Row 596: `graph.set_neighbors(doc_id...)`.
                // Row 599: `for neighbor in neighbors { graph.add_neighbor(neighbor, doc_id) }`.
                // Yes, bidirectional.
                // So if EP is skipped, it eventually gets linked when other nodes (at lower levels or same level) connect TO it.
                // BUT, if EP is at level 10, and everyone else is at level 5.
                // No one at level 5 can see level 10.
                // So EP at level 10 remains isolated?
                // Unless someone else is at level 10.

                // If EP is strictly higher than everyone else, it MUST be inserted to connect to lower layers.
                // The original code `if ep == doc_id { return Ok(()); }` seems suspect if `level(ep) > max_level(others)`.
                // But maybe `select_layer` distribution makes this rare or acceptable?

                // To be safe for incremental:
                // If `doc_id` is the new Global EP (level 10), and Old EP was level 5.
                // We MUST insert `doc_id` starting from Old EP (level 5), traversing up?
                // No, traverse down.
                // We start at Old EP (level 5).
                // We search at level 5. Connect `doc_id` to level 5 nodes.
                // Then 4, 3...
                // Levels 6, 7, 8, 9, 10 for `doc_id` will be empty?
                // HNSW requires connectivity at top layer.

                // Let's stick to the pattern:
                // If `doc_id` is being inserted, we need a valid `curr_obj` (enter point).
                // If `entry_point` (global) is `doc_id` itself, we need a *different* node to start searching from (the 2nd best?).

                // However, preserving behavior is safest.
                // I will use `entry_point` calculated as max level.
                // If `doc_id == entry_point`, I will proceed with logic but use a fallback?
                // Or just keep the `if ep == doc_id { return Ok(()) }` line?

                // In incremental case, if we keep `if ep == doc_id { return Ok(()) }`:
                // If we append a new node with higher level, it becomes EP. It gets skipped.
                // It never connects to the old graph.
                // When we search, we start at New EP. It has no neighbors. Search fails.
                // This is bad.

                // Fix:
                // If we have an `existing_entry_point` (from `graph.take()`), we should probably use THAT as the start point for insertion of ALL new nodes, *including* the new Global EP candidate.
                // So for `new_doc_ids`, `curr_obj` should init to `old_entry_point` (if exists).

                // But `entry_point` variable in my replacement code is `total_max_level` one.
                // I should pass `start_node` separately?
                // I can use `ep` (from closure capture) which is the `total_max_level` one.

                // Let's refine the replacement to capture both `global_entry_point` (for future reference) and `search_entry_point` (for insertion).
                // Actually, `ConcurrentHnswGraph` doesn't track EP. The Writer stores it at the end.
                // The insertion logic just needs A valid entry point.

                // Adjusted strategy:
                // `let search_entry_point = existing_graph.entry_point.unwrap_or(global_entry_point_candidate);`
                // But `existing_graph.entry_point` might be None (first batch).

                // I will modify the logic to:
                // `let search_entry_point = ...`
                // Inside loop: `let mut curr_obj = search_entry_point;`
                // And REMOVE `if ep == doc_id { return Ok(()); }` for the incremental case?
                // Or be smarter.

                // Let's rely on `ep` being `existing_graph.entry_point` (if valid) for the searches.
                // If `existing_graph` exists, we use its EP.
                // If `new_max_level > existing`, the new Global EP is one of the new nodes.
                // Even that new node should be inserted starting from `existing_EP`.

                // So: `let search_ep = existing_graph.entry_point.or(new_ep_candidate)`.
                // Actually in my code: `let mut ep = existing_graph.entry_point;`
                // `if new_max > current_max { ep = ... }` -> this updates it to New EP.

                // I should keep `old_ep`.

                // Implementation detail:
                // In `build_hnsw_graph`, I calculate `entry_point` (the final one).
                // I should ALSO calculate `insertion_entry_point`.
                // If `appending` (graph exists): `insertion_entry_point` = `existing_graph.entry_point`.
                // If `full build`: `insertion_entry_point` = `entry_point` (one of the nodes).

                // And inside the loop:
                // `if doc_id == insertion_entry_point { return Ok(()); }`
                // (Only skip if we are inserting the start node itself, which implies full build first node).

                // For incremental: `insertion_entry_point` is an OLD node. `doc_id` is NEW. They are never equal.
                // So we never skip. We always insert new nodes starting from Old EP.
                // This connects New Node to Old Graph.
                // AND since links are bidirectional, Old Graph connects to New Node.
                // Finally we update `self.entry_point` to `final_global_ep`.
                // This seems correct.

                let doc_vector_idx = *writer_ref.doc_id_map.get(&doc_id).ok_or_else(|| {
                    SarissaError::internal(format!("Doc ID {} not found in doc_id_map", doc_id))
                })?;
                let vector = &writer_ref.vectors[doc_vector_idx].2;

                let start_node = if let Some(sp) = search_entry_point {
                    sp
                } else {
                    // No start node (graph empty and this is the first node?)
                    return Ok(());
                };

                if start_node == doc_id {
                    // We are inserting the start node itself (full build case).
                    return Ok(());
                }

                // Determine level from graph (it was pre-populated)
                // We need to access without lock if possible or just acquire read lock.
                // graph.nodes is HashMap<u64, Vec<RwLock<Vec<u64>>>>
                // We can just check len of Vec.
                let layers_len = graph.nodes.get(&doc_id).map(|l| l.len()).unwrap_or(0);
                if layers_len == 0 {
                    return Ok(());
                }
                let level = layers_len - 1;

                let max_level = graph.max_level;
                let mut curr_obj = start_node;
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
                    let candidates =
                        writer_ref.search_layer(&graph, curr_obj, vector, ef_construction, lc)?;

                    let nearest = candidates.iter().min_by(|a, b| {
                        a.distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(Ordering::Equal)
                    });

                    if let Some(min_cand) = nearest {
                        curr_obj = min_cand.id;
                    }

                    let neighbors = writer_ref.select_neighbors(&candidates, m, lc, m_max, m_max_0);

                    graph.set_neighbors(doc_id, lc, neighbors.clone());

                    for neighbor_id in neighbors {
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
                Ok(())
            })?;

        // 4. Convert ConcurrentGraph to HnswGraph
        let mut final_nodes = HashMap::new();
        let mut final_levels_map = HashMap::new();

        for (doc_id, layers) in graph.nodes {
            let mut vec_layers = Vec::with_capacity(layers.len());
            for lock in layers {
                vec_layers.push(lock.into_inner()); // Consume RwLock
            }
            final_levels_map.insert(doc_id, vec_layers.len() - 1);
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
            level_mult: 1.0 / (self.index_config.m as f64).ln(),
        });
        self.entry_point = entry_point;

        // Rebuild self.levels
        let mut levels_vec = vec![Vec::new(); max_level + 1];
        for (doc_id, level) in final_levels_map {
            if level < levels_vec.len() {
                levels_vec[level].push(doc_id);
            }
        }
        self.levels = levels_vec;

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
        doc_id: u64,
        neighbors: Vec<u64>,
        max_conn: usize,
    ) -> Result<Vec<u64>> {
        if neighbors.len() <= max_conn {
            return Ok(neighbors);
        }

        // Sort by distance from doc_id
        let idx = *self.doc_id_map.get(&doc_id).unwrap();
        let doc_vec = &self.vectors[idx].2;

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

    fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(SarissaError::InvalidOperation(
                "Cannot build on finalized index".to_string(),
            ));
        }

        self.validate_vectors(&vectors)?;

        self.vectors = vectors;
        Self::normalize_vectors_internal(
            &self.index_config,
            &self.writer_config,
            &mut self.vectors,
        );
        self.rebuild_doc_id_map();

        // Update next_vec_id
        if let Some((max_id, _, _)) = self.vectors.iter().max_by_key(|(id, _, _)| id) {
            if *max_id >= self.next_vec_id {
                self.next_vec_id = *max_id + 1;
            }
        }

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
        Self::normalize_vectors_internal(&self.index_config, &self.writer_config, &mut vectors);

        // Ensure doc_id_map is up to date
        self.rebuild_doc_id_map();

        for (doc_id, field, vector) in vectors {
            if let Some(&idx) = self.doc_id_map.get(&doc_id) {
                // Update existing vector
                self.vectors[idx] = (doc_id, field, vector);
            } else {
                // Add new vector
                let idx = self.vectors.len();
                self.vectors.push((doc_id, field, vector));
                self.doc_id_map.insert(doc_id, idx);
            }
        }

        // Update next_vec_id
        if let Some((max_id, _, _)) = self.vectors.iter().max_by_key(|(id, _, _)| id) {
            if *max_id >= self.next_vec_id {
                self.next_vec_id = *max_id + 1;
            }
        }

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
                8 + // doc_id (tuple element)
            32 + // field_name string overhead (approx)
            self.index_config.dimension * 4
                // f32 values
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

    fn vectors(&self) -> &[(u64, String, Vector)] {
        &self.vectors
    }

    fn write(&self) -> Result<()> {
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
        let file_name = format!("{}.hnsw", self.path);
        let mut output = storage.create_output(&file_name)?;

        // Write metadata
        output.write_all(&(self.vectors.len() as u64 as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.dimension as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.m as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.ef_construction as u32).to_le_bytes())?;

        // Write vectors
        // Note: In a real implementation, we would write the graph structure here
        // For now, we just write the vectors like FlatIndexWriter but with HNSW metadata

        // Write vector count (again? metadata above has it) - sticking to Flat format + HNSW params

        // Write vectors with field names and metadata
        // Write vectors with field names and metadata
        // We need to iterate in some order. Sorted by doc_id is best.
        let mut sorted_vectors: Vec<_> = self.vectors.iter().collect();
        sorted_vectors.sort_by_key(|(doc_id, _, _)| *doc_id);

        for (doc_id, field_name, vector) in sorted_vectors {
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

    fn build_reader(&self) -> Result<Arc<dyn crate::vector::reader::VectorIndexReader>> {
        use crate::vector::index::hnsw::reader::HnswIndexReader;

        let storage = self.storage.as_ref().ok_or_else(|| {
            SarissaError::InvalidOperation(
                "Cannot build reader: storage not configured".to_string(),
            )
        })?;

        let reader = HnswIndexReader::load(
            storage.as_ref(),
            &self.path,
            self.index_config.distance_metric,
        )?;

        Ok(Arc::new(reader))
    }
}
