//! HNSW (Hierarchical Navigable Small World) index builder for approximate search.

use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::rerank::RerankStorageKind;
use crate::vector::core::vector::Vector;
use crate::vector::index::HnswIndexConfig;
use crate::vector::index::alloc_bounds::{checked_capacity, checked_len};
use crate::vector::index::field::LegacyVectorFieldWriter;
use crate::vector::index::format::{QuantHeader, VectorSegmentHeader};
use crate::vector::index::hnsw::graph::HnswGraph;
use crate::vector::index::quantized_io::{
    quantize_segment, quantized_record_payload_size, read_dequantized_vector,
    write_quantized_record,
};
use crate::vector::index::rerank_sidecar::{read_sidecar, write_sidecar};
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
use parking_lot::RwLock;
use rand::{RngExt, SeedableRng};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Fixed seed for the HNSW level-selection RNG (Issue #841).
///
/// Level selection needs no secret randomness — what matters is the
/// geometric level distribution, not its unpredictability — so the
/// build uses a deterministic generator seeded with this constant.
/// This makes graph topology reproducible for a given insertion order:
/// segment builds, merges, and topology-sensitive tests all become
/// deterministic instead of flaking on unlucky layouts. Precedent:
/// Lucene's `HnswGraphBuilder` builds with `DEFAULT_RAND_SEED = 42`.
const LEVEL_RNG_SEED: u64 = 42;

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
        if let Some(layers) = self.nodes.get(&doc_id)
            && let Some(lock) = layers.get(level)
        {
            *lock.write() = new_neighbors;
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
        if let Some(layers) = self.nodes.get(&doc_id)
            && let Some(lock) = layers.get(level)
        {
            // Acquire lock briefly to add the neighbor and snapshot the list
            // if pruning is needed.  Distance calculations happen outside the
            // lock to reduce contention across parallel threads.
            let needs_pruning = {
                let mut neighbors = lock.write();
                if !neighbors.contains(&neighbor_id) {
                    neighbors.push(neighbor_id);
                }
                if neighbors.len() > max_conn {
                    Some(neighbors.clone())
                } else {
                    None
                }
            };

            if let Some(snapshot) = needs_pruning {
                let pruned = writer.prune_neighbors(doc_id, snapshot, max_conn)?;
                *lock.write() = pruned;
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
        let mut nodes = HashMap::with_capacity(graph.node_count());
        for (doc_id, layered_neighbors) in graph.into_iter_nodes() {
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
    #[allow(dead_code)] // Maintained during build but not yet read; reserved for future use
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
        self.distance.total_cmp(&other.distance)
    }
}

impl HnswIndexWriter {
    /// Create a new HNSW index builder.
    pub fn new(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
        path: impl Into<String>,
    ) -> Result<Self> {
        if index_config.m < 2 {
            return Err(crate::error::LaurusError::InvalidOperation(
                "HNSW parameter m must be >= 2".to_string(),
            ));
        }
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
    ///
    /// If an existing index file is found on disk, its vectors are loaded
    /// into the writer so that the next commit preserves them. This
    /// prevents data loss across multiple commit cycles.
    pub fn with_storage(
        index_config: HnswIndexConfig,
        writer_config: VectorIndexWriterConfig,
        path: impl Into<String>,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let path = path.into();
        let file_name = format!("{}.hnsw", path);
        if storage.file_exists(&file_name) {
            return Self::load(index_config, writer_config, storage, &path);
        }

        if index_config.m < 2 {
            return Err(crate::error::LaurusError::InvalidOperation(
                "HNSW parameter m must be >= 2".to_string(),
            ));
        }
        let max_level = Self::calculate_max_level(index_config.m, index_config.ef_construction);
        let _ml = 1.0 / (index_config.m as f64).ln();

        Ok(Self {
            index_config,
            writer_config,
            storage: Some(storage),
            path,
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
        use std::io::{Read, Seek};

        // Open the index file
        let file_name = format!("{}.hnsw", path);
        let mut input = storage.open_input(&file_name)?;

        // Ground truth for bounding allocations sized from unverified header
        // counts below (Issue #806). Unlike the reader, this writer load path
        // runs no checksum footer verification, so every count — including
        // those of footer-carrying segments — reaches its allocation unverified.
        let file_size = input.size()?;

        // Read metadata (vector count stored as u64)
        let mut num_vectors_buf = [0u8; 8];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u64::from_le_bytes(num_vectors_buf) as usize;

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
            return Err(LaurusError::InvalidOperation(format!(
                "Dimension mismatch: expected {}, found {}",
                index_config.dimension, dimension
            )));
        }

        // Read the Issue #481 Stage 1 / Stage 3 vector segment header
        // (LVS1). Pre-Stage-1 segments are rejected with
        // IncompatibleFormat by the reader. Both Scalar8Bit and
        // ProductQuantization (Stage 3, #481) variants are reconstituted
        // back to f32 for the writer's in-memory state — the on-disk
        // form is rebuilt from scratch in `write()` once add_vector /
        // delete_document calls have replayed.
        let header = VectorSegmentHeader::read_from(&mut input)?;

        // Read quantized vectors and dequantize back to f32 for the
        // in-memory writer state. The dequantized values are a lossy
        // approximation of the originals; if a Stage 2 sidecar is
        // present we'll overwrite them below with the lossless f32
        // payload.
        // Bytes left for the per-vector records section (Issue #806). Each
        // record is at least doc_id (8) + field_name_len (4) + the per-kind
        // fixed payload, so `record_stride` also bounds the per-record payload
        // read decoded below.
        let records_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        let min_payload = match &header.quant {
            QuantHeader::Scalar8Bit(_) => quantized_record_payload_size(dimension) as u64,
            QuantHeader::ProductQuantization { params, .. } => params.m as u64,
            #[cfg(feature = "pq-fastscan")]
            QuantHeader::ProductQuantizationFastScan { .. } => 1,
        };
        let record_stride = 12 + min_payload;
        checked_capacity(
            num_vectors,
            record_stride,
            records_remaining,
            "hnsw num_vectors",
        )?;
        let mut vectors = Vec::with_capacity(num_vectors);
        for _ in 0..num_vectors {
            let mut doc_id_buf = [0u8; 8];
            input.read_exact(&mut doc_id_buf)?;
            let doc_id = u64::from_le_bytes(doc_id_buf);

            // Read field name
            let mut field_name_len_buf = [0u8; 4];
            input.read_exact(&mut field_name_len_buf)?;
            let field_name_len = u32::from_le_bytes(field_name_len_buf) as usize;
            checked_len(field_name_len, records_remaining, "hnsw field_name_len")?;

            let mut field_name_buf = vec![0u8; field_name_len];
            input.read_exact(&mut field_name_buf)?;
            let field_name = String::from_utf8(field_name_buf).map_err(|e| {
                LaurusError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
            })?;

            // Decode the per-vector payload according to the segment's
            // quantization kind.
            let values = match &header.quant {
                QuantHeader::Scalar8Bit(params) => {
                    read_dequantized_vector(&mut input, dimension, params)?
                }
                QuantHeader::ProductQuantization { params, codebook } => {
                    crate::vector::index::pq_io::read_dequantized_pq_vector(
                        &mut input, *params, codebook,
                    )?
                }
                #[cfg(feature = "pq-fastscan")]
                QuantHeader::ProductQuantizationFastScan { params, codebook } => {
                    crate::vector::index::pq_fastscan_io::read_dequantized_pq_fastscan_vector(
                        &mut input, *params, codebook,
                    )?
                }
            };

            vectors.push((doc_id, field_name, Vector::new(values)));
        }

        // Stage 2 (Issue #481): if the LRS1 sidecar exists alongside
        // the main `.hnsw` file, prefer its lossless f32 payload over
        // the dequantized int8 values. This keeps the in-memory writer
        // state byte-exact across load -> add -> write cycles, so a
        // re-emitted sidecar does not slowly bleed precision through
        // repeated dequant -> requantize round-trips.
        let sidecar_name = format!("{}.f32", file_name);
        if storage.file_exists(&sidecar_name) {
            let mut sidecar_in = storage.open_input(&sidecar_name)?;
            let sidecar_size = sidecar_in.size()?;
            let (header, payload) = read_sidecar(&mut sidecar_in, sidecar_size)?;
            if header.dim as usize != dimension {
                return Err(LaurusError::InvalidOperation(format!(
                    "rerank sidecar dim mismatch: LVS1 segment uses {dimension}, sidecar uses {}",
                    header.dim
                )));
            }
            if header.vector_count as usize != vectors.len() {
                return Err(LaurusError::InvalidOperation(format!(
                    "rerank sidecar vector_count mismatch: LVS1 segment has {} vectors, \
                     sidecar has {}",
                    vectors.len(),
                    header.vector_count
                )));
            }
            match header.storage_kind {
                RerankStorageKind::F32 => {
                    let bytes_per_vec = dimension * 4;
                    for (i, (_, _, vec)) in vectors.iter_mut().enumerate() {
                        let start = i * bytes_per_vec;
                        let mut data = Vec::with_capacity(dimension);
                        for j in 0..dimension {
                            let lo = start + j * 4;
                            data.push(f32::from_le_bytes([
                                payload[lo],
                                payload[lo + 1],
                                payload[lo + 2],
                                payload[lo + 3],
                            ]));
                        }
                        *vec = Vector::new(data);
                    }
                }
            }
        }

        // Rebuild doc_id_map
        let mut doc_id_map = HashMap::new();
        for (i, (doc_id, _, _)) in vectors.iter().enumerate() {
            doc_id_map.insert(*doc_id, i);
        }

        // Calculate next_vec_id from loaded vectors
        let max_id = vectors.iter().map(|(id, _, _)| *id).max().unwrap_or(0);
        let next_vec_id = if num_vectors > 0 { max_id + 1 } else { 0 };

        if index_config.m < 2 {
            return Err(LaurusError::InvalidOperation(
                "HNSW parameter m must be >= 2".to_string(),
            ));
        }
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

                // Read nodes (u64 to match write format)
                let mut node_count_buf = [0u8; 8];
                input.read_exact(&mut node_count_buf)?;
                let node_count = u64::from_le_bytes(node_count_buf) as usize;

                // Bound every graph allocation by the bytes left in the file
                // (Issue #806). Reused for the inner layer / neighbor counts so
                // no extra syscall is taken inside the per-node / per-layer
                // loops.
                let graph_remaining =
                    file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
                // Each node serializes at least doc_id (8) + layer_count (4).
                checked_capacity(node_count, 12, graph_remaining, "hnsw node_count")?;
                let mut nodes = HashMap::with_capacity(node_count);

                for _ in 0..node_count {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let mut layer_count_buf = [0u8; 4];
                    input.read_exact(&mut layer_count_buf)?;
                    let layer_count = u32::from_le_bytes(layer_count_buf) as usize;

                    // Each layer serializes at least its neighbor_count (4).
                    checked_capacity(layer_count, 4, graph_remaining, "hnsw layer_count")?;
                    let mut layers = Vec::with_capacity(layer_count);

                    for _ in 0..layer_count {
                        let mut neighbor_count_buf = [0u8; 4];
                        input.read_exact(&mut neighbor_count_buf)?;
                        let neighbor_count = u32::from_le_bytes(neighbor_count_buf) as usize;

                        // Each neighbor serializes as a u64 (8 bytes).
                        checked_capacity(
                            neighbor_count,
                            8,
                            graph_remaining,
                            "hnsw neighbor_count",
                        )?;
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

                Some(HnswGraph::new(
                    entry_point,
                    max_level,
                    nodes,
                    index_config.m,
                    index_config.m,
                    index_config.m * 2,
                    index_config.ef_construction,
                    _ml,
                ))
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
    ///
    /// # Arguments
    ///
    /// * `rng` - The build-scoped level RNG, seeded with
    ///   [`LEVEL_RNG_SEED`] so graph topology is deterministic for a
    ///   given insertion order (Issue #841).
    ///
    /// # Returns
    ///
    /// The layer index, geometrically distributed with ratio `_ml` and
    /// capped at 16.
    fn select_layer(&self, rng: &mut impl RngExt) -> usize {
        let mut layer = 0;

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
                return Err(LaurusError::InvalidOperation(format!(
                    "Vector {} has dimension {}, expected {}",
                    doc_id,
                    vector.dimension(),
                    self.index_config.dimension
                )));
            }

            if !vector.is_valid() {
                return Err(LaurusError::InvalidOperation(format!(
                    "Vector {doc_id} contains invalid values (NaN or infinity)"
                )));
            }
        }

        Ok(())
    }

    /// Normalize vectors if configured to do so.
    /// Normalize vectors if configured to do so.
    #[allow(unused_variables)]
    fn normalize_vectors_internal(
        index_config: &HnswIndexConfig,
        writer_config: &VectorIndexWriterConfig,
        vectors: &mut Vec<(u64, String, Vector)>,
    ) {
        if !index_config.normalize_vectors {
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        if writer_config.parallel_build && vectors.len() > 100 {
            vectors.par_iter_mut().for_each(|(_, _, vector)| {
                vector.normalize();
            });
            return;
        }

        for (_, _, vector) in vectors {
            vector.normalize();
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

        // TODO: replace with tracing::info! when a logging crate is added
        // "Building HNSW graph with {count} vectors (parallel), M={m}, efConstruction={ef}"

        // Ensure doc_id_map is up to date
        self.rebuild_doc_id_map();

        let m = self.index_config.m;
        let m_max = m;
        let m_max_0 = m * 2;
        let ef_construction = self.index_config.ef_construction;

        // Determine which vectors are new and need insertion
        let mut new_node_levels = Vec::new(); // (doc_id, level)
        let mut new_doc_ids_in_order = Vec::new();

        // Deterministic level RNG (Issue #841): one seeded generator per
        // build, threaded through the serial level-assignment loops below.
        let mut level_rng = rand::rngs::StdRng::seed_from_u64(LEVEL_RNG_SEED);

        // Check if we have an existing graph to append to
        let (graph, entry_point, max_level, search_entry_point) =
            if let Some(existing_graph) = self.graph.take() {
                // Identify new vectors
                for (doc_id, _, _) in &self.vectors {
                    if !existing_graph.contains_node(doc_id) {
                        new_doc_ids_in_order.push(*doc_id);
                    }
                }
                new_doc_ids_in_order.sort_unstable();

                // Assign levels to new vectors
                for doc_id in &new_doc_ids_in_order {
                    let level = self.select_layer(&mut level_rng);
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
                    let level = self.select_layer(&mut level_rng);
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
        // Each thread inserts one node using `search_entry_point` as the
        // starting node for greedy search.  The HashMap in ConcurrentHnswGraph
        // is pre-populated (immutable during this phase); individual neighbor
        // lists are protected by RwLock.
        let writer_ref = &*self;

        #[cfg(not(target_arch = "wasm32"))]
        #[allow(unused_mut)]
        let mut iter = new_doc_ids_in_order.into_par_iter();
        #[cfg(target_arch = "wasm32")]
        let mut iter = new_doc_ids_in_order.into_iter();

        iter.try_for_each(|doc_id| -> Result<()> {
            let doc_vector_idx = *writer_ref.doc_id_map.get(&doc_id).ok_or_else(|| {
                LaurusError::internal(format!("Doc ID {} not found in doc_id_map", doc_id))
            })?;
            let vector = &writer_ref.vectors[doc_vector_idx].2;

            // Determine the starting node for search.
            // For incremental builds `search_entry_point` is the OLD entry
            // point so new nodes (including a promoted EP) always get
            // connected to the existing graph.
            let start_node = match search_entry_point {
                Some(sp) => sp,
                None => return Ok(()), // No existing node to search from
            };

            // Skip insertion of the search start node itself (full-build
            // seed node — other nodes will link TO it via bidirectional edges).
            if start_node == doc_id {
                return Ok(());
            }

            // Determine the assigned level from the pre-populated graph
            let layers_len = graph.nodes.get(&doc_id).map(|l| l.len()).unwrap_or(0);
            if layers_len == 0 {
                return Ok(());
            }
            let level = layers_len - 1;

            let max_level = graph.max_level;
            let mut curr_obj = start_node;
            let mut dist = writer_ref.calc_dist(vector, curr_obj)?;

            // Phase A: Greedy descent from top layer down to level + 1
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

            // Phase B: Search & connect from min(max_level, level) down to 0
            let top_level = usize::min(max_level, level);
            for lc in (0..=top_level).rev() {
                let candidates =
                    writer_ref.search_layer(&graph, curr_obj, vector, ef_construction, lc)?;

                if let Some(min_cand) = candidates
                    .iter()
                    .min_by(|a, b| a.distance.total_cmp(&b.distance))
                {
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

        self.graph = Some(HnswGraph::new(
            entry_point,
            max_level,
            final_nodes,
            m,
            m_max,
            m_max_0,
            ef_construction,
            1.0 / (self.index_config.m as f64).ln(),
        ));
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
            .ok_or_else(|| LaurusError::internal(format!("Doc ID {} not found in map", doc_id)))?;
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
                other.distance.total_cmp(&self.distance)
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
            if let Some(furthest_found) = found.peek()
                && curr.distance > furthest_found.distance
                && found.len() >= ef
            {
                break;
            }

            if let Some(neighbors) = graph.get_neighbors_view(curr.id, level) {
                for neighbor_id in neighbors {
                    // Use insert() return value to avoid double hash lookup.
                    if !visited.insert(neighbor_id) {
                        continue;
                    }

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
        // Simple heuristic: take M nearest.
        // Collect without cloning the heap, then sort by ascending distance.
        let mut sorted: Vec<_> = candidates.iter().cloned().collect();
        sorted.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
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
        let idx = *self.doc_id_map.get(&doc_id).ok_or_else(|| {
            LaurusError::internal(format!(
                "Doc ID {} not found in doc_id_map during pruning",
                doc_id
            ))
        })?;
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
        candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        candidates.truncate(max_conn);

        Ok(candidates.into_iter().map(|c| c.id).collect())
    }

    /// Check for memory limits.
    fn check_memory_limit(&self) -> Result<()> {
        if let Some(limit) = self.writer_config.memory_limit {
            let current_usage = self.estimated_memory_usage();
            if current_usage > limit {
                return Err(LaurusError::ResourceExhausted(format!(
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

#[async_trait::async_trait]
impl VectorIndexWriter for HnswIndexWriter {
    fn next_vector_id(&self) -> u64 {
        self.next_vec_id
    }

    fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(LaurusError::InvalidOperation(
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
        if let Some((max_id, _, _)) = self.vectors.iter().max_by_key(|(id, _, _)| id)
            && *max_id >= self.next_vec_id
        {
            self.next_vec_id = *max_id + 1;
        }

        self.total_vectors_to_add = Some(self.vectors.len());

        self.check_memory_limit()?;
        Ok(())
    }

    fn add_vectors(&mut self, mut vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        if self.is_finalized {
            self.is_finalized = false;
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
        if let Some((max_id, _, _)) = self.vectors.iter().max_by_key(|(id, _, _)| id)
            && *max_id >= self.next_vec_id
        {
            self.next_vec_id = *max_id + 1;
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
            return Err(LaurusError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let storage = self
            .storage
            .as_ref()
            .ok_or_else(|| LaurusError::InvalidOperation("No storage configured".to_string()))?;

        // Write to a temp file and atomically rename into place (Issue #784)
        // so a crash mid-write leaves the previously committed `.hnsw` intact
        // instead of a truncated, unreadable segment.
        let file_name = format!("{}.hnsw", self.path);
        let tmp_name = format!("{}.hnsw.tmp", self.path);
        // Wrap the output so a CRC-32 accumulates over the segment bytes as
        // they are written; a checksum footer is appended below (Issue #786).
        let mut output =
            crate::storage::checksum::CrcWriter::new(storage.create_output(&tmp_name)?);

        // Write metadata (vector count as u64 to avoid truncation)
        output.write_all(&(self.vectors.len() as u64).to_le_bytes())?;
        output.write_all(&(self.index_config.dimension as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.m as u32).to_le_bytes())?;
        output.write_all(&(self.index_config.ef_construction as u32).to_le_bytes())?;

        // Write vectors using the Issue #481 quantized format. The
        // HNSW-specific 28-byte preamble above (count / dim / m / ef)
        // stays unchanged so the graph parameters are still readable
        // first; the vector payload is quantized to int8 (Stage 1) or
        // PQ codes (Stage 3, #481) according to the field's
        // `quantization_method`, prefixed by `VectorSegmentHeader`
        // (LVS1).

        // Sort by doc_id for deterministic serialization.
        let mut sorted_vectors: Vec<_> = self.vectors.iter().collect();
        sorted_vectors.sort_by_key(|(doc_id, _, _)| *doc_id);

        let f32_vectors: Vec<Vector> = sorted_vectors
            .iter()
            .map(|(_, _, v)| (*v).clone())
            .collect();

        match self.index_config.quantization_method {
            crate::vector::core::quantization::QuantizationMethod::Scalar8Bit => {
                // Empty segments fall back to neutral params (0.0, 1.0)
                // since there is nothing to train on; the LVS1 header
                // is still emitted so readers can dispatch on
                // quant_kind uniformly.
                let (params, records) = if f32_vectors.is_empty() {
                    (
                        crate::vector::core::quantization::ScalarQuantParams {
                            offset: 0.0,
                            scale: 1.0,
                        },
                        Vec::new(),
                    )
                } else {
                    quantize_segment(&f32_vectors, self.index_config.dimension)?
                };
                VectorSegmentHeader::scalar_8bit(params).write_to(&mut output)?;
                for ((doc_id, field_name, _), (int8, meta)) in
                    sorted_vectors.iter().zip(records.iter())
                {
                    output.write_all(&doc_id.to_le_bytes())?;
                    let field_name_bytes = field_name.as_bytes();
                    output.write_all(&(field_name_bytes.len() as u32).to_le_bytes())?;
                    output.write_all(field_name_bytes)?;
                    write_quantized_record(&mut output, int8, *meta)?;
                }
            }
            crate::vector::core::quantization::QuantizationMethod::ProductQuantization {
                subvector_count,
            } => {
                if f32_vectors.is_empty() {
                    // An empty segment still needs a well-formed LVS1
                    // header so the reader can dispatch on quant_kind.
                    // We pick a minimal (m=1, sub_dim=dim) codebook of
                    // a single zero centroid per sub-vector — readers
                    // will never index into it because there are no
                    // codes after it.
                    let params = crate::vector::core::quantization::PqParams::from_dim_and_m(
                        self.index_config.dimension,
                        subvector_count.max(1),
                    )?;
                    let codebook = vec![0.0_f32; params.codebook_len()];
                    VectorSegmentHeader::product_quantization(params, codebook)
                        .write_to(&mut output)?;
                } else {
                    let (params, codebook, codes) =
                        crate::vector::index::pq_io::quantize_segment_pq(
                            &f32_vectors,
                            self.index_config.dimension,
                            subvector_count,
                        )?;
                    VectorSegmentHeader::product_quantization(params, codebook)
                        .write_to(&mut output)?;
                    for ((doc_id, field_name, _), codes_i) in
                        sorted_vectors.iter().zip(codes.iter())
                    {
                        output.write_all(&doc_id.to_le_bytes())?;
                        let field_name_bytes = field_name.as_bytes();
                        output.write_all(&(field_name_bytes.len() as u32).to_le_bytes())?;
                        output.write_all(field_name_bytes)?;
                        crate::vector::index::pq_io::write_pq_record(&mut output, codes_i)?;
                    }
                }
            }
            #[cfg(feature = "pq-fastscan")]
            crate::vector::core::quantization::QuantizationMethod::ProductQuantizationFastScan {
                subvector_count,
            } => {
                if f32_vectors.is_empty() {
                    // Empty segment: emit a well-formed LVS1 header with a
                    // minimal zero-centroid K=16 codebook so the reader can
                    // dispatch on quant_kind. Mirrors the PQ-256 empty path.
                    let m = subvector_count.max(1);
                    let sub_dim = self.index_config.dimension / m;
                    let params = crate::vector::core::quantization::PqParams::new(
                        m as u16,
                        16,
                        sub_dim as u16,
                    )?;
                    let codebook = vec![0.0_f32; params.codebook_len()];
                    VectorSegmentHeader::product_quantization_fastscan(params, codebook)
                        .write_to(&mut output)?;
                } else {
                    let (params, codebook, codes) =
                        crate::vector::index::pq_fastscan_io::quantize_segment_pq_fastscan(
                            &f32_vectors,
                            self.index_config.dimension,
                            subvector_count,
                        )?;
                    VectorSegmentHeader::product_quantization_fastscan(params, codebook)
                        .write_to(&mut output)?;
                    for ((doc_id, field_name, _), codes_i) in
                        sorted_vectors.iter().zip(codes.iter())
                    {
                        output.write_all(&doc_id.to_le_bytes())?;
                        let field_name_bytes = field_name.as_bytes();
                        output.write_all(&(field_name_bytes.len() as u32).to_le_bytes())?;
                        output.write_all(field_name_bytes)?;
                        crate::vector::index::pq_fastscan_io::write_pq_fastscan_record(
                            &mut output,
                            codes_i,
                        )?;
                    }
                }
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

            // Nodes (u64 to avoid truncation for large graphs)
            let node_count = graph.node_count() as u64;
            output.write_all(&node_count.to_le_bytes())?;

            // Sort nodes by doc_id for deterministic serialization
            let sorted_nodes = graph.sorted_nodes();

            for (doc_id, layers) in sorted_nodes {
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

        // Append the CRC-32 footer (magic + checksum over all preceding bytes)
        // so a corrupted segment is rejected on load (Issue #786).
        let content_crc = output.checksum();
        output.write_all(&crate::vector::index::hnsw::HNSW_FOOTER_MAGIC.to_le_bytes())?;
        output.write_all(&content_crc.to_le_bytes())?;
        output.flush()?;
        drop(output);
        storage.rename_file(&tmp_name, &file_name)?;

        // Stage 2 (Issue #481): emit the optional LRS1 rerank sidecar
        // alongside the main int8 segment. The sidecar's payload order
        // matches `sorted_vectors` (the same doc_id ordering used for
        // the LVS1 records above), which keeps a (sidecar position) ->
        // (LVS1 position) mapping at the identity. Sidecar is written
        // only when explicitly enabled per field; absence keeps Stage 1
        // (int8-only) behavior intact.
        if let Some(rerank_kind) = self.index_config.rerank_storage {
            // Same temp-then-rename atomicity for the rerank sidecar (#784).
            let sidecar_name = format!("{}.f32", file_name);
            let sidecar_tmp = format!("{}.f32.tmp", file_name);
            let mut sidecar_out = storage.create_output(&sidecar_tmp)?;
            let mut payload: Vec<f32> =
                Vec::with_capacity(sorted_vectors.len() * self.index_config.dimension);
            for (_, _, v) in &sorted_vectors {
                payload.extend_from_slice(&v.data);
            }
            write_sidecar(
                &mut sidecar_out,
                rerank_kind,
                self.index_config.dimension as u32,
                &payload,
            )?;
            sidecar_out.flush()?;
            drop(sidecar_out);
            storage.rename_file(&sidecar_tmp, &sidecar_name)?;
        }

        Ok(())
    }

    fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        if self.is_finalized {
            self.is_finalized = false;
        }

        // Logical deletion from buffer
        let initial_len = self.vectors.len();
        self.vectors.retain(|(id, _, _)| *id != doc_id);

        if self.vectors.len() < initial_len {
            self.rebuild_doc_id_map();
            // Invalidate the HNSW graph — it still contains edges
            // referencing the deleted doc_id.  The graph will be rebuilt
            // on the next finalize().
            self.graph = None;
        }
        Ok(())
    }

    fn delete_documents(&mut self, _field: &str, _value: &str) -> Result<usize> {
        if self.is_finalized {
            return Err(LaurusError::InvalidOperation(
                "Cannot delete documents from finalized index".to_string(),
            ));
        }

        // Vectors no longer carry metadata; field-based deletion is not supported.
        // Use delete_document(doc_id) for document-level deletion.
        Ok(0)
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
            LaurusError::InvalidOperation("Cannot build reader: storage not configured".to_string())
        })?;

        let reader = HnswIndexReader::load(
            storage.clone(),
            &self.path,
            self.index_config.distance_metric,
        )?;

        Ok(Arc::new(reader))
    }
}
