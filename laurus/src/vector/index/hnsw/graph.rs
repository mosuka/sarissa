use ahash::AHashMap;
use serde::{Deserialize, Serialize};

/// Represents the HNSW graph structure.
///
/// This structure holds the connectivity information between vectors in the index.
/// It tracks the entry point, node connections at each layer, and configuration parameters.
///
/// Internally, nodes are stored in a contiguous `Vec` for O(1) index-based access,
/// with an `AHashMap` providing the mapping from document IDs to internal indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswGraph {
    /// Entry point node ID (doc_id).
    /// This is the starting point for searches, usually the node present in the highest level.
    pub entry_point: Option<u64>,

    /// Maximum level currently in the graph.
    pub max_level: usize,

    /// Mapping from document ID to internal graph index.
    id_to_index: AHashMap<u64, usize>,

    /// Mapping from internal graph index to document ID.
    index_to_id: Vec<u64>,

    /// nodes[index][level] -> neighbor doc_ids list.
    nodes: Vec<Vec<Vec<u64>>>,

    /// Examples of HNSW parameters that might be useful to store with the graph,
    /// though some are primarily construction-time parameters.
    pub m: usize,
    pub m_max: usize,   // Max neighbors per node for higher levels (usually M)
    pub m_max_0: usize, // Max neighbors for layer 0 (usually 2*M)
    pub ef_construction: usize,
    pub level_mult: f64,

    /// Largest `doc_id` ever inserted into the graph. Cached so search-
    /// time code can size a visited-set bitmap (`BitVec` of length
    /// `max_doc_id + 1`) without rescanning the graph. Updated by
    /// [`get_or_create_index`] when a new node is added.
    max_doc_id: u64,
}

impl HnswGraph {
    /// Create a new HnswGraph from a HashMap of doc_id -> layers.
    ///
    /// This converts the HashMap-based representation into the internal Vec-based storage
    /// for O(1) access by internal index.
    ///
    /// # Arguments
    /// * `entry_point` - The entry point node ID.
    /// * `max_level` - Maximum level in the graph.
    /// * `nodes_map` - HashMap from doc_id to layered neighbor lists.
    /// * `m` - HNSW M parameter.
    /// * `m_max` - Max neighbors for higher levels.
    /// * `m_max_0` - Max neighbors for layer 0.
    /// * `ef_construction` - ef_construction parameter.
    /// * `level_mult` - Level multiplier.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        entry_point: Option<u64>,
        max_level: usize,
        nodes_map: std::collections::HashMap<u64, Vec<Vec<u64>>>,
        m: usize,
        m_max: usize,
        m_max_0: usize,
        ef_construction: usize,
        level_mult: f64,
    ) -> Self {
        let mut id_to_index = AHashMap::with_capacity(nodes_map.len());
        let mut index_to_id = Vec::with_capacity(nodes_map.len());
        let mut nodes = Vec::with_capacity(nodes_map.len());
        let mut max_doc_id: u64 = 0;

        for (doc_id, layers) in nodes_map {
            let index = nodes.len();
            id_to_index.insert(doc_id, index);
            index_to_id.push(doc_id);
            nodes.push(layers);
            if doc_id > max_doc_id {
                max_doc_id = doc_id;
            }
        }

        Self {
            entry_point,
            max_level,
            id_to_index,
            index_to_id,
            nodes,
            m,
            m_max,
            m_max_0,
            ef_construction,
            level_mult,
            max_doc_id,
        }
    }

    /// Largest `doc_id` currently stored in the graph.
    ///
    /// Returns `0` for an empty graph; callers should sanity-check
    /// against [`node_count`](Self::node_count) before allocating.
    /// Used by `HnswSearcher` to size the visited-set bitmap.
    pub fn max_doc_id(&self) -> u64 {
        self.max_doc_id
    }

    /// Get neighbors of a node at a specific level.
    ///
    /// # Arguments
    /// * `doc_id` - The document ID.
    /// * `level` - The layer level.
    ///
    /// # Returns
    /// A reference to the neighbor list, or `None` if the node or level does not exist.
    pub fn get_neighbors(&self, doc_id: u64, level: usize) -> Option<&Vec<u64>> {
        let &index = self.id_to_index.get(&doc_id)?;
        self.nodes.get(index).and_then(|levels| levels.get(level))
    }

    /// Set neighbors for a node at a specific level (replacing existing ones).
    ///
    /// # Arguments
    /// * `doc_id` - The document ID.
    /// * `level` - The layer level.
    /// * `neighbors` - The new neighbor list.
    pub fn set_neighbors(&mut self, doc_id: u64, level: usize, neighbors: Vec<u64>) {
        let index = self.get_or_create_index(doc_id);
        if level < self.nodes[index].len() {
            self.nodes[index][level] = neighbors;
        }
    }

    /// Get or create an internal index for a document ID.
    fn get_or_create_index(&mut self, doc_id: u64) -> usize {
        if let Some(&index) = self.id_to_index.get(&doc_id) {
            index
        } else {
            let index = self.nodes.len();
            self.id_to_index.insert(doc_id, index);
            self.index_to_id.push(doc_id);
            self.nodes.push(Vec::new());
            if doc_id > self.max_doc_id {
                self.max_doc_id = doc_id;
            }
            index
        }
    }

    /// Check if a node exists in the graph.
    ///
    /// # Arguments
    /// * `doc_id` - The document ID to check.
    ///
    /// # Returns
    /// `true` if the node exists.
    pub fn contains_node(&self, doc_id: &u64) -> bool {
        self.id_to_index.contains_key(doc_id)
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the layers for a specific node by document ID.
    ///
    /// # Arguments
    /// * `doc_id` - The document ID.
    ///
    /// # Returns
    /// A reference to the layers, or `None` if the node does not exist.
    pub fn get_node_layers(&self, doc_id: &u64) -> Option<&Vec<Vec<u64>>> {
        let &index = self.id_to_index.get(doc_id)?;
        self.nodes.get(index)
    }

    /// Iterate over all nodes as (doc_id, layers) pairs.
    ///
    /// # Returns
    /// An iterator yielding `(u64, &Vec<Vec<u64>>)` pairs.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (u64, &Vec<Vec<u64>>)> {
        self.index_to_id
            .iter()
            .zip(self.nodes.iter())
            .map(|(&doc_id, layers)| (doc_id, layers))
    }

    /// Consume the graph and return an iterator over all nodes as (doc_id, layers) pairs.
    ///
    /// # Returns
    /// An iterator yielding owned `(u64, Vec<Vec<u64>>)` pairs.
    pub fn into_iter_nodes(self) -> impl Iterator<Item = (u64, Vec<Vec<u64>>)> {
        self.index_to_id.into_iter().zip(self.nodes)
    }

    /// Get a sorted iterator over all nodes (sorted by doc_id).
    ///
    /// Used for deterministic serialization.
    ///
    /// # Returns
    /// A vector of `(u64, &Vec<Vec<u64>>)` pairs sorted by doc_id.
    pub fn sorted_nodes(&self) -> Vec<(u64, &Vec<Vec<u64>>)> {
        let mut pairs: Vec<_> = self.iter_nodes().collect();
        pairs.sort_by_key(|(id, _)| *id);
        pairs
    }
}
