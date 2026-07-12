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

/// Read-only, ordinal-addressed HNSW graph used by the search path
/// (Issue #686).
///
/// Nodes are identified by their **segment-local u32 ordinal** — the rank
/// of the node's doc id in the ascending, deduplicated record doc-id
/// sequence of the segment. Adjacency lists store ordinals, so the search
/// hot loop expands neighbours with plain `Vec` indexing instead of the
/// per-hop hash probe that the doc_id-keyed [`HnswGraph`] pays, and the
/// visited bitmap can be sized by [`node_count`](Self::node_count) instead
/// of the global doc-id space.
///
/// This type is built by `HnswIndexReader::load` (from either the v1
/// doc_id-encoded graph block or the v2 ordinal-encoded one) and is never
/// mutated: the write path keeps using the doc_id-keyed [`HnswGraph`],
/// which materialises ordinals only at serialization time.
#[derive(Debug, Clone)]
pub struct OrdinalHnswGraph {
    /// Entry point ordinal, `None` for an empty graph.
    entry_point: Option<u32>,
    /// Maximum level currently in the graph.
    max_level: usize,
    /// Ordinal → doc id table, strictly ascending (= the deduplicated
    /// record doc-id sequence of the segment).
    doc_ids: std::sync::Arc<[u64]>,
    /// `nodes[ordinal][level]` → neighbour ordinals.
    nodes: Vec<Vec<Vec<u32>>>,
}

impl OrdinalHnswGraph {
    /// Build a validated ordinal graph from its parts.
    ///
    /// # Arguments
    ///
    /// * `entry_point` - Entry point ordinal, `None` for an empty graph.
    /// * `max_level` - Maximum level in the graph.
    /// * `doc_ids` - Ordinal → doc id table; must be strictly ascending
    ///   and have exactly one entry per graph node.
    /// * `nodes` - Per-ordinal layered neighbour lists (ordinals).
    ///
    /// # Returns
    ///
    /// The graph, or an error if `nodes.len() != doc_ids.len()`, the
    /// entry point is out of range, or any neighbour ordinal is out of
    /// range (corrupt segment).
    pub fn from_parts(
        entry_point: Option<u32>,
        max_level: usize,
        doc_ids: std::sync::Arc<[u64]>,
        nodes: Vec<Vec<Vec<u32>>>,
    ) -> crate::error::Result<Self> {
        let node_count = doc_ids.len();
        if nodes.len() != node_count {
            return Err(crate::error::LaurusError::index(format!(
                "HNSW ordinal graph corrupt: {} nodes for {} unique record doc ids",
                nodes.len(),
                node_count
            )));
        }
        if let Some(entry) = entry_point
            && entry as usize >= node_count
        {
            return Err(crate::error::LaurusError::index(format!(
                "HNSW ordinal graph corrupt: entry ordinal {entry} out of range \
                 (node count {node_count})"
            )));
        }
        for (ord, layers) in nodes.iter().enumerate() {
            for neighbors in layers {
                for &n in neighbors {
                    if n as usize >= node_count {
                        return Err(crate::error::LaurusError::index(format!(
                            "HNSW ordinal graph corrupt: node {ord} has neighbour \
                             ordinal {n} out of range (node count {node_count})"
                        )));
                    }
                }
            }
        }
        Ok(Self {
            entry_point,
            max_level,
            doc_ids,
            nodes,
        })
    }

    /// Entry point ordinal, `None` for an empty graph.
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Maximum level currently in the graph.
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Number of nodes in the graph.
    ///
    /// Also the length of the [`doc_ids`](Self::doc_ids) table and the
    /// exclusive upper bound of every valid ordinal, so the searcher can
    /// size its visited bitmap from this.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Translate an ordinal to its doc id.
    ///
    /// # Arguments
    ///
    /// * `ord` - A valid ordinal (`< node_count`); validated at
    ///   construction, so hot-path callers index without re-checking.
    ///
    /// # Returns
    ///
    /// The doc id at rank `ord`.
    #[inline]
    pub fn doc_id(&self, ord: u32) -> u64 {
        self.doc_ids[ord as usize]
    }

    /// Neighbour ordinals of `ord` at `level`.
    ///
    /// # Arguments
    ///
    /// * `ord` - A valid ordinal (`< node_count`).
    /// * `level` - The layer level.
    ///
    /// # Returns
    ///
    /// The neighbour slice, or `None` if the node has no such level.
    #[inline]
    pub fn neighbors(&self, ord: u32, level: usize) -> Option<&[u32]> {
        self.nodes[ord as usize].get(level).map(Vec::as_slice)
    }

    /// The full ordinal → doc id table (ascending).
    pub fn doc_ids(&self) -> &std::sync::Arc<[u64]> {
        &self.doc_ids
    }

    /// Translate a doc id to its ordinal via binary search.
    ///
    /// Cold paths and tests only — the hot loop never needs this
    /// direction.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The doc id to look up.
    ///
    /// # Returns
    ///
    /// The ordinal, or `None` if the doc id has no node in this graph.
    pub fn ord_of(&self, doc_id: u64) -> Option<u32> {
        self.doc_ids
            .binary_search(&doc_id)
            .ok()
            .map(|ord| ord as u32)
    }

    /// Iterate over all nodes as `(doc_id, layers-of-neighbour-ordinals)`
    /// pairs in ordinal (= ascending doc id) order.
    ///
    /// # Returns
    ///
    /// An iterator yielding `(u64, &Vec<Vec<u32>>)` pairs.
    pub fn iter_nodes(&self) -> impl Iterator<Item = (u64, &Vec<Vec<u32>>)> {
        self.doc_ids
            .iter()
            .zip(self.nodes.iter())
            .map(|(&doc_id, layers)| (doc_id, layers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn doc_ids(ids: &[u64]) -> Arc<[u64]> {
        Arc::from(ids.to_vec())
    }

    #[test]
    fn ordinal_graph_from_parts_roundtrips_accessors() {
        // 3 nodes: doc ids 10, 20, 30; node 0 has two levels.
        let nodes = vec![
            vec![vec![1, 2], vec![1]],
            vec![vec![0, 2]],
            vec![vec![0, 1]],
        ];
        let g = OrdinalHnswGraph::from_parts(Some(0), 1, doc_ids(&[10, 20, 30]), nodes).unwrap();

        assert_eq!(g.entry_point(), Some(0));
        assert_eq!(g.max_level(), 1);
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.doc_id(0), 10);
        assert_eq!(g.doc_id(2), 30);
        assert_eq!(g.neighbors(0, 0), Some(&[1u32, 2][..]));
        assert_eq!(g.neighbors(0, 1), Some(&[1u32][..]));
        assert_eq!(g.neighbors(1, 1), None);
        assert_eq!(g.ord_of(20), Some(1));
        assert_eq!(g.ord_of(15), None);

        let collected: Vec<u64> = g.iter_nodes().map(|(id, _)| id).collect();
        assert_eq!(collected, vec![10, 20, 30]);
    }

    #[test]
    fn ordinal_graph_empty_is_valid() {
        let g = OrdinalHnswGraph::from_parts(None, 0, doc_ids(&[]), Vec::new()).unwrap();
        assert_eq!(g.entry_point(), None);
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.ord_of(1), None);
    }

    #[test]
    fn ordinal_graph_rejects_node_count_mismatch() {
        let err =
            OrdinalHnswGraph::from_parts(None, 0, doc_ids(&[10, 20]), vec![vec![]]).unwrap_err();
        assert!(err.to_string().contains("unique record doc ids"));
    }

    #[test]
    fn ordinal_graph_rejects_out_of_range_entry_point() {
        let err =
            OrdinalHnswGraph::from_parts(Some(2), 0, doc_ids(&[10, 20]), vec![vec![], vec![]])
                .unwrap_err();
        assert!(err.to_string().contains("entry ordinal"));
    }

    #[test]
    fn ordinal_graph_rejects_out_of_range_neighbor() {
        let err = OrdinalHnswGraph::from_parts(
            None,
            0,
            doc_ids(&[10, 20]),
            vec![vec![vec![7]], vec![vec![0]]],
        )
        .unwrap_err();
        assert!(err.to_string().contains("neighbour ordinal 7"));
    }
}
