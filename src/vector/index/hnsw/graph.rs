use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Represents the HNSW graph structure.
///
/// This structure holds the connectivity information between vectors in the index.
/// It tracks the entry point, node connections at each layer, and configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswGraph {
    /// Entry point node ID (doc_id).
    /// This is the starting point for searches, usually the node present in the highest level.
    pub entry_point: Option<u64>,

    /// Maximum level currently in the graph.
    pub max_level: usize,

    /// Nodes: map from doc_id to its layers.
    /// Each layer contains a list of neighbor doc_ids.
    /// `nodes[doc_id][level]` -> list of neighbor doc_ids at that level.
    /// Level 0 is the base layer.
    pub nodes: HashMap<u64, Vec<Vec<u64>>>,

    /// Examples of HNSW parameters that might be useful to store with the graph,
    /// though some are primarily construction-time parameters.
    pub m: usize,
    pub m_max: usize,   // Max neighbors per node for higher levels (usually M)
    pub m_max_0: usize, // Max neighbors for layer 0 (usually 2*M)
    pub ef_construction: usize,
    pub level_mult: f64,
}

impl HnswGraph {
    /// Get neighbors of a node at a specific level.
    pub fn get_neighbors(&self, doc_id: u64, level: usize) -> Option<&Vec<u64>> {
        self.nodes.get(&doc_id).and_then(|layers| layers.get(level))
    }

    /// Set neighbors for a node at a specific level (replacing existing ones).
    pub fn set_neighbors(&mut self, doc_id: u64, level: usize, neighbors: Vec<u64>) {
        if let Some(layers) = self.nodes.get_mut(&doc_id) {
            if level < layers.len() {
                layers[level] = neighbors;
            }
        }
    }
}
