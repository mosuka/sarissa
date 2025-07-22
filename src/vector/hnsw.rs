//! HNSW (Hierarchical Navigable Small World) implementation for approximate vector search.
//!
//! HNSW is a state-of-the-art algorithm for approximate nearest neighbor search that provides
//! excellent performance with logarithmic search complexity. It builds a multi-layer graph
//! where each layer contains a subset of the nodes from the layer below, enabling efficient
//! navigation through the vector space.
//!
//! Key features:
//! - Sub-linear search complexity: O(log N)
//! - High recall rates (>95% for most datasets)
//! - Excellent performance for high-dimensional vectors
//! - Incremental updates (add/remove vectors)

use crate::error::{SarissaError, Result};
use crate::vector::index::VectorIndex;
use crate::vector::{
    DistanceMetric, Vector, VectorSearchConfig, VectorSearchResult, VectorSearchResults,
    VectorStats,
};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::RwLock;
use std::time::Instant;

/// Configuration for HNSW index construction and search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per node in layer 0.
    pub m: usize,
    /// Maximum number of connections per node in higher layers (typically m/2).
    pub m_l: usize,
    /// Multiplier that controls the probability of layer assignment.
    pub ml: f64,
    /// Size of the candidate set during construction.
    pub ef_construction: usize,
    /// Size of the candidate set during search.
    pub ef_search: usize,
    /// Random seed for reproducible results.
    pub seed: u64,
    /// Vector dimension.
    pub dimension: usize,
    /// Distance metric to use.
    pub distance_metric: DistanceMetric,
    /// Whether to normalize vectors.
    pub normalize_vectors: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,                    // Typical value for good performance
            m_l: 8,                   // m/2
            ml: 1.0 / (2.0_f64.ln()), // 1/ln(2) ≈ 1.44
            ef_construction: 200,     // Higher values improve quality but slow construction
            ef_search: 50,            // Can be adjusted per query
            seed: 42,
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: true,
        }
    }
}

impl HnswConfig {
    /// Create a new HNSW configuration with the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    /// Set the M parameter (connections per node).
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m_l = m / 2;
        self
    }

    /// Set the ef_construction parameter.
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set the distance metric.
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Validate the configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.dimension == 0 {
            return Err(SarissaError::InvalidOperation(
                "Dimension must be > 0".to_string(),
            ));
        }
        if self.m == 0 {
            return Err(SarissaError::InvalidOperation("M must be > 0".to_string()));
        }
        if self.ef_construction < self.m {
            return Err(SarissaError::InvalidOperation(
                "ef_construction must be >= M".to_string(),
            ));
        }
        Ok(())
    }
}

/// A node in the HNSW graph representing a vector.
#[derive(Debug, Clone)]
struct HnswNode {
    /// Document ID associated with this vector.
    doc_id: u64,
    /// The vector data.
    vector: Vector,
    /// Connections to other nodes, organized by layer.
    /// connections[layer] contains the set of node IDs connected at that layer.
    connections: Vec<HashSet<usize>>,
    /// The maximum layer this node exists in.
    max_layer: usize,
}

impl HnswNode {
    /// Create a new HNSW node.
    fn new(doc_id: u64, vector: Vector, max_layer: usize) -> Self {
        let mut connections = Vec::with_capacity(max_layer + 1);
        for _ in 0..=max_layer {
            connections.push(HashSet::new());
        }

        Self {
            doc_id,
            vector,
            connections,
            max_layer,
        }
    }

    /// Add a connection to another node at the specified layer.
    fn add_connection(&mut self, layer: usize, node_id: usize) {
        if layer <= self.max_layer {
            self.connections[layer].insert(node_id);
        }
    }

    /// Remove a connection from the specified layer.
    fn remove_connection(&mut self, layer: usize, node_id: usize) {
        if layer <= self.max_layer {
            self.connections[layer].remove(&node_id);
        }
    }

    /// Get connections at the specified layer.
    fn get_connections(&self, layer: usize) -> &HashSet<usize> {
        if layer <= self.max_layer {
            &self.connections[layer]
        } else {
            // Return empty set for layers that don't exist
            use std::sync::LazyLock;
            static EMPTY_SET: LazyLock<HashSet<usize>> = LazyLock::new(HashSet::new);
            &EMPTY_SET
        }
    }

    /// Get the number of connections at the specified layer.
    fn connection_count(&self, layer: usize) -> usize {
        if layer <= self.max_layer {
            self.connections[layer].len()
        } else {
            0
        }
    }
}

/// Priority queue entry for HNSW search.
#[derive(Debug, Clone, PartialEq)]
struct SearchCandidate {
    /// Distance to the query vector.
    distance: f32,
    /// Node ID in the graph.
    node_id: usize,
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // For min-heap behavior, we want smaller distances first
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// HNSW index implementation for approximate nearest neighbor search.
pub struct HnswIndex {
    /// Configuration parameters.
    config: HnswConfig,
    /// All nodes in the graph, indexed by node ID.
    nodes: RwLock<Vec<Option<HnswNode>>>,
    /// Map from document ID to node ID.
    doc_to_node: RwLock<HashMap<u64, usize>>,
    /// Entry point for search (node with highest layer).
    entry_point: RwLock<Option<usize>>,
    /// Current maximum layer in the graph.
    max_layer: RwLock<usize>,
    /// Random number generator for layer assignment.
    rng: RwLock<StdRng>,
    /// Next available node ID.
    next_node_id: RwLock<usize>,
}

impl HnswIndex {
    /// Create a new HNSW index with the given configuration.
    pub fn new(config: HnswConfig) -> Result<Self> {
        config.validate()?;

        let rng = StdRng::seed_from_u64(config.seed);

        Ok(Self {
            config,
            nodes: RwLock::new(Vec::new()),
            doc_to_node: RwLock::new(HashMap::new()),
            entry_point: RwLock::new(None),
            max_layer: RwLock::new(0),
            rng: RwLock::new(rng),
            next_node_id: RwLock::new(0),
        })
    }

    /// Create a new HNSW index with default configuration for the given dimension.
    pub fn with_dimension(dimension: usize) -> Result<Self> {
        let config = HnswConfig::new(dimension);
        Self::new(config)
    }

    /// Calculate the distance between two vectors using the configured metric.
    fn calculate_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        self.config.distance_metric.distance(a, b)
    }

    /// Randomly select a layer for a new node using the ml parameter.
    fn select_layer(&self) -> usize {
        let mut rng = self.rng.write().unwrap();
        let uniform_random: f64 = rng.random();
        (-uniform_random.ln() * self.config.ml).floor() as usize
    }

    /// Search for the closest nodes to a query vector starting from an entry point.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: Vec<usize>,
        num_closest: usize,
        layer: usize,
    ) -> Result<Vec<SearchCandidate>> {
        let nodes = self.nodes.read().unwrap();
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Max-heap for farthest candidates
        let mut dynamic_candidates = BinaryHeap::new(); // Min-heap for closest candidates

        // Initialize with entry points
        for entry_id in entry_points {
            if let Some(Some(node)) = nodes.get(entry_id) {
                let distance = self.calculate_distance(query, &node.vector.data)?;
                let candidate = SearchCandidate {
                    distance,
                    node_id: entry_id,
                };

                visited.insert(entry_id);
                candidates.push(Reverse(candidate.clone())); // Use Reverse for min-heap behavior
                dynamic_candidates.push(candidate);
            }
        }

        while let Some(Reverse(current)) = candidates.pop() {
            // Check if we should continue (pruning condition)
            if let Some(farthest) = dynamic_candidates.peek() {
                if current.distance > farthest.distance && dynamic_candidates.len() >= num_closest {
                    break;
                }
            }

            // Explore neighbors of current node
            if let Some(Some(current_node)) = nodes.get(current.node_id) {
                for &neighbor_id in current_node.get_connections(layer) {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id);

                        if let Some(Some(neighbor_node)) = nodes.get(neighbor_id) {
                            let distance =
                                self.calculate_distance(query, &neighbor_node.vector.data)?;
                            let candidate = SearchCandidate {
                                distance,
                                node_id: neighbor_id,
                            };

                            // Add to dynamic candidates if we need more or if it's closer than the farthest
                            if dynamic_candidates.len() < num_closest {
                                dynamic_candidates.push(candidate.clone());
                                candidates.push(Reverse(candidate));
                            } else if let Some(farthest) = dynamic_candidates.peek() {
                                if candidate.distance < farthest.distance {
                                    dynamic_candidates.pop(); // Remove farthest
                                    dynamic_candidates.push(candidate.clone());
                                    candidates.push(Reverse(candidate));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert max-heap to sorted vector (closest first)
        let mut result: Vec<_> = dynamic_candidates.into_sorted_vec();
        result.reverse(); // Sort by distance (ascending)
        Ok(result)
    }

    /// Select M diverse neighbors from candidates using a simple heuristic.
    fn select_neighbors_simple(&self, candidates: Vec<SearchCandidate>, m: usize) -> Vec<usize> {
        candidates.into_iter().take(m).map(|c| c.node_id).collect()
    }

    /// Prune connections for a node to maintain the maximum connection limit.
    fn prune_connections(
        &self,
        node_id: usize,
        layer: usize,
        max_connections: usize,
    ) -> Result<()> {
        let mut nodes = self.nodes.write().unwrap();

        // First, get the connections and query vector without borrowing mutably
        let (connections, query_vector) = {
            if let Some(Some(node)) = nodes.get(node_id) {
                let connections: Vec<_> = node.get_connections(layer).iter().cloned().collect();
                let query_vector = node.vector.data.clone();
                (connections, query_vector)
            } else {
                return Ok(());
            }
        };

        if connections.len() > max_connections {
            // Calculate distances to all neighbors
            let mut candidates = Vec::new();

            for &neighbor_id in &connections {
                if let Some(Some(neighbor)) = nodes.get(neighbor_id) {
                    let distance = self
                        .config
                        .distance_metric
                        .distance(&query_vector, &neighbor.vector.data)?;
                    candidates.push(SearchCandidate {
                        distance,
                        node_id: neighbor_id,
                    });
                }
            }

            candidates.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Update the node's connections
            if let Some(Some(node)) = nodes.get_mut(node_id) {
                node.connections[layer].clear();
                for candidate in candidates.into_iter().take(max_connections) {
                    node.add_connection(layer, candidate.node_id);
                }
            }
        }

        Ok(())
    }

    /// Get the current entry point for search.
    fn get_entry_point(&self) -> Option<usize> {
        *self.entry_point.read().unwrap()
    }

    /// Update the entry point if the new node has a higher layer.
    fn update_entry_point(&self, node_id: usize, layer: usize) {
        let mut entry_point = self.entry_point.write().unwrap();
        let mut max_layer = self.max_layer.write().unwrap();

        if layer >= *max_layer || entry_point.is_none() {
            *entry_point = Some(node_id);
            *max_layer = layer;
        }
    }
}

impl VectorIndex for HnswIndex {
    fn add_vector(&mut self, doc_id: u64, mut vector: Vector) -> Result<()> {
        // Validate vector
        vector.validate_dimension(self.config.dimension)?;
        if !vector.is_valid() {
            return Err(SarissaError::InvalidOperation(
                "Vector contains NaN or infinite values".to_string(),
            ));
        }

        // Normalize if requested
        if self.config.normalize_vectors {
            vector.normalize();
        }

        // Check if document already exists
        {
            let doc_to_node = self.doc_to_node.read().unwrap();
            if doc_to_node.contains_key(&doc_id) {
                return Err(SarissaError::InvalidOperation(format!(
                    "Document {doc_id} already exists in the index"
                )));
            }
        }

        // Assign node ID and layer
        let node_id = {
            let mut next_id = self.next_node_id.write().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let layer = self.select_layer();
        let node = HnswNode::new(doc_id, vector, layer);

        // Add node to the graph
        {
            let mut nodes = self.nodes.write().unwrap();

            // Expand the nodes vector if necessary
            while nodes.len() <= node_id {
                nodes.push(None);
            }
            nodes[node_id] = Some(node);
        }

        // Update document to node mapping
        {
            let mut doc_to_node = self.doc_to_node.write().unwrap();
            doc_to_node.insert(doc_id, node_id);
        }

        // Update entry point if this is the first node or has higher layer
        self.update_entry_point(node_id, layer);

        // Connect the new node to the graph
        if let Some(entry_point) = self.get_entry_point() {
            if entry_point != node_id {
                self.connect_new_node(node_id, entry_point)?;
            }
        }

        Ok(())
    }

    fn remove_vector(&mut self, doc_id: u64) -> Result<bool> {
        // Find the node ID for this document
        let node_id = {
            let mut doc_to_node = self.doc_to_node.write().unwrap();
            if let Some(node_id) = doc_to_node.remove(&doc_id) {
                node_id
            } else {
                return Ok(false); // Document not found
            }
        };

        // Remove the node and update connections
        {
            let mut nodes = self.nodes.write().unwrap();
            if let Some(Some(node)) = nodes.get(node_id).cloned() {
                // Remove all connections to this node from other nodes
                for layer in 0..=node.max_layer {
                    for &neighbor_id in node.get_connections(layer) {
                        if let Some(Some(neighbor)) = nodes.get_mut(neighbor_id) {
                            neighbor.remove_connection(layer, node_id);
                        }
                    }
                }

                // Mark the node as removed
                nodes[node_id] = None;
            }
        }

        // Update entry point if necessary
        if Some(node_id) == self.get_entry_point() {
            self.find_new_entry_point();
        }

        Ok(true)
    }

    fn search(&self, query: &Vector, config: &VectorSearchConfig) -> Result<VectorSearchResults> {
        let start_time = Instant::now();

        query.validate_dimension(self.config.dimension)?;

        let mut query_vector = query.clone();
        if self.config.normalize_vectors {
            query_vector.normalize();
        }

        let entry_point = match self.get_entry_point() {
            Some(ep) => ep,
            None => {
                // Empty index
                return Ok(VectorSearchResults::empty());
            }
        };

        let max_layer = *self.max_layer.read().unwrap();
        let ef = self.config.ef_search.max(config.top_k);

        // Search from top layer down to layer 1
        let mut current_closest = vec![entry_point];
        for layer in (1..=max_layer).rev() {
            current_closest = self
                .search_layer(&query_vector.data, current_closest, 1, layer)?
                .into_iter()
                .map(|c| c.node_id)
                .collect();
        }

        // Search layer 0 with larger candidate set
        let candidates = self.search_layer(&query_vector.data, current_closest, ef, 0)?;

        // Convert candidates to search results
        let nodes = self.nodes.read().unwrap();
        let mut results = Vec::new();

        for candidate in candidates.into_iter().take(config.top_k) {
            if let Some(Some(node)) = nodes.get(candidate.node_id) {
                let similarity = self
                    .config
                    .distance_metric
                    .similarity(&query_vector.data, &node.vector.data)?;

                if similarity >= config.min_similarity {
                    let vector_data = if config.include_vectors {
                        Some(node.vector.clone())
                    } else {
                        None
                    };

                    let metadata = if config.include_metadata {
                        node.vector.metadata.clone()
                    } else {
                        HashMap::new()
                    };

                    results.push(VectorSearchResult::new(
                        node.doc_id,
                        similarity,
                        candidate.distance,
                        vector_data,
                        metadata,
                    ));
                }
            }
        }

        let query_time_ms = start_time.elapsed().as_millis() as u64;
        let total_searched = self.len();

        let query_vector = if config.include_vectors {
            Some(query_vector)
        } else {
            None
        };

        Ok(VectorSearchResults::new(
            results,
            total_searched,
            query_time_ms,
            query_vector,
        ))
    }

    fn get_vector(&self, doc_id: u64) -> Result<Option<Vector>> {
        let doc_to_node = self.doc_to_node.read().unwrap();
        let nodes = self.nodes.read().unwrap();

        if let Some(&node_id) = doc_to_node.get(&doc_id) {
            if let Some(Some(node)) = nodes.get(node_id) {
                return Ok(Some(node.vector.clone()));
            }
        }

        Ok(None)
    }

    fn len(&self) -> usize {
        let doc_to_node = self.doc_to_node.read().unwrap();
        doc_to_node.len()
    }

    fn stats(&self) -> VectorStats {
        let nodes = self.nodes.read().unwrap();
        let doc_to_node = self.doc_to_node.read().unwrap();
        let total = doc_to_node.len();

        if total == 0 {
            return VectorStats::new(0, self.config.dimension, 0.0, 0.0, 0.0, 0, 0);
        }

        let mut sum_norm = 0.0;
        let mut min_norm = f32::INFINITY;
        let mut max_norm: f32 = 0.0;
        let mut total_connections = 0;

        for &node_id in doc_to_node.values() {
            if let Some(Some(node)) = nodes.get(node_id) {
                let norm = node.vector.norm();
                sum_norm += norm;
                min_norm = min_norm.min(norm);
                max_norm = max_norm.max(norm);

                // Count total connections across all layers
                for layer in 0..=node.max_layer {
                    total_connections += node.connection_count(layer);
                }
            }
        }

        let avg_norm = sum_norm / total as f32;

        // Estimate memory usage
        let vector_size = total * self.config.dimension * 4; // 4 bytes per f32
        let graph_size = total_connections * 8; // 8 bytes per connection (rough estimate)
        let metadata_size = total * 64; // Rough estimate
        let memory_usage_bytes = vector_size + graph_size + metadata_size;

        VectorStats::new(
            total,
            self.config.dimension,
            avg_norm,
            min_norm,
            max_norm,
            memory_usage_bytes,
            memory_usage_bytes, // Index size same as memory usage for in-memory index
        )
    }

    fn clear(&mut self) {
        let mut nodes = self.nodes.write().unwrap();
        let mut doc_to_node = self.doc_to_node.write().unwrap();
        let mut entry_point = self.entry_point.write().unwrap();
        let mut max_layer = self.max_layer.write().unwrap();
        let mut next_node_id = self.next_node_id.write().unwrap();

        nodes.clear();
        doc_to_node.clear();
        *entry_point = None;
        *max_layer = 0;
        *next_node_id = 0;
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.config.distance_metric
    }
}

impl HnswIndex {
    /// Connect a new node to the existing graph.
    fn connect_new_node(&self, new_node_id: usize, entry_point: usize) -> Result<()> {
        let nodes = self.nodes.read().unwrap();
        let new_node = if let Some(Some(node)) = nodes.get(new_node_id) {
            node.clone()
        } else {
            return Err(SarissaError::InvalidOperation(
                "New node not found".to_string(),
            ));
        };
        drop(nodes);

        let max_layer = *self.max_layer.read().unwrap();

        // Search and connect for each layer
        let mut current_closest = vec![entry_point];

        for layer in (1..=max_layer.min(new_node.max_layer)).rev() {
            current_closest = self
                .search_layer(&new_node.vector.data, current_closest, 1, layer)?
                .into_iter()
                .map(|c| c.node_id)
                .collect();
        }

        // Connect in layer 0 and up to the node's max layer
        for layer in 0..=new_node.max_layer {
            let m = if layer == 0 {
                self.config.m
            } else {
                self.config.m_l
            };
            let ef = self.config.ef_construction.max(m);

            let candidates =
                self.search_layer(&new_node.vector.data, current_closest.clone(), ef, layer)?;
            let selected = self.select_neighbors_simple(candidates, m);

            // Add bidirectional connections
            {
                let mut nodes = self.nodes.write().unwrap();

                // Add connections from new node to selected neighbors
                if let Some(Some(new_node_mut)) = nodes.get_mut(new_node_id) {
                    for &neighbor_id in &selected {
                        new_node_mut.add_connection(layer, neighbor_id);
                    }
                }

                // Add connections from selected neighbors to new node
                for &neighbor_id in &selected {
                    if let Some(Some(neighbor)) = nodes.get_mut(neighbor_id) {
                        neighbor.add_connection(layer, new_node_id);

                        // Prune connections if necessary
                        let max_conn = if layer == 0 {
                            self.config.m
                        } else {
                            self.config.m_l
                        };
                        if neighbor.connection_count(layer) > max_conn {
                            drop(nodes);
                            self.prune_connections(neighbor_id, layer, max_conn)?;
                            nodes = self.nodes.write().unwrap();
                        }
                    }
                }
            }

            current_closest = selected;
        }

        Ok(())
    }

    /// Find a new entry point when the current one is removed.
    fn find_new_entry_point(&self) {
        let mut entry_point = self.entry_point.write().unwrap();
        let mut max_layer = self.max_layer.write().unwrap();
        let nodes = self.nodes.read().unwrap();

        *entry_point = None;
        *max_layer = 0;

        // Find the node with the highest layer
        for (node_id, node_opt) in nodes.iter().enumerate() {
            if let Some(node) = node_opt {
                if node.max_layer > *max_layer {
                    *max_layer = node.max_layer;
                    *entry_point = Some(node_id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_config() {
        let config = HnswConfig::new(128);
        assert_eq!(config.dimension, 128);
        assert_eq!(config.m, 16);
        assert_eq!(config.m_l, 8);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_hnsw_config_validation() {
        let mut config = HnswConfig::new(0);
        assert!(config.validate().is_err());

        config.dimension = 128;
        config.m = 0;
        assert!(config.validate().is_err());

        config.m = 16;
        config.ef_construction = 8; // Less than M
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_hnsw_index_creation() {
        let index = HnswIndex::with_dimension(4).unwrap();
        assert_eq!(index.dimension(), 4);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_add_vector() {
        let mut index = HnswIndex::with_dimension(3).unwrap();
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        assert!(index.add_vector(1, vector).is_ok());
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_hnsw_duplicate_document() {
        let mut index = HnswIndex::with_dimension(3).unwrap();
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        assert!(index.add_vector(1, vector.clone()).is_ok());
        assert!(index.add_vector(1, vector).is_err()); // Duplicate
    }

    #[test]
    fn test_hnsw_search_single_vector() {
        let mut index = HnswIndex::with_dimension(2).unwrap();
        let vector = Vector::new(vec![1.0, 0.0]);

        index.add_vector(1, vector.clone()).unwrap();

        let config = VectorSearchConfig {
            min_similarity: 0.0, // Lower threshold to ensure we find results
            ..Default::default()
        };
        let results = index.search(&vector, &config).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results.best_result().unwrap().doc_id, 1);
        assert!(results.best_result().unwrap().similarity > 0.99);
    }

    #[test]
    fn test_hnsw_search_multiple_vectors() {
        let mut index = HnswIndex::with_dimension(2).unwrap();

        // Add several vectors
        index.add_vector(1, Vector::new(vec![1.0, 0.0])).unwrap();
        index.add_vector(2, Vector::new(vec![0.0, 1.0])).unwrap();
        index.add_vector(3, Vector::new(vec![0.9, 0.1])).unwrap(); // Similar to vector 1

        // Search for vector similar to [1, 0]
        let query = Vector::new(vec![1.0, 0.0]);
        let config = VectorSearchConfig {
            min_similarity: 0.0, // Lower threshold
            ..Default::default()
        };
        let results = index.search(&query, &config).unwrap();

        assert!(!results.is_empty());
        // The most similar should be doc_id 1 or 3
        let best = results.best_result().unwrap();
        assert!(best.doc_id == 1 || best.doc_id == 3);
    }

    #[test]
    fn test_hnsw_get_vector() {
        let config = HnswConfig::new(2).with_distance_metric(DistanceMetric::Cosine);
        let mut index = HnswIndex::new(config).unwrap();
        let vector = Vector::new(vec![1.0, 2.0]);

        index.add_vector(42, vector.clone()).unwrap();

        let retrieved = index.get_vector(42).unwrap().unwrap();
        // Vector will be normalized, so check that it's a valid normalized version
        let expected_norm = (1.0_f32 * 1.0 + 2.0 * 2.0).sqrt();
        let expected = [1.0 / expected_norm, 2.0 / expected_norm];
        assert!((retrieved.data[0] - expected[0]).abs() < 1e-6);
        assert!((retrieved.data[1] - expected[1]).abs() < 1e-6);

        assert!(index.get_vector(999).unwrap().is_none());
    }

    #[test]
    fn test_hnsw_remove_vector() {
        let mut index = HnswIndex::with_dimension(2).unwrap();
        let vector = Vector::new(vec![1.0, 2.0]);

        index.add_vector(1, vector).unwrap();
        assert_eq!(index.len(), 1);

        assert!(index.remove_vector(1).unwrap());
        assert_eq!(index.len(), 0);

        assert!(!index.remove_vector(1).unwrap()); // Already removed
    }

    #[test]
    fn test_hnsw_clear() {
        let mut index = HnswIndex::with_dimension(2).unwrap();

        index.add_vector(1, Vector::new(vec![1.0, 0.0])).unwrap();
        index.add_vector(2, Vector::new(vec![0.0, 1.0])).unwrap();
        assert_eq!(index.len(), 2);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_stats() {
        let mut index = HnswIndex::with_dimension(2).unwrap();

        // Empty index
        let stats = index.stats();
        assert_eq!(stats.total_vectors, 0);

        // Add some vectors
        index.add_vector(1, Vector::new(vec![1.0, 0.0])).unwrap();
        index.add_vector(2, Vector::new(vec![0.0, 1.0])).unwrap();

        let stats = index.stats();
        assert_eq!(stats.total_vectors, 2);
        assert_eq!(stats.dimension, 2);
        assert!(stats.avg_norm > 0.0);
        assert!(stats.memory_usage_bytes > 0);
    }

    #[test]
    fn test_search_candidate_ordering() {
        let mut candidates = [
            SearchCandidate {
                distance: 0.5,
                node_id: 1,
            },
            SearchCandidate {
                distance: 0.2,
                node_id: 2,
            },
            SearchCandidate {
                distance: 0.8,
                node_id: 3,
            },
        ];

        candidates.sort();

        // Should be sorted by distance (ascending)
        assert_eq!(candidates[0].node_id, 2); // distance 0.2
        assert_eq!(candidates[1].node_id, 1); // distance 0.5
        assert_eq!(candidates[2].node_id, 3); // distance 0.8
    }
}
