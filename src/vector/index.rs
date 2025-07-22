//! Vector index implementations for high-performance similarity search.
//!
//! This module provides different vector indexing strategies optimized for various use cases:
//! - Flat index for small datasets with exact search
//! - HNSW index for large datasets with approximate search
//! - IVF (Inverted File) index for memory-efficient approximate search

use crate::error::{SarissaError, Result};
use crate::vector::{
    DistanceMetric, Vector, VectorSearchConfig, VectorSearchResult, VectorSearchResults,
    VectorStats,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;

/// Trait for vector index implementations.
pub trait VectorIndex: Send + Sync {
    /// Add a vector to the index with the given document ID.
    fn add_vector(&mut self, doc_id: u64, vector: Vector) -> Result<()>;

    /// Add multiple vectors to the index in batch.
    fn add_vectors(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()> {
        for (doc_id, vector) in vectors {
            self.add_vector(doc_id, vector)?;
        }
        Ok(())
    }

    /// Remove a vector from the index by document ID.
    fn remove_vector(&mut self, doc_id: u64) -> Result<bool>;

    /// Search for similar vectors to the query vector.
    fn search(&self, query: &Vector, config: &VectorSearchConfig) -> Result<VectorSearchResults>;

    /// Get a vector by document ID.
    fn get_vector(&self, doc_id: u64) -> Result<Option<Vector>>;

    /// Get the total number of vectors in the index.
    fn len(&self) -> usize;

    /// Check if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get statistics about the index.
    fn stats(&self) -> VectorStats;

    /// Clear all vectors from the index.
    fn clear(&mut self);

    /// Optimize the index for better search performance.
    fn optimize(&mut self) -> Result<()> {
        // Default implementation does nothing
        Ok(())
    }

    /// Get the expected vector dimension for this index.
    fn dimension(&self) -> usize;

    /// Get the distance metric used by this index.
    fn distance_metric(&self) -> DistanceMetric;
}

/// Configuration for vector index creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexConfig {
    /// Vector dimension (must be consistent for all vectors).
    pub dimension: usize,
    /// Distance metric to use for similarity calculation.
    pub distance_metric: DistanceMetric,
    /// Index type to create.
    pub index_type: VectorIndexType,
    /// Whether to normalize vectors before indexing.
    pub normalize_vectors: bool,
    /// Initial capacity hint for the index.
    pub initial_capacity: usize,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            index_type: VectorIndexType::Flat,
            normalize_vectors: true,
            initial_capacity: 1000,
        }
    }
}

/// Different types of vector indexes available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorIndexType {
    /// Flat index with exact search (brute force).
    Flat,
    /// HNSW (Hierarchical Navigable Small World) index for approximate search.
    HNSW,
    /// IVF (Inverted File) index for memory-efficient approximate search.
    IVF,
}

impl VectorIndexType {
    /// Get the name of this index type.
    pub fn name(&self) -> &'static str {
        match self {
            VectorIndexType::Flat => "flat",
            VectorIndexType::HNSW => "hnsw",
            VectorIndexType::IVF => "ivf",
        }
    }

    /// Parse an index type from a string.
    pub fn parse_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "flat" => Ok(VectorIndexType::Flat),
            "hnsw" => Ok(VectorIndexType::HNSW),
            "ivf" => Ok(VectorIndexType::IVF),
            _ => Err(SarissaError::InvalidOperation(format!(
                "Unknown vector index type: {s}"
            ))),
        }
    }
}

/// A flat vector index that performs exact search using brute force.
///
/// This index stores all vectors in memory and compares the query vector
/// against every indexed vector. It provides exact results but has O(n)
/// search complexity, making it suitable for small datasets (< 10k vectors).
#[derive(Debug)]
pub struct FlatVectorIndex {
    /// Configuration for this index.
    config: VectorIndexConfig,
    /// Map from document ID to vector data.
    vectors: RwLock<HashMap<u64, Vector>>,
    /// Total number of vectors indexed.
    total_vectors: RwLock<usize>,
}

impl FlatVectorIndex {
    /// Create a new flat vector index.
    pub fn new(config: VectorIndexConfig) -> Self {
        let initial_capacity = config.initial_capacity;
        Self {
            config,
            vectors: RwLock::new(HashMap::with_capacity(initial_capacity)),
            total_vectors: RwLock::new(0),
        }
    }

    /// Create a new flat vector index with default configuration.
    pub fn with_dimension(dimension: usize) -> Self {
        let config = VectorIndexConfig {
            dimension,
            ..Default::default()
        };
        Self::new(config)
    }
}

impl VectorIndex for FlatVectorIndex {
    fn add_vector(&mut self, doc_id: u64, mut vector: Vector) -> Result<()> {
        // Validate vector dimension
        vector.validate_dimension(self.config.dimension)?;

        // Check for invalid values
        if !vector.is_valid() {
            return Err(SarissaError::InvalidOperation(
                "Vector contains NaN or infinite values".to_string(),
            ));
        }

        // Normalize if requested
        if self.config.normalize_vectors {
            vector.normalize();
        }

        let mut vectors = self.vectors.write().unwrap();
        let is_new = !vectors.contains_key(&doc_id);

        vectors.insert(doc_id, vector);

        if is_new {
            let mut total = self.total_vectors.write().unwrap();
            *total += 1;
        }

        Ok(())
    }

    fn remove_vector(&mut self, doc_id: u64) -> Result<bool> {
        let mut vectors = self.vectors.write().unwrap();
        let existed = vectors.remove(&doc_id).is_some();

        if existed {
            let mut total = self.total_vectors.write().unwrap();
            *total -= 1;
        }

        Ok(existed)
    }

    fn search(&self, query: &Vector, config: &VectorSearchConfig) -> Result<VectorSearchResults> {
        let start_time = Instant::now();

        // Validate query vector
        query.validate_dimension(self.config.dimension)?;

        let mut query_vector = query.clone();
        if self.config.normalize_vectors {
            query_vector.normalize();
        }

        let vectors = self.vectors.read().unwrap();
        let mut results = Vec::new();

        // Brute force search through all vectors
        for (doc_id, vector) in vectors.iter() {
            let distance = config
                .distance_metric
                .distance(&query_vector.data, &vector.data)?;
            let similarity = config
                .distance_metric
                .similarity(&query_vector.data, &vector.data)?;

            // Filter by minimum similarity threshold
            if similarity >= config.min_similarity {
                let vector_data = if config.include_vectors {
                    Some(vector.clone())
                } else {
                    None
                };

                let metadata = if config.include_metadata {
                    vector.metadata.clone()
                } else {
                    HashMap::new()
                };

                results.push(VectorSearchResult::new(
                    *doc_id,
                    similarity,
                    distance,
                    vector_data,
                    metadata,
                ));
            }
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to top k results
        if results.len() > config.top_k {
            results.truncate(config.top_k);
        }

        let query_time_ms = start_time.elapsed().as_millis() as u64;
        let total_searched = vectors.len();

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
        let vectors = self.vectors.read().unwrap();
        Ok(vectors.get(&doc_id).cloned())
    }

    fn len(&self) -> usize {
        let total = self.total_vectors.read().unwrap();
        *total
    }

    fn stats(&self) -> VectorStats {
        let vectors = self.vectors.read().unwrap();
        let total = *self.total_vectors.read().unwrap();

        if total == 0 {
            return VectorStats::new(0, self.config.dimension, 0.0, 0.0, 0.0, 0, 0);
        }

        let mut sum_norm = 0.0;
        let mut min_norm = f32::INFINITY;
        let mut max_norm: f32 = 0.0;

        for vector in vectors.values() {
            let norm = vector.norm();
            sum_norm += norm;
            min_norm = min_norm.min(norm);
            max_norm = max_norm.max(norm);
        }

        let avg_norm = sum_norm / total as f32;
        let index_size_bytes = total * self.config.dimension * 4; // 4 bytes per f32
        let memory_usage_bytes = index_size_bytes + (total * 64); // Estimate metadata overhead

        VectorStats::new(
            total,
            self.config.dimension,
            avg_norm,
            min_norm,
            max_norm,
            index_size_bytes,
            memory_usage_bytes,
        )
    }

    fn clear(&mut self) {
        let mut vectors = self.vectors.write().unwrap();
        let mut total = self.total_vectors.write().unwrap();

        vectors.clear();
        *total = 0;
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.config.distance_metric
    }
}

/// Factory for creating vector indexes.
pub struct VectorIndexFactory;

impl VectorIndexFactory {
    /// Create a new vector index based on the configuration.
    pub fn create(config: VectorIndexConfig) -> Result<Box<dyn VectorIndex>> {
        match config.index_type {
            VectorIndexType::Flat => Ok(Box::new(FlatVectorIndex::new(config))),
            VectorIndexType::HNSW => {
                use crate::vector::hnsw::{HnswConfig, HnswIndex};
                let hnsw_config =
                    HnswConfig::new(config.dimension).with_distance_metric(config.distance_metric);
                Ok(Box::new(HnswIndex::new(hnsw_config)?))
            }
            VectorIndexType::IVF => {
                // TODO: Implement IVF index
                Err(SarissaError::InvalidOperation(
                    "IVF index not yet implemented".to_string(),
                ))
            }
        }
    }

    /// Create a flat vector index with the given dimension.
    pub fn create_flat(dimension: usize) -> Box<dyn VectorIndex> {
        let config = VectorIndexConfig {
            dimension,
            index_type: VectorIndexType::Flat,
            ..Default::default()
        };
        Box::new(FlatVectorIndex::new(config))
    }

    /// Create a flat vector index with custom distance metric.
    pub fn create_flat_with_metric(
        dimension: usize,
        metric: DistanceMetric,
    ) -> Box<dyn VectorIndex> {
        let config = VectorIndexConfig {
            dimension,
            distance_metric: metric,
            index_type: VectorIndexType::Flat,
            ..Default::default()
        };
        Box::new(FlatVectorIndex::new(config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index_config() {
        let config = VectorIndexConfig::default();

        assert_eq!(config.dimension, 128);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert_eq!(config.index_type, VectorIndexType::Flat);
        assert!(config.normalize_vectors);
        assert_eq!(config.initial_capacity, 1000);
    }

    #[test]
    fn test_vector_index_type_parsing() {
        assert_eq!(
            VectorIndexType::parse_str("flat").unwrap(),
            VectorIndexType::Flat
        );
        assert_eq!(
            VectorIndexType::parse_str("hnsw").unwrap(),
            VectorIndexType::HNSW
        );
        assert_eq!(
            VectorIndexType::parse_str("ivf").unwrap(),
            VectorIndexType::IVF
        );
        assert!(VectorIndexType::parse_str("unknown").is_err());
    }

    #[test]
    fn test_flat_vector_index_creation() {
        let index = FlatVectorIndex::with_dimension(3);

        assert_eq!(index.dimension(), 3);
        assert_eq!(index.distance_metric(), DistanceMetric::Cosine);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_flat_vector_index_add_vector() {
        let mut index = FlatVectorIndex::with_dimension(3);
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        assert!(index.add_vector(1, vector).is_ok());
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_flat_vector_index_dimension_validation() {
        let mut index = FlatVectorIndex::with_dimension(3);
        let wrong_dimension_vector = Vector::new(vec![1.0, 2.0]); // Only 2 dimensions

        assert!(index.add_vector(1, wrong_dimension_vector).is_err());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_flat_vector_index_invalid_vector() {
        let mut index = FlatVectorIndex::with_dimension(3);
        let invalid_vector = Vector::new(vec![1.0, f32::NAN, 3.0]);

        assert!(index.add_vector(1, invalid_vector).is_err());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_flat_vector_index_search() {
        let mut index = FlatVectorIndex::with_dimension(2);

        // Add some vectors
        index.add_vector(1, Vector::new(vec![1.0, 0.0])).unwrap();
        index.add_vector(2, Vector::new(vec![0.0, 1.0])).unwrap();
        index.add_vector(3, Vector::new(vec![1.0, 1.0])).unwrap();

        // Search for vector similar to [1, 0]
        let query = Vector::new(vec![1.0, 0.0]);
        let config = VectorSearchConfig::default();
        let results = index.search(&query, &config).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results.best_result().unwrap().doc_id, 1); // Should be exact match
        assert!(results.best_result().unwrap().similarity > 0.9);
    }

    #[test]
    fn test_flat_vector_index_get_vector() {
        let mut index = FlatVectorIndex::with_dimension(2);
        let vector = Vector::new(vec![1.0, 2.0]);

        index.add_vector(1, vector.clone()).unwrap();

        let retrieved = index.get_vector(1).unwrap().unwrap();
        // Vector will be normalized by default, so check normalized version
        let expected_norm = (1.0_f32 * 1.0 + 2.0 * 2.0).sqrt();
        let expected = [1.0 / expected_norm, 2.0 / expected_norm];
        assert!((retrieved.data[0] - expected[0]).abs() < 1e-6);
        assert!((retrieved.data[1] - expected[1]).abs() < 1e-6);

        assert!(index.get_vector(999).unwrap().is_none());
    }

    #[test]
    fn test_flat_vector_index_remove_vector() {
        let mut index = FlatVectorIndex::with_dimension(2);
        let vector = Vector::new(vec![1.0, 2.0]);

        index.add_vector(1, vector).unwrap();
        assert_eq!(index.len(), 1);

        assert!(index.remove_vector(1).unwrap());
        assert_eq!(index.len(), 0);

        assert!(!index.remove_vector(1).unwrap()); // Already removed
    }

    #[test]
    fn test_flat_vector_index_clear() {
        let mut index = FlatVectorIndex::with_dimension(2);

        index.add_vector(1, Vector::new(vec![1.0, 2.0])).unwrap();
        index.add_vector(2, Vector::new(vec![3.0, 4.0])).unwrap();
        assert_eq!(index.len(), 2);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_flat_vector_index_stats() {
        let mut index = FlatVectorIndex::with_dimension(2);

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
        assert!(stats.index_size_bytes > 0);
        assert!(stats.memory_usage_bytes > 0);
    }

    #[test]
    fn test_vector_index_factory() {
        let config = VectorIndexConfig {
            dimension: 4,
            index_type: VectorIndexType::Flat,
            ..Default::default()
        };

        let index = VectorIndexFactory::create(config).unwrap();
        assert_eq!(index.dimension(), 4);
        assert_eq!(index.distance_metric(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_vector_index_factory_shortcuts() {
        let flat_index = VectorIndexFactory::create_flat(3);
        assert_eq!(flat_index.dimension(), 3);

        let euclidean_index =
            VectorIndexFactory::create_flat_with_metric(5, DistanceMetric::Euclidean);
        assert_eq!(euclidean_index.dimension(), 5);
        assert_eq!(euclidean_index.distance_metric(), DistanceMetric::Euclidean);
    }

    #[test]
    fn test_search_with_config() {
        let mut index = FlatVectorIndex::with_dimension(2);

        // Add vectors with different similarities
        index.add_vector(1, Vector::new(vec![1.0, 0.0])).unwrap();
        index.add_vector(2, Vector::new(vec![0.9, 0.1])).unwrap();
        index.add_vector(3, Vector::new(vec![0.0, 1.0])).unwrap();

        let query = Vector::new(vec![1.0, 0.0]);
        let config = VectorSearchConfig {
            top_k: 2,
            min_similarity: 0.5,
            include_vectors: true,
            include_metadata: true,
            ..Default::default()
        };

        let results = index.search(&query, &config).unwrap();

        assert!(results.len() <= 2); // Top-k limit
        assert!(results.results.iter().all(|r| r.similarity >= 0.5)); // Min similarity
        assert!(results.results[0].vector.is_some()); // Include vectors
    }
}
