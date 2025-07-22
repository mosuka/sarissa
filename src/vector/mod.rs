//! Vector search module for semantic and similarity-based search capabilities.
//!
//! This module provides comprehensive vector search functionality including:
//! - Dense vector indexing with HNSW (Hierarchical Navigable Small World)
//! - Text embedding generation and management
//! - Hybrid search combining keyword and vector search
//! - Multiple similarity metrics (cosine, euclidean, dot product)

pub mod embeddings;
pub mod hnsw;
pub mod index;
pub mod similarity;

use crate::error::{SarissaError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A dense vector representation for similarity search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    /// The vector dimensions as floating point values.
    pub data: Vec<f32>,
    /// Optional metadata associated with this vector.
    pub metadata: HashMap<String, String>,
}

impl Vector {
    /// Create a new vector with the given dimensions.
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data,
            metadata: HashMap::new(),
        }
    }

    /// Create a new vector with metadata.
    pub fn with_metadata(data: Vec<f32>, metadata: HashMap<String, String>) -> Self {
        Self { data, metadata }
    }

    /// Get the dimensionality of this vector.
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Calculate the L2 norm (magnitude) of this vector.
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize this vector to unit length.
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 0.0 {
            for value in &mut self.data {
                *value /= norm;
            }
        }
    }

    /// Get a normalized copy of this vector.
    pub fn normalized(&self) -> Self {
        let mut normalized = self.clone();
        normalized.normalize();
        normalized
    }

    /// Add metadata to this vector.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata by key.
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Validate that this vector has the expected dimension.
    pub fn validate_dimension(&self, expected_dim: usize) -> Result<()> {
        if self.data.len() != expected_dim {
            return Err(SarissaError::InvalidOperation(format!(
                "Vector dimension mismatch: expected {}, got {}",
                expected_dim,
                self.data.len()
            )));
        }
        Ok(())
    }

    /// Check if this vector contains any NaN or infinite values.
    pub fn is_valid(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }

    /// Calculate the L2 norm using parallel processing for large vectors.
    pub fn norm_parallel(&self) -> f32 {
        if self.data.len() > 10000 {
            // Use parallel processing for large vectors
            self.data.par_iter().map(|x| x * x).sum::<f32>().sqrt()
        } else {
            self.norm()
        }
    }

    /// Normalize this vector using parallel processing for large vectors.
    pub fn normalize_parallel(&mut self) {
        let norm = self.norm_parallel();
        if norm > 0.0 {
            if self.data.len() > 10000 {
                // Use parallel processing for large vectors
                self.data.par_iter_mut().for_each(|value| *value /= norm);
            } else {
                for value in &mut self.data {
                    *value /= norm;
                }
            }
        }
    }

    /// Normalize multiple vectors in parallel.
    pub fn normalize_batch_parallel(vectors: &mut [Vector]) {
        if vectors.len() > 10 {
            vectors
                .par_iter_mut()
                .for_each(|vector| vector.normalize_parallel());
        } else {
            for vector in vectors {
                vector.normalize();
            }
        }
    }
}

/// Distance metrics for vector similarity calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DistanceMetric {
    /// Cosine distance (1 - cosine similarity)
    #[default]
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Manhattan (L1) distance  
    Manhattan,
    /// Dot product similarity (higher is more similar)
    DotProduct,
    /// Angular distance
    Angular,
}


impl DistanceMetric {
    /// Calculate the distance between two vectors using this metric.
    pub fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(SarissaError::InvalidOperation(
                "Vector dimensions must match for distance calculation".to_string(),
            ));
        }

        let result = match self {
            DistanceMetric::Cosine => {
                let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0 // Maximum distance for zero vectors
                } else {
                    1.0 - (dot_product / (norm_a * norm_b))
                }
            }
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
            DistanceMetric::DotProduct => {
                // For dot product, we return the negative to make higher values indicate closer similarity
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
            DistanceMetric::Angular => {
                let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    std::f32::consts::PI
                } else {
                    let cosine = (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0);
                    cosine.acos()
                }
            }
        };

        Ok(result)
    }

    /// Calculate similarity (0-1, higher is more similar) between two vectors.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let distance = self.distance(a, b)?;

        let similarity = match self {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::Euclidean => {
                // Convert distance to similarity using exponential decay
                (-distance).exp()
            }
            DistanceMetric::Manhattan => {
                // Convert distance to similarity using exponential decay
                (-distance).exp()
            }
            DistanceMetric::DotProduct => {
                // Dot product returns negative distance, so negate it
                -distance
            }
            DistanceMetric::Angular => {
                // Convert angular distance to similarity
                1.0 - (distance / std::f32::consts::PI)
            }
        };

        Ok(similarity.clamp(0.0, 1.0))
    }

    /// Get the name of this distance metric.
    pub fn name(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Euclidean => "euclidean",
            DistanceMetric::Manhattan => "manhattan",
            DistanceMetric::DotProduct => "dot_product",
            DistanceMetric::Angular => "angular",
        }
    }

    /// Parse a distance metric from a string.
    pub fn parse_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(DistanceMetric::Cosine),
            "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
            "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
            "dot_product" | "dot" => Ok(DistanceMetric::DotProduct),
            "angular" => Ok(DistanceMetric::Angular),
            _ => Err(SarissaError::InvalidOperation(format!(
                "Unknown distance metric: {s}"
            ))),
        }
    }

    /// Calculate distance between a query vector and multiple vectors in parallel.
    pub fn batch_distance_parallel(&self, query: &[f32], vectors: &[&[f32]]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // For small batches, use sequential processing
        if vectors.len() < 100 {
            return vectors
                .iter()
                .map(|v| self.distance(query, v))
                .collect::<Result<Vec<_>>>();
        }

        // Use parallel processing for larger batches
        vectors
            .par_iter()
            .map(|v| self.distance(query, v))
            .collect::<Result<Vec<_>>>()
    }

    /// Calculate similarities between a query vector and multiple vectors in parallel.
    pub fn batch_similarity_parallel(&self, query: &[f32], vectors: &[&[f32]]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // For small batches, use sequential processing
        if vectors.len() < 100 {
            return vectors
                .iter()
                .map(|v| self.similarity(query, v))
                .collect::<Result<Vec<_>>>();
        }

        // Use parallel processing for larger batches
        vectors
            .par_iter()
            .map(|v| self.similarity(query, v))
            .collect::<Result<Vec<_>>>()
    }
}

/// Configuration for vector search operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchConfig {
    /// Distance metric to use for similarity calculation.
    pub distance_metric: DistanceMetric,
    /// Number of top results to return.
    pub top_k: usize,
    /// Minimum similarity threshold (0.0-1.0).
    pub min_similarity: f32,
    /// Whether to normalize vectors before indexing.
    pub normalize_vectors: bool,
    /// Whether to include vector data in search results.
    pub include_vectors: bool,
    /// Whether to include metadata in search results.
    pub include_metadata: bool,
    /// Whether to use parallel processing for vector operations.
    pub parallel: bool,
}

impl Default for VectorSearchConfig {
    fn default() -> Self {
        Self {
            distance_metric: DistanceMetric::Cosine,
            top_k: 10,
            min_similarity: 0.0,
            normalize_vectors: true,
            include_vectors: false,
            include_metadata: true,
            parallel: true, // Enable parallel processing by default
        }
    }
}

/// A search result from vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// Document ID associated with this vector.
    pub doc_id: u64,
    /// Similarity score (0.0-1.0, higher is more similar).
    pub similarity: f32,
    /// Distance value (metric-dependent).
    pub distance: f32,
    /// The vector data (if requested).
    pub vector: Option<Vector>,
    /// Metadata associated with this result.
    pub metadata: HashMap<String, String>,
}

impl VectorSearchResult {
    /// Create a new vector search result.
    pub fn new(
        doc_id: u64,
        similarity: f32,
        distance: f32,
        vector: Option<Vector>,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            doc_id,
            similarity,
            distance,
            vector,
            metadata,
        }
    }
}

/// Collection of vector search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResults {
    /// List of matching results, sorted by similarity (descending).
    pub results: Vec<VectorSearchResult>,
    /// Total number of vectors searched.
    pub total_searched: usize,
    /// Query processing time in milliseconds.
    pub query_time_ms: u64,
    /// The query vector used for this search.
    pub query_vector: Option<Vector>,
}

impl VectorSearchResults {
    /// Create new empty search results.
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            total_searched: 0,
            query_time_ms: 0,
            query_vector: None,
        }
    }

    /// Create new search results with data.
    pub fn new(
        results: Vec<VectorSearchResult>,
        total_searched: usize,
        query_time_ms: u64,
        query_vector: Option<Vector>,
    ) -> Self {
        Self {
            results,
            total_searched,
            query_time_ms,
            query_vector,
        }
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the best (highest similarity) result.
    pub fn best_result(&self) -> Option<&VectorSearchResult> {
        self.results.first()
    }

    /// Filter results by minimum similarity threshold.
    pub fn filter_by_similarity(&mut self, min_similarity: f32) {
        self.results
            .retain(|result| result.similarity >= min_similarity);
    }

    /// Sort results by similarity in descending order.
    pub fn sort_by_similarity(&mut self) {
        self.results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Limit the number of results to top k.
    pub fn limit(&mut self, k: usize) {
        if self.results.len() > k {
            self.results.truncate(k);
        }
    }

    /// Sort results by similarity in parallel (for large result sets).
    pub fn sort_by_similarity_parallel(&mut self) {
        if self.results.len() > 1000 {
            // Use parallel sort for large result sets
            self.results.par_sort_by(|a, b| {
                b.similarity
                    .partial_cmp(&a.similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Use regular sort for smaller sets
            self.sort_by_similarity();
        }
    }

    /// Filter and process results in parallel for better performance.
    pub fn filter_and_limit_parallel(&mut self, min_similarity: f32, top_k: usize) {
        if self.results.len() > 1000 {
            // Use parallel filtering for large result sets
            let filtered: Vec<_> = self
                .results
                .par_iter()
                .cloned()
                .filter(|result| result.similarity >= min_similarity)
                .collect();

            self.results = filtered;
            self.sort_by_similarity_parallel();
        } else {
            // Use sequential processing for smaller sets
            self.filter_by_similarity(min_similarity);
            self.sort_by_similarity();
        }

        self.limit(top_k);
    }
}

/// Vector statistics for analysis and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStats {
    /// Total number of vectors indexed.
    pub total_vectors: usize,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Average vector norm.
    pub avg_norm: f32,
    /// Minimum vector norm.
    pub min_norm: f32,
    /// Maximum vector norm.
    pub max_norm: f32,
    /// Index size in bytes.
    pub index_size_bytes: usize,
    /// Memory usage in bytes.
    pub memory_usage_bytes: usize,
}

impl VectorStats {
    /// Create new vector statistics.
    pub fn new(
        total_vectors: usize,
        dimension: usize,
        avg_norm: f32,
        min_norm: f32,
        max_norm: f32,
        index_size_bytes: usize,
        memory_usage_bytes: usize,
    ) -> Self {
        Self {
            total_vectors,
            dimension,
            avg_norm,
            min_norm,
            max_norm,
            index_size_bytes,
            memory_usage_bytes,
        }
    }

    /// Calculate memory efficiency (vectors per MB).
    pub fn vectors_per_mb(&self) -> f32 {
        if self.memory_usage_bytes > 0 {
            self.total_vectors as f32 / (self.memory_usage_bytes as f32 / 1024.0 / 1024.0)
        } else {
            0.0
        }
    }

    /// Calculate storage efficiency (compression ratio).
    pub fn compression_ratio(&self) -> f32 {
        let raw_size = self.total_vectors * self.dimension * 4; // 4 bytes per f32
        if raw_size > 0 {
            raw_size as f32 / self.index_size_bytes as f32
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let data = vec![1.0, 2.0, 3.0];
        let vector = Vector::new(data.clone());

        assert_eq!(vector.data, data);
        assert_eq!(vector.dimension(), 3);
        assert!(vector.metadata.is_empty());
    }

    #[test]
    fn test_vector_with_metadata() {
        let data = vec![1.0, 2.0, 3.0];
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "text".to_string());

        let vector = Vector::with_metadata(data.clone(), metadata.clone());

        assert_eq!(vector.data, data);
        assert_eq!(vector.metadata, metadata);
        assert_eq!(vector.get_metadata("type"), Some(&"text".to_string()));
    }

    #[test]
    fn test_vector_norm() {
        let vector = Vector::new(vec![3.0, 4.0]);
        assert_eq!(vector.norm(), 5.0); // 3-4-5 triangle
    }

    #[test]
    fn test_vector_normalization() {
        let mut vector = Vector::new(vec![3.0, 4.0]);
        vector.normalize();

        assert!((vector.norm() - 1.0).abs() < 1e-6);
        assert!((vector.data[0] - 0.6).abs() < 1e-6);
        assert!((vector.data[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_vector_validation() {
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        assert!(vector.validate_dimension(3).is_ok());
        assert!(vector.validate_dimension(4).is_err());
    }

    #[test]
    fn test_vector_validity() {
        let valid_vector = Vector::new(vec![1.0, 2.0, 3.0]);
        assert!(valid_vector.is_valid());

        let invalid_vector = Vector::new(vec![1.0, f32::NAN, 3.0]);
        assert!(!invalid_vector.is_valid());

        let infinite_vector = Vector::new(vec![1.0, f32::INFINITY, 3.0]);
        assert!(!infinite_vector.is_valid());
    }

    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        // Test cosine distance
        let cosine_dist = DistanceMetric::Cosine.distance(&a, &b).unwrap();
        assert!((cosine_dist - 1.0).abs() < 1e-6); // Orthogonal vectors

        // Test euclidean distance
        let euclidean_dist = DistanceMetric::Euclidean.distance(&a, &b).unwrap();
        assert!((euclidean_dist - 2.0_f32.sqrt()).abs() < 1e-6);

        // Test manhattan distance
        let manhattan_dist = DistanceMetric::Manhattan.distance(&a, &b).unwrap();
        assert!((manhattan_dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_calculation() {
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0]; // Same vector

        let similarity = DistanceMetric::Cosine.similarity(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6); // Perfect similarity
    }

    #[test]
    fn test_distance_metric_parsing() {
        assert_eq!(
            DistanceMetric::parse_str("cosine").unwrap(),
            DistanceMetric::Cosine
        );
        assert_eq!(
            DistanceMetric::parse_str("euclidean").unwrap(),
            DistanceMetric::Euclidean
        );
        assert_eq!(
            DistanceMetric::parse_str("l2").unwrap(),
            DistanceMetric::Euclidean
        );
        assert!(DistanceMetric::parse_str("unknown").is_err());
    }

    #[test]
    fn test_vector_search_config() {
        let config = VectorSearchConfig::default();

        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert_eq!(config.top_k, 10);
        assert_eq!(config.min_similarity, 0.0);
        assert!(config.normalize_vectors);
        assert!(!config.include_vectors);
        assert!(config.include_metadata);
    }

    #[test]
    fn test_vector_search_results() {
        let mut results = VectorSearchResults::empty();
        assert!(results.is_empty());
        assert_eq!(results.len(), 0);

        let result = VectorSearchResult::new(1, 0.95, 0.05, None, HashMap::new());
        results.results.push(result);

        assert!(!results.is_empty());
        assert_eq!(results.len(), 1);
        assert_eq!(results.best_result().unwrap().doc_id, 1);
    }

    #[test]
    fn test_vector_search_results_filtering() {
        let mut results = VectorSearchResults::empty();

        results
            .results
            .push(VectorSearchResult::new(1, 0.9, 0.1, None, HashMap::new()));
        results
            .results
            .push(VectorSearchResult::new(2, 0.5, 0.5, None, HashMap::new()));
        results
            .results
            .push(VectorSearchResult::new(3, 0.8, 0.2, None, HashMap::new()));

        results.filter_by_similarity(0.7);
        assert_eq!(results.len(), 2); // Only results with similarity >= 0.7

        results.limit(1);
        assert_eq!(results.len(), 1); // Top 1 result
    }

    #[test]
    fn test_vector_stats() {
        let stats = VectorStats::new(1000, 128, 1.0, 0.5, 2.0, 512000, 1024000);

        assert_eq!(stats.total_vectors, 1000);
        assert_eq!(stats.dimension, 128);
        assert!(stats.vectors_per_mb() > 0.0);
        assert!(stats.compression_ratio() > 0.0);
    }

    #[test]
    fn test_parallel_distance_calculation() {
        let query = vec![1.0, 0.0];
        let v1 = vec![0.0, 1.0];
        let v2 = vec![1.0, 1.0];
        let v3 = vec![0.5, 0.5];
        let vectors = vec![v1.as_slice(), v2.as_slice(), v3.as_slice()];

        let distances = DistanceMetric::Cosine
            .batch_distance_parallel(&query, &vectors)
            .unwrap();
        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 1.0).abs() < 1e-6); // Orthogonal
    }

    #[test]
    fn test_parallel_similarity_calculation() {
        let query = vec![1.0, 1.0];
        let v1 = vec![1.0, 1.0];
        let v2 = vec![0.0, 1.0];
        let v3 = vec![1.0, 0.0];
        let vectors = vec![v1.as_slice(), v2.as_slice(), v3.as_slice()];

        let similarities = DistanceMetric::Cosine
            .batch_similarity_parallel(&query, &vectors)
            .unwrap();
        assert_eq!(similarities.len(), 3);
        assert!((similarities[0] - 1.0).abs() < 1e-6); // Perfect match
    }

    #[test]
    fn test_parallel_vector_normalization() {
        let mut vector = Vector::new(vec![3.0, 4.0]); // Small vector that will use regular normalization path
        vector.normalize_parallel();

        assert!((vector.norm_parallel() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_batch_normalization() {
        let mut vectors = vec![
            Vector::new(vec![3.0, 4.0]),
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 5.0]),
        ];

        Vector::normalize_batch_parallel(&mut vectors);

        for vector in &vectors {
            assert!((vector.norm() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_parallel_result_filtering() {
        let mut results = VectorSearchResults::empty();

        for i in 0..10 {
            let similarity = (i as f32) / 10.0;
            results.results.push(VectorSearchResult::new(
                i as u64,
                similarity,
                1.0 - similarity,
                None,
                HashMap::new(),
            ));
        }

        results.filter_and_limit_parallel(0.5, 3);
        assert!(results.len() <= 3);

        // Results should be sorted by similarity (descending)
        if results.len() > 1 {
            assert!(results.results[0].similarity >= results.results[1].similarity);
        }
    }

    #[test]
    fn test_vector_search_config_with_parallel() {
        let mut config = VectorSearchConfig::default();
        assert!(config.parallel); // Should be enabled by default

        config.parallel = false;
        assert!(!config.parallel);
    }
}
