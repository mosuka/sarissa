//! Common types used by both vector indexing and search modules.

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Configuration for vector search operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchConfig {
    /// Number of results to return.
    pub top_k: usize,
    /// Minimum similarity threshold.
    pub min_similarity: f32,
    /// Whether to return similarity scores.
    pub include_scores: bool,
    /// Whether to include vector data in results.
    pub include_vectors: bool,
    /// Search timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

impl Default for VectorSearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            include_scores: true,
            include_vectors: false,
            timeout_ms: None,
        }
    }
}

/// A single vector search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// Document ID.
    pub doc_id: u64,
    /// Similarity score (higher is more similar).
    pub similarity: f32,
    /// Distance score (lower is more similar).
    pub distance: f32,
    /// Optional vector data.
    pub vector: Option<crate::vector::Vector>,
    /// Result metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

/// Collection of vector search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResults {
    /// Individual search results.
    pub results: Vec<VectorSearchResult>,
    /// Total number of candidates examined.
    pub candidates_examined: usize,
    /// Search execution time in milliseconds.
    pub search_time_ms: f64,
    /// Query metadata.
    pub query_metadata: std::collections::HashMap<String, String>,
}

impl VectorSearchResults {
    /// Create new empty search results.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            candidates_examined: 0,
            search_time_ms: 0.0,
            query_metadata: std::collections::HashMap::new(),
        }
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Sort results by similarity (descending).
    pub fn sort_by_similarity(&mut self) {
        self.results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sort results by distance (ascending).
    pub fn sort_by_distance(&mut self) {
        self.results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Take the top k results.
    pub fn take_top_k(&mut self, k: usize) {
        if self.results.len() > k {
            self.results.truncate(k);
        }
    }

    /// Filter results by minimum similarity.
    pub fn filter_by_similarity(&mut self, min_similarity: f32) {
        self.results
            .retain(|result| result.similarity >= min_similarity);
    }

    /// Get the best (highest similarity) result.
    pub fn best_result(&self) -> Option<&VectorSearchResult> {
        self.results.iter().max_by(|a, b| {
            a.similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl Default for VectorSearchResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Vector normalization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorNormalization {
    /// No normalization.
    None,
    /// L2 normalization (unit length).
    L2,
    /// L1 normalization.
    L1,
    /// Min-max normalization.
    MinMax,
}

/// Vector validation error types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorValidationError {
    /// Vector dimension mismatch.
    DimensionMismatch { expected: usize, actual: usize },
    /// Vector contains invalid values (NaN, infinity).
    InvalidValues,
    /// Vector is empty.
    Empty,
    /// Custom validation error.
    Custom(String),
}

impl std::fmt::Display for VectorValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorValidationError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Vector dimension mismatch: expected {expected}, got {actual}"
                )
            }
            VectorValidationError::InvalidValues => {
                write!(f, "Vector contains invalid values (NaN or infinity)")
            }
            VectorValidationError::Empty => {
                write!(f, "Vector is empty")
            }
            VectorValidationError::Custom(msg) => write!(f, "Custom validation error: {msg}"),
        }
    }
}

impl std::error::Error for VectorValidationError {}

/// Helper functions for vector operations.
pub mod utils {
    use super::*;
    use crate::vector::{DistanceMetric, Vector};

    /// Validate a vector against requirements.
    pub fn validate_vector(vector: &Vector, expected_dimension: Option<usize>) -> Result<()> {
        if vector.data.is_empty() {
            return Err(crate::error::SageError::InvalidOperation(
                VectorValidationError::Empty.to_string(),
            ));
        }

        if let Some(expected_dim) = expected_dimension
            && vector.data.len() != expected_dim
        {
            return Err(crate::error::SageError::InvalidOperation(
                VectorValidationError::DimensionMismatch {
                    expected: expected_dim,
                    actual: vector.data.len(),
                }
                .to_string(),
            ));
        }

        if !vector.is_valid() {
            return Err(crate::error::SageError::InvalidOperation(
                VectorValidationError::InvalidValues.to_string(),
            ));
        }

        Ok(())
    }

    /// Normalize a batch of vectors in parallel.
    pub fn normalize_vectors_parallel(vectors: &mut [Vector], method: VectorNormalization) {
        use rayon::prelude::*;

        match method {
            VectorNormalization::None => {
                // No normalization needed
            }
            VectorNormalization::L2 => {
                vectors.par_iter_mut().for_each(|vector| {
                    vector.normalize();
                });
            }
            VectorNormalization::L1 => {
                vectors.par_iter_mut().for_each(|vector| {
                    let l1_norm: f32 = vector.data.iter().map(|x| x.abs()).sum();
                    if l1_norm > 0.0 {
                        for value in &mut vector.data {
                            *value /= l1_norm;
                        }
                    }
                });
            }
            VectorNormalization::MinMax => {
                vectors.par_iter_mut().for_each(|vector| {
                    if let (Some(&min_val), Some(&max_val)) = (
                        vector.data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
                        vector.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
                    ) {
                        let range = max_val - min_val;
                        if range > 0.0 {
                            for value in &mut vector.data {
                                *value = (*value - min_val) / range;
                            }
                        }
                    }
                });
            }
        }
    }

    /// Calculate batch similarities between a query and multiple vectors.
    pub fn batch_similarities(
        query: &Vector,
        vectors: &[Vector],
        metric: DistanceMetric,
    ) -> Result<Vec<f32>> {
        vectors
            .iter()
            .map(|vector| metric.similarity(&query.data, &vector.data))
            .collect()
    }

    /// Calculate batch distances between a query and multiple vectors.
    pub fn batch_distances(
        query: &Vector,
        vectors: &[Vector],
        metric: DistanceMetric,
    ) -> Result<Vec<f32>> {
        vectors
            .iter()
            .map(|vector| metric.distance(&query.data, &vector.data))
            .collect()
    }
}
