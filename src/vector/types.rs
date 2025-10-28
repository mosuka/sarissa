//! Common types used by both vector indexing and search modules.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::Vector;

/// Vector search request combining query vector and configuration.
#[derive(Debug, Clone)]
pub struct VectorSearchRequest {
    /// The query vector.
    pub query: Vector,
    /// Search configuration.
    pub config: VectorSearchConfig,
}

impl VectorSearchRequest {
    /// Create a new vector search request.
    pub fn new(query: Vector) -> Self {
        VectorSearchRequest {
            query,
            config: VectorSearchConfig::default(),
        }
    }

    /// Set the number of results to return.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.config.top_k = top_k;
        self
    }

    /// Set minimum similarity threshold.
    pub fn min_similarity(mut self, threshold: f32) -> Self {
        self.config.min_similarity = threshold;
        self
    }

    /// Set whether to include scores in results.
    pub fn include_scores(mut self, include: bool) -> Self {
        self.config.include_scores = include;
        self
    }

    /// Set whether to include vectors in results.
    pub fn include_vectors(mut self, include: bool) -> Self {
        self.config.include_vectors = include;
        self
    }

    /// Set search timeout in milliseconds.
    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        self.config.timeout_ms = Some(timeout);
        self
    }
}

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
    /// Reranking configuration.
    pub reranking: Option<crate::vector::search::ranking::RankingConfig>,
}

impl Default for VectorSearchConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            include_scores: true,
            include_vectors: false,
            timeout_ms: None,
            reranking: None,
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

/// Statistics about a vector index.
#[derive(Debug, Clone)]
pub struct VectorStats {
    /// Total number of vectors in the index.
    pub vector_count: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Index memory usage in bytes.
    pub memory_usage: usize,
    /// Build time in milliseconds.
    pub build_time_ms: u64,
}

/// Metadata about a vector index.
#[derive(Debug, Clone)]
pub struct VectorIndexMetadata {
    /// Index type (HNSW, Flat, IVF, etc.).
    pub index_type: String,
    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modified timestamp.
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Index version.
    pub version: String,
    /// Build configuration.
    pub build_config: serde_json::Value,
    /// Custom metadata.
    pub custom_metadata: std::collections::HashMap<String, String>,
}

/// Index validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether the index is valid.
    pub is_valid: bool,
    /// Validation errors found.
    pub errors: Vec<String>,
    /// Validation warnings.
    pub warnings: Vec<String>,
    /// Repair suggestions.
    pub repair_suggestions: Vec<String>,
}
