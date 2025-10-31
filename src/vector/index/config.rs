//! Configuration types for vector indexes.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::Vector;

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
    use crate::vector::DistanceMetric;

    /// Validate a vector against requirements.
    pub fn validate_vector(vector: &Vector, expected_dimension: Option<usize>) -> Result<()> {
        if vector.data.is_empty() {
            return Err(crate::error::YatagarasuError::InvalidOperation(
                VectorValidationError::Empty.to_string(),
            ));
        }

        if let Some(expected_dim) = expected_dimension
            && vector.data.len() != expected_dim
        {
            return Err(crate::error::YatagarasuError::InvalidOperation(
                VectorValidationError::DimensionMismatch {
                    expected: expected_dim,
                    actual: vector.data.len(),
                }
                .to_string(),
            ));
        }

        if !vector.is_valid() {
            return Err(crate::error::YatagarasuError::InvalidOperation(
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
