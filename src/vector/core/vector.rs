//! Core vector data structure.

use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{PlatypusError, Result};

/// Metadata key used to store the original (pre-embedded) text.
pub const ORIGINAL_TEXT_METADATA_KEY: &str = "original_text";

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

    /// Store the original text representation for this vector.
    pub fn set_original_text<T: Into<String>>(&mut self, text: T) {
        self.metadata
            .insert(ORIGINAL_TEXT_METADATA_KEY.to_string(), text.into());
    }

    /// Get metadata by key.
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Convenience accessor for the stored original text.
    pub fn original_text(&self) -> Option<&str> {
        self.metadata
            .get(ORIGINAL_TEXT_METADATA_KEY)
            .map(|s| s.as_str())
    }

    /// Validate that this vector has the expected dimension.
    pub fn validate_dimension(&self, expected_dim: usize) -> Result<()> {
        if self.data.len() != expected_dim {
            return Err(PlatypusError::InvalidOperation(format!(
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
