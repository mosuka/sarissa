//! Vector data structures and common types.
//!
//! This module provides the core vector data structures and common types
//! shared between vector indexing and search modules.

pub mod engine; // Unified vector engine (indexing + search)
pub mod reader;
pub mod types;

// Sub-modules
pub mod index; // Vector indexing
pub mod search; // Vector search

use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SageError};

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
            return Err(SageError::InvalidOperation(format!(
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
            return Err(SageError::InvalidOperation(
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
            DistanceMetric::DotProduct => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
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
            DistanceMetric::Euclidean => (-distance).exp(),
            DistanceMetric::Manhattan => (-distance).exp(),
            DistanceMetric::DotProduct => -distance,
            DistanceMetric::Angular => 1.0 - (distance / std::f32::consts::PI),
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
            _ => Err(SageError::InvalidOperation(format!(
                "Unknown distance metric: {s}"
            ))),
        }
    }

    /// Calculate distance between a query vector and multiple vectors in parallel.
    pub fn batch_distance_parallel(&self, query: &[f32], vectors: &[&[f32]]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        if vectors.len() < 100 {
            return vectors
                .iter()
                .map(|v| self.distance(query, v))
                .collect::<Result<Vec<_>>>();
        }

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

        if vectors.len() < 100 {
            return vectors
                .iter()
                .map(|v| self.similarity(query, v))
                .collect::<Result<Vec<_>>>();
        }

        vectors
            .par_iter()
            .map(|v| self.similarity(query, v))
            .collect::<Result<Vec<_>>>()
    }
}

// Re-export commonly used types
pub use types::{VectorSearchConfig, VectorSearchRequest, VectorSearchResult, VectorSearchResults};
