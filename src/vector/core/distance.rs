//! Distance metrics for vector similarity calculation.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SarissaError};

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
