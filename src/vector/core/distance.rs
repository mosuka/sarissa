//! Distance metrics for vector similarity calculation.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SarissaError};

/// Distance metrics for vector similarity calculation.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    Default,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]

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
                let (dot_product, norm_a_sq, norm_b_sq) = self.simd_dot_and_norms(a, b);
                let norm_a = norm_a_sq.sqrt();
                let norm_b = norm_b_sq.sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0 // Maximum distance for zero vectors
                } else {
                    1.0 - (dot_product / (norm_a * norm_b))
                }
            }
            DistanceMetric::Euclidean => self.simd_euclidean_sq(a, b).sqrt(),
            DistanceMetric::Manhattan => self.simd_manhattan(a, b),
            DistanceMetric::DotProduct => -self.simd_dot_product(a, b),
            DistanceMetric::Angular => {
                let (dot_product, norm_a_sq, norm_b_sq) = self.simd_dot_and_norms(a, b);
                let norm_a = norm_a_sq.sqrt();
                let norm_b = norm_b_sq.sqrt();

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

    /// Calculate dot product and squared norms in a single pass using SIMD.
    fn simd_dot_and_norms(&self, a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        use wide::f32x8;

        let mut dot_sum = f32x8::ZERO;
        let mut norm_a_sum = f32x8::ZERO;
        let mut norm_b_sum = f32x8::ZERO;

        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let rem_a = chunks_a.remainder();
        let rem_b = chunks_b.remainder();

        for (ca, cb) in chunks_a.zip(chunks_b) {
            let va = f32x8::from(ca);
            let vb = f32x8::from(cb);
            dot_sum += va * vb;
            norm_a_sum += va * va;
            norm_b_sum += vb * vb;
        }

        let mut dot_product: f32 = dot_sum.reduce_add();
        let mut norm_a_sq: f32 = norm_a_sum.reduce_add();
        let mut norm_b_sq: f32 = norm_b_sum.reduce_add();

        // Tail
        for (x, y) in rem_a.iter().zip(rem_b.iter()) {
            dot_product += x * y;
            norm_a_sq += x * x;
            norm_b_sq += y * y;
        }

        (dot_product, norm_a_sq, norm_b_sq)
    }

    /// Calculate dot product using SIMD.
    fn simd_dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        use wide::f32x8;

        let mut sum = f32x8::ZERO;
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let rem_a = chunks_a.remainder();
        let rem_b = chunks_b.remainder();

        for (ca, cb) in chunks_a.zip(chunks_b) {
            sum += f32x8::from(ca) * f32x8::from(cb);
        }

        let mut dot_product: f32 = sum.reduce_add();
        for (x, y) in rem_a.iter().zip(rem_b.iter()) {
            dot_product += x * y;
        }
        dot_product
    }

    /// Calculate squared Euclidean distance using SIMD.
    fn simd_euclidean_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        use wide::f32x8;

        let mut sum = f32x8::ZERO;
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let rem_a = chunks_a.remainder();
        let rem_b = chunks_b.remainder();

        for (ca, cb) in chunks_a.zip(chunks_b) {
            let diff = f32x8::from(ca) - f32x8::from(cb);
            sum += diff * diff;
        }

        let mut dist_sq: f32 = sum.reduce_add();
        for (x, y) in rem_a.iter().zip(rem_b.iter()) {
            dist_sq += (x - y).powi(2);
        }
        dist_sq
    }

    /// Calculate Manhattan distance using SIMD.
    fn simd_manhattan(&self, a: &[f32], b: &[f32]) -> f32 {
        use wide::f32x8;

        let mut sum = f32x8::ZERO;
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let rem_a = chunks_a.remainder();
        let rem_b = chunks_b.remainder();

        for (ca, cb) in chunks_a.zip(chunks_b) {
            let va = f32x8::from(ca);
            let vb = f32x8::from(cb);
            sum += (va - vb).abs();
        }

        let mut dist: f32 = sum.reduce_add();
        for (x, y) in rem_a.iter().zip(rem_b.iter()) {
            dist += (x - y).abs();
        }
        dist
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
