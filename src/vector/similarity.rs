//! Similarity calculation utilities for vector search.
//!
//! This module provides additional similarity metrics and utilities that complement
//! the basic distance metrics in the main vector module. It includes:
//! - Advanced similarity measures for specialized use cases
//! - Similarity aggregation and combination functions
//! - Utilities for normalizing and comparing similarity scores

use crate::error::{SarissaError, Result};
use crate::vector::{DistanceMetric, Vector, VectorSearchResult};
use serde::{Deserialize, Serialize};

/// Advanced similarity metrics for specialized vector comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Jaccard similarity for binary vectors.
    Jaccard,
    /// Pearson correlation coefficient.
    Pearson,
    /// Spearman rank correlation.
    Spearman,
    /// Chebyshev distance (L-infinity norm).
    Chebyshev,
    /// Canberra distance.
    Canberra,
    /// Bray-Curtis dissimilarity.
    BrayCurtis,
}

impl SimilarityMetric {
    /// Calculate similarity using this metric.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(SarissaError::InvalidOperation(
                "Vector dimensions must match for similarity calculation".to_string(),
            ));
        }

        let result = match self {
            SimilarityMetric::Jaccard => Self::jaccard_similarity(a, b)?,
            SimilarityMetric::Pearson => Self::pearson_correlation(a, b)?,
            SimilarityMetric::Spearman => Self::spearman_correlation(a, b)?,
            SimilarityMetric::Chebyshev => Self::chebyshev_similarity(a, b)?,
            SimilarityMetric::Canberra => Self::canberra_similarity(a, b)?,
            SimilarityMetric::BrayCurtis => Self::bray_curtis_similarity(a, b)?,
        };

        Ok(result.clamp(0.0, 1.0))
    }

    /// Calculate Jaccard similarity (for binary or sparse vectors).
    fn jaccard_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
        let mut intersection = 0.0;
        let mut union = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            let x_nonzero = *x > 0.0;
            let y_nonzero = *y > 0.0;

            if x_nonzero || y_nonzero {
                union += 1.0;
                if x_nonzero && y_nonzero {
                    intersection += 1.0;
                }
            }
        }

        if union == 0.0 {
            Ok(1.0) // Both vectors are zero
        } else {
            Ok(intersection / union)
        }
    }

    /// Calculate Pearson correlation coefficient.
    fn pearson_correlation(a: &[f32], b: &[f32]) -> Result<f32> {
        let n = a.len() as f32;
        if n == 0.0 {
            return Ok(0.0);
        }

        let mean_a = a.iter().sum::<f32>() / n;
        let mean_b = b.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            let diff_a = x - mean_a;
            let diff_b = y - mean_b;

            numerator += diff_a * diff_b;
            sum_sq_a += diff_a * diff_a;
            sum_sq_b += diff_b * diff_b;
        }

        let denominator = (sum_sq_a * sum_sq_b).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok((numerator / denominator + 1.0) / 2.0) // Normalize to [0, 1]
        }
    }

    /// Calculate Spearman rank correlation.
    fn spearman_correlation(a: &[f32], b: &[f32]) -> Result<f32> {
        let ranks_a = Self::calculate_ranks(a);
        let ranks_b = Self::calculate_ranks(b);

        Self::pearson_correlation(&ranks_a, &ranks_b)
    }

    /// Calculate ranks for Spearman correlation.
    fn calculate_ranks(values: &[f32]) -> Vec<f32> {
        let mut indexed_values: Vec<(usize, f32)> = values.iter().cloned().enumerate().collect();
        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; values.len()];
        for (rank, (original_index, _)) in indexed_values.iter().enumerate() {
            ranks[*original_index] = rank as f32 + 1.0;
        }

        ranks
    }

    /// Calculate Chebyshev similarity.
    fn chebyshev_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);

        // Convert distance to similarity using exponential decay
        Ok((-max_diff).exp())
    }

    /// Calculate Canberra similarity.
    fn canberra_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
        let distance: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let sum = x.abs() + y.abs();
                if sum > 0.0 {
                    (x - y).abs() / sum
                } else {
                    0.0
                }
            })
            .sum();

        // Convert distance to similarity
        Ok((-distance / a.len() as f32).exp())
    }

    /// Calculate Bray-Curtis similarity.
    fn bray_curtis_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
        let numerator: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        let denominator: f32 = a.iter().zip(b.iter()).map(|(x, y)| x + y).sum();

        if denominator == 0.0 {
            Ok(1.0) // Both vectors are zero
        } else {
            Ok(1.0 - (numerator / denominator))
        }
    }

    /// Get the name of this similarity metric.
    pub fn name(&self) -> &'static str {
        match self {
            SimilarityMetric::Jaccard => "jaccard",
            SimilarityMetric::Pearson => "pearson",
            SimilarityMetric::Spearman => "spearman",
            SimilarityMetric::Chebyshev => "chebyshev",
            SimilarityMetric::Canberra => "canberra",
            SimilarityMetric::BrayCurtis => "bray_curtis",
        }
    }
}

/// Combination strategies for merging different similarity scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityCombination {
    /// Weighted average of similarities.
    WeightedAverage,
    /// Maximum similarity score.
    Maximum,
    /// Minimum similarity score.
    Minimum,
    /// Harmonic mean of similarities.
    HarmonicMean,
    /// Geometric mean of similarities.
    GeometricMean,
    /// Product of similarities.
    Product,
}

impl SimilarityCombination {
    /// Combine multiple similarity scores using this strategy.
    pub fn combine(&self, scores: &[(f32, f32)]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }

        match self {
            SimilarityCombination::WeightedAverage => {
                let total_weight: f32 = scores.iter().map(|(_, weight)| weight).sum();
                if total_weight > 0.0 {
                    scores
                        .iter()
                        .map(|(score, weight)| score * weight)
                        .sum::<f32>()
                        / total_weight
                } else {
                    0.0
                }
            }
            SimilarityCombination::Maximum => {
                scores.iter().map(|(score, _)| *score).fold(0.0, f32::max)
            }
            SimilarityCombination::Minimum => {
                scores.iter().map(|(score, _)| *score).fold(1.0, f32::min)
            }
            SimilarityCombination::HarmonicMean => {
                let n = scores.len() as f32;
                let sum_reciprocals: f32 =
                    scores.iter().map(|(score, _)| 1.0 / score.max(1e-10)).sum();
                n / sum_reciprocals
            }
            SimilarityCombination::GeometricMean => {
                let product: f32 = scores.iter().map(|(score, _)| *score).product();
                product.powf(1.0 / scores.len() as f32)
            }
            SimilarityCombination::Product => scores.iter().map(|(score, _)| *score).product(),
        }
    }
}

/// Hybrid similarity calculator that combines multiple similarity measures.
#[derive(Debug, Clone)]
pub struct HybridSimilarity {
    /// Distance metric configurations with weights.
    distance_configs: Vec<(DistanceMetric, f32)>,
    /// Advanced similarity metric configurations with weights.
    similarity_configs: Vec<(SimilarityMetric, f32)>,
    /// Combination strategy for merging scores.
    combination: SimilarityCombination,
    /// Normalization strategy.
    normalize: bool,
}

impl HybridSimilarity {
    /// Create a new hybrid similarity calculator.
    pub fn new(combination: SimilarityCombination) -> Self {
        Self {
            distance_configs: Vec::new(),
            similarity_configs: Vec::new(),
            combination,
            normalize: true,
        }
    }

    /// Add a distance metric with weight.
    pub fn add_distance_metric(mut self, metric: DistanceMetric, weight: f32) -> Self {
        self.distance_configs.push((metric, weight));
        self
    }

    /// Add a similarity metric with weight.
    pub fn add_similarity_metric(mut self, metric: SimilarityMetric, weight: f32) -> Self {
        self.similarity_configs.push((metric, weight));
        self
    }

    /// Set normalization strategy.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Calculate hybrid similarity between two vectors.
    pub fn calculate(&self, a: &Vector, b: &Vector) -> Result<f32> {
        let mut scores = Vec::new();

        // Calculate distance-based similarities
        for (metric, weight) in &self.distance_configs {
            let similarity = metric.similarity(&a.data, &b.data)?;
            scores.push((similarity, *weight));
        }

        // Calculate advanced similarities
        for (metric, weight) in &self.similarity_configs {
            let similarity = metric.similarity(&a.data, &b.data)?;
            scores.push((similarity, *weight));
        }

        let combined = self.combination.combine(&scores);

        if self.normalize {
            Ok(combined.clamp(0.0, 1.0))
        } else {
            Ok(combined)
        }
    }

    /// Calculate hybrid similarity for a query against multiple candidates.
    pub fn calculate_batch(&self, query: &Vector, candidates: &[Vector]) -> Result<Vec<f32>> {
        candidates
            .iter()
            .map(|candidate| self.calculate(query, candidate))
            .collect()
    }

    /// Rank search results using hybrid similarity.
    pub fn rank_results(&self, query: &Vector, results: &mut [VectorSearchResult]) -> Result<()> {
        for result in results.iter_mut() {
            if let Some(ref vector) = result.vector {
                let hybrid_score = self.calculate(query, vector)?;
                result.similarity = hybrid_score;
            }
        }

        // Sort by hybrid similarity (descending)
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(())
    }
}

/// Similarity aggregation utilities.
pub struct SimilarityAggregator;

impl SimilarityAggregator {
    /// Calculate centroid vector for a collection of vectors.
    pub fn centroid(vectors: &[Vector]) -> Result<Vector> {
        if vectors.is_empty() {
            return Err(SarissaError::InvalidOperation(
                "Cannot calculate centroid of empty vector collection".to_string(),
            ));
        }

        let dimension = vectors[0].dimension();
        for vector in vectors {
            if vector.dimension() != dimension {
                return Err(SarissaError::InvalidOperation(
                    "All vectors must have the same dimension".to_string(),
                ));
            }
        }

        let mut centroid_data = vec![0.0; dimension];
        for vector in vectors {
            for (i, value) in vector.data.iter().enumerate() {
                centroid_data[i] += value;
            }
        }

        let n = vectors.len() as f32;
        for value in &mut centroid_data {
            *value /= n;
        }

        Ok(Vector::new(centroid_data))
    }

    /// Calculate average similarity to a set of reference vectors.
    pub fn average_similarity(
        query: &Vector,
        references: &[Vector],
        metric: DistanceMetric,
    ) -> Result<f32> {
        if references.is_empty() {
            return Ok(0.0);
        }

        let mut total_similarity = 0.0;
        for reference in references {
            let similarity = metric.similarity(&query.data, &reference.data)?;
            total_similarity += similarity;
        }

        Ok(total_similarity / references.len() as f32)
    }

    /// Find the most similar vector from a collection.
    pub fn most_similar(
        query: &Vector,
        candidates: &[Vector],
        metric: DistanceMetric,
    ) -> Result<Option<(usize, f32)>> {
        if candidates.is_empty() {
            return Ok(None);
        }

        let mut best_index = 0;
        let mut best_similarity = metric.similarity(&query.data, &candidates[0].data)?;

        for (i, candidate) in candidates.iter().enumerate().skip(1) {
            let similarity = metric.similarity(&query.data, &candidate.data)?;
            if similarity > best_similarity {
                best_similarity = similarity;
                best_index = i;
            }
        }

        Ok(Some((best_index, best_similarity)))
    }

    /// Calculate pairwise similarity matrix for a collection of vectors.
    pub fn similarity_matrix(vectors: &[Vector], metric: DistanceMetric) -> Result<Vec<Vec<f32>>> {
        let n = vectors.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            matrix[i][i] = 1.0; // Self-similarity is 1.0
            for j in (i + 1)..n {
                let similarity = metric.similarity(&vectors[i].data, &vectors[j].data)?;
                matrix[i][j] = similarity;
                matrix[j][i] = similarity; // Symmetric
            }
        }

        Ok(matrix)
    }

    /// Calculate diversity score for a set of vectors.
    pub fn diversity_score(vectors: &[Vector], metric: DistanceMetric) -> Result<f32> {
        if vectors.len() < 2 {
            return Ok(0.0);
        }

        let matrix = Self::similarity_matrix(vectors, metric)?;
        let mut total_dissimilarity = 0.0;
        let mut count = 0;

        for i in 0..matrix.len() {
            for j in (i + 1)..matrix.len() {
                total_dissimilarity += 1.0 - matrix[i][j];
                count += 1;
            }
        }

        Ok(total_dissimilarity / count as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_similarity() {
        let a = vec![1.0, 0.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0, 0.0];

        let similarity = SimilarityMetric::Jaccard.similarity(&a, &b).unwrap();
        assert!((similarity - 0.333333).abs() < 0.001); // 1/3 intersection over union
    }

    #[test]
    fn test_pearson_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 4.0, 6.0, 8.0]; // Perfect positive correlation

        let similarity = SimilarityMetric::Pearson.similarity(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_spearman_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0]; // Perfect negative rank correlation

        let similarity = SimilarityMetric::Spearman.similarity(&a, &b).unwrap();
        assert!(similarity < 0.1); // Should be close to 0 after normalization
    }

    #[test]
    fn test_similarity_combination() {
        let scores = vec![(0.8, 1.0), (0.6, 0.5), (0.9, 2.0)];

        let weighted_avg = SimilarityCombination::WeightedAverage.combine(&scores);
        let maximum = SimilarityCombination::Maximum.combine(&scores);
        let minimum = SimilarityCombination::Minimum.combine(&scores);

        assert!(weighted_avg > 0.7 && weighted_avg < 0.9);
        assert_eq!(maximum, 0.9);
        assert_eq!(minimum, 0.6);
    }

    #[test]
    fn test_hybrid_similarity() {
        let hybrid = HybridSimilarity::new(SimilarityCombination::WeightedAverage)
            .add_distance_metric(DistanceMetric::Cosine, 1.0)
            .add_distance_metric(DistanceMetric::Euclidean, 0.5)
            .add_similarity_metric(SimilarityMetric::Pearson, 0.3);

        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![2.0, 4.0, 6.0]);

        let similarity = hybrid.calculate(&a, &b).unwrap();
        assert!(similarity > 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_centroid_calculation() {
        let vectors = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![1.0, 1.0]),
        ];

        let centroid = SimilarityAggregator::centroid(&vectors).unwrap();
        assert!((centroid.data[0] - 0.666667).abs() < 0.001);
        assert!((centroid.data[1] - 0.666667).abs() < 0.001);
    }

    #[test]
    fn test_average_similarity() {
        let query = Vector::new(vec![1.0, 0.0]);
        let references = vec![
            Vector::new(vec![1.0, 0.0]), // Same vector
            Vector::new(vec![0.0, 1.0]), // Orthogonal
        ];

        let avg_sim =
            SimilarityAggregator::average_similarity(&query, &references, DistanceMetric::Cosine)
                .unwrap();

        assert!(avg_sim > 0.4 && avg_sim < 0.6); // Should be around 0.5
    }

    #[test]
    fn test_most_similar() {
        let query = Vector::new(vec![1.0, 0.0]);
        let candidates = vec![
            Vector::new(vec![0.0, 1.0]),  // Orthogonal
            Vector::new(vec![0.9, 0.1]),  // Very similar
            Vector::new(vec![-1.0, 0.0]), // Opposite
        ];

        let (best_index, best_sim) =
            SimilarityAggregator::most_similar(&query, &candidates, DistanceMetric::Cosine)
                .unwrap()
                .unwrap();

        assert_eq!(best_index, 1); // Should be the similar vector
        assert!(best_sim > 0.9);
    }

    #[test]
    fn test_similarity_matrix() {
        let vectors = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![1.0, 1.0]),
        ];

        let matrix =
            SimilarityAggregator::similarity_matrix(&vectors, DistanceMetric::Cosine).unwrap();

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        assert_eq!(matrix[0][0], 1.0); // Self-similarity
        assert_eq!(matrix[1][1], 1.0);
        assert_eq!(matrix[2][2], 1.0);
        assert_eq!(matrix[0][1], matrix[1][0]); // Symmetry
    }

    #[test]
    fn test_diversity_score() {
        let vectors = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]), // Orthogonal - high diversity
        ];

        let diversity =
            SimilarityAggregator::diversity_score(&vectors, DistanceMetric::Cosine).unwrap();
        assert!(diversity > 0.8); // Should be high for orthogonal vectors

        let similar_vectors = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.99, 0.01]), // Very similar - low diversity
        ];

        let low_diversity =
            SimilarityAggregator::diversity_score(&similar_vectors, DistanceMetric::Cosine)
                .unwrap();
        assert!(low_diversity < 0.2); // Should be low for similar vectors
    }

    #[test]
    fn test_hybrid_similarity_batch() {
        let hybrid = HybridSimilarity::new(SimilarityCombination::Maximum)
            .add_distance_metric(DistanceMetric::Cosine, 1.0);

        let query = Vector::new(vec![1.0, 0.0]);
        let candidates = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![0.5, 0.5]),
        ];

        let similarities = hybrid.calculate_batch(&query, &candidates).unwrap();
        assert_eq!(similarities.len(), 3);
        assert!(similarities[0] > similarities[1]); // First should be most similar
    }

    #[test]
    fn test_empty_vectors() {
        let empty_vectors: Vec<Vector> = vec![];
        assert!(SimilarityAggregator::centroid(&empty_vectors).is_err());

        let query = Vector::new(vec![1.0, 0.0]);
        let result =
            SimilarityAggregator::most_similar(&query, &empty_vectors, DistanceMetric::Cosine)
                .unwrap();
        assert!(result.is_none());
    }
}
