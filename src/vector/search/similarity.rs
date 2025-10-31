//! Advanced similarity metrics and aggregation functions.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::{DistanceMetric, Vector};

/// Advanced similarity metrics beyond basic distance functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdvancedSimilarityMetric {
    /// Weighted cosine similarity.
    WeightedCosine,
    /// Jaccard similarity for binary vectors.
    Jaccard,
    /// Tanimoto similarity.
    Tanimoto,
    /// Pearson correlation coefficient.
    Pearson,
    /// Spearman rank correlation.
    Spearman,
}

impl AdvancedSimilarityMetric {
    /// Calculate advanced similarity between two vectors.
    pub fn similarity(&self, a: &Vector, b: &Vector, weights: Option<&[f32]>) -> Result<f32> {
        if a.dimension() != b.dimension() {
            return Err(crate::error::YatagarasuError::InvalidOperation(
                "Vector dimensions must match".to_string(),
            ));
        }

        let result = match self {
            AdvancedSimilarityMetric::WeightedCosine => {
                self.weighted_cosine_similarity(&a.data, &b.data, weights)?
            }
            AdvancedSimilarityMetric::Jaccard => self.jaccard_similarity(&a.data, &b.data)?,
            AdvancedSimilarityMetric::Tanimoto => self.tanimoto_similarity(&a.data, &b.data)?,
            AdvancedSimilarityMetric::Pearson => self.pearson_correlation(&a.data, &b.data)?,
            AdvancedSimilarityMetric::Spearman => self.spearman_correlation(&a.data, &b.data)?,
        };

        Ok(result.clamp(0.0, 1.0))
    }

    /// Calculate weighted cosine similarity.
    fn weighted_cosine_similarity(
        &self,
        a: &[f32],
        b: &[f32],
        weights: Option<&[f32]>,
    ) -> Result<f32> {
        let default_weights = vec![1.0; a.len()];
        let weights = weights.unwrap_or(&default_weights);

        if weights.len() != a.len() {
            return Err(crate::error::YatagarasuError::InvalidOperation(
                "Weight vector dimension mismatch".to_string(),
            ));
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..a.len() {
            let weighted_a = a[i] * weights[i];
            let weighted_b = b[i] * weights[i];

            dot_product += weighted_a * weighted_b;
            norm_a += weighted_a * weighted_a;
            norm_b += weighted_b * weighted_b;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm_a.sqrt() * norm_b.sqrt()))
        }
    }

    /// Calculate Jaccard similarity for binary vectors.
    fn jaccard_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let mut intersection = 0;
        let mut union = 0;

        for i in 0..a.len() {
            let a_bin = if a[i] > 0.0 { 1 } else { 0 };
            let b_bin = if b[i] > 0.0 { 1 } else { 0 };

            intersection += a_bin & b_bin;
            union += a_bin | b_bin;
        }

        if union == 0 {
            Ok(1.0) // Both vectors are zero
        } else {
            Ok(intersection as f32 / union as f32)
        }
    }

    /// Calculate Tanimoto similarity.
    fn tanimoto_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..a.len() {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let denominator = norm_a + norm_b - dot_product;
        if denominator == 0.0 {
            Ok(1.0)
        } else {
            Ok(dot_product / denominator)
        }
    }

    /// Calculate Pearson correlation coefficient.
    fn pearson_correlation(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let n = a.len() as f32;

        let mean_a = a.iter().sum::<f32>() / n;
        let mean_b = b.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_a = 0.0;
        let mut sum_sq_b = 0.0;

        for i in 0..a.len() {
            let diff_a = a[i] - mean_a;
            let diff_b = b[i] - mean_b;

            numerator += diff_a * diff_b;
            sum_sq_a += diff_a * diff_a;
            sum_sq_b += diff_b * diff_b;
        }

        let denominator = (sum_sq_a * sum_sq_b).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok((numerator / denominator + 1.0) / 2.0) // Normalize to 0-1
        }
    }

    /// Calculate Spearman rank correlation.
    fn spearman_correlation(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        // Convert to ranks
        let ranks_a = self.convert_to_ranks(a);
        let ranks_b = self.convert_to_ranks(b);

        // Calculate Pearson correlation on ranks
        self.pearson_correlation(&ranks_a, &ranks_b)
    }

    /// Convert values to ranks.
    fn convert_to_ranks(&self, values: &[f32]) -> Vec<f32> {
        let mut indexed_values: Vec<(f32, usize)> =
            values.iter().enumerate().map(|(i, &v)| (v, i)).collect();

        indexed_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; values.len()];
        for (rank, (_, original_index)) in indexed_values.iter().enumerate() {
            ranks[*original_index] = rank as f32;
        }

        ranks
    }
}

/// Aggregator for combining multiple similarity scores.
pub struct SimilarityAggregator {
    weights: Vec<f32>,
    metrics: Vec<DistanceMetric>,
}

impl SimilarityAggregator {
    /// Create a new similarity aggregator.
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            metrics: Vec::new(),
        }
    }

    /// Add a metric with its weight.
    pub fn add_metric(&mut self, metric: DistanceMetric, weight: f32) {
        self.metrics.push(metric);
        self.weights.push(weight);
    }

    /// Calculate aggregated similarity.
    pub fn aggregate_similarity(&self, a: &Vector, b: &Vector) -> Result<f32> {
        if self.metrics.is_empty() {
            return Ok(0.0);
        }

        let mut weighted_sum = 0.0;
        let total_weight: f32 = self.weights.iter().sum();

        for (metric, weight) in self.metrics.iter().zip(self.weights.iter()) {
            let similarity = metric.similarity(&a.data, &b.data)?;
            weighted_sum += similarity * weight;
        }

        Ok(weighted_sum / total_weight)
    }
}

impl Default for SimilarityAggregator {
    fn default() -> Self {
        Self::new()
    }
}
