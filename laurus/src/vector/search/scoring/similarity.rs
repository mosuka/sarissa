//! Advanced similarity metrics and aggregation functions.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;

/// Pre-computed query side of a weighted cosine similarity, allowing
/// per-candidate calls to skip the redundant `weighted_a` and
/// `||weighted_a||²` work (#414).
///
/// Build it once per search via
/// [`AdvancedSimilarityMetric::prepare_weighted_cosine`] and call
/// [`AdvancedSimilarityMetric::weighted_cosine_with_prepared`] for
/// each candidate vector.
#[derive(Debug, Clone)]
pub struct PreparedWeightedCosine {
    /// `query[i] * weights[i]` precomputed once per search.
    pub weighted_query: Vec<f32>,
    /// `Σ weighted_query[i]²`, the squared norm of `weighted_query`.
    pub norm_sq: f32,
}

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
    ///
    /// Computes similarity using the configured metric, with optional weighting.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector
    /// * `b` - Second vector
    /// * `weights` - Optional weight vector for weighted metrics
    ///
    /// # Returns
    ///
    /// Similarity score clamped to [0, 1] range
    ///
    /// # Errors
    ///
    /// Returns error if vector dimensions don't match
    pub fn similarity(&self, a: &Vector, b: &Vector, weights: Option<&[f32]>) -> Result<f32> {
        if a.dimension() != b.dimension() {
            return Err(crate::error::LaurusError::InvalidOperation(
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
    ///
    /// Computes cosine similarity with per-dimension weights.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector data
    /// * `b` - Second vector data
    /// * `weights` - Optional per-dimension weight vector
    ///
    /// # Returns
    ///
    /// Weighted cosine similarity score
    ///
    /// # Errors
    ///
    /// Returns error if weight vector dimension doesn't match
    fn weighted_cosine_similarity(
        &self,
        a: &[f32],
        b: &[f32],
        weights: Option<&[f32]>,
    ) -> Result<f32> {
        let default_weights = vec![1.0; a.len()];
        let weights = weights.unwrap_or(&default_weights);

        if weights.len() != a.len() {
            return Err(crate::error::LaurusError::InvalidOperation(
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

    /// Pre-compute the query side of a weighted cosine similarity (#414).
    ///
    /// Returns `weighted_query[i] = query[i] * weights[i]` and its
    /// squared norm. Pair with
    /// [`Self::weighted_cosine_with_prepared`] to avoid the per-
    /// candidate `weighted_a` and `||weighted_a||²` recomputation that
    /// [`Self::weighted_cosine_similarity`] does on every call.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector.
    /// * `weights` - Optional per-dimension weights; `None` falls back
    ///   to a uniform `1.0` vector matching the existing
    ///   [`Self::weighted_cosine_similarity`] semantics.
    ///
    /// # Errors
    ///
    /// Returns an error when `weights` is `Some` but its length does
    /// not match `query.len()`.
    pub fn prepare_weighted_cosine(
        query: &[f32],
        weights: Option<&[f32]>,
    ) -> Result<PreparedWeightedCosine> {
        if let Some(w) = weights
            && w.len() != query.len()
        {
            return Err(crate::error::LaurusError::InvalidOperation(
                "Weight vector dimension mismatch".to_string(),
            ));
        }

        let mut weighted_query = Vec::with_capacity(query.len());
        let mut norm_sq = 0.0_f32;
        for i in 0..query.len() {
            let w = weights.map_or(1.0, |ws| ws[i]);
            let wq = query[i] * w;
            weighted_query.push(wq);
            norm_sq += wq * wq;
        }
        Ok(PreparedWeightedCosine {
            weighted_query,
            norm_sq,
        })
    }

    /// Weighted cosine similarity that consumes the prepared query
    /// side from [`Self::prepare_weighted_cosine`] and the same
    /// per-dimension `weights` vector (#414). Only `b`-side
    /// per-dimension weighted values, the dot product, and the
    /// candidate norm are computed per call.
    ///
    /// # Arguments
    ///
    /// * `prepared` - Output of [`Self::prepare_weighted_cosine`].
    /// * `b` - Candidate vector.
    /// * `weights` - The same `weights` slice that produced
    ///   `prepared`. The function is unchecked against accidental
    ///   weight-mismatch — callers are expected to reuse the value
    ///   they passed to `prepare_weighted_cosine`.
    ///
    /// # Errors
    ///
    /// Returns an error when `b.len() != prepared.weighted_query.len()`
    /// or `weights.is_some()` and `weights.unwrap().len() != b.len()`.
    pub fn weighted_cosine_with_prepared(
        prepared: &PreparedWeightedCosine,
        b: &[f32],
        weights: Option<&[f32]>,
    ) -> Result<f32> {
        if b.len() != prepared.weighted_query.len() {
            return Err(crate::error::LaurusError::InvalidOperation(
                "Vector dimensions must match".to_string(),
            ));
        }
        if let Some(w) = weights
            && w.len() != b.len()
        {
            return Err(crate::error::LaurusError::InvalidOperation(
                "Weight vector dimension mismatch".to_string(),
            ));
        }

        let mut dot_product = 0.0_f32;
        let mut norm_b = 0.0_f32;
        for i in 0..b.len() {
            let w = weights.map_or(1.0, |ws| ws[i]);
            let weighted_b = b[i] * w;
            dot_product += prepared.weighted_query[i] * weighted_b;
            norm_b += weighted_b * weighted_b;
        }

        if prepared.norm_sq == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (prepared.norm_sq.sqrt() * norm_b.sqrt()))
        }
    }

    /// Calculate Jaccard similarity for binary vectors.
    ///
    /// Treats non-zero values as 1, zero values as 0, then computes Jaccard index.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector data
    /// * `b` - Second vector data
    ///
    /// # Returns
    ///
    /// Jaccard similarity: intersection/union ratio
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
    ///
    /// Also known as extended Jaccard, handles continuous values.
    /// Formula: (a·b) / (||a||² + ||b||² - a·b)
    ///
    /// # Arguments
    ///
    /// * `a` - First vector data
    /// * `b` - Second vector data
    ///
    /// # Returns
    ///
    /// Tanimoto similarity coefficient
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
    ///
    /// Measures linear correlation between vectors, normalized to [0, 1].
    ///
    /// # Arguments
    ///
    /// * `a` - First vector data
    /// * `b` - Second vector data
    ///
    /// # Returns
    ///
    /// Pearson correlation normalized from [-1, 1] to [0, 1]
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
    ///
    /// Non-parametric measure of rank correlation.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector data
    /// * `b` - Second vector data
    ///
    /// # Returns
    ///
    /// Spearman correlation coefficient
    fn spearman_correlation(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        // Convert to ranks
        let ranks_a = self.convert_to_ranks(a);
        let ranks_b = self.convert_to_ranks(b);

        // Calculate Pearson correlation on ranks
        self.pearson_correlation(&ranks_a, &ranks_b)
    }

    /// Convert values to ranks.
    ///
    /// Maps each value to its position in sorted order.
    ///
    /// # Arguments
    ///
    /// * `values` - Values to convert to ranks
    ///
    /// # Returns
    ///
    /// Vector of rank positions
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
///
/// Allows combining results from different distance metrics with
/// configurable weighting for each metric.
pub struct SimilarityAggregator {
    /// Weights for each metric.
    weights: Vec<f32>,
    /// Distance metrics to aggregate.
    metrics: Vec<DistanceMetric>,
}

impl SimilarityAggregator {
    /// Create a new similarity aggregator.
    ///
    /// Creates an empty aggregator. Use [`add_metric`] to add metrics.
    ///
    /// [`add_metric`]: Self::add_metric
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            metrics: Vec::new(),
        }
    }

    /// Add a metric with its weight.
    ///
    /// # Arguments
    ///
    /// * `metric` - Distance metric to include in aggregation
    /// * `weight` - Weight for this metric in final calculation
    pub fn add_metric(&mut self, metric: DistanceMetric, weight: f32) {
        self.metrics.push(metric);
        self.weights.push(weight);
    }

    /// Calculate aggregated similarity.
    ///
    /// Computes weighted average of similarities from all added metrics.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    ///
    /// Weighted average similarity score
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

        if total_weight == 0.0 {
            return Ok(0.0);
        }

        Ok(weighted_sum / total_weight)
    }
}

impl Default for SimilarityAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `weighted_cosine_with_prepared` must produce the same value as
    /// the in-place `weighted_cosine_similarity` (#414). Tested with
    /// both an explicit weight vector and the implicit uniform-1.0
    /// weights path.
    #[test]
    fn weighted_cosine_prepared_matches_inline() {
        let metric = AdvancedSimilarityMetric::WeightedCosine;
        let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.013 + 0.7).collect();
        let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.021 - 0.4).collect();
        let weights: Vec<f32> = (0..256).map(|i| 1.0 + (i as f32) * 0.002).collect();

        for w in [None, Some(weights.as_slice())] {
            let direct = metric.weighted_cosine_similarity(&a, &b, w).unwrap();
            let prepared = AdvancedSimilarityMetric::prepare_weighted_cosine(&a, w).unwrap();
            let via_prep =
                AdvancedSimilarityMetric::weighted_cosine_with_prepared(&prepared, &b, w).unwrap();
            assert!(
                (direct - via_prep).abs() < 1e-5,
                "weights={:?}: direct={direct}, prepared={via_prep}",
                w.map(|s| s.len())
            );
        }
    }
}
