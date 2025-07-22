//! Scoring implementations for ranking search results.

use crate::util::simd;
use std::fmt::Debug;

/// Trait for document scorers.
pub trait Scorer: Send + Debug {
    /// Calculate the score for a document.
    fn score(&self, doc_id: u64, term_freq: f32) -> f32;

    /// Get the boost factor for this scorer.
    fn boost(&self) -> f32;

    /// Set the boost factor for this scorer.
    fn set_boost(&mut self, boost: f32);

    /// Get the maximum possible score.
    fn max_score(&self) -> f32;

    /// Get the name of this scorer.
    fn name(&self) -> &'static str;
}

/// BM25 scorer implementation.
#[derive(Debug, Clone)]
pub struct BM25Scorer {
    /// Document frequency of the term.
    doc_freq: u64,
    /// Total term frequency across all documents.
    #[allow(dead_code)]
    total_term_freq: u64,
    /// Number of documents containing the field.
    #[allow(dead_code)]
    field_doc_count: u64,
    /// Average field length.
    avg_field_length: f64,
    /// Total number of documents in the index.
    total_docs: u64,
    /// Boost factor.
    boost: f32,
    /// BM25 k1 parameter.
    k1: f32,
    /// BM25 b parameter.
    b: f32,
}

impl BM25Scorer {
    /// Create a new BM25 scorer.
    pub fn new(
        doc_freq: u64,
        total_term_freq: u64,
        field_doc_count: u64,
        avg_field_length: f64,
        total_docs: u64,
        boost: f32,
    ) -> Self {
        BM25Scorer {
            doc_freq,
            total_term_freq,
            field_doc_count,
            avg_field_length,
            total_docs,
            boost,
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Create a new BM25 scorer with custom parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        doc_freq: u64,
        total_term_freq: u64,
        field_doc_count: u64,
        avg_field_length: f64,
        total_docs: u64,
        boost: f32,
        k1: f32,
        b: f32,
    ) -> Self {
        BM25Scorer {
            doc_freq,
            total_term_freq,
            field_doc_count,
            avg_field_length,
            total_docs,
            boost,
            k1,
            b,
        }
    }

    /// Calculate the IDF (Inverse Document Frequency) component.
    fn idf(&self) -> f32 {
        if self.doc_freq == 0 || self.total_docs == 0 {
            return 0.0;
        }

        let n = self.total_docs as f32;
        let df = self.doc_freq as f32;

        // IDF = log((N - df + 0.5) / (df + 0.5))
        ((n - df + 0.5) / (df + 0.5)).ln()
    }

    /// Calculate the TF (Term Frequency) component.
    fn tf(&self, term_freq: f32, field_length: f32) -> f32 {
        if term_freq == 0.0 {
            return 0.0;
        }

        let avg_len = self.avg_field_length as f32;
        let norm_factor = 1.0 - self.b + self.b * (field_length / avg_len);

        // TF = (tf * (k1 + 1)) / (tf + k1 * norm_factor)
        (term_freq * (self.k1 + 1.0)) / (term_freq + self.k1 * norm_factor)
    }

    /// Get the k1 parameter.
    pub fn k1(&self) -> f32 {
        self.k1
    }

    /// Get the b parameter.
    pub fn b(&self) -> f32 {
        self.b
    }

    /// Set the k1 parameter.
    pub fn set_k1(&mut self, k1: f32) {
        self.k1 = k1;
    }

    /// Set the b parameter.
    pub fn set_b(&mut self, b: f32) {
        self.b = b;
    }
}

impl Scorer for BM25Scorer {
    fn score(&self, _doc_id: u64, term_freq: f32) -> f32 {
        if self.doc_freq == 0 || self.total_docs == 0 {
            return 0.0;
        }

        let idf = self.idf();

        // For now, assume field length is average field length
        // TODO: Get actual field length from the index
        let field_length = self.avg_field_length as f32;
        let tf = self.tf(term_freq, field_length);

        self.boost * idf * tf
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        if self.doc_freq == 0 || self.total_docs == 0 {
            return 0.0;
        }

        let idf = self.idf();
        let max_tf = self.k1 + 1.0; // Maximum possible TF component

        self.boost * idf * max_tf
    }

    fn name(&self) -> &'static str {
        "BM25"
    }
}

impl BM25Scorer {
    /// Batch score calculation for multiple documents using SIMD optimization.
    ///
    /// This method processes multiple documents simultaneously for better performance.
    pub fn batch_score(&self, term_freqs: &[f32], field_lengths: &[f32]) -> Vec<f32> {
        assert_eq!(term_freqs.len(), field_lengths.len());

        if term_freqs.len() >= 4 {
            self.batch_score_optimized(term_freqs, field_lengths)
        } else {
            // Fallback for small batches
            term_freqs
                .iter()
                .zip(field_lengths.iter())
                .map(|(&tf, &_fl)| self.score(0, tf)) // doc_id not used in current implementation
                .collect()
        }
    }

    /// Optimized batch scoring using SIMD operations.
    fn batch_score_optimized(&self, term_freqs: &[f32], field_lengths: &[f32]) -> Vec<f32> {
        let avg_len = self.avg_field_length as f32;

        // Calculate normalization factors
        let norm_factors: Vec<f32> = field_lengths
            .iter()
            .map(|&field_len| 1.0 - self.b + self.b * (field_len / avg_len))
            .collect();

        // Calculate TF scores using SIMD
        let tf_scores = simd::numeric::batch_bm25_tf(term_freqs, self.k1, &norm_factors);

        // Calculate IDF (same for all documents in this term)
        let idf = self.idf();
        let idf_scores = vec![idf; tf_scores.len()];

        // Apply boost
        let boosts = vec![self.boost; tf_scores.len()];

        // Final score calculation using SIMD
        simd::numeric::batch_bm25_final_score(&tf_scores, &idf_scores, &boosts)
    }

    /// Calculate scores for multiple terms and documents.
    ///
    /// This is useful for complex queries with multiple terms.
    pub fn batch_multi_term_score(
        &self,
        term_data: &[(Vec<f32>, Vec<f32>)], // (term_freqs, field_lengths) for each term
    ) -> Vec<f32> {
        let mut final_scores = Vec::new();

        for (term_freqs, field_lengths) in term_data {
            let term_scores = self.batch_score(term_freqs, field_lengths);

            if final_scores.is_empty() {
                final_scores = term_scores;
            } else {
                // Add scores from multiple terms using optimized sum
                for (i, score) in term_scores.into_iter().enumerate() {
                    if i < final_scores.len() {
                        final_scores[i] += score;
                    } else {
                        final_scores.push(score);
                    }
                }
            }
        }

        final_scores
    }
}

/// A constant scorer that always returns the same score.
#[derive(Debug, Clone)]
pub struct ConstantScorer {
    /// The constant score value.
    score: f32,
    /// The boost factor.
    boost: f32,
}

impl ConstantScorer {
    /// Create a new constant scorer.
    pub fn new(score: f32) -> Self {
        ConstantScorer { score, boost: 1.0 }
    }

    /// Create a new constant scorer with boost.
    pub fn with_boost(score: f32, boost: f32) -> Self {
        ConstantScorer { score, boost }
    }

    /// Get the constant score value.
    pub fn score_value(&self) -> f32 {
        self.score
    }

    /// Set the constant score value.
    pub fn set_score_value(&mut self, score: f32) {
        self.score = score;
    }
}

impl Scorer for ConstantScorer {
    fn score(&self, _doc_id: u64, _term_freq: f32) -> f32 {
        self.score * self.boost
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        self.score * self.boost
    }

    fn name(&self) -> &'static str {
        "Constant"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_scorer_creation() {
        let scorer = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);

        assert_eq!(scorer.boost(), 1.0);
        assert_eq!(scorer.k1(), 1.2);
        assert_eq!(scorer.b(), 0.75);
        assert_eq!(scorer.name(), "BM25");
    }

    #[test]
    fn test_bm25_scorer_with_params() {
        let scorer = BM25Scorer::with_params(10, 100, 50, 10.0, 1000, 2.0, 1.5, 0.8);

        assert_eq!(scorer.boost(), 2.0);
        assert_eq!(scorer.k1(), 1.5);
        assert_eq!(scorer.b(), 0.8);
    }

    #[test]
    fn test_bm25_scorer_idf() {
        let scorer = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);
        let idf = scorer.idf();

        // IDF should be positive for normal cases
        assert!(idf > 0.0);

        // Test edge case: no documents
        let scorer_zero = BM25Scorer::new(0, 0, 0, 0.0, 0, 1.0);
        assert_eq!(scorer_zero.idf(), 0.0);
    }

    #[test]
    fn test_bm25_scorer_tf() {
        let scorer = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);

        let tf1 = scorer.tf(1.0, 10.0);
        let tf2 = scorer.tf(2.0, 10.0);

        // Higher term frequency should give higher TF score
        assert!(tf2 > tf1);

        // Zero term frequency should give zero TF
        assert_eq!(scorer.tf(0.0, 10.0), 0.0);
    }

    #[test]
    fn test_bm25_scorer_score() {
        let scorer = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);

        let score1 = scorer.score(0, 1.0);
        let score2 = scorer.score(0, 2.0);

        // Higher term frequency should give higher score
        assert!(score2 > score1);

        // Zero term frequency should give zero score
        assert_eq!(scorer.score(0, 0.0), 0.0);
    }

    #[test]
    fn test_bm25_scorer_boost() {
        let mut scorer = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);

        let original_score = scorer.score(0, 1.0);

        scorer.set_boost(2.0);
        let boosted_score = scorer.score(0, 1.0);

        assert_eq!(scorer.boost(), 2.0);
        assert_eq!(boosted_score, original_score * 2.0);
    }

    #[test]
    fn test_bm25_scorer_max_score() {
        let scorer = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);

        let max_score = scorer.max_score();
        let actual_score = scorer.score(0, 1.0);

        // Max score should be >= actual score
        assert!(max_score >= actual_score);
    }

    #[test]
    fn test_constant_scorer() {
        let scorer = ConstantScorer::new(5.0);

        assert_eq!(scorer.score_value(), 5.0);
        assert_eq!(scorer.boost(), 1.0);
        assert_eq!(scorer.name(), "Constant");

        // Should return the same score for any input
        assert_eq!(scorer.score(0, 1.0), 5.0);
        assert_eq!(scorer.score(100, 10.0), 5.0);
        assert_eq!(scorer.score(0, 0.0), 5.0);
    }

    #[test]
    fn test_constant_scorer_with_boost() {
        let scorer = ConstantScorer::with_boost(5.0, 2.0);

        assert_eq!(scorer.score_value(), 5.0);
        assert_eq!(scorer.boost(), 2.0);

        // Should return score * boost
        assert_eq!(scorer.score(0, 1.0), 10.0);
        assert_eq!(scorer.max_score(), 10.0);
    }

    #[test]
    fn test_constant_scorer_mutation() {
        let mut scorer = ConstantScorer::new(5.0);

        scorer.set_score_value(3.0);
        assert_eq!(scorer.score_value(), 3.0);
        assert_eq!(scorer.score(0, 1.0), 3.0);

        scorer.set_boost(2.0);
        assert_eq!(scorer.boost(), 2.0);
        assert_eq!(scorer.score(0, 1.0), 6.0);
    }

    #[test]
    fn test_bm25_batch_score() {
        let scorer = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);

        let term_freqs = vec![1.0, 2.0, 3.0, 4.0];
        let field_lengths = vec![10.0, 15.0, 8.0, 12.0];

        let batch_scores = scorer.batch_score(&term_freqs, &field_lengths);

        // Verify that batch scores are reasonable
        for &score in &batch_scores {
            assert!(score > 0.0);
        }

        assert_eq!(batch_scores.len(), term_freqs.len());
    }

    #[test]
    fn test_bm25_batch_small() {
        let scorer = BM25Scorer::new(5, 50, 25, 10.0, 500, 1.5);

        // Test with small batch (should use fallback)
        let term_freqs = vec![1.5, 2.5];
        let field_lengths = vec![8.0, 12.0];

        let batch_scores = scorer.batch_score(&term_freqs, &field_lengths);

        assert_eq!(batch_scores.len(), 2);
        assert!(batch_scores[0] > 0.0);
        assert!(batch_scores[1] > 0.0);
    }

    #[test]
    fn test_bm25_multi_term_score() {
        let scorer = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);

        let term_data = vec![
            (vec![1.0, 2.0], vec![10.0, 15.0]),
            (vec![2.0, 1.0], vec![10.0, 15.0]),
        ];

        let multi_scores = scorer.batch_multi_term_score(&term_data);

        assert_eq!(multi_scores.len(), 2);
        assert!(multi_scores[0] > 0.0);
        assert!(multi_scores[1] > 0.0);
    }
}
