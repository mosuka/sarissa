//! Scoring implementations for ranking search results.

use std::fmt::Debug;

use crate::error::Result;
use crate::lexical::query::Query;
use crate::lexical::query::matcher::Matcher;
use crate::util::simd;

/// Type alias for boolean scorer clauses.
type BooleanScorerClauses = std::cell::RefCell<Vec<(Box<dyn Scorer>, Box<dyn Matcher>)>>;

/// Trait for document scorers.
pub trait Scorer: Send + Debug {
    /// Calculate the score for a document.
    ///
    /// # Arguments
    /// * `doc_id` - Document ID
    /// * `term_freq` - Term frequency in the document
    /// * `field_length` - Length of the field (number of tokens). If None, uses average field length.
    fn score(&self, doc_id: u64, term_freq: f32, field_length: Option<f32>) -> f32;

    /// Get the boost factor for this scorer.
    fn boost(&self) -> f32;

    /// Set the boost factor for this scorer.
    fn set_boost(&mut self, boost: f32);

    /// Get the maximum possible score.
    fn max_score(&self) -> f32;

    /// Upper bound on the score of the **next-block** of postings.
    ///
    /// For scorers that do not yet expose per-block max-score metadata,
    /// the default impl forwards to [`Scorer::max_score`], i.e. it
    /// returns the global upper bound — correct but loose. A future
    /// per-posting block-max index format (#403 PR-B2) will let scorers
    /// override this with a tightened bound that varies as the matcher
    /// advances, enabling Block-Max-WAND skips.
    ///
    /// # Returns
    ///
    /// An upper bound on any score the scorer can produce for documents
    /// in the matcher's current block. The returned value is **at most**
    /// [`Scorer::max_score`] but may be tighter.
    fn block_max_score(&self) -> f32 {
        self.max_score()
    }

    /// Get the name of this scorer.
    fn name(&self) -> &'static str;
}

/// BM25 scorer implementation.
#[derive(Debug, Clone)]
pub struct BM25Scorer {
    /// Document frequency of the term.
    doc_freq: u64,
    /// Total term frequency across all documents (reserved for BM25F).
    #[allow(dead_code)]
    total_term_freq: u64,
    /// Number of documents containing the field (reserved for BM25F).
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
    /// Cached IDF value computed at construction time.
    cached_idf: f32,
    /// Tightened TF-component upper bound precomputed at index time
    /// using the default `(k1, b) = (1.2, 0.75)` parameters and the
    /// segment's average field length (#403 PR-B2). Read from
    /// [`crate::lexical::reader::ReaderTermInfo::max_score_factor`].
    ///
    /// `0.0` is treated as "unset" — [`Self::max_score`] then falls
    /// back to the loose synthetic `k1 + 1` ceiling. Also bypassed
    /// when the caller overrides `(k1, b)` via [`Self::with_params`],
    /// since the precomputed factor is only valid for the defaults.
    max_score_factor: f32,
}

impl BM25Scorer {
    /// Default BM25 `k1` parameter; matches what the indexer uses to
    /// precompute [`crate::lexical::index::structures::dictionary::TermInfo::max_score_factor`].
    const DEFAULT_K1: f32 = 1.2;
    /// Default BM25 `b` parameter; matches the indexer's choice (see
    /// [`Self::DEFAULT_K1`]).
    const DEFAULT_B: f32 = 0.75;

    /// Compute the IDF (Inverse Document Frequency) value.
    fn compute_idf(doc_freq: u64, total_docs: u64) -> f32 {
        if doc_freq == 0 || total_docs == 0 {
            return 0.0;
        }
        let n = total_docs as f32;
        let df = doc_freq as f32;
        let base_idf = ((n - df + 0.5) / (df + 0.5)).ln();
        let epsilon = 0.01;
        base_idf.max(epsilon)
    }

    /// Create a new BM25 scorer with no precomputed `max_score_factor`
    /// (loose `k1 + 1` upper bound).
    pub fn new(
        doc_freq: u64,
        total_term_freq: u64,
        field_doc_count: u64,
        avg_field_length: f64,
        total_docs: u64,
        boost: f32,
    ) -> Self {
        let cached_idf = Self::compute_idf(doc_freq, total_docs);
        BM25Scorer {
            doc_freq,
            total_term_freq,
            field_doc_count,
            avg_field_length,
            total_docs,
            boost,
            k1: Self::DEFAULT_K1,
            b: Self::DEFAULT_B,
            cached_idf,
            max_score_factor: 0.0,
        }
    }

    /// Create a new BM25 scorer with the index-side tightened TF
    /// upper bound (#403 PR-B2). Pass `0.0` for `max_score_factor` if
    /// the index does not carry the precomputed value (legacy v1
    /// dictionaries / aggregated cross-segment views) — [`Self::max_score`]
    /// will fall back to the loose `k1 + 1` ceiling.
    #[allow(clippy::too_many_arguments)]
    pub fn with_max_score_factor(
        doc_freq: u64,
        total_term_freq: u64,
        field_doc_count: u64,
        avg_field_length: f64,
        total_docs: u64,
        boost: f32,
        max_score_factor: f32,
    ) -> Self {
        let cached_idf = Self::compute_idf(doc_freq, total_docs);
        BM25Scorer {
            doc_freq,
            total_term_freq,
            field_doc_count,
            avg_field_length,
            total_docs,
            boost,
            k1: Self::DEFAULT_K1,
            b: Self::DEFAULT_B,
            cached_idf,
            max_score_factor,
        }
    }

    /// Create a new BM25 scorer with custom `(k1, b)` parameters. The
    /// precomputed `max_score_factor` is **not** wired in here because
    /// the index-side factor is only valid for the default
    /// `(k1, b) = (1.2, 0.75)` parameters.
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
        let cached_idf = Self::compute_idf(doc_freq, total_docs);
        BM25Scorer {
            doc_freq,
            total_term_freq,
            field_doc_count,
            avg_field_length,
            total_docs,
            boost,
            k1,
            b,
            cached_idf,
            max_score_factor: 0.0,
        }
    }

    /// Return the cached IDF (Inverse Document Frequency) value.
    #[inline(always)]
    fn idf(&self) -> f32 {
        self.cached_idf
    }

    /// Calculate the TF (Term Frequency) component.
    fn tf(&self, term_freq: f32, field_length: f32) -> f32 {
        if term_freq == 0.0 {
            return 0.0;
        }

        let avg_len = self.avg_field_length as f32;

        // Handle zero average length case
        let norm_factor = if avg_len == 0.0 {
            // When avg is unknown but we have individual field lengths,
            // disable length normalization (set factor to 1.0)
            1.0
        } else {
            1.0 - self.b + self.b * (field_length / avg_len)
        };

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
    fn score(&self, _doc_id: u64, term_freq: f32, field_length: Option<f32>) -> f32 {
        if self.doc_freq == 0 || self.total_docs == 0 || term_freq == 0.0 {
            return 0.0;
        }

        // Standard BM25 formula: score = boost × IDF × TF
        let idf = self.idf();

        // Use provided field length, or fall back to average
        let field_len = field_length.unwrap_or(self.avg_field_length as f32);
        let tf = self.tf(term_freq, field_len);

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
        // Use the index-side tightened bound (#403 PR-B2) when it is
        // available **and** the scorer is running with the default
        // BM25 parameters that the index-time precomputation assumed.
        // Custom `(k1, b)` callers fall back to the synthetic
        // `k1 + 1` upper bound on the TF component.
        let tf_upper_bound = if self.max_score_factor > 0.0
            && self.k1 == Self::DEFAULT_K1
            && self.b == Self::DEFAULT_B
        {
            self.max_score_factor
        } else {
            self.k1 + 1.0
        };

        self.boost * idf * tf_upper_bound
    }

    fn block_max_score(&self) -> f32 {
        // Today we do not yet store per-block bounds — return the
        // tightest term-level upper bound, which is the same value
        // [`Self::max_score`] returns. A future per-posting block-max
        // index format (#403 PR-C) will let this method override with
        // a bound that varies as the matcher advances.
        self.max_score()
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
            // Fallback for small batches - use actual field lengths
            term_freqs
                .iter()
                .enumerate()
                .map(|(i, &tf)| {
                    let idf = self.idf();
                    let tf_score = self.tf(tf, field_lengths[i]);
                    self.boost * idf * tf_score
                })
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
    fn score(&self, _doc_id: u64, _term_freq: f32, _field_length: Option<f32>) -> f32 {
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

/// A scorer that combines multiple scorers by summing their scores.
#[derive(Debug)]
pub struct BooleanScorer {
    /// The sub-queries and their scorers/matchers.
    /// We use a Mutex for matchers since they are mutable.
    clauses: BooleanScorerClauses,
    /// The boost factor for this scorer.
    boost: f32,
}

// SAFETY: BooleanScorer is only used within single-threaded search execution paths.
// The RefCell is never shared across threads.
unsafe impl Send for BooleanScorer {}

impl BooleanScorer {
    /// Create a new boolean scorer.
    pub fn new(
        reader: &dyn crate::lexical::reader::LexicalIndexReader,
        queries: Vec<Box<dyn Query>>,
    ) -> Result<Self> {
        let mut clauses = Vec::new();
        for query in queries {
            let matcher = query.matcher(reader)?;
            let scorer = query.scorer(reader)?;
            clauses.push((scorer, matcher));
        }
        Ok(BooleanScorer {
            clauses: std::cell::RefCell::new(clauses),
            boost: 1.0,
        })
    }
}

impl Scorer for BooleanScorer {
    fn score(&self, doc_id: u64, _term_freq: f32, field_length: Option<f32>) -> f32 {
        let mut total_score = 0.0;
        let mut clauses = self.clauses.borrow_mut();

        for (scorer, matcher) in clauses.iter_mut() {
            // Skip to the target document
            match matcher.skip_to(doc_id) {
                Ok(true) if matcher.doc_id() == doc_id => {
                    // This clause matches the document
                    let tf = matcher.term_freq() as f32;
                    total_score += scorer.score(doc_id, tf, field_length);
                }
                _ => {
                    // This clause doesn't match, contributes zero
                }
            }
        }
        total_score * self.boost
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        let mut total_max = 0.0;
        let clauses = self.clauses.borrow();
        for (scorer, _) in clauses.iter() {
            total_max += scorer.max_score();
        }
        total_max * self.boost
    }

    fn name(&self) -> &'static str {
        "Boolean"
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

        let score1 = scorer.score(0, 1.0, None);
        let score2 = scorer.score(0, 2.0, None);

        // Higher term frequency should give higher score
        assert!(score2 > score1);

        // Zero term frequency should give zero score
        assert_eq!(scorer.score(0, 0.0, None), 0.0);
    }

    #[test]
    fn test_bm25_scorer_boost() {
        let mut scorer = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);

        let original_score = scorer.score(0, 1.0, None);

        scorer.set_boost(2.0);
        let boosted_score = scorer.score(0, 1.0, None);

        assert_eq!(scorer.boost(), 2.0);
        assert_eq!(boosted_score, original_score * 2.0);
    }

    #[test]
    fn test_bm25_scorer_max_score() {
        let scorer = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);

        let max_score = scorer.max_score();
        let actual_score = scorer.score(0, 1.0, None);

        // Max score should be >= actual score
        assert!(max_score >= actual_score);
    }

    #[test]
    fn test_bm25_scorer_max_score_factor_tightens_bound() {
        // #403 PR-B2: when the index supplies a precomputed
        // `max_score_factor`, the scorer's `max_score()` upper bound
        // is `boost * idf * factor` (tight) rather than the synthetic
        // `boost * idf * (k1 + 1)` ceiling. With `factor < k1 + 1`
        // (the realistic case) the new bound must be strictly
        // smaller.
        let loose = BM25Scorer::new(10, 100, 50, 10.0, 1000, 1.0);
        // Average doc length 10, so a typical TF=1 / L=10 posting has
        // factor ≈ 1.0 — well below the loose `k1 + 1 = 2.2` ceiling.
        let tight = BM25Scorer::with_max_score_factor(10, 100, 50, 10.0, 1000, 1.0, 1.0);

        assert!(
            tight.max_score() < loose.max_score(),
            "tight bound ({}) must be < loose bound ({})",
            tight.max_score(),
            loose.max_score()
        );
        assert!(tight.max_score() > 0.0, "tight bound must remain positive");
    }

    #[test]
    fn test_bm25_scorer_max_score_factor_ignored_with_custom_params() {
        // The precomputed `max_score_factor` is only valid for the
        // default `(k1, b) = (1.2, 0.75)` parameters used at index
        // time. When the caller overrides `(k1, b)` the scorer must
        // fall back to the loose `k1 + 1` synthetic bound — verified
        // here by constructing two scorers (one with custom params,
        // one default) and confirming the custom-params scorer still
        // emits the loose ceiling.
        let custom = BM25Scorer::with_params(10, 100, 50, 10.0, 1000, 1.0, 2.0, 0.5);
        let custom_loose = custom.boost() * custom.max_score() / custom.boost();
        // Loose bound is `boost * idf * (k1 + 1) = 1.0 * idf * 3.0`.
        // With factor unset (`with_params`) the scorer returns the
        // loose bound, so `max_score / (boost * idf) = k1 + 1 = 3.0`.
        let idf = ((1000.0_f32 - 10.0 + 0.5) / (10.0 + 0.5)).ln().max(0.01);
        let expected = 1.0 * idf * 3.0;
        assert!((custom_loose - expected).abs() < 1e-4);
    }

    #[test]
    fn test_bm25_scorer_block_max_score_matches_max_score() {
        // PR-B2 leaves `block_max_score()` forwarding to `max_score()`.
        // PR-C will introduce per-block bounds; the contract that the
        // block bound never exceeds the term-level bound is asserted
        // here so a future override doesn't accidentally regress.
        let scorer = BM25Scorer::with_max_score_factor(10, 100, 50, 10.0, 1000, 1.0, 1.0);
        assert_eq!(scorer.block_max_score(), scorer.max_score());
    }

    #[test]
    fn test_constant_scorer() {
        let scorer = ConstantScorer::new(5.0);

        assert_eq!(scorer.score_value(), 5.0);
        assert_eq!(scorer.boost(), 1.0);
        assert_eq!(scorer.name(), "Constant");

        // Should return the same score for any input
        assert_eq!(scorer.score(0, 1.0, None), 5.0);
        assert_eq!(scorer.score(100, 10.0, None), 5.0);
        assert_eq!(scorer.score(0, 0.0, None), 5.0);
    }

    #[test]
    fn test_constant_scorer_with_boost() {
        let scorer = ConstantScorer::with_boost(5.0, 2.0);

        assert_eq!(scorer.score_value(), 5.0);
        assert_eq!(scorer.boost(), 2.0);

        // Should return score * boost
        assert_eq!(scorer.score(0, 1.0, None), 10.0);
        assert_eq!(scorer.max_score(), 10.0);
    }

    #[test]
    fn test_constant_scorer_mutation() {
        let mut scorer = ConstantScorer::new(5.0);

        scorer.set_score_value(3.0);
        assert_eq!(scorer.score_value(), 3.0);
        assert_eq!(scorer.score(0, 1.0, None), 3.0);

        scorer.set_boost(2.0);
        assert_eq!(scorer.boost(), 2.0);
        assert_eq!(scorer.score(0, 1.0, None), 6.0);
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
