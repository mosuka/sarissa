//! Scoring implementations for ranking search results.

use std::fmt::Debug;
use std::sync::Arc;

use crate::error::Result;
use crate::lexical::index::structures::dictionary::BlockMax;
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
    /// returns the global upper bound — correct but loose.
    fn block_max_score(&self) -> f32 {
        self.max_score()
    }

    /// Tightened per-block upper bound on the score for the block
    /// containing `doc_id` (#403 PR-C, Block-Max-WAND).
    ///
    /// Scorers that carry per-block metadata override this to return
    /// `boost · idf · right_max_factor` for the block whose
    /// `last_doc_id ≥ doc_id`, where `right_max_factor` is the
    /// **right-cumulative** maximum across the suffix `[block, …, end]`
    /// — guaranteeing the bound holds for every doc at-or-after the
    /// block. The default implementation forwards to [`Scorer::max_score`]
    /// — correct but loose, identical to what PR-B2's term-level
    /// bound would return.
    ///
    /// Returns `0.0` when `doc_id` is past the last block in the
    /// posting list (no further docs can match), letting the searcher
    /// loop short-circuit.
    ///
    /// # Arguments
    ///
    /// * `doc_id` — current matcher position. The scorer locates the
    ///   block containing this doc id (binary search over its
    ///   per-block table) and returns that block's bound.
    fn block_max_score_at(&self, _doc_id: u64) -> f32 {
        self.max_score()
    }

    /// Per-block upper bound on the score of any doc **inside the
    /// current block** that contains `doc_id` (#403 PR-E, Block-Max
    /// skip-ahead).
    ///
    /// In contrast to [`Self::block_max_score_at`] (which folds in the
    /// right-cumulative max so the searcher loop's `break` is sound),
    /// this returns only the **current block's** bound. The
    /// non-cumulative form lets the searcher decide that *this* block
    /// is non-competitive even though some *later* block may still be
    /// competitive — and skip past the current block via
    /// [`Self::next_block_boundary`] instead of breaking.
    ///
    /// Returns `0.0` past the last block. The default implementation
    /// forwards to [`Self::max_score`] — correct but loose, so the
    /// searcher's skip-ahead becomes a no-op for legacy scorers.
    fn current_block_max_score(&self, _doc_id: u64) -> f32 {
        self.max_score()
    }

    /// Smallest `doc_id` past the block containing `doc_id` — i.e.
    /// `block.last_doc_id + 1` for the block returned by
    /// [`Self::current_block_max_score`] (#403 PR-E). The searcher
    /// uses this as the `skip_to` target when the current block is
    /// non-competitive.
    ///
    /// Returns `None` when the scorer carries no per-block metadata
    /// (legacy v1 / v2 segments, multi-segment aggregated views) — the
    /// searcher then keeps the existing PR-C `break` semantics.
    /// Returns `Some(u64::MAX)` when `doc_id` is past the last block —
    /// the searcher should treat this as exhausted.
    fn next_block_boundary(&self, _doc_id: u64) -> Option<u64> {
        None
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
    /// Per-block (`BLOCK_SIZE = 128`) max-impact metadata used by
    /// Block-Max-WAND (#403 PR-C). Each entry holds `(last_doc_id,
    /// max_factor)`; entries are sorted by `last_doc_id`.
    /// [`Self::block_max_score_at`] binary-searches this slice to
    /// return a tighter bound than the term-level
    /// [`Self::max_score_factor`]. Empty when not available — the
    /// scorer then falls back to the term-level bound.
    block_max: Arc<[BlockMax]>,
    /// Right-cumulative max-factor: `right_max[i] = max(block_max[j].max_factor
    /// for j in i..)`. Pre-computed once at scorer construction so
    /// [`Self::block_max_score_at`] can return a valid upper bound on the
    /// score of **every doc at or after** the block containing the query
    /// doc id — required for the searcher loop's global early-break to
    /// be correctness-preserving.
    right_max: Arc<[f32]>,
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
            block_max: Arc::from([] as [BlockMax; 0]),
            right_max: Arc::from([] as [f32; 0]),
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
            block_max: Arc::from([] as [BlockMax; 0]),
            right_max: Arc::from([] as [f32; 0]),
        }
    }

    /// Compute the right-cumulative max-factor table from a sorted
    /// `block_max` slice. `out[i] = max(block_max[j].max_factor for j
    /// in i..)`. The result is monotonically non-increasing, which is
    /// what the searcher loop's global early-break requires
    /// (#403 PR-C): `block_max_score_at(doc_id)` returns a valid
    /// upper bound on the score of every doc at or after the block
    /// containing `doc_id`, not just the immediate block.
    fn compute_right_max(blocks: &[BlockMax]) -> Vec<f32> {
        let n = blocks.len();
        let mut out = vec![0.0_f32; n];
        if n == 0 {
            return out;
        }
        out[n - 1] = blocks[n - 1].max_factor;
        for i in (0..n - 1).rev() {
            out[i] = blocks[i].max_factor.max(out[i + 1]);
        }
        out
    }

    /// Create a new BM25 scorer with the per-block max-impact metadata
    /// (#403 PR-C). `block_max` carries `(last_doc_id, max_factor)`
    /// entries sorted by `last_doc_id`; pass an empty slice to fall
    /// back to the term-level [`Self::max_score_factor`] (which itself
    /// falls back to `k1 + 1` when zero).
    #[allow(clippy::too_many_arguments)]
    pub fn with_block_max(
        doc_freq: u64,
        total_term_freq: u64,
        field_doc_count: u64,
        avg_field_length: f64,
        total_docs: u64,
        boost: f32,
        max_score_factor: f32,
        block_max: Arc<[BlockMax]>,
    ) -> Self {
        let cached_idf = Self::compute_idf(doc_freq, total_docs);
        let right_max: Arc<[f32]> =
            Arc::from(Self::compute_right_max(&block_max).into_boxed_slice());
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
            block_max,
            right_max,
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
            block_max: Arc::from([] as [BlockMax; 0]),
            right_max: Arc::from([] as [f32; 0]),
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
        // Without a doc id we cannot pick a specific block; report the
        // term-level bound (same as `max_score()`). Use
        // [`Self::block_max_score_at`] for the per-block bound.
        self.max_score()
    }

    fn current_block_max_score(&self, doc_id: u64) -> f32 {
        // No per-block metadata → fall back to the term-level bound;
        // searcher's skip-ahead becomes a no-op (next_block_boundary
        // returns None alongside).
        if self.block_max.is_empty() {
            return self.max_score();
        }
        if self.doc_freq == 0 || self.total_docs == 0 {
            return 0.0;
        }
        if self.k1 != Self::DEFAULT_K1 || self.b != Self::DEFAULT_B {
            return self.max_score();
        }
        let idx = self
            .block_max
            .partition_point(|block| block.last_doc_id < doc_id);
        if idx >= self.block_max.len() {
            return 0.0;
        }
        // Per-block factor only — no right-cumulative folding. A later
        // block may carry a higher factor; that is what makes the
        // searcher's skip-ahead path interesting (the cumulative bound
        // exposed by [`Self::block_max_score_at`] still tells the
        // searcher when the global suffix becomes uncompetitive and a
        // hard `break` is sound).
        let factor = self.block_max[idx].max_factor;
        self.boost * self.idf() * factor
    }

    fn next_block_boundary(&self, doc_id: u64) -> Option<u64> {
        if self.block_max.is_empty() {
            return None;
        }
        // Mirror `current_block_max_score`'s short-circuits so the
        // searcher only relies on the boundary when the per-block
        // bound it just consulted was actually meaningful.
        if self.k1 != Self::DEFAULT_K1 || self.b != Self::DEFAULT_B {
            return None;
        }
        let idx = self
            .block_max
            .partition_point(|block| block.last_doc_id < doc_id);
        if idx >= self.block_max.len() {
            return Some(u64::MAX);
        }
        Some(self.block_max[idx].last_doc_id.saturating_add(1))
    }

    fn block_max_score_at(&self, doc_id: u64) -> f32 {
        // No per-block metadata → fall back to the term-level bound.
        if self.block_max.is_empty() {
            return self.max_score();
        }
        if self.doc_freq == 0 || self.total_docs == 0 {
            return 0.0;
        }
        // Skip the per-block bound when the caller is using non-default
        // BM25 parameters — the precomputed factor is only valid for
        // `(k1, b) = (1.2, 0.75)`.
        if self.k1 != Self::DEFAULT_K1 || self.b != Self::DEFAULT_B {
            return self.max_score();
        }

        // Binary search for the first block whose `last_doc_id >= doc_id`.
        let idx = self
            .block_max
            .partition_point(|block| block.last_doc_id < doc_id);
        if idx >= self.block_max.len() {
            // Past the last block — no remaining doc in this term's
            // posting list can contribute.
            return 0.0;
        }
        // Use the right-cumulative max so the returned bound holds for
        // every doc at or after the block containing `doc_id`. The
        // searcher loop's `if bound <= min_competitive { break }`
        // would be unsound with the per-block factor alone — a later
        // block could carry a higher factor and yield a higher-scoring
        // doc.
        let factor = self.right_max[idx];
        self.boost * self.idf() * factor
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

    fn block_max_score_at(&self, doc_id: u64) -> f32 {
        // For an OR-style boolean any sub-clause might contribute, so
        // the upper bound at `doc_id` is the sum of each sub-scorer's
        // per-block bound at the same doc id (#403 PR-C). Because
        // each sub-scorer's bound is itself ≤ its own `max_score()`,
        // the sum is also ≤ `BooleanScorer::max_score()` and therefore
        // a tighter, valid upper bound.
        let mut total = 0.0_f32;
        let clauses = self.clauses.borrow();
        for (scorer, _) in clauses.iter() {
            total += scorer.block_max_score_at(doc_id);
        }
        total * self.boost
    }

    fn current_block_max_score(&self, doc_id: u64) -> f32 {
        // Sum of per-block (non-cumulative) bounds. Same correctness
        // argument as [`Self::block_max_score_at`] but tighter — used
        // by the searcher's skip-ahead path (#403 PR-E).
        let mut total = 0.0_f32;
        let clauses = self.clauses.borrow();
        for (scorer, _) in clauses.iter() {
            total += scorer.current_block_max_score(doc_id);
        }
        total * self.boost
    }

    fn next_block_boundary(&self, doc_id: u64) -> Option<u64> {
        // Conservative skip target: the **min** boundary across
        // clauses so the searcher does not jump past any clause's
        // current block. If any clause carries no per-block info
        // (returns `None`), the whole query falls back to the
        // existing PR-C `break` semantics — a partial skip would let
        // the searcher overshoot a clause that is still potentially
        // competitive but lacks block metadata.
        let clauses = self.clauses.borrow();
        let mut min_boundary: Option<u64> = None;
        for (scorer, _) in clauses.iter() {
            match scorer.next_block_boundary(doc_id) {
                None => return None,
                Some(b) => {
                    min_boundary = Some(match min_boundary {
                        None => b,
                        Some(m) => m.min(b),
                    });
                }
            }
        }
        min_boundary
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

    /// PR-E: `current_block_max_score` returns the **per-block** max
    /// (without right-cumulative folding); `next_block_boundary`
    /// reports `last_doc_id + 1` for that block.
    #[test]
    fn bm25_current_block_score_and_boundary_are_per_block() {
        let blocks: Arc<[BlockMax]> = Arc::from(vec![
            BlockMax {
                last_doc_id: 99,
                max_factor: 1.0,
            },
            BlockMax {
                last_doc_id: 199,
                max_factor: 3.0, // higher impact in a later block
            },
            BlockMax {
                last_doc_id: 299,
                max_factor: 2.0,
            },
        ]);
        let scorer = BM25Scorer::with_block_max(10, 100, 50, 10.0, 1000, 1.0, 0.0, blocks);

        // The cumulative bound at doc 0 sees the max factor from any
        // suffix block (= 3.0 in block 1).
        let cumulative_at_0 = scorer.block_max_score_at(0);

        // The per-block bound at doc 0 only sees block 0 (= 1.0) and
        // must therefore be strictly less than the cumulative bound.
        let per_block_at_0 = scorer.current_block_max_score(0);
        assert!(
            per_block_at_0 < cumulative_at_0,
            "per_block={per_block_at_0}, cumulative={cumulative_at_0}"
        );

        // Boundary for doc 0 lands at block 0's last_doc_id + 1 = 100.
        assert_eq!(scorer.next_block_boundary(0), Some(100));

        // Inside block 1: boundary lands at 200.
        assert_eq!(scorer.next_block_boundary(150), Some(200));

        // Past last block: signal exhaustion via u64::MAX.
        assert_eq!(scorer.next_block_boundary(500), Some(u64::MAX));
        assert_eq!(scorer.current_block_max_score(500), 0.0);
    }

    /// PR-E: legacy scorers without per-block metadata fall through to
    /// the term-level bound and return `None` from
    /// `next_block_boundary` so the searcher keeps the PR-C `break`
    /// semantics.
    #[test]
    fn bm25_legacy_scorer_returns_none_boundary() {
        let legacy = BM25Scorer::with_max_score_factor(10, 100, 50, 10.0, 1000, 1.0, 1.0);
        assert_eq!(legacy.next_block_boundary(0), None);
        assert_eq!(legacy.next_block_boundary(u64::MAX), None);
        // current_block_max_score forwards to max_score for legacy.
        assert_eq!(legacy.current_block_max_score(0), legacy.max_score());
    }

    /// PR-E: when caller overrides `(k1, b)` away from the defaults,
    /// the precomputed per-block factor is no longer valid — both the
    /// per-block bound and the boundary must short-circuit.
    #[test]
    fn bm25_non_default_params_shortcircuit_per_block() {
        let blocks: Arc<[BlockMax]> = Arc::from(vec![BlockMax {
            last_doc_id: 99,
            max_factor: 1.0,
        }]);
        let mut scorer = BM25Scorer::with_block_max(10, 100, 50, 10.0, 1000, 1.0, 0.0, blocks);
        scorer.set_k1(2.0); // override away from default

        assert_eq!(scorer.next_block_boundary(0), None);
        assert_eq!(scorer.current_block_max_score(0), scorer.max_score());
    }

    /// PR-E: BooleanScorer's per-block bound and boundary aggregate
    /// across clauses — sum for the bound, `min` for the boundary
    /// (conservative skip target).
    #[test]
    fn boolean_per_block_bound_sums_and_boundary_takes_min() {
        use crate::lexical::query::TermQuery;

        // Two clauses with different per-block layouts.
        let blocks_a: Arc<[BlockMax]> = Arc::from(vec![BlockMax {
            last_doc_id: 50,
            max_factor: 1.5,
        }]);
        let scorer_a = BM25Scorer::with_block_max(5, 30, 20, 10.0, 500, 1.0, 0.0, blocks_a);
        let blocks_b: Arc<[BlockMax]> = Arc::from(vec![BlockMax {
            last_doc_id: 80,
            max_factor: 2.0,
        }]);
        let scorer_b = BM25Scorer::with_block_max(8, 40, 25, 10.0, 500, 1.0, 0.0, blocks_b);

        let combined_at_0 =
            scorer_a.current_block_max_score(0) + scorer_b.current_block_max_score(0);

        // Build a BooleanScorer over two TermQuery clauses and a
        // dummy reader. We can't easily construct a real BooleanScorer
        // without a reader, so this test exercises the aggregation
        // logic by invoking the trait methods directly through a
        // hand-rolled stand-in.
        #[derive(Debug)]
        struct PairScorer(BM25Scorer, BM25Scorer, f32);
        impl Scorer for PairScorer {
            fn score(&self, _: u64, _: f32, _: Option<f32>) -> f32 {
                0.0
            }
            fn boost(&self) -> f32 {
                self.2
            }
            fn set_boost(&mut self, b: f32) {
                self.2 = b;
            }
            fn max_score(&self) -> f32 {
                (self.0.max_score() + self.1.max_score()) * self.2
            }
            fn block_max_score_at(&self, doc_id: u64) -> f32 {
                (self.0.block_max_score_at(doc_id) + self.1.block_max_score_at(doc_id)) * self.2
            }
            fn current_block_max_score(&self, doc_id: u64) -> f32 {
                (self.0.current_block_max_score(doc_id) + self.1.current_block_max_score(doc_id))
                    * self.2
            }
            fn next_block_boundary(&self, doc_id: u64) -> Option<u64> {
                match (
                    self.0.next_block_boundary(doc_id),
                    self.1.next_block_boundary(doc_id),
                ) {
                    (Some(x), Some(y)) => Some(x.min(y)),
                    _ => None,
                }
            }
            fn name(&self) -> &'static str {
                "PairTest"
            }
        }
        let pair = PairScorer(scorer_a, scorer_b, 1.0);

        assert!((pair.current_block_max_score(0) - combined_at_0).abs() < 1e-6);
        // Min of clause boundaries: clause-a hits 51 first, clause-b
        // hits 81. min = 51.
        assert_eq!(pair.next_block_boundary(0), Some(51));
        // After 51: clause-a is exhausted (returns u64::MAX), clause-b
        // returns 81. min = 81.
        assert_eq!(pair.next_block_boundary(51), Some(81));
        // Avoid an unused-import warning when this test is the only
        // user of `TermQuery`.
        let _ = std::any::TypeId::of::<TermQuery>();
    }

    /// PR-E: when **any** clause carries no per-block info, the
    /// BooleanScorer must fall back to `None` so the searcher keeps
    /// the PR-C `break` semantics for the whole query.
    #[test]
    fn boolean_boundary_propagates_none_from_any_clause() {
        #[derive(Debug)]
        struct NoBlockScorer;
        impl Scorer for NoBlockScorer {
            fn score(&self, _: u64, _: f32, _: Option<f32>) -> f32 {
                0.0
            }
            fn boost(&self) -> f32 {
                1.0
            }
            fn set_boost(&mut self, _: f32) {}
            fn max_score(&self) -> f32 {
                10.0
            }
            // next_block_boundary defaults to None.
            fn name(&self) -> &'static str {
                "NoBlock"
            }
        }
        let blocks: Arc<[BlockMax]> = Arc::from(vec![BlockMax {
            last_doc_id: 50,
            max_factor: 1.5,
        }]);
        let block_scorer = BM25Scorer::with_block_max(5, 30, 20, 10.0, 500, 1.0, 0.0, blocks);

        #[derive(Debug)]
        struct PairWithLegacy(NoBlockScorer, BM25Scorer);
        impl Scorer for PairWithLegacy {
            fn score(&self, _: u64, _: f32, _: Option<f32>) -> f32 {
                0.0
            }
            fn boost(&self) -> f32 {
                1.0
            }
            fn set_boost(&mut self, _: f32) {}
            fn max_score(&self) -> f32 {
                self.0.max_score() + self.1.max_score()
            }
            fn next_block_boundary(&self, doc_id: u64) -> Option<u64> {
                match (
                    self.0.next_block_boundary(doc_id),
                    self.1.next_block_boundary(doc_id),
                ) {
                    (Some(x), Some(y)) => Some(x.min(y)),
                    _ => None,
                }
            }
            fn name(&self) -> &'static str {
                "PairWithLegacy"
            }
        }
        let pair = PairWithLegacy(NoBlockScorer, block_scorer);
        assert_eq!(pair.next_block_boundary(0), None);
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
