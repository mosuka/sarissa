//! Result processing and ranking.
//!
//! This module provides utilities for post-processing hybrid search results,
//! including re-ranking, filtering, and result formatting.

use crate::error::Result;
use crate::hybrid::search::searcher::HybridSearchResults;

/// Processes and enhances hybrid search results.
///
/// This processor applies various transformations to search results,
/// such as re-ranking, score normalization, and result filtering.
pub struct ResultProcessor {
    /// Minimum score threshold for results
    min_score: Option<f32>,
    /// Maximum number of results to return
    max_results: Option<usize>,
}

impl ResultProcessor {
    /// Create a new result processor with default settings.
    pub fn new() -> Self {
        Self {
            min_score: None,
            max_results: None,
        }
    }

    /// Set the minimum score threshold.
    ///
    /// Results with scores below this threshold will be filtered out.
    ///
    /// # Arguments
    ///
    /// * `min_score` - Minimum hybrid score threshold
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Set the maximum number of results.
    ///
    /// # Arguments
    ///
    /// * `max_results` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = Some(max_results);
        self
    }

    /// Process search results.
    ///
    /// # Arguments
    ///
    /// * `results` - The search results to process
    ///
    /// # Returns
    ///
    /// Processed and filtered search results
    pub fn process(&self, mut results: HybridSearchResults) -> Result<HybridSearchResults> {
        // Filter by minimum score
        if let Some(min_score) = self.min_score {
            results.results.retain(|r| r.hybrid_score >= min_score);
        }

        // Limit number of results
        if let Some(max_results) = self.max_results {
            results.results.truncate(max_results);
        }

        Ok(results)
    }

    /// Normalize scores to 0-1 range.
    ///
    /// Applies min-max normalization to bring all hybrid scores to [0, 1] scale.
    ///
    /// # Arguments
    ///
    /// * `results` - Mutable reference to search results
    pub fn normalize_scores(&self, results: &mut HybridSearchResults) {
        if results.results.is_empty() {
            return;
        }

        let max_score = results
            .results
            .iter()
            .map(|r| r.hybrid_score)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        if max_score > 0.0 {
            for result in &mut results.results {
                result.hybrid_score /= max_score;
            }
        }
    }
}

impl Default for ResultProcessor {
    fn default() -> Self {
        Self::new()
    }
}
