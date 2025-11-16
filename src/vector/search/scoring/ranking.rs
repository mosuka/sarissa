//! Vector search result ranking and reranking utilities.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::search::searcher::VectorSearchResults;

/// Configuration for result ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    /// Ranking method to use.
    pub method: RankingMethod,
    /// Whether to normalize scores.
    pub normalize_scores: bool,
    /// Score boost factors.
    pub boost_factors: std::collections::HashMap<String, f32>,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            method: RankingMethod::Similarity,
            normalize_scores: true,
            boost_factors: std::collections::HashMap::new(),
        }
    }
}

/// Different ranking methods available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankingMethod {
    /// Rank by similarity score.
    Similarity,
    /// Rank by distance (inverse).
    Distance,
    /// Combine multiple factors.
    Weighted,
    /// Custom ranking function.
    Custom,
}

/// Vector result ranker.
///
/// Provides functionality to rank and reorder vector search results
/// using different ranking strategies and score normalization.
pub struct VectorRanker {
    /// Ranking configuration.
    config: RankingConfig,
}

impl VectorRanker {
    /// Create a new vector ranker.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying ranking behavior
    ///
    /// # Examples
    ///
    /// ```
    /// use yatagarasu::vector::search::scoring::ranking::{VectorRanker, RankingConfig};
    ///
    /// let config = RankingConfig::default();
    /// let ranker = VectorRanker::new(config);
    /// ```
    pub fn new(config: RankingConfig) -> Self {
        Self { config }
    }

    /// Rank and reorder search results.
    ///
    /// Applies the configured ranking method and optionally normalizes scores.
    ///
    /// # Arguments
    ///
    /// * `results` - Mutable reference to search results to be ranked
    ///
    /// # Returns
    ///
    /// `Ok(())` if ranking succeeds
    pub fn rank_results(&self, results: &mut VectorSearchResults) -> Result<()> {
        match self.config.method {
            RankingMethod::Similarity => {
                results.sort_by_similarity();
            }
            RankingMethod::Distance => {
                results.sort_by_distance();
            }
            RankingMethod::Weighted => {
                self.apply_weighted_ranking(results)?;
            }
            RankingMethod::Custom => {
                self.apply_custom_ranking(results)?;
            }
        }

        if self.config.normalize_scores {
            self.normalize_scores(results)?;
        }

        Ok(())
    }

    /// Apply weighted ranking based on multiple factors.
    ///
    /// Adjusts similarity scores using boost factors from metadata.
    ///
    /// # Arguments
    ///
    /// * `results` - Mutable reference to search results
    ///
    /// # Returns
    ///
    /// `Ok(())` if weighting succeeds
    fn apply_weighted_ranking(&self, results: &mut VectorSearchResults) -> Result<()> {
        for result in &mut results.results {
            let mut weighted_score = result.similarity;

            // Apply boost factors based on metadata
            for (key, boost) in &self.config.boost_factors {
                if result.metadata.contains_key(key) {
                    weighted_score *= boost;
                }
            }

            result.similarity = weighted_score;
        }

        results.sort_by_similarity();
        Ok(())
    }

    /// Apply custom ranking logic.
    ///
    /// Placeholder for user-defined custom ranking implementations.
    ///
    /// # Arguments
    ///
    /// * `_results` - Mutable reference to search results
    ///
    /// # Returns
    ///
    /// `Ok(())` - Currently a no-op placeholder
    fn apply_custom_ranking(&self, _results: &mut VectorSearchResults) -> Result<()> {
        // Placeholder for custom ranking implementation
        Ok(())
    }

    /// Normalize scores to 0-1 range.
    ///
    /// Applies min-max normalization to bring all scores to a [0, 1] scale.
    ///
    /// # Arguments
    ///
    /// * `results` - Mutable reference to search results
    ///
    /// # Returns
    ///
    /// `Ok(())` if normalization succeeds
    fn normalize_scores(&self, results: &mut VectorSearchResults) -> Result<()> {
        if results.results.is_empty() {
            return Ok(());
        }

        let max_score = results
            .results
            .iter()
            .map(|r| r.similarity)
            .fold(f32::NEG_INFINITY, f32::max);

        let min_score = results
            .results
            .iter()
            .map(|r| r.similarity)
            .fold(f32::INFINITY, f32::min);

        let range = max_score - min_score;

        if range > 0.0 {
            for result in &mut results.results {
                result.similarity = (result.similarity - min_score) / range;
            }
        }

        Ok(())
    }

    /// Get the ranking configuration.
    ///
    /// # Returns
    ///
    /// Reference to the current ranking configuration
    pub fn config(&self) -> &RankingConfig {
        &self.config
    }
}
