//! Configuration for hybrid search.
//!
//! This module provides configuration structures for controlling how keyword
//! and vector search results are combined in hybrid search operations.
//!
//! # Examples
//!
//! ```
//! use yatagarasu::hybrid::config::{HybridSearchConfig, ScoreNormalization};
//!
//! // Use default configuration
//! let config = HybridSearchConfig::default();
//! assert_eq!(config.keyword_weight, 0.6);
//! assert_eq!(config.vector_weight, 0.4);
//!
//! // Create custom configuration
//! let mut custom_config = HybridSearchConfig::default();
//! custom_config.keyword_weight = 0.8;  // Favor keyword search
//! custom_config.vector_weight = 0.2;
//! custom_config.normalization = ScoreNormalization::ZScore;
//! custom_config.require_both = true;  // Require both matches
//! ```

use serde::{Deserialize, Serialize};

use crate::vector::search::searcher::VectorSearchParams;

/// Configuration for hybrid search combining keyword and vector search.
///
/// This structure defines how keyword (lexical) and vector (semantic) search
/// results are combined and weighted. The weights should typically sum to 1.0
/// for proper score normalization.
///
/// # Weight Guidelines
///
/// - **Keyword-focused** (0.7-0.8): Good for exact term matching
/// - **Balanced** (0.5-0.6): Mix of exact and semantic matching
/// - **Semantic-focused** (0.3-0.4): Emphasize meaning over exact terms
///
/// # Examples
///
/// ```
/// use yatagarasu::hybrid::config::{HybridSearchConfig, ScoreNormalization};
///
/// // Balanced search (default)
/// let balanced = HybridSearchConfig::default();
///
/// // Keyword-focused search
/// let mut keyword_focused = HybridSearchConfig::default();
/// keyword_focused.keyword_weight = 0.8;
/// keyword_focused.vector_weight = 0.2;
///
/// // Semantic-focused search
/// let mut semantic_focused = HybridSearchConfig::default();
/// semantic_focused.keyword_weight = 0.3;
/// semantic_focused.vector_weight = 0.7;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// Weight for keyword search results (0.0-1.0).
    pub keyword_weight: f32,
    /// Weight for vector search results (0.0-1.0).
    pub vector_weight: f32,
    /// Minimum keyword score threshold.
    pub min_keyword_score: f32,
    /// Minimum vector similarity threshold.
    pub min_vector_similarity: f32,
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Whether to require both keyword and vector matches.
    pub require_both: bool,
    /// Normalization strategy for combining scores.
    pub normalization: ScoreNormalization,
    /// Vector search configuration.
    pub vector_config: VectorSearchParams,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            keyword_weight: 0.6,
            vector_weight: 0.4,
            min_keyword_score: 0.0,
            min_vector_similarity: 0.3,
            max_results: 50,
            require_both: false,
            normalization: ScoreNormalization::MinMax,
            vector_config: VectorSearchParams::default(),
        }
    }
}

/// Score normalization strategies for combining keyword and vector scores.
///
/// Different normalization strategies are appropriate for different scenarios:
///
/// - **None**: Use raw scores directly (fastest, but may favor one type)
/// - **MinMax**: Scale scores to [0,1] range (good for balanced weighting)
/// - **ZScore**: Standardize using mean and std dev (robust to outliers)
/// - **Rank**: Use relative ranking positions (ignores score magnitudes)
///
/// # Examples
///
/// ```
/// use yatagarasu::hybrid::config::ScoreNormalization;
///
/// let norm = ScoreNormalization::MinMax;
/// assert_eq!(norm, ScoreNormalization::MinMax);
///
/// // Different strategies for different use cases
/// let no_norm = ScoreNormalization::None;      // Fast, raw scores
/// let minmax = ScoreNormalization::MinMax;     // Balanced weighting
/// let zscore = ScoreNormalization::ZScore;     // Statistical normalization
/// let rank = ScoreNormalization::Rank;         // Position-based
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreNormalization {
    /// No normalization - use raw scores directly.
    ///
    /// This is the fastest option but may lead to imbalanced results
    /// if keyword and vector scores have different scales.
    None,
    /// Min-max normalization to [0, 1] range.
    ///
    /// Scales scores linearly: `(score - min) / (max - min)`.
    /// Good for balanced weighting when combining scores.
    MinMax,
    /// Z-score normalization (standardization).
    ///
    /// Transforms scores using: `(score - mean) / std_dev`.
    /// More robust to outliers than min-max normalization.
    ZScore,
    /// Rank-based normalization.
    ///
    /// Uses relative ranking positions instead of score magnitudes.
    /// Useful when score distributions are very different.
    Rank,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_search_config_default() {
        let config = HybridSearchConfig::default();
        assert_eq!(config.keyword_weight, 0.6);
        assert_eq!(config.vector_weight, 0.4);
        assert_eq!(config.min_keyword_score, 0.0);
        assert_eq!(config.min_vector_similarity, 0.3);
        assert_eq!(config.max_results, 50);
        assert!(!config.require_both);
        assert_eq!(config.normalization, ScoreNormalization::MinMax);
    }

    #[test]
    fn test_score_normalization_values() {
        assert_eq!(ScoreNormalization::None, ScoreNormalization::None);
        assert_eq!(ScoreNormalization::MinMax, ScoreNormalization::MinMax);
        assert_eq!(ScoreNormalization::ZScore, ScoreNormalization::ZScore);
        assert_eq!(ScoreNormalization::Rank, ScoreNormalization::Rank);
    }

    #[test]
    fn test_config_clone() {
        let config = HybridSearchConfig::default();
        let cloned = config.clone();
        assert_eq!(config.keyword_weight, cloned.keyword_weight);
        assert_eq!(config.vector_weight, cloned.vector_weight);
        assert_eq!(config.normalization, cloned.normalization);
    }
}
