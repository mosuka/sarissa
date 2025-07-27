//! Configuration for hybrid search.

use crate::vector::types::VectorSearchConfig;
use crate::embeding::EmbeddingConfig;
use serde::{Deserialize, Serialize};

/// Configuration for hybrid search combining keyword and vector search.
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
    pub vector_config: VectorSearchConfig,
    /// Embedding configuration for text processing.
    pub embedding_config: EmbeddingConfig,
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
            vector_config: VectorSearchConfig::default(),
            embedding_config: EmbeddingConfig::default(),
        }
    }
}

/// Score normalization strategies for combining keyword and vector scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreNormalization {
    /// No normalization - use raw scores.
    None,
    /// Min-max normalization to [0, 1] range.
    MinMax,
    /// Z-score normalization.
    ZScore,
    /// Rank-based normalization.
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
