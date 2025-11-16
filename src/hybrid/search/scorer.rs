//! Score normalization for hybrid search.
//!
//! This module provides score normalization functionality for combining
//! keyword and vector search scores with different scales.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::error::Result;
use crate::hybrid::search::searcher::HybridSearchResult;
use crate::hybrid::search::searcher::ScoreNormalization;

/// Score normalizer for hybrid search results.
///
/// Normalizes keyword and vector scores to a common scale before
/// combining them into hybrid scores.
pub struct ScoreNormalizer {
    /// The normalization strategy to use.
    strategy: ScoreNormalization,
}

impl ScoreNormalizer {
    /// Create a new score normalizer.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The normalization strategy to apply
    ///
    /// # Examples
    ///
    /// ```
    /// use platypus::hybrid::search::searcher::ScoreNormalization;
    /// use platypus::hybrid::search::scorer::ScoreNormalizer;
    ///
    /// let normalizer = ScoreNormalizer::new(ScoreNormalization::MinMax);
    /// ```
    pub fn new(strategy: ScoreNormalization) -> Self {
        Self { strategy }
    }

    /// Normalize scores based on the configured normalization strategy.
    ///
    /// Applies the configured normalization to both keyword and vector scores
    /// to bring them to a common scale.
    ///
    /// # Arguments
    ///
    /// * `results` - Map of document IDs to hybrid search results to normalize
    /// * `keyword_scores` - All keyword scores for statistical normalization
    /// * `vector_similarities` - All vector scores for statistical normalization
    ///
    /// # Returns
    ///
    /// `Ok(())` if normalization succeeds
    pub fn normalize_scores(
        &self,
        results: &mut HashMap<u64, HybridSearchResult>,
        keyword_scores: &[f32],
        vector_similarities: &[f32],
    ) -> Result<()> {
        match self.strategy {
            ScoreNormalization::None => {
                // No normalization needed
            }
            ScoreNormalization::MinMax => {
                self.normalize_min_max(results, keyword_scores, vector_similarities)?;
            }
            ScoreNormalization::ZScore => {
                self.normalize_z_score(results, keyword_scores, vector_similarities)?;
            }
            ScoreNormalization::Rank => {
                self.normalize_rank(results, keyword_scores, vector_similarities)?;
            }
        }

        Ok(())
    }

    /// Min-max normalization to [0, 1] range.
    ///
    /// Scales scores linearly using: `(score - min) / (max - min)`.
    ///
    /// # Arguments
    ///
    /// * `results` - Results to normalize
    /// * `keyword_scores` - All keyword scores
    /// * `vector_similarities` - All vector scores
    fn normalize_min_max(
        &self,
        results: &mut HashMap<u64, HybridSearchResult>,
        keyword_scores: &[f32],
        vector_similarities: &[f32],
    ) -> Result<()> {
        // Normalize keyword scores
        if !keyword_scores.is_empty() {
            let min_keyword = keyword_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_keyword = keyword_scores
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let keyword_range = max_keyword - min_keyword;

            if keyword_range > 0.0 {
                for result in results.values_mut() {
                    if let Some(score) = result.keyword_score {
                        result.keyword_score = Some((score - min_keyword) / keyword_range);
                    }
                }
            }
        }

        // Normalize vector similarities
        if !vector_similarities.is_empty() {
            let min_vector = vector_similarities
                .iter()
                .fold(f32::INFINITY, |a, &b| a.min(b));
            let max_vector = vector_similarities
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let vector_range = max_vector - min_vector;

            if vector_range > 0.0 {
                for result in results.values_mut() {
                    if let Some(similarity) = result.vector_similarity {
                        result.vector_similarity = Some((similarity - min_vector) / vector_range);
                    }
                }
            }
        }

        Ok(())
    }

    /// Z-score normalization.
    fn normalize_z_score(
        &self,
        results: &mut HashMap<u64, HybridSearchResult>,
        keyword_scores: &[f32],
        vector_similarities: &[f32],
    ) -> Result<()> {
        // Calculate keyword statistics
        if !keyword_scores.is_empty() {
            let keyword_mean = keyword_scores.iter().sum::<f32>() / keyword_scores.len() as f32;
            let keyword_variance = keyword_scores
                .iter()
                .map(|&score| (score - keyword_mean).powi(2))
                .sum::<f32>()
                / keyword_scores.len() as f32;
            let keyword_std = keyword_variance.sqrt();

            if keyword_std > 0.0 {
                for result in results.values_mut() {
                    if let Some(score) = result.keyword_score {
                        let z_score = (score - keyword_mean) / keyword_std;
                        result.keyword_score = Some((z_score + 3.0) / 6.0); // Normalize to [0, 1] approximately
                    }
                }
            }
        }

        // Calculate vector statistics
        if !vector_similarities.is_empty() {
            let vector_mean =
                vector_similarities.iter().sum::<f32>() / vector_similarities.len() as f32;
            let vector_variance = vector_similarities
                .iter()
                .map(|&sim| (sim - vector_mean).powi(2))
                .sum::<f32>()
                / vector_similarities.len() as f32;
            let vector_std = vector_variance.sqrt();

            if vector_std > 0.0 {
                for result in results.values_mut() {
                    if let Some(similarity) = result.vector_similarity {
                        let z_score = (similarity - vector_mean) / vector_std;
                        result.vector_similarity = Some((z_score + 3.0) / 6.0); // Normalize to [0, 1] approximately
                    }
                }
            }
        }

        Ok(())
    }

    /// Rank-based normalization.
    fn normalize_rank(
        &self,
        results: &mut HashMap<u64, HybridSearchResult>,
        keyword_scores: &[f32],
        vector_similarities: &[f32],
    ) -> Result<()> {
        // Create rank mappings
        let keyword_ranks = self.create_rank_mapping(keyword_scores);
        let vector_ranks = self.create_rank_mapping(vector_similarities);

        // Apply keyword ranks
        for result in results.values_mut() {
            if let Some(score) = result.keyword_score
                && let Some(&rank) = keyword_ranks.get(&((score * 1000000.0) as i32))
            {
                result.keyword_score = Some(rank);
            }
        }

        // Apply vector ranks
        for result in results.values_mut() {
            if let Some(similarity) = result.vector_similarity
                && let Some(&rank) = vector_ranks.get(&((similarity * 1000000.0) as i32))
            {
                result.vector_similarity = Some(rank);
            }
        }

        Ok(())
    }

    /// Create rank mapping for scores.
    fn create_rank_mapping(&self, scores: &[f32]) -> HashMap<i32, f32> {
        let mut unique_scores: Vec<f32> = scores.to_vec();
        unique_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        unique_scores.dedup();

        let mut rank_map = HashMap::new();
        for (rank, &score) in unique_scores.iter().enumerate() {
            let normalized_rank = 1.0 - (rank as f32 / unique_scores.len() as f32);
            rank_map.insert((score * 1000000.0) as i32, normalized_rank);
        }

        rank_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_results() -> HashMap<u64, HybridSearchResult> {
        let mut results = HashMap::new();
        results.insert(1, HybridSearchResult::new(1, 0.0).with_keyword_score(0.8));
        results.insert(2, HybridSearchResult::new(2, 0.0).with_keyword_score(0.4));
        results.insert(
            3,
            HybridSearchResult::new(3, 0.0).with_vector_similarity(0.9),
        );
        results.insert(
            4,
            HybridSearchResult::new(4, 0.0).with_vector_similarity(0.3),
        );
        results
    }

    #[test]
    fn test_score_normalizer_creation() {
        let normalizer = ScoreNormalizer::new(ScoreNormalization::MinMax);
        assert_eq!(normalizer.strategy, ScoreNormalization::MinMax);
    }

    #[test]
    fn test_no_normalization() {
        let normalizer = ScoreNormalizer::new(ScoreNormalization::None);
        let mut results = create_test_results();
        let keyword_scores = vec![0.8, 0.4];
        let vector_similarities = vec![0.9, 0.3];

        assert!(
            normalizer
                .normalize_scores(&mut results, &keyword_scores, &vector_similarities)
                .is_ok()
        );

        // Scores should remain unchanged
        assert_eq!(results.get(&1).unwrap().keyword_score, Some(0.8));
        assert_eq!(results.get(&2).unwrap().keyword_score, Some(0.4));
        assert_eq!(results.get(&3).unwrap().vector_similarity, Some(0.9));
        assert_eq!(results.get(&4).unwrap().vector_similarity, Some(0.3));
    }

    #[test]
    fn test_min_max_normalization() {
        let normalizer = ScoreNormalizer::new(ScoreNormalization::MinMax);
        let mut results = create_test_results();
        let keyword_scores = vec![0.8, 0.4];
        let vector_similarities = vec![0.9, 0.3];

        assert!(
            normalizer
                .normalize_scores(&mut results, &keyword_scores, &vector_similarities)
                .is_ok()
        );

        // Min-max normalized scores should be in [0, 1] range
        assert_eq!(results.get(&1).unwrap().keyword_score, Some(1.0)); // (0.8 - 0.4) / (0.8 - 0.4) = 1.0
        assert_eq!(results.get(&2).unwrap().keyword_score, Some(0.0)); // (0.4 - 0.4) / (0.8 - 0.4) = 0.0
        assert_eq!(results.get(&3).unwrap().vector_similarity, Some(1.0)); // (0.9 - 0.3) / (0.9 - 0.3) = 1.0
        assert_eq!(results.get(&4).unwrap().vector_similarity, Some(0.0)); // (0.3 - 0.3) / (0.9 - 0.3) = 0.0
    }

    #[test]
    fn test_z_score_normalization() {
        let normalizer = ScoreNormalizer::new(ScoreNormalization::ZScore);
        let mut results = create_test_results();
        let keyword_scores = vec![0.8, 0.4];
        let vector_similarities = vec![0.9, 0.3];

        assert!(
            normalizer
                .normalize_scores(&mut results, &keyword_scores, &vector_similarities)
                .is_ok()
        );

        // Z-score normalized values should be different from original
        assert_ne!(results.get(&1).unwrap().keyword_score, Some(0.8));
        assert_ne!(results.get(&2).unwrap().keyword_score, Some(0.4));
        assert_ne!(results.get(&3).unwrap().vector_similarity, Some(0.9));
        assert_ne!(results.get(&4).unwrap().vector_similarity, Some(0.3));

        // Should be in approximate [0, 1] range after transformation
        let keyword_1 = results.get(&1).unwrap().keyword_score.unwrap();
        let keyword_2 = results.get(&2).unwrap().keyword_score.unwrap();
        assert!((0.0..=1.0).contains(&keyword_1));
        assert!((0.0..=1.0).contains(&keyword_2));
    }

    #[test]
    fn test_rank_normalization() {
        let normalizer = ScoreNormalizer::new(ScoreNormalization::Rank);
        let mut results = create_test_results();
        let keyword_scores = vec![0.8, 0.4];
        let vector_similarities = vec![0.9, 0.3];

        assert!(
            normalizer
                .normalize_scores(&mut results, &keyword_scores, &vector_similarities)
                .is_ok()
        );

        // Rank normalized scores should be different from original
        assert_ne!(results.get(&1).unwrap().keyword_score, Some(0.8));
        assert_ne!(results.get(&2).unwrap().keyword_score, Some(0.4));
    }

    #[test]
    fn test_empty_scores() {
        let normalizer = ScoreNormalizer::new(ScoreNormalization::MinMax);
        let mut results = HashMap::new();
        results.insert(1, HybridSearchResult::new(1, 0.0));

        assert!(normalizer.normalize_scores(&mut results, &[], &[]).is_ok());
    }

    #[test]
    fn test_single_score() {
        let normalizer = ScoreNormalizer::new(ScoreNormalization::MinMax);
        let mut results = HashMap::new();
        results.insert(1, HybridSearchResult::new(1, 0.0).with_keyword_score(0.8));

        let keyword_scores = vec![0.8];
        assert!(
            normalizer
                .normalize_scores(&mut results, &keyword_scores, &[])
                .is_ok()
        );

        // Single score should remain unchanged (range is 0)
        assert_eq!(results.get(&1).unwrap().keyword_score, Some(0.8));
    }

    #[test]
    fn test_create_rank_mapping() {
        let normalizer = ScoreNormalizer::new(ScoreNormalization::Rank);
        let scores = vec![0.8, 0.4, 0.6];
        let rank_map = normalizer.create_rank_mapping(&scores);

        assert_eq!(rank_map.len(), 3);
        // Should contain mappings for all unique scores
        assert!(rank_map.contains_key(&800000)); // 0.8 * 1000000
        assert!(rank_map.contains_key(&400000)); // 0.4 * 1000000
        assert!(rank_map.contains_key(&600000)); // 0.6 * 1000000
    }
}
