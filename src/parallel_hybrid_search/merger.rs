//! Result merger for parallel hybrid search.

use super::config::{MergeStrategy, ParallelHybridSearchConfig};
use super::types::{ParallelHybridSearchResult, ScoreExplanation};
use crate::query::SearchHit;
use crate::vector::types::VectorSearchResult;
use std::collections::HashMap;

/// Merger for combining keyword and vector search results.
pub struct ParallelHybridResultMerger {
    config: ParallelHybridSearchConfig,
}

impl ParallelHybridResultMerger {
    /// Create a new result merger.
    pub fn new(config: ParallelHybridSearchConfig) -> Self {
        Self { config }
    }
    
    /// Merge keyword and vector search results.
    pub fn merge(
        &self,
        keyword_results: Vec<(String, Vec<SearchHit>)>,
        vector_results: Vec<(String, Vec<VectorSearchResult>)>,
        document_store: &HashMap<u64, HashMap<String, String>>,
    ) -> Vec<ParallelHybridSearchResult> {
        // Create maps for efficient lookup
        let mut keyword_map: HashMap<u64, (String, f32, usize)> = HashMap::new();
        let mut vector_map: HashMap<u64, (String, f32, usize)> = HashMap::new();
        
        // Process keyword results
        for (index_id, hits) in keyword_results {
            for (rank, hit) in hits.iter().enumerate() {
                keyword_map.insert(hit.doc_id, (index_id.clone(), hit.score, rank));
            }
        }
        
        // Process vector results
        for (index_id, results) in vector_results {
            for (rank, result) in results.iter().enumerate() {
                vector_map.insert(result.doc_id, (index_id.clone(), result.similarity, rank));
            }
        }
        
        // Collect all unique document IDs
        let all_doc_ids: Vec<u64> = keyword_map.keys()
            .chain(vector_map.keys())
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        
        // Calculate combined scores
        let mut merged_results: Vec<ParallelHybridSearchResult> = all_doc_ids
            .into_iter()
            .filter_map(|doc_id| {
                let keyword_info = keyword_map.get(&doc_id);
                let vector_info = vector_map.get(&doc_id);
                
                // Apply minimum score thresholds
                let keyword_score = keyword_info.map(|(_, score, _)| *score);
                let vector_similarity = vector_info.map(|(_, sim, _)| *sim);
                
                if let Some(score) = keyword_score {
                    if score < self.config.min_keyword_score {
                        return None;
                    }
                }
                
                if let Some(sim) = vector_similarity {
                    if sim < self.config.min_vector_similarity {
                        return None;
                    }
                }
                
                // Calculate combined score based on strategy
                let (combined_score, explanation) = self.calculate_combined_score(
                    keyword_score,
                    vector_similarity,
                    keyword_info.map(|(_, _, rank)| *rank),
                    vector_info.map(|(_, _, rank)| *rank),
                );
                
                // Determine index ID (prefer keyword if both exist)
                let index_id = keyword_info
                    .map(|(id, _, _)| id.clone())
                    .or_else(|| vector_info.map(|(id, _, _)| id.clone()))
                    .unwrap();
                
                // Get document fields
                let fields = document_store
                    .get(&doc_id)
                    .cloned()
                    .unwrap_or_default();
                
                Some(ParallelHybridSearchResult {
                    doc_id,
                    combined_score,
                    keyword_score,
                    vector_similarity,
                    keyword_rank: keyword_info.map(|(_, _, rank)| *rank),
                    vector_rank: vector_info.map(|(_, _, rank)| *rank),
                    index_id,
                    fields,
                    explanation: Some(explanation),
                })
            })
            .collect();
        
        // Sort by combined score (descending)
        merged_results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Limit results
        merged_results.truncate(self.config.max_final_results);
        
        merged_results
    }
    
    /// Calculate combined score based on merge strategy.
    fn calculate_combined_score(
        &self,
        keyword_score: Option<f32>,
        vector_similarity: Option<f32>,
        keyword_rank: Option<usize>,
        vector_rank: Option<usize>,
    ) -> (f32, ScoreExplanation) {
        match self.config.merge_strategy {
            MergeStrategy::LinearCombination => {
                self.linear_combination(keyword_score, vector_similarity)
            }
            MergeStrategy::ReciprocalRankFusion => {
                self.reciprocal_rank_fusion(keyword_rank, vector_rank)
            }
            MergeStrategy::MaxScore => {
                self.max_score(keyword_score, vector_similarity)
            }
            MergeStrategy::ScoreProduct => {
                self.score_product(keyword_score, vector_similarity)
            }
            MergeStrategy::Adaptive => {
                self.adaptive_merge(keyword_score, vector_similarity, keyword_rank, vector_rank)
            }
        }
    }
    
    /// Linear weighted combination of scores.
    fn linear_combination(
        &self,
        keyword_score: Option<f32>,
        vector_similarity: Option<f32>,
    ) -> (f32, ScoreExplanation) {
        let kw_contrib = keyword_score.unwrap_or(0.0) * self.config.keyword_weight;
        let vec_contrib = vector_similarity.unwrap_or(0.0) * self.config.vector_weight;
        let combined = kw_contrib + vec_contrib;
        
        let explanation = ScoreExplanation {
            method: "Linear Combination".to_string(),
            keyword_contribution: kw_contrib,
            vector_contribution: vec_contrib,
            details: HashMap::from([
                ("keyword_weight".to_string(), self.config.keyword_weight.to_string()),
                ("vector_weight".to_string(), self.config.vector_weight.to_string()),
            ]),
        };
        
        (combined, explanation)
    }
    
    /// Reciprocal Rank Fusion (RRF) scoring.
    fn reciprocal_rank_fusion(
        &self,
        keyword_rank: Option<usize>,
        vector_rank: Option<usize>,
    ) -> (f32, ScoreExplanation) {
        const K: f32 = 60.0; // RRF constant
        
        let keyword_rrf = keyword_rank
            .map(|rank| 1.0 / (K + rank as f32 + 1.0))
            .unwrap_or(0.0);
        let vector_rrf = vector_rank
            .map(|rank| 1.0 / (K + rank as f32 + 1.0))
            .unwrap_or(0.0);
        
        let kw_contrib = keyword_rrf * self.config.keyword_weight;
        let vec_contrib = vector_rrf * self.config.vector_weight;
        let combined = kw_contrib + vec_contrib;
        
        let explanation = ScoreExplanation {
            method: "Reciprocal Rank Fusion".to_string(),
            keyword_contribution: kw_contrib,
            vector_contribution: vec_contrib,
            details: HashMap::from([
                ("k_constant".to_string(), K.to_string()),
                ("keyword_rrf".to_string(), keyword_rrf.to_string()),
                ("vector_rrf".to_string(), vector_rrf.to_string()),
            ]),
        };
        
        (combined, explanation)
    }
    
    /// Maximum score from either search.
    fn max_score(
        &self,
        keyword_score: Option<f32>,
        vector_similarity: Option<f32>,
    ) -> (f32, ScoreExplanation) {
        let kw_weighted = keyword_score.unwrap_or(0.0) * self.config.keyword_weight;
        let vec_weighted = vector_similarity.unwrap_or(0.0) * self.config.vector_weight;
        let combined = kw_weighted.max(vec_weighted);
        
        let explanation = ScoreExplanation {
            method: "Max Score".to_string(),
            keyword_contribution: if kw_weighted >= vec_weighted { kw_weighted } else { 0.0 },
            vector_contribution: if vec_weighted > kw_weighted { vec_weighted } else { 0.0 },
            details: HashMap::from([
                ("selected".to_string(), 
                 if kw_weighted >= vec_weighted { "keyword".to_string() } 
                 else { "vector".to_string() }),
            ]),
        };
        
        (combined, explanation)
    }
    
    /// Product of scores.
    fn score_product(
        &self,
        keyword_score: Option<f32>,
        vector_similarity: Option<f32>,
    ) -> (f32, ScoreExplanation) {
        let kw_score = keyword_score.unwrap_or(0.0);
        let vec_score = vector_similarity.unwrap_or(0.0);
        
        // Avoid zero multiplication
        let combined = if kw_score > 0.0 && vec_score > 0.0 {
            kw_score * vec_score
        } else {
            // Fallback to weighted average if one is zero
            kw_score * self.config.keyword_weight + vec_score * self.config.vector_weight
        };
        
        let explanation = ScoreExplanation {
            method: "Score Product".to_string(),
            keyword_contribution: kw_score,
            vector_contribution: vec_score,
            details: HashMap::from([
                ("product".to_string(), combined.to_string()),
            ]),
        };
        
        (combined, explanation)
    }
    
    /// Adaptive merge based on result characteristics.
    fn adaptive_merge(
        &self,
        keyword_score: Option<f32>,
        vector_similarity: Option<f32>,
        keyword_rank: Option<usize>,
        vector_rank: Option<usize>,
    ) -> (f32, ScoreExplanation) {
        // Use different strategies based on what's available
        match (keyword_score, vector_similarity) {
            (Some(_), Some(_)) => {
                // Both available: use linear combination
                self.linear_combination(keyword_score, vector_similarity)
            }
            (Some(_), None) => {
                // Only keyword: use full keyword score
                let score = keyword_score.unwrap();
                let explanation = ScoreExplanation {
                    method: "Adaptive (Keyword Only)".to_string(),
                    keyword_contribution: score,
                    vector_contribution: 0.0,
                    details: HashMap::new(),
                };
                (score, explanation)
            }
            (None, Some(_)) => {
                // Only vector: use full vector score
                let score = vector_similarity.unwrap();
                let explanation = ScoreExplanation {
                    method: "Adaptive (Vector Only)".to_string(),
                    keyword_contribution: 0.0,
                    vector_contribution: score,
                    details: HashMap::new(),
                };
                (score, explanation)
            }
            (None, None) => {
                // Neither available but ranks exist: use RRF
                if keyword_rank.is_some() || vector_rank.is_some() {
                    self.reciprocal_rank_fusion(keyword_rank, vector_rank)
                } else {
                    // No information: return zero
                    let explanation = ScoreExplanation {
                        method: "Adaptive (No Data)".to_string(),
                        keyword_contribution: 0.0,
                        vector_contribution: 0.0,
                        details: HashMap::new(),
                    };
                    (0.0, explanation)
                }
            }
        }
    }
}