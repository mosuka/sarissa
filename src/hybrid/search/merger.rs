//! Result merging functionality for hybrid search.
//!
//! This module provides the `ResultMerger` for combining keyword and vector
//! search results into unified hybrid search results.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::error::Result;
use crate::hybrid::search::scorer::ScoreNormalizer;
use crate::hybrid::search::searcher::HybridSearchParams;
use crate::hybrid::search::searcher::{HybridSearchResult, HybridSearchResults};
use crate::lexical::index::inverted::query::LexicalSearchResults;
use crate::vector::engine::VectorSearchResults;

/// Result merger for combining keyword and vector search results.
///
/// This structure merges results from lexical (keyword) and vector (semantic)
/// search, applying normalization and weighting to produce unified hybrid scores.
pub struct ResultMerger {
    /// Configuration for merging behavior.
    config: HybridSearchParams,
    /// Score normalizer for bringing scores to common scale.
    normalizer: ScoreNormalizer,
}

impl ResultMerger {
    /// Create a new result merger.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for hybrid search merging
    ///
    /// # Examples
    ///
    /// ```
    /// use sarissa::hybrid::search::searcher::HybridSearchParams;
    /// use sarissa::hybrid::search::merger::ResultMerger;
    ///
    /// let config = HybridSearchParams::default();
    /// let merger = ResultMerger::new(config);
    /// ```
    pub fn new(config: HybridSearchParams) -> Self {
        let normalizer = ScoreNormalizer::new(config.normalization);
        Self { config, normalizer }
    }

    /// Merge keyword and vector search results into hybrid results.
    ///
    /// Combines results from both search types, normalizes scores, and
    /// calculates final hybrid scores using configured weights.
    ///
    /// # Arguments
    ///
    /// * `keyword_results` - Results from lexical (keyword) search
    /// * `vector_results` - Optional results from vector (semantic) search
    /// * `query_text` - The original query text
    /// * `query_time_ms` - Time taken for the query in milliseconds
    /// * `document_store` - Document storage for retrieving full content
    ///
    /// # Returns
    ///
    /// Unified hybrid search results with combined scores
    pub async fn merge_results(
        &self,
        keyword_results: LexicalSearchResults,
        vector_results: Option<VectorSearchResults>,
        query_text: String,
        query_time_ms: u64,
        document_store: &HashMap<u64, HashMap<String, String>>,
    ) -> Result<HybridSearchResults> {
        let mut result_map: HashMap<u64, HybridSearchResult> = HashMap::new();
        let mut keyword_scores = Vec::new();
        let mut vector_similarities = Vec::new();

        // Process keyword results
        for hit in &keyword_results.hits {
            keyword_scores.push(hit.score);
            let result = HybridSearchResult::new(hit.doc_id, 0.0).with_keyword_score(hit.score);
            result_map.insert(hit.doc_id, result);
        }

        // Process vector results
        if let Some(ref vector_results) = vector_results {
            for hit in &vector_results.hits {
                vector_similarities.push(hit.score);

                if let Some(existing) = result_map.get_mut(&hit.doc_id) {
                    existing.vector_similarity = Some(hit.score);
                    if existing.vector_field_hits.is_empty() {
                        existing.vector_field_hits = hit.field_hits.clone();
                    } else {
                        existing.vector_field_hits.extend(hit.field_hits.clone());
                    }
                } else {
                    let mut hybrid_result =
                        HybridSearchResult::new(hit.doc_id, 0.0).with_vector_similarity(hit.score);
                    hybrid_result.vector_field_hits = hit.field_hits.clone();
                    result_map.insert(hit.doc_id, hybrid_result);
                }
            }
        }

        // Filter results based on requirements
        if self.config.require_both {
            result_map.retain(|_, result| {
                result.keyword_score.is_some() && result.vector_similarity.is_some()
            });
        }

        // Normalize scores
        self.normalizer
            .normalize_scores(&mut result_map, &keyword_scores, &vector_similarities)?;

        // Calculate hybrid scores
        use crate::hybrid::search::searcher::ScoreCombination;
        match self.config.combination {
            ScoreCombination::WeightedSum => {
                for result in result_map.values_mut() {
                    let keyword_component =
                        result.keyword_score.unwrap_or(0.0) * self.config.keyword_weight;
                    let vector_component =
                        result.vector_similarity.unwrap_or(0.0) * self.config.vector_weight;
                    result.hybrid_score = keyword_component + vector_component;
                }
            }
            ScoreCombination::Rrf => {
                // reciprocal rank fusion: 1 / (k + rank)
                // We assume self.normalizer.normalize_scores with ScoreNormalization::Rank
                // already converted scores to 1.0 - (rank / total).
                // However, RRF is usually calculated directly from ranks.
                // If Rank normalization was used, keyword_score is 1.0 - (rank_k / count_k)
                // Let's re-calculate RRF properly if we have the original ranks,
                // or use the normalized rank if that's all we have.
                // Given the current ScoreNormalizer::normalize_rank, we can derive the rank.

                let k = 60.0;
                for result in result_map.values_mut() {
                    let mut score = 0.0;
                    if let Some(norm_rank) = result.keyword_score {
                        // norm_rank = 1.0 - (rank / count) => rank = (1.0 - norm_rank) * count
                        // Since we don't have 'count' here easily if filtered,
                        // and RRF is often used on the top-K results:
                        // If we didn't use ScoreNormalization::Rank, this might be tricky.
                        // Standard RRF implementation usually takes two sorted lists.

                        // If ScoreNormalization::Rank was used:
                        let rank = (1.0 - norm_rank) * (keyword_scores.len() as f32);
                        score += self.config.keyword_weight * (1.0 / (k + rank));
                    }
                    if let Some(norm_rank) = result.vector_similarity {
                        let rank = (1.0 - norm_rank) * (vector_similarities.len() as f32);
                        score += self.config.vector_weight * (1.0 / (k + rank));
                    }
                    result.hybrid_score = score;
                }
            }
            ScoreCombination::HarmonicMean => {
                for result in result_map.values_mut() {
                    let k_score = result.keyword_score.unwrap_or(0.0);
                    let v_score = result.vector_similarity.unwrap_or(0.0);
                    if k_score > 0.0 && v_score > 0.0 {
                        result.hybrid_score = 2.0 * (k_score * v_score) / (k_score + v_score);
                    } else {
                        result.hybrid_score = 0.0;
                    }
                }
            }
        }

        // Add document content if available
        self.add_document_content(&mut result_map, document_store)
            .await?;

        // Convert to vector and sort
        let mut results: Vec<HybridSearchResult> = result_map.into_values().collect();
        results.sort_by(|a, b| {
            b.hybrid_score
                .partial_cmp(&a.hybrid_score)
                .unwrap_or(Ordering::Equal)
        });

        // Limit results
        if results.len() > self.config.max_results {
            results.truncate(self.config.max_results);
        }

        let total_searched = document_store.len();
        let keyword_matches = keyword_results.hits.len();
        let vector_matches = vector_results.as_ref().map(|vr| vr.hits.len()).unwrap_or(0);

        Ok(HybridSearchResults::new(
            results,
            total_searched,
            keyword_matches,
            vector_matches,
            query_time_ms,
            query_text,
        ))
    }

    /// Add document content to results.
    ///
    /// Enriches search results with full document content from storage.
    ///
    /// # Arguments
    ///
    /// * `results` - Map of results to enrich
    /// * `document_store` - Document storage to retrieve content from
    async fn add_document_content(
        &self,
        results: &mut HashMap<u64, HybridSearchResult>,
        document_store: &HashMap<u64, HashMap<String, String>>,
    ) -> Result<()> {
        for (doc_id, result) in results.iter_mut() {
            if let Some(document) = document_store.get(doc_id) {
                result.document = Some(document.clone());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::search::searcher::{ScoreCombination, ScoreNormalization};
    use crate::lexical::index::inverted::query::SearchHit;
    use crate::vector::engine::VectorHit;
    use crate::vector::field::FieldHit;

    #[tokio::test]
    async fn merge_results_preserves_vector_field_hits() {
        let mut keyword_results = LexicalSearchResults {
            hits: Vec::new(),
            total_hits: 0,
            max_score: 0.0,
        };
        keyword_results.hits.push(SearchHit {
            doc_id: 7,
            score: 0.42,
            document: None,
        });
        keyword_results.total_hits = 1;
        keyword_results.max_score = 0.42;

        let vector_results = VectorSearchResults {
            hits: vec![VectorHit {
                doc_id: 7,
                score: 0.91,
                field_hits: vec![FieldHit {
                    doc_id: 7,
                    field: "title_embedding".into(),
                    score: 0.91,
                    distance: 0.08,
                    metadata: Default::default(),
                }],
            }],
        };

        let params = HybridSearchParams::default();
        let merger = ResultMerger::new(params);
        let merged = merger
            .merge_results(
                keyword_results,
                Some(vector_results),
                "rust".to_string(),
                1,
                &HashMap::new(),
            )
            .await
            .expect("merge");

        assert_eq!(merged.results.len(), 1);
        let result = &merged.results[0];
        assert_eq!(result.vector_similarity, Some(0.91));
        assert_eq!(result.vector_field_hits.len(), 1);
        assert_eq!(result.vector_field_hits[0].field, "title_embedding");
    }

    #[tokio::test]
    async fn merge_results_rrf_combination() {
        let keyword_results = LexicalSearchResults {
            hits: vec![SearchHit {
                doc_id: 1,
                score: 1.0, // Best keyword
                document: None,
            }],
            total_hits: 1,
            max_score: 1.0,
        };

        let vector_results = VectorSearchResults {
            hits: vec![VectorHit {
                doc_id: 2,
                score: 1.0, // Best vector
                field_hits: Vec::new(),
            }],
        };

        let mut params = HybridSearchParams::default();
        params.combination = ScoreCombination::Rrf;
        params.normalization = ScoreNormalization::Rank;

        let merger = ResultMerger::new(params);
        let merged = merger
            .merge_results(
                keyword_results,
                Some(vector_results),
                "test".to_string(),
                1,
                &HashMap::new(),
            )
            .await
            .expect("merge");

        assert_eq!(merged.results.len(), 2);
        // Both should have some RRF score
        assert!(merged.results[0].hybrid_score > 0.0);
        assert!(merged.results[1].hybrid_score > 0.0);
    }

    #[tokio::test]
    async fn merge_results_harmonic_mean() {
        let keyword_results = LexicalSearchResults {
            hits: vec![SearchHit {
                doc_id: 1,
                score: 0.8,
                document: None,
            }],
            total_hits: 1,
            max_score: 0.8,
        };

        let vector_results = VectorSearchResults {
            hits: vec![VectorHit {
                doc_id: 1,
                score: 0.4,
                field_hits: Vec::new(),
            }],
        };

        let mut params = HybridSearchParams::default();
        params.combination = ScoreCombination::HarmonicMean;
        params.normalization = ScoreNormalization::None;

        let merger = ResultMerger::new(params);
        let merged = merger
            .merge_results(
                keyword_results,
                Some(vector_results),
                "test".to_string(),
                1,
                &HashMap::new(),
            )
            .await
            .expect("merge");

        assert_eq!(merged.results.len(), 1);
        // 2 * (0.8 * 0.4) / (0.8 + 0.4) = 0.64 / 1.2 = 0.533
        assert!((merged.results[0].hybrid_score - 0.533).abs() < 0.001);
    }

    #[tokio::test]
    async fn merge_results_enriches_document_content() {
        let keyword_results = LexicalSearchResults {
            hits: vec![SearchHit {
                doc_id: 42,
                score: 0.5,
                document: None,
            }],
            total_hits: 1,
            max_score: 0.5,
        };

        let mut doc_store = HashMap::new();
        let mut content = HashMap::new();
        content.insert("title".to_string(), "Deep Learning".to_string());
        doc_store.insert(42, content);

        let params = HybridSearchParams::default();
        let merger = ResultMerger::new(params);
        let merged = merger
            .merge_results(keyword_results, None, "test".to_string(), 1, &doc_store)
            .await
            .expect("merge");

        assert_eq!(merged.results.len(), 1);
        assert!(merged.results[0].document.is_some());
        assert_eq!(
            merged.results[0]
                .document
                .as_ref()
                .unwrap()
                .get("title")
                .unwrap(),
            "Deep Learning"
        );
    }
}
