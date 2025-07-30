//! Result merging functionality for hybrid search.

use std::collections::HashMap;

use super::config::HybridSearchConfig;
use super::scorer::ScoreNormalizer;
use super::types::{HybridSearchResult, HybridSearchResults};

use crate::error::Result;
use crate::query::SearchResults;
use crate::vector::types::VectorSearchResults;

/// Result merger for combining keyword and vector search results.
pub struct ResultMerger {
    config: HybridSearchConfig,
    normalizer: ScoreNormalizer,
}

impl ResultMerger {
    /// Create a new result merger.
    pub fn new(config: HybridSearchConfig) -> Self {
        let normalizer = ScoreNormalizer::new(config.normalization);
        Self { config, normalizer }
    }

    /// Merge keyword and vector search results into hybrid results.
    pub async fn merge_results(
        &self,
        keyword_results: SearchResults,
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
            for result in &vector_results.results {
                vector_similarities.push(result.similarity);

                if let Some(existing) = result_map.get_mut(&result.doc_id) {
                    existing.vector_similarity = Some(result.similarity);
                } else {
                    let hybrid_result = HybridSearchResult::new(result.doc_id, 0.0)
                        .with_vector_similarity(result.similarity);
                    result_map.insert(result.doc_id, hybrid_result);
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
        for result in result_map.values_mut() {
            let keyword_component =
                result.keyword_score.unwrap_or(0.0) * self.config.keyword_weight;
            let vector_component =
                result.vector_similarity.unwrap_or(0.0) * self.config.vector_weight;
            result.hybrid_score = keyword_component + vector_component;
        }

        // Add document content if available
        self.add_document_content(&mut result_map, document_store)
            .await?;

        // Convert to vector and sort
        let mut results: Vec<HybridSearchResult> = result_map.into_values().collect();
        results.sort_by(|a, b| {
            b.hybrid_score
                .partial_cmp(&a.hybrid_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        if results.len() > self.config.max_results {
            results.truncate(self.config.max_results);
        }

        let total_searched = document_store.len();
        let keyword_matches = keyword_results.hits.len();
        let vector_matches = vector_results.map(|vr| vr.results.len()).unwrap_or(0);

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
