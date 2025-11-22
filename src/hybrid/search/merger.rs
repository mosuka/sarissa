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
use crate::lexical::index::inverted::query::SearchResults;
use crate::vector::collection::VectorCollectionSearchResults;

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
    /// use platypus::hybrid::search::searcher::HybridSearchParams;
    /// use platypus::hybrid::search::merger::ResultMerger;
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
        keyword_results: SearchResults,
        vector_results: Option<VectorCollectionSearchResults>,
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
                .unwrap_or(Ordering::Equal)
        });

        // Limit results
        if results.len() > self.config.max_results {
            results.truncate(self.config.max_results);
        }

        let total_searched = document_store.len();
        let keyword_matches = keyword_results.hits.len();
        let vector_matches = vector_results
            .as_ref()
            .map(|vr| vr.hits.len())
            .unwrap_or(0);

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
    use crate::lexical::index::inverted::query::SearchHit;
    use crate::vector::collection::VectorCollectionHit;
    use crate::vector::field::FieldHit;

    #[tokio::test]
    async fn merge_results_preserves_vector_field_hits() {
        let mut keyword_results = SearchResults {
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

        let vector_results = VectorCollectionSearchResults {
            hits: vec![VectorCollectionHit {
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
}
