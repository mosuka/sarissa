//! Hybrid search execution, requests, and results.
//!
//! This module provides all the structures and functionality for hybrid search,
//! combining lexical (keyword) and vector (semantic) search.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::lexical::search::searcher::LexicalSearchParams;
use crate::vector::Vector;
use crate::vector::search::searcher::VectorSearchParams;

/// Hybrid search request combining text query, optional vector, and search parameters.
///
/// This is the main structure for executing hybrid searches, similar to
/// `LexicalSearchRequest` and `VectorSearchRequest`.
///
/// # Examples
///
/// ```
/// use yatagarasu::hybrid::search::searcher::HybridSearchRequest;
/// use yatagarasu::vector::Vector;
///
/// // Text-only search
/// let request = HybridSearchRequest::new("rust programming");
///
/// // Hybrid search with vector
/// let vector = Vector::new(vec![1.0, 2.0, 3.0]);
/// let request = HybridSearchRequest::new("machine learning")
///     .with_vector(vector)
///     .keyword_weight(0.6)
///     .vector_weight(0.4)
///     .max_results(20);
/// ```
#[derive(Debug, Clone)]
pub struct HybridSearchRequest {
    /// Text query for lexical search.
    pub text_query: String,
    /// Optional vector for semantic search.
    pub vector_query: Option<Vector>,
    /// Hybrid search parameters.
    pub params: HybridSearchParams,
    /// Lexical search parameters.
    pub lexical_params: LexicalSearchParams,
    /// Vector search parameters.
    pub vector_params: VectorSearchParams,
}

impl HybridSearchRequest {
    /// Create a new hybrid search request with default parameters.
    pub fn new(text_query: impl Into<String>) -> Self {
        Self {
            text_query: text_query.into(),
            vector_query: None,
            params: HybridSearchParams::default(),
            lexical_params: LexicalSearchParams::default(),
            vector_params: VectorSearchParams::default(),
        }
    }

    /// Add a vector query for semantic search.
    pub fn with_vector(mut self, vector: Vector) -> Self {
        self.vector_query = Some(vector);
        self
    }

    /// Set the keyword weight (0.0-1.0).
    pub fn keyword_weight(mut self, weight: f32) -> Self {
        self.params.keyword_weight = weight;
        self
    }

    /// Set the vector weight (0.0-1.0).
    pub fn vector_weight(mut self, weight: f32) -> Self {
        self.params.vector_weight = weight;
        self
    }

    /// Set the maximum number of results to return.
    pub fn max_results(mut self, max: usize) -> Self {
        self.params.max_results = max;
        self
    }

    /// Set minimum keyword score threshold.
    pub fn min_keyword_score(mut self, score: f32) -> Self {
        self.params.min_keyword_score = score;
        self
    }

    /// Set minimum vector similarity threshold.
    pub fn min_vector_similarity(mut self, similarity: f32) -> Self {
        self.params.min_vector_similarity = similarity;
        self
    }

    /// Require both keyword and vector matches.
    pub fn require_both(mut self, require: bool) -> Self {
        self.params.require_both = require;
        self
    }

    /// Set the score normalization strategy.
    pub fn normalization(mut self, norm: ScoreNormalization) -> Self {
        self.params.normalization = norm;
        self
    }

    /// Set lexical search parameters.
    pub fn lexical_params(mut self, params: LexicalSearchParams) -> Self {
        self.lexical_params = params;
        self
    }

    /// Set vector search parameters.
    pub fn vector_params(mut self, params: VectorSearchParams) -> Self {
        self.vector_params = params;
        self
    }
}

/// Parameters for hybrid search combining keyword and vector search.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchParams {
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
}

impl Default for HybridSearchParams {
    fn default() -> Self {
        Self {
            keyword_weight: 0.6,
            vector_weight: 0.4,
            min_keyword_score: 0.0,
            min_vector_similarity: 0.3,
            max_results: 50,
            require_both: false,
            normalization: ScoreNormalization::MinMax,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreNormalization {
    /// No normalization - use raw scores directly.
    None,
    /// Min-max normalization to [0, 1] range.
    MinMax,
    /// Z-score normalization (standardization).
    ZScore,
    /// Rank-based normalization.
    Rank,
}

/// A single result from hybrid search containing both keyword and vector scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult {
    /// Document ID.
    pub doc_id: u64,
    /// Combined hybrid score.
    pub hybrid_score: f32,
    /// Keyword search score (if available).
    pub keyword_score: Option<f32>,
    /// Vector similarity score (if available).
    pub vector_similarity: Option<f32>,
    /// Document content (if requested).
    pub document: Option<HashMap<String, String>>,
    /// Vector data (if requested).
    pub vector: Option<Vector>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl HybridSearchResult {
    /// Create a new hybrid search result.
    pub fn new(doc_id: u64, hybrid_score: f32) -> Self {
        Self {
            doc_id,
            hybrid_score,
            keyword_score: None,
            vector_similarity: None,
            document: None,
            vector: None,
            metadata: HashMap::new(),
        }
    }

    /// Set keyword search score.
    pub fn with_keyword_score(mut self, score: f32) -> Self {
        self.keyword_score = Some(score);
        self
    }

    /// Set vector similarity score.
    pub fn with_vector_similarity(mut self, similarity: f32) -> Self {
        self.vector_similarity = Some(similarity);
        self
    }

    /// Add document content.
    pub fn with_document(mut self, document: HashMap<String, String>) -> Self {
        self.document = Some(document);
        self
    }

    /// Add vector data.
    pub fn with_vector(mut self, vector: Vector) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Collection of hybrid search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResults {
    /// List of results, sorted by hybrid score (descending).
    pub results: Vec<HybridSearchResult>,
    /// Total number of documents searched.
    pub total_searched: usize,
    /// Number of keyword matches.
    pub keyword_matches: usize,
    /// Number of vector matches.
    pub vector_matches: usize,
    /// Query processing time in milliseconds.
    pub query_time_ms: u64,
    /// Query text used for search.
    pub query_text: String,
}

impl HybridSearchResults {
    /// Create new empty hybrid search results.
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            total_searched: 0,
            keyword_matches: 0,
            vector_matches: 0,
            query_time_ms: 0,
            query_text: String::new(),
        }
    }

    /// Create new hybrid search results.
    pub fn new(
        results: Vec<HybridSearchResult>,
        total_searched: usize,
        keyword_matches: usize,
        vector_matches: usize,
        query_time_ms: u64,
        query_text: String,
    ) -> Self {
        Self {
            results,
            total_searched,
            keyword_matches,
            vector_matches,
            query_time_ms,
            query_text,
        }
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the best result.
    pub fn best_result(&self) -> Option<&HybridSearchResult> {
        self.results.first()
    }

    /// Filter results by minimum hybrid score.
    pub fn filter_by_score(&mut self, min_score: f32) {
        self.results
            .retain(|result| result.hybrid_score >= min_score);
    }

    /// Sort results by hybrid score (descending).
    pub fn sort_by_score(&mut self) {
        self.results.sort_by(|a, b| {
            b.hybrid_score
                .partial_cmp(&a.hybrid_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Limit the number of results.
    pub fn limit(&mut self, max_results: usize) {
        if self.results.len() > max_results {
            self.results.truncate(max_results);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_search_params_default() {
        let params = HybridSearchParams::default();
        assert_eq!(params.keyword_weight, 0.6);
        assert_eq!(params.vector_weight, 0.4);
        assert_eq!(params.min_keyword_score, 0.0);
        assert_eq!(params.min_vector_similarity, 0.3);
        assert_eq!(params.max_results, 50);
        assert!(!params.require_both);
        assert_eq!(params.normalization, ScoreNormalization::MinMax);
    }

    #[test]
    fn test_score_normalization_values() {
        assert_eq!(ScoreNormalization::None, ScoreNormalization::None);
        assert_eq!(ScoreNormalization::MinMax, ScoreNormalization::MinMax);
        assert_eq!(ScoreNormalization::ZScore, ScoreNormalization::ZScore);
        assert_eq!(ScoreNormalization::Rank, ScoreNormalization::Rank);
    }

    #[test]
    fn test_hybrid_search_result_creation() {
        let result = HybridSearchResult::new(1, 0.8);
        assert_eq!(result.doc_id, 1);
        assert_eq!(result.hybrid_score, 0.8);
        assert_eq!(result.keyword_score, None);
        assert_eq!(result.vector_similarity, None);
        assert!(result.document.is_none());
        assert!(result.vector.is_none());
        assert!(result.metadata.is_empty());
    }

    #[test]
    fn test_hybrid_search_result_builder() {
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "Test".to_string());

        let result = HybridSearchResult::new(1, 0.8)
            .with_keyword_score(0.7)
            .with_vector_similarity(0.9)
            .with_document(fields.clone())
            .with_metadata(fields);

        assert_eq!(result.doc_id, 1);
        assert_eq!(result.hybrid_score, 0.8);
        assert_eq!(result.keyword_score, Some(0.7));
        assert_eq!(result.vector_similarity, Some(0.9));
        assert!(result.document.is_some());
        assert!(!result.metadata.is_empty());
    }

    #[test]
    fn test_hybrid_search_results_empty() {
        let results = HybridSearchResults::empty();
        assert!(results.is_empty());
        assert_eq!(results.len(), 0);
        assert_eq!(results.total_searched, 0);
        assert_eq!(results.keyword_matches, 0);
        assert_eq!(results.vector_matches, 0);
        assert_eq!(results.query_time_ms, 0);
        assert!(results.query_text.is_empty());
        assert!(results.best_result().is_none());
    }

    #[test]
    fn test_hybrid_search_results_operations() {
        let mut results = HybridSearchResults::empty();

        // Add some test results
        results.results.push(HybridSearchResult::new(1, 0.9));
        results.results.push(HybridSearchResult::new(2, 0.7));
        results.results.push(HybridSearchResult::new(3, 0.5));

        assert_eq!(results.len(), 3);
        assert!(!results.is_empty());

        // Test best result
        assert_eq!(results.best_result().unwrap().doc_id, 1);

        // Test sorting
        results.sort_by_score();
        assert_eq!(results.results[0].hybrid_score, 0.9);
        assert_eq!(results.results[1].hybrid_score, 0.7);
        assert_eq!(results.results[2].hybrid_score, 0.5);

        // Test filtering
        results.filter_by_score(0.6);
        assert_eq!(results.len(), 2);

        // Test limiting
        results.limit(1);
        assert_eq!(results.len(), 1);
        assert_eq!(results.results[0].doc_id, 1);
    }

    #[test]
    fn test_hybrid_search_results_constructor() {
        let results_vec = vec![
            HybridSearchResult::new(1, 0.8),
            HybridSearchResult::new(2, 0.6),
        ];

        let results =
            HybridSearchResults::new(results_vec, 100, 10, 5, 250, "test query".to_string());

        assert_eq!(results.len(), 2);
        assert_eq!(results.total_searched, 100);
        assert_eq!(results.keyword_matches, 10);
        assert_eq!(results.vector_matches, 5);
        assert_eq!(results.query_time_ms, 250);
        assert_eq!(results.query_text, "test query");
    }
}
