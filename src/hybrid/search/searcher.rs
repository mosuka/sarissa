//! Hybrid search execution, requests, and results.
//!
//! This module provides all the structures and functionality for hybrid search,
//! combining lexical (keyword) and vector (semantic) search.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::lexical::search::searcher::LexicalSearchParams;
use crate::vector::core::document::{Payload, StoredVector};
use crate::vector::core::vector::Vector;
use crate::vector::engine::{
    FieldSelector, QueryVector, VectorFilter, VectorScoreMode, VectorSearchRequest,
};
use crate::vector::field::FieldHit;
use crate::vector::search::searcher::VectorIndexSearchParams;

/// Hybrid search request combining text query, optional vector, and search parameters.
///
/// This is the main structure for executing hybrid searches, similar to
/// `LexicalSearchRequest` and `VectorEngineSearchRequest`.
///
/// # Examples
///
/// ```
/// use platypus::hybrid::search::searcher::HybridSearchRequest;
/// use platypus::vector::core::vector::Vector;
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
#[derive(Debug, Clone, Default)]
pub struct HybridVectorOptions {
    pub fields: Option<Vec<FieldSelector>>,
    pub score_mode: Option<VectorScoreMode>,
    pub overfetch: Option<f32>,
    pub filter: Option<VectorFilter>,
}

#[derive(Debug, Clone)]
pub struct HybridSearchRequest {
    /// Text query for lexical search.
    pub text_query: String,
    /// Optional vector for semantic search.
    pub vector_query: Option<VectorSearchRequest>,
    /// Raw payloads that require embedding before executing the vector search.
    /// Each entry maps a field name to a single `Payload`.
    pub vector_payloads: HashMap<String, Payload>,
    /// Hybrid search parameters.
    pub params: HybridSearchParams,
    /// Lexical search parameters.
    pub lexical_params: LexicalSearchParams,
    /// Vector search parameters.
    pub vector_params: VectorIndexSearchParams,
    /// Doc-centric overrides applied when generating vector queries.
    pub vector_overrides: HybridVectorOptions,
}

impl HybridSearchRequest {
    /// Create a new hybrid search request with default parameters.
    pub fn new(text_query: impl Into<String>) -> Self {
        Self {
            text_query: text_query.into(),
            vector_query: None,
            vector_payloads: HashMap::new(),
            params: HybridSearchParams::default(),
            lexical_params: LexicalSearchParams::default(),
            vector_params: VectorIndexSearchParams::default(),
            vector_overrides: HybridVectorOptions::default(),
        }
    }

    /// Add a vector query for semantic search.
    pub fn with_vector(mut self, vector: Vector) -> Self {
        let mut query = Self::build_query_from_vector(vector, self.vector_params.top_k);
        Self::apply_overrides_to_query(&self.vector_overrides, &mut query);
        self.vector_query = Some(query);
        self
    }

    /// Provide a fully-specified VectorEngineSearchRequest.
    pub fn with_vector_engine_search_request(
        mut self,
        mut vector_query: VectorSearchRequest,
    ) -> Self {
        Self::apply_overrides_to_query(&self.vector_overrides, &mut vector_query);
        self.vector_query = Some(vector_query);
        self
    }

    /// Provide a raw payload that will be embedded for the specified field at query time.
    pub fn with_vector_payload(mut self, field_name: impl Into<String>, payload: Payload) -> Self {
        self.vector_payloads.insert(field_name.into(), payload);
        self
    }

    /// Convenience helper to embed a single text snippet for a specific field.
    pub fn with_vector_text(self, field_name: impl Into<String>, text: impl Into<String>) -> Self {
        self.with_vector_payload(field_name, Payload::text(text))
    }

    /// Override vector field selectors for doc-centric search.
    pub fn vector_fields(mut self, selectors: Vec<FieldSelector>) -> Self {
        self.vector_overrides.fields = if selectors.is_empty() {
            None
        } else {
            Some(selectors)
        };
        self.update_active_query();
        self
    }

    /// Override vector score mode for doc-centric search.
    pub fn vector_score_mode(mut self, mode: VectorScoreMode) -> Self {
        self.vector_overrides.score_mode = Some(mode);
        self.update_active_query();
        self
    }

    /// Override vector overfetch factor for doc-centric search.
    pub fn vector_overfetch(mut self, factor: f32) -> Self {
        self.vector_overrides.overfetch = Some(factor);
        self.update_active_query();
        self
    }

    /// Override vector metadata filters for doc-centric search.
    pub fn vector_filter(mut self, filter: VectorFilter) -> Self {
        self.vector_overrides.filter = Some(filter);
        self.update_active_query();
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

    /// Set the score combination strategy.
    pub fn combination(mut self, comb: ScoreCombination) -> Self {
        self.params.combination = comb;
        self
    }

    /// Internal helper to update the active vector query with current overrides.
    fn update_active_query(&mut self) {
        if let Some(ref mut query) = self.vector_query {
            Self::apply_overrides_to_query(&self.vector_overrides, query);
        }
    }

    pub(crate) fn build_query_from_vector(vector: Vector, limit: usize) -> VectorSearchRequest {
        let stored = StoredVector::from(&vector);
        let query_vector = QueryVector {
            vector: stored,
            weight: 1.0,
        };
        VectorSearchRequest {
            query_vectors: vec![query_vector],
            limit,
            ..Default::default()
        }
    }

    pub(crate) fn apply_overrides_to_query(
        overrides: &HybridVectorOptions,
        query: &mut VectorSearchRequest,
    ) {
        if let Some(ref fields) = overrides.fields {
            query.fields = Some(fields.clone());
        }
        if let Some(mode) = overrides.score_mode {
            query.score_mode = mode;
        }
        if let Some(factor) = overrides.overfetch {
            query.overfetch = factor;
        }
        if let Some(ref filter) = overrides.filter {
            query.filter = Some(filter.clone());
        }
    }
}

impl Default for HybridSearchRequest {
    fn default() -> Self {
        Self::new("")
    }
}

/// Parameters for hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchParams {
    /// Weight for keyword scores (0.0-1.0).
    pub keyword_weight: f32,
    /// Weight for vector similarity scores (0.0-1.0).
    pub vector_weight: f32,
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Minimum keyword score threshold.
    pub min_keyword_score: f32,
    /// Minimum vector similarity threshold.
    pub min_vector_similarity: f32,
    /// Whether to require both keyword and vector matches.
    pub require_both: bool,
    /// Score normalization strategy.
    pub normalization: ScoreNormalization,
    /// Score combination strategy.
    pub combination: ScoreCombination,
}

impl Default for HybridSearchParams {
    fn default() -> Self {
        Self {
            keyword_weight: 0.5,
            vector_weight: 0.5,
            max_results: 10,
            min_keyword_score: 0.0,
            min_vector_similarity: 0.0,
            require_both: false,
            normalization: ScoreNormalization::MinMax,
            combination: ScoreCombination::WeightedSum,
        }
    }
}

/// Score normalization strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScoreNormalization {
    /// No normalization.
    None,
    /// Min-max normalization to [0, 1].
    #[default]
    MinMax,
    /// Z-score normalization.
    ZScore,
    /// Rank-based normalization.
    Rank,
}

/// Score combination strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScoreCombination {
    /// Weighted sum of scores.
    #[default]
    WeightedSum,
    /// Reciprocal rank fusion.
    Rrf,
    /// Harmonic mean of scores.
    HarmonicMean,
}

/// Result of a hybrid search operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResults {
    /// The search results.
    pub results: Vec<HybridSearchResult>,
    /// Total number of documents searched.
    pub total_searched: usize,
    /// Number of keyword matches.
    pub keyword_matches: usize,
    /// Number of vector matches.
    pub vector_matches: usize,
    /// Time taken for the search in milliseconds.
    pub took_ms: u64,
    /// The original query text.
    pub query_text: String,
}

impl HybridSearchResults {
    /// Create a new HybridSearchResults.
    pub fn new(
        results: Vec<HybridSearchResult>,
        total_searched: usize,
        keyword_matches: usize,
        vector_matches: usize,
        took_ms: u64,
        query_text: String,
    ) -> Self {
        Self {
            results,
            total_searched,
            keyword_matches,
            vector_matches,
            took_ms,
            query_text,
        }
    }
}

/// A single result from hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult {
    /// Document ID.
    pub doc_id: u64,
    /// Combined hybrid score.
    pub hybrid_score: f32,
    /// Keyword search score (if matched).
    pub keyword_score: Option<f32>,
    /// Vector similarity score (if matched).
    pub vector_similarity: Option<f32>,
    /// Per-field vector hits contributing to this document's score.
    pub vector_field_hits: Vec<FieldHit>,
    /// Optional document content.
    pub document: Option<HashMap<String, String>>,
}

impl HybridSearchResult {
    /// Create a new HybridSearchResult.
    pub fn new(doc_id: u64, hybrid_score: f32) -> Self {
        Self {
            doc_id,
            hybrid_score,
            keyword_score: None,
            vector_similarity: None,
            vector_field_hits: Vec::new(),
            document: None,
        }
    }

    /// Set the keyword score.
    pub fn with_keyword_score(mut self, score: f32) -> Self {
        self.keyword_score = Some(score);
        self
    }

    /// Set the vector similarity.
    pub fn with_vector_similarity(mut self, similarity: f32) -> Self {
        self.vector_similarity = Some(similarity);
        self
    }
}
