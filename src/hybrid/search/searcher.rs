//! Hybrid search execution, requests, and results.
//!
//! This module provides all the structures and functionality for hybrid search,
//! combining lexical (keyword) and vector (semantic) search.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::lexical::search::searcher::LexicalSearchRequest;
use crate::vector::core::document::StoredVector;
use crate::vector::core::vector::Vector;
use crate::vector::engine::request::{QueryVector, VectorSearchRequest};
use crate::vector::field::FieldHit;

/// Hybrid search request combining text query, optional vector, and search parameters.
///
/// This is the main structure for executing hybrid searches, similar to
/// `LexicalSearchRequest` and `VectorEngineSearchRequest`.
///
/// # Examples
///
/// ```
/// use sarissa::hybrid::search::searcher::HybridSearchRequest;
/// use sarissa::vector::core::vector::Vector;
///
/// // Create an empty request and add components
/// let request = HybridSearchRequest::new()
///     .with_text("rust programming")
///     .keyword_weight(0.6)
///     .vector_weight(0.4)
///     .max_results(20);
/// ```
#[derive(Debug, Clone)]
pub struct HybridSearchRequest {
    /// Lexical search request.
    pub lexical_request: Option<LexicalSearchRequest>,
    /// Vector search request.
    pub vector_request: Option<VectorSearchRequest>,
    /// Hybrid search parameters (merging strategy etc).
    pub params: HybridSearchParams,
}

impl HybridSearchRequest {
    /// Create a new hybrid search request.
    pub fn new() -> Self {
        Self {
            lexical_request: None,
            vector_request: None,
            params: HybridSearchParams::default(),
        }
    }

    /// Set or update the lexical request.
    pub fn with_lexical_request(mut self, req: LexicalSearchRequest) -> Self {
        self.lexical_request = Some(req);
        self
    }

    /// Set or update the vector request.
    pub fn with_vector_request(mut self, req: VectorSearchRequest) -> Self {
        self.vector_request = Some(req);
        self
    }

    /// Helper to set text query (creates default LexicalSearchRequest).
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        let text = text.into();
        self.lexical_request = Some(LexicalSearchRequest::new(text));
        self
    }

    /// Helper to set vector (creates default VectorSearchRequest).
    pub fn with_vector(mut self, vector: Vector) -> Self {
        let mut req = VectorSearchRequest::default();
        let stored = StoredVector::from(&vector);
        let q_vec = QueryVector {
            vector: stored,
            weight: 1.0,
            fields: None, // Will be handled by engine default
        };
        req.query_vectors.push(q_vec);
        self.vector_request = Some(req);
        self
    }

    /// Set hybrid params.
    pub fn with_params(mut self, params: HybridSearchParams) -> Self {
        self.params = params;
        self
    }
}

impl Default for HybridSearchRequest {
    fn default() -> Self {
        Self::new()
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
