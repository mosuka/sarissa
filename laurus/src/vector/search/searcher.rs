//! Vector searcher trait and query/response types.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::core::vector::Vector;

/// Low-level query for a single-vector search against a vector index.
///
/// This type represents a single nearest-neighbor query at the index level,
/// in contrast to the high-level [`VectorSearchRequest`] which can contain
/// multiple query vectors and aggregation settings.
///
/// Naming convention: low-level index operations use "Query" (e.g.,
/// `VectorIndexQuery`, `VectorIndexQueryParams`), while high-level
/// store/engine operations use "Request" (e.g., `VectorSearchRequest`).
#[derive(Debug, Clone)]
pub struct VectorIndexQuery {
    /// The query vector.
    pub query: Vector,
    /// Search configuration.
    pub params: VectorIndexQueryParams,
    /// Optional field name to filter search results.
    /// If None, searches across all fields.
    pub field_name: Option<String>,
}

impl VectorIndexQuery {
    /// Create a new vector search request.
    pub fn new(query: Vector) -> Self {
        VectorIndexQuery {
            query,
            params: VectorIndexQueryParams::default(),
            field_name: None,
        }
    }

    /// Set the number of results to return.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.params.top_k = top_k;
        self
    }

    /// Set minimum similarity threshold.
    pub fn min_similarity(mut self, threshold: f32) -> Self {
        self.params.min_similarity = threshold;
        self
    }

    /// Set whether to include scores in results.
    pub fn include_scores(mut self, include: bool) -> Self {
        self.params.include_scores = include;
        self
    }

    /// Set whether to include vectors in results.
    pub fn include_vectors(mut self, include: bool) -> Self {
        self.params.include_vectors = include;
        self
    }

    /// Set search timeout in milliseconds.
    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        self.params.timeout_ms = Some(timeout);
        self
    }

    /// Set field name to filter search results.
    pub fn field_name(mut self, field_name: String) -> Self {
        self.field_name = Some(field_name);
        self
    }

    /// Set two-stage rerank multiplier (Issue #481 Stage 2 — pre-design).
    ///
    /// Currently the searcher returns
    /// [`crate::error::LaurusError::NotImplemented`] when this is set
    /// to `Some(_)`. The API surface is reserved here so existing
    /// callers can opt in once Stage 2 lands without further proto /
    /// binding revisions.
    pub fn rerank_factor(mut self, factor: usize) -> Self {
        self.params.rerank_factor = Some(factor);
        self
    }
}

/// Configuration for low-level vector index query operations.
///
/// Used with [`VectorIndexQuery`] to configure nearest-neighbor search
/// parameters at the index level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexQueryParams {
    /// Number of results to return.
    pub top_k: usize,
    /// Minimum similarity threshold.
    pub min_similarity: f32,
    /// Whether to return similarity scores.
    pub include_scores: bool,
    /// Whether to include vector data in results.
    pub include_vectors: bool,
    /// Search timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Two-stage rerank multiplier (Issue #481 Stage 2 — pre-design).
    ///
    /// When set to `Some(n)`, the index searcher will first fetch
    /// `top_k * n` candidates via the int8 quantized hot path and then
    /// re-score them against the full f32 vectors before returning
    /// the top `top_k`. This recovers the recall lost to scalar
    /// quantization at modest extra cost.
    ///
    /// **Stage 2 of Issue #481 — currently returns
    /// [`crate::error::LaurusError::NotImplemented`] when set to
    /// `Some(_)`.** `None` (the default) runs the Stage 1
    /// quantized-only search.
    #[serde(default)]
    pub rerank_factor: Option<usize>,
}

impl Default for VectorIndexQueryParams {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            include_scores: true,
            include_vectors: false,
            timeout_ms: None,
            rerank_factor: None,
        }
    }
}

/// A single result from a low-level vector index query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexQueryResult {
    /// Document ID.
    pub doc_id: u64,
    /// Field name of the matched vector.
    pub field_name: String,
    /// Similarity score (higher is more similar).
    pub similarity: f32,
    /// Distance score (lower is more similar).
    pub distance: f32,
    /// Optional vector data.
    pub vector: Option<Vector>,
}

/// Collection of results from a low-level vector index query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexQueryResults {
    /// Individual search results.
    pub results: Vec<VectorIndexQueryResult>,
    /// Total number of candidates examined.
    pub candidates_examined: usize,
    /// Search execution time in milliseconds.
    pub search_time_ms: f64,
    /// Query metadata.
    pub query_metadata: std::collections::HashMap<String, String>,
}

impl VectorIndexQueryResults {
    /// Create new empty search results.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            candidates_examined: 0,
            search_time_ms: 0.0,
            query_metadata: std::collections::HashMap::new(),
        }
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Sort results by similarity (descending).
    pub fn sort_by_similarity(&mut self) {
        self.results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sort results by distance (ascending).
    pub fn sort_by_distance(&mut self) {
        self.results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Take the top k results.
    pub fn take_top_k(&mut self, k: usize) {
        if self.results.len() > k {
            self.results.truncate(k);
        }
    }

    /// Filter results by minimum similarity.
    pub fn filter_by_similarity(&mut self, min_similarity: f32) {
        self.results
            .retain(|result| result.similarity >= min_similarity);
    }

    /// Get the best (highest similarity) result.
    pub fn best_result(&self) -> Option<&VectorIndexQueryResult> {
        self.results.iter().max_by(|a, b| {
            a.similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl Default for VectorIndexQueryResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for vector searchers.
pub trait VectorIndexSearcher: Send + Sync + std::fmt::Debug {
    /// Execute a vector similarity search.
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults>;

    /// Count the number of vectors matching the query.
    fn count(&self, request: VectorIndexQuery) -> Result<u64>;

    /// Warm up the searcher (pre-load data, etc.).
    fn warmup(&mut self) -> Result<()> {
        // No-op by default. Implementations can override this method to perform
        // any necessary warm-up steps, such as loading index data into memory.
        Ok(())
    }
}

// ── High-level search request types ──────────────────────────────────────────

/// How a vector search query is specified.
///
/// Mirrors [`LexicalSearchQuery`](crate::lexical::search::searcher::LexicalSearchQuery)
/// for symmetry:
///
/// | | Lexical | Vector |
/// |---|---|---|
/// | Deferred resolution | [`Dsl(String)`](crate::lexical::search::searcher::LexicalSearchQuery::Dsl) | [`Payloads`](Self::Payloads) |
/// | Pre-built | [`Obj(Box<dyn Query>)`](crate::lexical::search::searcher::LexicalSearchQuery::Obj) | [`Vectors`](Self::Vectors) |
#[derive(Debug, Clone)]
pub enum VectorSearchQuery {
    /// Raw payloads (text, bytes, etc.) to be embedded into vectors at
    /// search time by the engine's configured embedder.
    Payloads(Vec<crate::vector::store::request::QueryPayload>),

    /// Pre-embedded query vectors, ready for nearest-neighbor search.
    Vectors(Vec<crate::vector::store::request::QueryVector>),
}

fn default_query_limit() -> usize {
    10
}

fn default_overfetch() -> f32 {
    1.0
}

/// Parameters for vector search operations.
///
/// Analogous to
/// [`LexicalSearchParams`](crate::lexical::search::searcher::LexicalSearchParams),
/// this struct groups all configuration knobs for a vector search independently
/// of the query specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchParams {
    /// Fields to search in.
    #[serde(default)]
    pub fields: Option<Vec<crate::vector::store::request::FieldSelector>>,
    /// Maximum number of results to return.
    #[serde(default = "default_query_limit")]
    pub limit: usize,
    /// How to combine scores from multiple query vectors.
    #[serde(default)]
    pub score_mode: crate::vector::store::request::VectorScoreMode,
    /// Overfetch factor for better result quality.
    #[serde(default = "default_overfetch")]
    pub overfetch: f32,
    /// Minimum score threshold. Results below this score are filtered out.
    #[serde(default)]
    pub min_score: f32,
    /// List of allowed document IDs (for internal use by Engine filtering).
    #[serde(skip)]
    pub allowed_ids: Option<Vec<u64>>,
    /// Optional Stage 2 rerank factor (Issue #481).
    ///
    /// When `Some(factor)`, the HNSW searcher widens the int8 candidate
    /// fetch to `top_k * factor` and rescores against the LRS1 sidecar's
    /// f32 vectors. Honored only on HNSW fields with `rerank_storage`
    /// configured at index time; otherwise silently ignored.
    #[serde(default)]
    pub rerank_factor: Option<usize>,
}

impl Default for VectorSearchParams {
    fn default() -> Self {
        Self {
            fields: None,
            limit: default_query_limit(),
            score_mode: crate::vector::store::request::VectorScoreMode::default(),
            overfetch: default_overfetch(),
            min_score: 0.0,
            allowed_ids: None,
            rerank_factor: None,
        }
    }
}

/// Request model for collection-level vector search.
///
/// Mirrors
/// [`LexicalSearchRequest`](crate::lexical::search::searcher::LexicalSearchRequest)
/// structure: a query enum paired with a params struct.
#[derive(Debug, Clone)]
pub struct VectorSearchRequest {
    /// The query to execute.
    pub query: VectorSearchQuery,
    /// Search configuration.
    pub params: VectorSearchParams,
}

impl Default for VectorSearchRequest {
    fn default() -> Self {
        Self {
            query: VectorSearchQuery::Vectors(Vec::new()),
            params: VectorSearchParams::default(),
        }
    }
}

// ── High-level searcher trait ────────────────────────────────────────────────

/// Trait for high-level vector search implementations.
///
/// This trait defines the interface for executing searches against vector indexes,
/// analogous to [`crate::lexical::search::searcher::LexicalSearcher`] for lexical search.
///
/// Unlike [`VectorIndexSearcher`] which operates at the low-level (single vector queries),
/// `VectorSearcher` handles high-level search requests with multiple query vectors,
/// field selection, filters, and score aggregation.
pub trait VectorSearcher: Send + Sync + std::fmt::Debug {
    /// Execute a search with the given request.
    ///
    /// This method processes a high-level search request that may contain
    /// multiple query vectors across different fields, applies filters,
    /// and aggregates scores according to the specified score mode.
    fn search(
        &self,
        request: &VectorSearchRequest,
    ) -> crate::error::Result<crate::vector::store::response::VectorSearchResults>;

    /// Count the number of matching documents for a request.
    ///
    /// Returns the number of documents that match the given search request,
    /// applying the min_score threshold if specified in the request.
    fn count(&self, request: &VectorSearchRequest) -> crate::error::Result<u64>;
}
