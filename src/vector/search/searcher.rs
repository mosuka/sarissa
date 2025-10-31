//! Vector searcher trait and search request/response types.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::Vector;

/// Vector search request combining query vector and configuration.
#[derive(Debug, Clone)]
pub struct VectorSearchRequest {
    /// The query vector.
    pub query: Vector,
    /// Search configuration.
    pub params: VectorSearchParams,
}

impl VectorSearchRequest {
    /// Create a new vector search request.
    pub fn new(query: Vector) -> Self {
        VectorSearchRequest {
            query,
            params: VectorSearchParams::default(),
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
}

/// Configuration for vector search operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchParams {
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
    /// Reranking configuration.
    pub reranking: Option<crate::vector::search::ranking::RankingConfig>,
}

impl Default for VectorSearchParams {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            include_scores: true,
            include_vectors: false,
            timeout_ms: None,
            reranking: None,
        }
    }
}

/// A single vector search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// Document ID.
    pub doc_id: u64,
    /// Similarity score (higher is more similar).
    pub similarity: f32,
    /// Distance score (lower is more similar).
    pub distance: f32,
    /// Optional vector data.
    pub vector: Option<crate::vector::Vector>,
    /// Result metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

/// Collection of vector search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchResults {
    /// Individual search results.
    pub results: Vec<VectorSearchResult>,
    /// Total number of candidates examined.
    pub candidates_examined: usize,
    /// Search execution time in milliseconds.
    pub search_time_ms: f64,
    /// Query metadata.
    pub query_metadata: std::collections::HashMap<String, String>,
}

impl VectorSearchResults {
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
    pub fn best_result(&self) -> Option<&VectorSearchResult> {
        self.results.iter().max_by(|a, b| {
            a.similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl Default for VectorSearchResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for vector searchers.
pub trait VectorSearcher: Send + Sync {
    /// Execute a vector similarity search.
    fn search(&self, request: &VectorSearchRequest) -> Result<VectorSearchResults>;

    /// Warm up the searcher (pre-load data, etc.).
    fn warmup(&mut self) -> Result<()> {
        // デフォルト実装: 何もしない
        Ok(())
    }
}
