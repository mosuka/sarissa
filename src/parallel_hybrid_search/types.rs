//! Types for parallel hybrid search.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::query::SearchHit;

/// Result from parallel hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelHybridSearchResult {
    /// Document ID.
    pub doc_id: u64,

    /// Combined score from keyword and vector search.
    pub combined_score: f32,

    /// Keyword search score (if available).
    pub keyword_score: Option<f32>,

    /// Vector search similarity (if available).
    pub vector_similarity: Option<f32>,

    /// Rank in keyword results.
    pub keyword_rank: Option<usize>,

    /// Rank in vector results.
    pub vector_rank: Option<usize>,

    /// Index ID this result came from.
    pub index_id: String,

    /// Document fields.
    pub fields: HashMap<String, String>,

    /// Explanation of score calculation.
    pub explanation: Option<ScoreExplanation>,
}

/// Explanation of how the combined score was calculated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreExplanation {
    /// Description of the scoring method.
    pub method: String,

    /// Keyword contribution to final score.
    pub keyword_contribution: f32,

    /// Vector contribution to final score.
    pub vector_contribution: f32,

    /// Additional details.
    pub details: HashMap<String, String>,
}

/// Results from parallel hybrid search execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelHybridSearchResults {
    /// Merged and ranked results.
    pub results: Vec<ParallelHybridSearchResult>,

    /// Total keyword matches across all indices.
    pub total_keyword_matches: u64,

    /// Total vector matches across all indices.
    pub total_vector_matches: u64,

    /// Number of indices searched.
    pub indices_searched: usize,

    /// Search execution time in milliseconds.
    pub search_time_ms: f64,

    /// Breakdown of search times.
    pub time_breakdown: SearchTimeBreakdown,

    /// Cache statistics.
    pub cache_stats: CacheStats,

    /// Per-index statistics.
    pub index_stats: Vec<IndexSearchStats>,
}

/// Breakdown of search execution times.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchTimeBreakdown {
    /// Time spent on keyword search.
    pub keyword_search_ms: f64,

    /// Time spent on vector search.
    pub vector_search_ms: f64,

    /// Time spent merging results.
    pub merge_ms: f64,

    /// Time spent on query expansion.
    pub expansion_ms: f64,

    /// Time spent ranking results.
    pub ranking_ms: f64,
}

/// Cache statistics for the search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cache hits for keyword search.
    pub keyword_hits: usize,

    /// Number of cache misses for keyword search.
    pub keyword_misses: usize,

    /// Number of cache hits for vector search.
    pub vector_hits: usize,

    /// Number of cache misses for vector search.
    pub vector_misses: usize,
}

/// Statistics for a single index search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSearchStats {
    /// Index identifier.
    pub index_id: String,

    /// Number of keyword matches.
    pub keyword_matches: u64,

    /// Number of vector matches.
    pub vector_matches: u64,

    /// Time spent searching this index.
    pub search_time_ms: f64,

    /// Whether the search timed out.
    pub timed_out: bool,

    /// Error message if search failed.
    pub error: Option<String>,
}

/// Task for parallel hybrid search execution.
#[derive(Debug)]
pub struct HybridSearchTask {
    /// Task identifier.
    pub task_id: usize,

    /// Index identifier.
    pub index_id: String,

    /// Query text.
    pub query_text: String,

    /// Keyword query.
    pub keyword_query: Box<dyn crate::query::Query>,

    /// Query vector (if available).
    pub query_vector: Option<crate::vector::Vector>,
}

/// Result from a hybrid search task.
#[derive(Debug)]
pub struct HybridSearchTaskResult {
    /// Task identifier.
    pub task_id: usize,

    /// Index identifier.
    pub index_id: String,

    /// Keyword search results.
    pub keyword_results: Option<Vec<SearchHit>>,

    /// Vector search results.
    pub vector_results: Option<Vec<crate::vector::types::VectorSearchResult>>,

    /// Execution time.
    pub execution_time_ms: f64,

    /// Error if the task failed.
    pub error: Option<crate::error::SarissaError>,
}

impl ParallelHybridSearchResults {
    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Get the top result.
    pub fn top_result(&self) -> Option<&ParallelHybridSearchResult> {
        self.results.first()
    }

    /// Get cache hit rate.
    pub fn cache_hit_rate(&self) -> f32 {
        let total_queries = (self.cache_stats.keyword_hits
            + self.cache_stats.keyword_misses
            + self.cache_stats.vector_hits
            + self.cache_stats.vector_misses) as f32;
        if total_queries > 0.0 {
            let hits = (self.cache_stats.keyword_hits + self.cache_stats.vector_hits) as f32;
            hits / total_queries
        } else {
            0.0
        }
    }
}
