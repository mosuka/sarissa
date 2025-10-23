//! Vector search module for executing similarity searches on vector indexes.
//!
//! This module handles all vector search operations:
//! - Approximate and exact nearest neighbor search
//! - Hybrid search combining keyword and vector search
//! - Advanced similarity metrics and ranking
//! - Search result processing and filtering

pub mod hybrid;
pub mod ranking;
pub mod searcher;
pub mod similarity;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::Vector;
use crate::vector::types::{VectorSearchConfig, VectorSearchResults};

/// Advanced search configuration with multiple search strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSearchConfig {
    /// Base search configuration.
    pub base_config: VectorSearchConfig,
    /// Search strategy to use.
    pub search_strategy: SearchStrategy,
    /// Reranking configuration.
    pub reranking: Option<ranking::RankingConfig>,
    /// Post-processing filters.
    pub filters: Vec<SearchFilter>,
    /// Search result explanation.
    pub explain: bool,
}

/// Different search strategies available.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Exact search (brute force).
    Exact,
    /// Approximate search with quality/speed tradeoff.
    Approximate {
        /// Quality vs speed parameter (0.0 = fastest, 1.0 = highest quality).
        quality: f32,
    },
    /// Multi-stage search with coarse-to-fine refinement.
    MultiStage {
        /// Number of candidates in coarse stage.
        coarse_candidates: usize,
        /// Refinement factor.
        refinement_factor: f32,
    },
    /// Adaptive search that adjusts based on query characteristics.
    Adaptive,
}

/// Search result filters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchFilter {
    /// Filter by similarity threshold.
    SimilarityThreshold(f32),
    /// Filter by metadata condition.
    MetadataFilter {
        key: String,
        condition: FilterCondition,
    },
    /// Filter by document ID range.
    DocIdRange {
        min_doc_id: Option<u64>,
        max_doc_id: Option<u64>,
    },
    /// Custom filter function.
    Custom(String), // Serialized filter logic
}

/// Filter conditions for metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterCondition {
    Equals(String),
    Contains(String),
    StartsWith(String),
    EndsWith(String),
    MatchesRegex(String),
}

/// Trait for vector searchers.
pub trait VectorSearcher: Send + Sync {
    /// Execute a vector similarity search.
    ///
    /// This method handles all search operations including basic search,
    /// reranking, filtering, and strategy selection based on the config.
    fn search(&self, query: &Vector, config: &VectorSearchConfig) -> Result<VectorSearchResults>;

    /// Execute a batch search for multiple queries.
    fn batch_search(
        &self,
        queries: &[Vector],
        config: &VectorSearchConfig,
    ) -> Result<Vec<VectorSearchResults>> {
        // デフォルト実装: 各クエリを順次実行
        queries
            .iter()
            .map(|query| self.search(query, config))
            .collect()
    }

    /// Get search statistics.
    fn search_stats(&self) -> SearchStats {
        // デフォルト実装: 空の統計
        SearchStats::default()
    }

    /// Warm up the searcher (pre-load data, etc.).
    fn warmup(&mut self) -> Result<()> {
        // デフォルト実装: 何もしない
        Ok(())
    }
}

/// Search execution statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStats {
    /// Total number of searches executed.
    pub total_searches: u64,
    /// Average search time in milliseconds.
    pub avg_search_time_ms: f64,
    /// Cache hit rate (if caching is enabled).
    pub cache_hit_rate: f32,
    /// Number of vectors examined per search on average.
    pub avg_vectors_examined: f64,
    /// Memory usage in bytes.
    pub memory_usage_bytes: usize,
}

impl Default for SearchStats {
    fn default() -> Self {
        Self {
            total_searches: 0,
            avg_search_time_ms: 0.0,
            cache_hit_rate: 0.0,
            avg_vectors_examined: 0.0,
            memory_usage_bytes: 0,
        }
    }
}

/// Search result explanation for debugging and optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchExplanation {
    /// Strategy used for this search.
    pub strategy_used: SearchStrategy,
    /// Number of candidates examined.
    pub candidates_examined: usize,
    /// Search time breakdown.
    pub time_breakdown: SearchTimeBreakdown,
    /// Distance calculations performed.
    pub distance_calculations: usize,
    /// Whether cache was used.
    pub cache_used: bool,
}

/// Breakdown of search time by operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchTimeBreakdown {
    /// Time spent on candidate generation.
    pub candidate_generation_ms: f64,
    /// Time spent on distance calculations.
    pub distance_calculation_ms: f64,
    /// Time spent on result ranking.
    pub ranking_ms: f64,
    /// Time spent on post-processing.
    pub post_processing_ms: f64,
}

/// Enhanced search results with explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainedSearchResults {
    /// Base search results.
    pub results: VectorSearchResults,
    /// Search explanation.
    pub explanation: Option<SearchExplanation>,
}
