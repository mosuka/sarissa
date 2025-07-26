//! Base vector searcher configuration and utilities.

use crate::error::Result;
use crate::vector::types::{VectorSearchConfig, VectorSearchResults};
use serde::{Deserialize, Serialize};

/// Configuration for vector searchers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearcherConfig {
    /// Search algorithm to use.
    pub algorithm: SearchAlgorithm,
    /// Maximum number of candidates to examine.
    pub max_candidates: usize,
    /// Search quality vs speed tradeoff (0.0 = fastest, 1.0 = best quality).
    pub quality: f32,
    /// Enable parallel search.
    pub parallel_search: bool,
    /// Search timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

impl Default for VectorSearcherConfig {
    fn default() -> Self {
        Self {
            algorithm: SearchAlgorithm::Auto,
            max_candidates: 1000,
            quality: 0.8,
            parallel_search: true,
            timeout_ms: Some(5000),
        }
    }
}

/// Available search algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchAlgorithm {
    /// Automatically select best algorithm.
    Auto,
    /// Exact brute force search.
    Exact,
    /// HNSW approximate search.
    HNSW,
    /// IVF approximate search.
    IVF,
}

/// Search performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Search execution time in milliseconds.
    pub search_time_ms: f64,
    /// Number of distance calculations performed.
    pub distance_calculations: usize,
    /// Number of candidates examined.
    pub candidates_examined: usize,
    /// Memory usage during search.
    pub memory_usage_bytes: usize,
    /// Cache hit rate (if applicable).
    pub cache_hit_rate: f32,
}

impl Default for SearchMetrics {
    fn default() -> Self {
        Self {
            search_time_ms: 0.0,
            distance_calculations: 0,
            candidates_examined: 0,
            memory_usage_bytes: 0,
            cache_hit_rate: 0.0,
        }
    }
}

/// Utility functions for vector search operations.
pub mod utils {
    use super::*;

    /// Validate search configuration.
    pub fn validate_search_config(config: &VectorSearchConfig) -> Result<()> {
        if config.top_k == 0 {
            return Err(crate::error::SarissaError::InvalidOperation(
                "top_k must be greater than 0".to_string(),
            ));
        }

        if config.min_similarity < 0.0 || config.min_similarity > 1.0 {
            return Err(crate::error::SarissaError::InvalidOperation(
                "min_similarity must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    /// Merge search results from multiple sources.
    pub fn merge_search_results(
        results_list: Vec<VectorSearchResults>,
        final_top_k: usize,
    ) -> Result<VectorSearchResults> {
        if results_list.is_empty() {
            return Ok(VectorSearchResults::new());
        }

        let mut merged = VectorSearchResults::new();

        // Collect all results
        for mut results in results_list {
            merged.results.append(&mut results.results);
            merged.candidates_examined += results.candidates_examined;
            merged.search_time_ms = merged.search_time_ms.max(results.search_time_ms);
        }

        // Sort and take top k
        merged.sort_by_similarity();
        merged.take_top_k(final_top_k);

        Ok(merged)
    }

    /// Calculate recall between two result sets.
    pub fn calculate_recall(ground_truth: &[u64], retrieved: &[u64]) -> f32 {
        if ground_truth.is_empty() {
            return if retrieved.is_empty() { 1.0 } else { 0.0 };
        }

        let truth_set: std::collections::HashSet<_> = ground_truth.iter().collect();
        let retrieved_set: std::collections::HashSet<_> = retrieved.iter().collect();

        let intersection_count = truth_set.intersection(&retrieved_set).count();
        intersection_count as f32 / ground_truth.len() as f32
    }

    /// Calculate precision between two result sets.
    pub fn calculate_precision(ground_truth: &[u64], retrieved: &[u64]) -> f32 {
        if retrieved.is_empty() {
            return if ground_truth.is_empty() { 1.0 } else { 0.0 };
        }

        let truth_set: std::collections::HashSet<_> = ground_truth.iter().collect();
        let retrieved_set: std::collections::HashSet<_> = retrieved.iter().collect();

        let intersection_count = truth_set.intersection(&retrieved_set).count();
        intersection_count as f32 / retrieved.len() as f32
    }
}
