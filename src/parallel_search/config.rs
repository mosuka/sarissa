//! Configuration for parallel search operations.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for parallel search engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelSearchConfig {
    /// Maximum number of concurrent search tasks.
    pub max_concurrent_tasks: usize,
    
    /// Default timeout for individual search tasks.
    pub default_timeout: Duration,
    
    /// Maximum number of results to collect from each index.
    pub max_results_per_index: usize,
    
    /// Whether to enable metrics collection.
    pub enable_metrics: bool,
    
    /// Default merge strategy type.
    pub default_merge_strategy: MergeStrategyType,
    
    /// Thread pool size for parallel execution.
    /// If None, uses the number of CPU cores.
    pub thread_pool_size: Option<usize>,
    
    /// Memory limit for result buffering (in bytes).
    pub max_memory_usage: usize,
    
    /// Whether to continue on partial failures.
    pub allow_partial_results: bool,
}

impl Default for ParallelSearchConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: num_cpus::get() * 2,
            default_timeout: Duration::from_secs(30),
            max_results_per_index: 1000,
            enable_metrics: true,
            default_merge_strategy: MergeStrategyType::ScoreBased,
            thread_pool_size: None,
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
            allow_partial_results: true,
        }
    }
}

/// Type of merge strategy to use for combining results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategyType {
    /// Merge based on document scores.
    ScoreBased,
    
    /// Round-robin merge for diversity.
    RoundRobin,
    
    /// Weighted merge based on index importance.
    Weighted,
    
    /// Custom merge strategy.
    Custom,
}

/// Options for a specific search request.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of documents to return.
    pub max_docs: usize,
    
    /// Minimum score threshold.
    pub min_score: Option<f32>,
    
    /// Timeout for this specific search.
    pub timeout: Option<Duration>,
    
    /// Override merge strategy for this search.
    pub merge_strategy: Option<MergeStrategyType>,
    
    /// Whether to load full documents.
    pub load_documents: bool,
    
    /// Whether to collect detailed metrics.
    pub collect_metrics: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_docs: 100,
            min_score: None,
            timeout: None,
            merge_strategy: None,
            load_documents: true,
            collect_metrics: false,
        }
    }
}

impl SearchOptions {
    /// Create a new SearchOptions with the specified max_docs.
    pub fn new(max_docs: usize) -> Self {
        Self {
            max_docs,
            ..Default::default()
        }
    }
    
    /// Set the minimum score threshold.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }
    
    /// Set the timeout for this search.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    
    /// Set the merge strategy for this search.
    pub fn with_merge_strategy(mut self, strategy: MergeStrategyType) -> Self {
        self.merge_strategy = Some(strategy);
        self
    }
    
    /// Set whether to load full documents.
    pub fn with_load_documents(mut self, load: bool) -> Self {
        self.load_documents = load;
        self
    }
    
    /// Set whether to collect metrics.
    pub fn with_metrics(mut self, collect: bool) -> Self {
        self.collect_metrics = collect;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = ParallelSearchConfig::default();
        assert!(config.max_concurrent_tasks > 0);
        assert_eq!(config.default_timeout, Duration::from_secs(30));
        assert_eq!(config.max_results_per_index, 1000);
        assert!(config.enable_metrics);
        assert_eq!(config.default_merge_strategy, MergeStrategyType::ScoreBased);
    }
    
    #[test]
    fn test_search_options_builder() {
        let options = SearchOptions::new(50)
            .with_min_score(0.5)
            .with_timeout(Duration::from_secs(10))
            .with_merge_strategy(MergeStrategyType::Weighted)
            .with_load_documents(false)
            .with_metrics(true);
        
        assert_eq!(options.max_docs, 50);
        assert_eq!(options.min_score, Some(0.5));
        assert_eq!(options.timeout, Some(Duration::from_secs(10)));
        assert_eq!(options.merge_strategy, Some(MergeStrategyType::Weighted));
        assert!(!options.load_documents);
        assert!(options.collect_metrics);
    }
}