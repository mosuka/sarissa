//! Configuration for parallel hybrid search.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for parallel hybrid search engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelHybridSearchConfig {
    /// Number of worker threads for parallel execution.
    pub num_threads: usize,
    
    /// Maximum number of indices to search concurrently.
    pub max_concurrent_indices: usize,
    
    /// Batch size for processing queries.
    pub batch_size: usize,
    
    /// Enable result caching.
    pub enable_result_caching: bool,
    
    /// Cache size limit (number of entries).
    pub cache_size_limit: usize,
    
    /// Timeout for individual index searches.
    pub index_timeout: Duration,
    
    /// Overall timeout for the entire search.
    pub total_timeout: Duration,
    
    /// Keyword search weight (0.0 to 1.0).
    pub keyword_weight: f32,
    
    /// Vector search weight (0.0 to 1.0).
    pub vector_weight: f32,
    
    /// Minimum keyword score threshold.
    pub min_keyword_score: f32,
    
    /// Minimum vector similarity threshold.
    pub min_vector_similarity: f32,
    
    /// Maximum results per index for keyword search.
    pub max_keyword_results_per_index: usize,
    
    /// Maximum results per index for vector search.
    pub max_vector_results_per_index: usize,
    
    /// Final maximum results to return.
    pub max_final_results: usize,
    
    /// Enable query expansion for keyword search.
    pub enable_query_expansion: bool,
    
    /// Enable semantic expansion for vector search.
    pub enable_semantic_expansion: bool,
    
    /// Load balancing strategy for index selection.
    pub load_balancing_strategy: LoadBalancingStrategy,
    
    /// Result merge strategy.
    pub merge_strategy: MergeStrategy,
}

/// Load balancing strategy for distributing searches across indices.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution.
    RoundRobin,
    /// Random distribution.
    Random,
    /// Choose least loaded index.
    LeastLoaded,
    /// Choose based on index characteristics.
    IndexAware,
}

/// Strategy for merging results from different search types.
#[derive(Debug, Clone, Copy, Hash, Serialize, Deserialize, PartialEq)]
pub enum MergeStrategy {
    /// Linear weighted combination.
    LinearCombination,
    /// Reciprocal rank fusion.
    ReciprocalRankFusion,
    /// Maximum score from either search.
    MaxScore,
    /// Multiply scores together.
    ScoreProduct,
    /// Adaptive based on result characteristics.
    Adaptive,
}

impl Default for ParallelHybridSearchConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            max_concurrent_indices: 8,
            batch_size: 100,
            enable_result_caching: true,
            cache_size_limit: 10000,
            index_timeout: Duration::from_secs(5),
            total_timeout: Duration::from_secs(30),
            keyword_weight: 0.5,
            vector_weight: 0.5,
            min_keyword_score: 0.0,
            min_vector_similarity: 0.0,
            max_keyword_results_per_index: 100,
            max_vector_results_per_index: 100,
            max_final_results: 10,
            enable_query_expansion: false,
            enable_semantic_expansion: false,
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            merge_strategy: MergeStrategy::LinearCombination,
        }
    }
}

impl ParallelHybridSearchConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the number of worker threads.
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }
    
    /// Set the weights for keyword and vector search.
    pub fn with_weights(mut self, keyword_weight: f32, vector_weight: f32) -> Self {
        self.keyword_weight = keyword_weight;
        self.vector_weight = vector_weight;
        self
    }
    
    /// Set the merge strategy.
    pub fn with_merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }
    
    /// Set the maximum final results.
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_final_results = max_results;
        self
    }
    
    /// Enable or disable result caching.
    pub fn with_caching(mut self, enable: bool) -> Self {
        self.enable_result_caching = enable;
        self
    }
    
    /// Validate the configuration.
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::SarissaError;
        
        if self.num_threads == 0 {
            return Err(SarissaError::invalid_config(
                "Number of threads must be greater than 0".to_string(),
            ));
        }
        
        if self.keyword_weight < 0.0 || self.keyword_weight > 1.0 {
            return Err(SarissaError::invalid_config(
                "Keyword weight must be between 0.0 and 1.0".to_string(),
            ));
        }
        
        if self.vector_weight < 0.0 || self.vector_weight > 1.0 {
            return Err(SarissaError::invalid_config(
                "Vector weight must be between 0.0 and 1.0".to_string(),
            ));
        }
        
        let total_weight = self.keyword_weight + self.vector_weight;
        if (total_weight - 1.0).abs() > 0.001 {
            return Err(SarissaError::invalid_config(
                format!("Keyword and vector weights must sum to 1.0, got {total_weight}"),
            ));
        }
        
        Ok(())
    }
}