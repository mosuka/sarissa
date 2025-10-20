//! Parallel vector search module for high-performance similarity searches.
//!
//! This module provides parallel implementations for vector search operations:
//! - Batch query processing with load balancing
//! - Multi-threaded search execution
//! - Result merging and ranking
//! - Advanced caching strategies

pub mod executor;
pub mod merger;

use std::sync::Arc;

use rayon::ThreadPool;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::search::VectorSearchEngineConfig;

use crate::parallel_vector_search::executor::ParallelVectorSearchExecutor;
use crate::parallel_vector_search::merger::MergeStrategy as SearchMergeStrategy;

/// Configuration for parallel vector search operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelVectorSearchConfig {
    /// Number of worker threads.
    pub num_threads: usize,
    /// Batch size for query processing.
    pub batch_size: usize,
    /// Maximum concurrent searches.
    pub max_concurrent_searches: usize,
    /// Enable result caching across searches.
    pub enable_result_caching: bool,
    /// Cache size limit.
    pub cache_size_limit: usize,
    /// Merge strategy for combining results.
    pub merge_strategy: SearchMergeStrategy,
    /// Base configuration for individual search engines.
    pub base_config: VectorSearchEngineConfig,
    /// Load balancing strategy.
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for ParallelVectorSearchConfig {
    fn default() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            batch_size: 100,
            max_concurrent_searches: 16,
            enable_result_caching: true,
            cache_size_limit: 10000,
            merge_strategy: SearchMergeStrategy::ScoreBased,
            base_config: VectorSearchEngineConfig::default(),
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Load balancing strategies for distributing search tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution.
    RoundRobin,
    /// Assign to least loaded worker.
    LeastLoaded,
    /// Random assignment.
    Random,
    /// Assign based on query characteristics.
    QueryAware,
}

/// Factory for creating parallel vector search executors.
pub struct ParallelVectorSearchExecutorFactory;

impl ParallelVectorSearchExecutorFactory {
    /// Create a new parallel vector search executor.
    pub fn create_executor(
        config: ParallelVectorSearchConfig,
    ) -> Result<ParallelVectorSearchExecutor> {
        ParallelVectorSearchExecutor::new(config)
    }

    /// Create an executor with a specific thread pool.
    pub fn create_executor_with_pool(
        config: ParallelVectorSearchConfig,
        thread_pool: Arc<ThreadPool>,
    ) -> Result<ParallelVectorSearchExecutor> {
        ParallelVectorSearchExecutor::with_thread_pool(config, thread_pool)
    }
}

/// Statistics for parallel vector search operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelSearchStats {
    /// Total searches executed.
    pub total_searches: u64,
    /// Total batch searches executed.
    pub total_batch_searches: u64,
    /// Average search time in milliseconds.
    pub avg_search_time_ms: f64,
    /// Average batch processing time in milliseconds.
    pub avg_batch_time_ms: f64,
    /// Cache hit rate.
    pub cache_hit_rate: f32,
    /// Parallel efficiency ratio (0.0 to 1.0).
    pub parallel_efficiency: f32,
    /// Average queue length.
    pub avg_queue_length: f32,
    /// Total vectors searched.
    pub total_vectors_searched: u64,
    /// Memory usage in bytes.
    pub memory_usage_bytes: usize,
}

impl Default for ParallelSearchStats {
    fn default() -> Self {
        Self {
            total_searches: 0,
            total_batch_searches: 0,
            avg_search_time_ms: 0.0,
            avg_batch_time_ms: 0.0,
            cache_hit_rate: 0.0,
            parallel_efficiency: 0.0,
            avg_queue_length: 0.0,
            total_vectors_searched: 0,
            memory_usage_bytes: 0,
        }
    }
}
