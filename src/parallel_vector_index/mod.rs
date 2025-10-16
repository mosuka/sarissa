//! Parallel vector indexing module for high-performance vector index construction.
//!
//! This module provides parallel implementations for building vector indexes:
//! - Segmented parallel construction
//! - Multi-threaded vector processing
//! - Memory-efficient segment merging
//! - Background optimization

pub mod builder;
pub mod executor;
pub mod merger;
pub mod segment;

use std::sync::Arc;

use rayon::ThreadPool;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector_index::VectorIndexBuildConfig;

use crate::parallel_vector_index::builder::ParallelVectorIndexBuilder;
use crate::parallel_vector_index::merger::MergeStrategy;

/// Configuration for parallel vector index construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelVectorIndexConfig {
    /// Number of worker threads.
    pub num_threads: usize,
    /// Maximum vectors per segment.
    pub segment_size: usize,
    /// Minimum segments to trigger merge.
    pub merge_threshold: usize,
    /// Maximum memory usage per segment (bytes).
    pub max_segment_memory: usize,
    /// Base configuration for individual segments.
    pub base_config: VectorIndexBuildConfig,
    /// Enable background optimization.
    pub background_optimization: bool,
    /// Merge strategy for segments.
    pub merge_strategy: MergeStrategy,
}

impl Default for ParallelVectorIndexConfig {
    fn default() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
            segment_size: 10000,
            merge_threshold: 4,
            max_segment_memory: 512 * 1024 * 1024, // 512MB
            base_config: VectorIndexBuildConfig::default(),
            background_optimization: true,
            merge_strategy: MergeStrategy::SizeBased,
        }
    }
}

/// Factory for creating parallel vector index builders.
pub struct ParallelVectorIndexBuilderFactory;

impl ParallelVectorIndexBuilderFactory {
    /// Create a new parallel vector index builder.
    pub fn create_builder(config: ParallelVectorIndexConfig) -> Result<ParallelVectorIndexBuilder> {
        ParallelVectorIndexBuilder::new(config)
    }

    /// Create a builder with specific thread pool.
    pub fn create_builder_with_pool(
        config: ParallelVectorIndexConfig,
        thread_pool: Arc<ThreadPool>,
    ) -> Result<ParallelVectorIndexBuilder> {
        ParallelVectorIndexBuilder::with_thread_pool(config, thread_pool)
    }
}

/// Statistics for parallel vector index construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelIndexStats {
    /// Total vectors processed.
    pub total_vectors: usize,
    /// Number of segments created.
    pub segment_count: usize,
    /// Total construction time in milliseconds.
    pub total_time_ms: f64,
    /// Average time per segment in milliseconds.
    pub avg_segment_time_ms: f64,
    /// Memory usage in bytes.
    pub memory_usage_bytes: usize,
    /// Number of merge operations performed.
    pub merge_operations: usize,
    /// Parallel efficiency ratio (0.0 to 1.0).
    pub parallel_efficiency: f32,
}

impl Default for ParallelIndexStats {
    fn default() -> Self {
        Self {
            total_vectors: 0,
            segment_count: 0,
            total_time_ms: 0.0,
            avg_segment_time_ms: 0.0,
            memory_usage_bytes: 0,
            merge_operations: 0,
            parallel_efficiency: 0.0,
        }
    }
}
