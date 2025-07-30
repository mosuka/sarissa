//! Parallel execution engine for vector index tasks.

use std::sync::Arc;
use std::time::Instant;

use rayon::ThreadPool;

use super::{SegmentMetadata, VectorIndexSegment};

use crate::error::{Result, SarissaError};
use crate::vector::Vector;
use crate::vector_index::{VectorIndexBuildConfig, VectorIndexBuilderFactory};

/// Task for parallel vector index construction.
#[derive(Debug, Clone)]
pub struct IndexTask {
    /// Unique segment identifier.
    pub segment_id: usize,
    /// Vectors to index in this segment.
    pub vectors: Vec<(u64, Vector)>,
    /// Configuration for this segment.
    pub config: VectorIndexBuildConfig,
}

/// Result of an index task execution.
pub struct IndexTaskResult {
    /// Task identifier.
    pub task_id: usize,
    /// Execution result.
    pub result: Result<VectorIndexSegment>,
    /// Execution time in milliseconds.
    pub execution_time_ms: f64,
    /// Memory usage in bytes.
    pub memory_usage_bytes: usize,
}

/// Parallel executor for vector index construction tasks.
pub struct ParallelIndexExecutor {
    thread_pool: Arc<ThreadPool>,
}

impl ParallelIndexExecutor {
    /// Create a new parallel index executor.
    pub fn new(thread_pool: Arc<ThreadPool>) -> Result<Self> {
        Ok(Self { thread_pool })
    }

    /// Execute multiple index tasks in parallel.
    pub fn execute_parallel(&self, tasks: Vec<IndexTask>) -> Result<Vec<IndexTaskResult>> {
        let task_count = tasks.len();

        // Use scope to ensure all tasks complete before returning
        let results_arc = Arc::new(std::sync::Mutex::new(Vec::with_capacity(task_count)));

        self.thread_pool.scope(|scope| {
            for task in tasks {
                let results_clone = Arc::clone(&results_arc);
                scope.spawn(move |_| {
                    let result = self.execute_task(task);
                    results_clone.lock().unwrap().push(result);
                });
            }
        });

        // Extract results from Arc<Mutex<Vec<_>>>
        let results = Arc::try_unwrap(results_arc)
            .map_err(|_| {
                SarissaError::InvalidOperation("Failed to unwrap results Arc".to_string())
            })?
            .into_inner()
            .map_err(|_| {
                SarissaError::InvalidOperation("Failed to unwrap results Mutex".to_string())
            })?;

        Ok(results)
    }

    /// Execute a single index task.
    fn execute_task(&self, task: IndexTask) -> IndexTaskResult {
        let start_time = Instant::now();
        let task_id = task.segment_id;

        let result = self.build_segment(task);
        let execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        let memory_usage_bytes = match &result {
            Ok(segment) => segment.memory_usage(),
            Err(_) => 0,
        };

        IndexTaskResult {
            task_id,
            result,
            execution_time_ms,
            memory_usage_bytes,
        }
    }

    /// Build a single segment from the task.
    fn build_segment(&self, task: IndexTask) -> Result<VectorIndexSegment> {
        // Create a builder for this segment
        let mut builder = VectorIndexBuilderFactory::create_builder(task.config.clone())?;

        // Build the segment
        builder.build(task.vectors.clone())?;
        builder.finalize()?;
        builder.optimize()?;

        // Create segment metadata
        let metadata = SegmentMetadata {
            segment_id: task.segment_id,
            vector_count: task.vectors.len(),
            dimension: task.config.dimension,
            index_type: task.config.index_type,
            distance_metric: task.config.distance_metric,
            memory_usage_bytes: builder.estimated_memory_usage(),
            created_at: std::time::SystemTime::now(),
        };

        // Create the segment
        VectorIndexSegment::new(metadata, builder)
    }

    /// Execute tasks with custom parallelism level.
    pub fn execute_with_parallelism(
        &self,
        tasks: Vec<IndexTask>,
        max_parallel: usize,
    ) -> Result<Vec<IndexTaskResult>> {
        let chunk_size = max_parallel;
        let mut all_results = Vec::new();

        // Process tasks in chunks to limit parallelism
        for task_chunk in tasks.chunks(chunk_size) {
            let chunk_results = self.execute_parallel(task_chunk.to_vec())?;
            all_results.extend(chunk_results);
        }

        Ok(all_results)
    }

    /// Execute tasks with progress callback.
    pub fn execute_with_progress<F>(
        &self,
        tasks: Vec<IndexTask>,
        mut progress_callback: F,
    ) -> Result<Vec<IndexTaskResult>>
    where
        F: FnMut(usize, usize) + Send + Sync,
    {
        let total_tasks = tasks.len();
        let mut completed = 0;
        let mut results = Vec::with_capacity(total_tasks);

        // Process tasks one by one to track progress
        for task in tasks {
            let task_results = self.execute_parallel(vec![task])?;
            results.extend(task_results);

            completed += 1;
            progress_callback(completed, total_tasks);
        }

        Ok(results)
    }

    /// Get thread pool statistics.
    pub fn thread_pool_stats(&self) -> ThreadPoolStats {
        ThreadPoolStats {
            num_threads: self.thread_pool.current_num_threads(),
            active_threads: 0, // Rayon doesn't expose this directly
        }
    }
}

/// Statistics for the thread pool.
#[derive(Debug, Clone)]
pub struct ThreadPoolStats {
    /// Number of threads in the pool.
    pub num_threads: usize,
    /// Number of currently active threads.
    pub active_threads: usize,
}
