//! Parallel vector index builder implementation.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::ThreadPool;

use super::executor::{IndexTask, ParallelIndexExecutor};
use super::merger::SegmentMerger;
use super::segment::VectorIndexSegment;
use super::{ParallelIndexStats, ParallelVectorIndexConfig};
use crate::error::{Result, SageError};
use crate::vector::Vector;
use crate::vector::index::VectorIndexBuilder;

/// Parallel vector index builder for high-performance construction.
pub struct ParallelVectorIndexBuilder {
    config: ParallelVectorIndexConfig,
    _thread_pool: Arc<ThreadPool>,
    executor: ParallelIndexExecutor,
    segments: Arc<Mutex<Vec<VectorIndexSegment>>>,
    merger: SegmentMerger,
    stats: Arc<Mutex<ParallelIndexStats>>,
    is_finalized: bool,
}

impl ParallelVectorIndexBuilder {
    /// Create a new parallel vector index builder.
    pub fn new(config: ParallelVectorIndexConfig) -> Result<Self> {
        let thread_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_threads)
                .build()
                .map_err(|e| {
                    SageError::InvalidOperation(format!("Failed to create thread pool: {e}"))
                })?,
        );

        Self::with_thread_pool(config, thread_pool)
    }

    /// Create a builder with an existing thread pool.
    pub fn with_thread_pool(
        config: ParallelVectorIndexConfig,
        thread_pool: Arc<ThreadPool>,
    ) -> Result<Self> {
        let executor = ParallelIndexExecutor::new(thread_pool.clone())?;
        let merger = SegmentMerger::new(config.merge_strategy)?;
        let segments = Arc::new(Mutex::new(Vec::new()));
        let stats = Arc::new(Mutex::new(ParallelIndexStats::default()));

        Ok(Self {
            config,
            _thread_pool: thread_pool,
            executor,
            segments,
            merger,
            stats,
            is_finalized: false,
        })
    }

    /// Build index from vectors in parallel segments.
    pub fn build_parallel(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(SageError::InvalidOperation(
                "Cannot build on finalized index".to_string(),
            ));
        }

        let start_time = Instant::now();
        let total_vectors = vectors.len();

        // Split vectors into segments
        let segments_data = self.create_segments(vectors)?;

        // Build segments in parallel
        let tasks = self.create_build_tasks(segments_data)?;
        let segment_results = self.executor.execute_parallel(tasks)?;

        // Store completed segments
        {
            let mut segments = self.segments.lock().unwrap();
            for result in segment_results {
                if let Ok(segment) = result.result {
                    segments.push(segment);
                }
            }
        }

        // Trigger merge if necessary
        if self.should_merge()? {
            self.merge_segments()?;
        }

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_vectors += total_vectors;
            stats.total_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
            stats.segment_count = self.segments.lock().unwrap().len();
        }

        Ok(())
    }

    /// Add vectors incrementally.
    pub fn add_vectors_parallel(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()> {
        if self.is_finalized {
            return Err(SageError::InvalidOperation(
                "Cannot add vectors to finalized index".to_string(),
            ));
        }

        // For incremental addition, we can build smaller segments
        let mut small_config = self.config.clone();
        small_config.segment_size = small_config.segment_size.min(vectors.len());

        self.build_parallel(vectors)
    }

    /// Finalize the parallel index construction.
    pub fn finalize_parallel(&mut self) -> Result<()> {
        if self.is_finalized {
            return Ok(());
        }

        // Final merge of all segments
        self.merge_all_segments()?;

        // Optimize final segments
        if self.config.background_optimization {
            self.optimize_segments()?;
        }

        self.is_finalized = true;
        Ok(())
    }

    /// Get construction statistics.
    pub fn stats(&self) -> ParallelIndexStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get current progress (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        let stats = self.stats.lock().unwrap();
        if self.is_finalized {
            1.0
        } else if stats.total_vectors == 0 {
            0.0
        } else {
            // Estimate progress based on segments built
            let segments_count = self.segments.lock().unwrap().len();
            let estimated_final_segments = (stats.total_vectors / self.config.segment_size).max(1);
            (segments_count as f32 / estimated_final_segments as f32).min(0.99)
        }
    }

    /// Get estimated memory usage.
    pub fn estimated_memory_usage(&self) -> usize {
        let segments = self.segments.lock().unwrap();
        segments.iter().map(|s| s.memory_usage()).sum()
    }

    /// Split vectors into segments for parallel processing.
    fn create_segments(&self, vectors: Vec<(u64, Vector)>) -> Result<Vec<Vec<(u64, Vector)>>> {
        let segment_size = self.config.segment_size;
        let mut segments = Vec::new();

        for chunk in vectors.chunks(segment_size) {
            segments.push(chunk.to_vec());
        }

        Ok(segments)
    }

    /// Create build tasks for parallel execution.
    fn create_build_tasks(&self, segments_data: Vec<Vec<(u64, Vector)>>) -> Result<Vec<IndexTask>> {
        let mut tasks = Vec::new();

        for (index, vectors) in segments_data.into_iter().enumerate() {
            let task = IndexTask {
                segment_id: index,
                vectors,
                config: self.config.base_config.clone(),
            };
            tasks.push(task);
        }

        Ok(tasks)
    }

    /// Check if segments should be merged.
    fn should_merge(&self) -> Result<bool> {
        let segments = self.segments.lock().unwrap();
        Ok(segments.len() >= self.config.merge_threshold)
    }

    /// Merge segments when threshold is reached.
    fn merge_segments(&mut self) -> Result<()> {
        let mut segments = self.segments.lock().unwrap();

        if segments.len() < self.config.merge_threshold {
            return Ok(());
        }

        // Take segments for merging
        let segments_to_merge = segments.drain(0..self.config.merge_threshold).collect();
        drop(segments); // Release lock

        // Perform merge
        let merged_segment = self.merger.merge_segments(segments_to_merge)?;

        // Add merged segment back
        let mut segments = self.segments.lock().unwrap();
        segments.push(merged_segment);

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.merge_operations += 1;
        }

        Ok(())
    }

    /// Merge all remaining segments.
    fn merge_all_segments(&mut self) -> Result<()> {
        let mut segments = self.segments.lock().unwrap();

        if segments.len() <= 1 {
            return Ok(());
        }

        let all_segments = segments.drain(..).collect();
        drop(segments); // Release lock

        let final_segment = self.merger.merge_segments(all_segments)?;

        let mut segments = self.segments.lock().unwrap();
        segments.push(final_segment);

        Ok(())
    }

    /// Optimize all segments.
    fn optimize_segments(&mut self) -> Result<()> {
        let mut segments = self.segments.lock().unwrap();

        for segment in segments.iter_mut() {
            segment.optimize()?;
        }

        Ok(())
    }
}

impl VectorIndexBuilder for ParallelVectorIndexBuilder {
    fn build(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()> {
        self.build_parallel(vectors)
    }

    fn add_vectors(&mut self, vectors: Vec<(u64, Vector)>) -> Result<()> {
        self.add_vectors_parallel(vectors)
    }

    fn finalize(&mut self) -> Result<()> {
        self.finalize_parallel()
    }

    fn progress(&self) -> f32 {
        self.progress()
    }

    fn estimated_memory_usage(&self) -> usize {
        self.estimated_memory_usage()
    }

    fn optimize(&mut self) -> Result<()> {
        self.optimize_segments()
    }

    fn vectors(&self) -> &[(u64, Vector)] {
        // ParallelVectorIndexBuilder stores vectors across segments
        // For simplicity, return an empty slice for now
        // TODO: Collect vectors from all segments
        &[]
    }
}
