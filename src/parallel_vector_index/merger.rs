//! Segment merging functionality for parallel vector indexes.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{SegmentMetadata, VectorIndexSegment};

use crate::error::{Result, SageError};
use crate::vector::Vector;
use crate::vector_index::{VectorIndexBuildConfig, VectorIndexBuilderFactory};

/// Strategy for merging vector index segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Merge based on segment size.
    SizeBased,
    /// Merge based on time since creation.
    TimeBased,
    /// Merge based on memory usage.
    MemoryBased,
    /// Round-robin merging.
    RoundRobin,
}

/// Segment merger for combining multiple vector index segments.
pub struct SegmentMerger {
    strategy: MergeStrategy,
    merge_stats: MergeStats,
}

/// Statistics for merge operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeStats {
    /// Total number of merge operations.
    pub total_merges: usize,
    /// Total time spent merging (milliseconds).
    pub total_merge_time_ms: f64,
    /// Average merge time (milliseconds).
    pub avg_merge_time_ms: f64,
    /// Total vectors merged.
    pub total_vectors_merged: usize,
    /// Memory savings from merging (bytes).
    pub memory_savings_bytes: usize,
}

impl Default for MergeStats {
    fn default() -> Self {
        Self {
            total_merges: 0,
            total_merge_time_ms: 0.0,
            avg_merge_time_ms: 0.0,
            total_vectors_merged: 0,
            memory_savings_bytes: 0,
        }
    }
}

impl SegmentMerger {
    /// Create a new segment merger.
    pub fn new(strategy: MergeStrategy) -> Result<Self> {
        Ok(Self {
            strategy,
            merge_stats: MergeStats::default(),
        })
    }

    /// Merge multiple segments into a single segment.
    pub fn merge_segments(
        &mut self,
        segments: Vec<VectorIndexSegment>,
    ) -> Result<VectorIndexSegment> {
        if segments.is_empty() {
            return Err(SageError::InvalidOperation(
                "Cannot merge empty segments".to_string(),
            ));
        }

        if segments.len() == 1 {
            return Ok(segments.into_iter().next().unwrap());
        }

        let start_time = std::time::Instant::now();

        // Sort segments based on merge strategy
        let sorted_segments = self.sort_segments_for_merge(segments)?;

        // Extract all vectors from segments
        let all_vectors = self.extract_vectors_from_segments(&sorted_segments)?;

        // Create merged segment configuration
        let merge_config = self.create_merge_config(&sorted_segments)?;

        // Build the merged segment
        let merged_segment = self.build_merged_segment(all_vectors, merge_config)?;

        // Update statistics
        self.update_merge_stats(&sorted_segments, &merged_segment, start_time.elapsed());

        Ok(merged_segment)
    }

    /// Merge segments with custom configuration.
    pub fn merge_segments_with_config(
        &mut self,
        segments: Vec<VectorIndexSegment>,
        config: VectorIndexBuildConfig,
    ) -> Result<VectorIndexSegment> {
        if segments.is_empty() {
            return Err(SageError::InvalidOperation(
                "Cannot merge empty segments".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();
        let all_vectors = self.extract_vectors_from_segments(&segments)?;
        let merged_segment = self.build_merged_segment(all_vectors, config)?;

        self.update_merge_stats(&segments, &merged_segment, start_time.elapsed());

        Ok(merged_segment)
    }

    /// Get merge statistics.
    pub fn stats(&self) -> &MergeStats {
        &self.merge_stats
    }

    /// Sort segments based on the merge strategy.
    fn sort_segments_for_merge(
        &self,
        mut segments: Vec<VectorIndexSegment>,
    ) -> Result<Vec<VectorIndexSegment>> {
        match self.strategy {
            MergeStrategy::SizeBased => {
                segments.sort_by_key(|s| s.metadata().vector_count);
            }
            MergeStrategy::TimeBased => {
                segments.sort_by_key(|s| s.metadata().created_at);
            }
            MergeStrategy::MemoryBased => {
                segments.sort_by_key(|s| s.metadata().memory_usage_bytes);
            }
            MergeStrategy::RoundRobin => {
                // No sorting needed for round-robin
            }
        }

        Ok(segments)
    }

    /// Extract all vectors from segments.
    fn extract_vectors_from_segments(
        &self,
        segments: &[VectorIndexSegment],
    ) -> Result<Vec<(u64, Vector)>> {
        let mut all_vectors = Vec::new();

        for segment in segments {
            let segment_vectors = segment.extract_vectors()?;
            all_vectors.extend(segment_vectors);
        }

        // Remove duplicates by document ID (keeping the latest)
        self.deduplicate_vectors(all_vectors)
    }

    /// Remove duplicate vectors by document ID.
    fn deduplicate_vectors(&self, mut vectors: Vec<(u64, Vector)>) -> Result<Vec<(u64, Vector)>> {
        // Sort by document ID
        vectors.sort_by_key(|(doc_id, _)| *doc_id);

        // Use HashMap to keep only the latest vector for each document ID
        let mut unique_vectors = HashMap::new();
        for (doc_id, vector) in vectors {
            unique_vectors.insert(doc_id, vector);
        }

        // Convert back to Vec
        let mut result: Vec<(u64, Vector)> = unique_vectors.into_iter().collect();
        result.sort_by_key(|(doc_id, _)| *doc_id);

        Ok(result)
    }

    /// Create configuration for the merged segment.
    fn create_merge_config(
        &self,
        segments: &[VectorIndexSegment],
    ) -> Result<VectorIndexBuildConfig> {
        if segments.is_empty() {
            return Err(SageError::InvalidOperation(
                "No segments to merge".to_string(),
            ));
        }

        // Use configuration from the first segment as base
        let first_metadata = segments[0].metadata();

        // Verify all segments have compatible configurations
        for segment in segments {
            let metadata = segment.metadata();
            if metadata.dimension != first_metadata.dimension {
                return Err(SageError::InvalidOperation(
                    "Cannot merge segments with different dimensions".to_string(),
                ));
            }
            if metadata.distance_metric != first_metadata.distance_metric {
                return Err(SageError::InvalidOperation(
                    "Cannot merge segments with different distance metrics".to_string(),
                ));
            }
        }

        Ok(VectorIndexBuildConfig {
            dimension: first_metadata.dimension,
            index_type: first_metadata.index_type,
            distance_metric: first_metadata.distance_metric,
            normalize_vectors: true,
            use_quantization: false,
            quantization_method: crate::vector_index::QuantizationMethod::None,
            parallel_build: true,
            memory_limit: None,
        })
    }

    /// Build the merged segment.
    fn build_merged_segment(
        &self,
        vectors: Vec<(u64, Vector)>,
        config: VectorIndexBuildConfig,
    ) -> Result<VectorIndexSegment> {
        // Create a builder for the merged segment
        let mut builder = VectorIndexBuilderFactory::create_builder(config.clone())?;

        // Build the segment
        builder.build(vectors.clone())?;
        builder.finalize()?;
        builder.optimize()?;

        // Create metadata for the merged segment
        let metadata = SegmentMetadata {
            segment_id: 0, // Will be assigned later
            vector_count: vectors.len(),
            dimension: config.dimension,
            index_type: config.index_type,
            distance_metric: config.distance_metric,
            memory_usage_bytes: builder.estimated_memory_usage(),
            created_at: std::time::SystemTime::now(),
        };

        VectorIndexSegment::new(metadata, builder)
    }

    /// Update merge statistics.
    fn update_merge_stats(
        &mut self,
        original_segments: &[VectorIndexSegment],
        merged_segment: &VectorIndexSegment,
        duration: std::time::Duration,
    ) {
        let merge_time_ms = duration.as_secs_f64() * 1000.0;
        let total_vectors: usize = original_segments
            .iter()
            .map(|s| s.metadata().vector_count)
            .sum();
        let original_memory: usize = original_segments
            .iter()
            .map(|s| s.metadata().memory_usage_bytes)
            .sum();
        let merged_memory = merged_segment.metadata().memory_usage_bytes;

        self.merge_stats.total_merges += 1;
        self.merge_stats.total_merge_time_ms += merge_time_ms;
        self.merge_stats.avg_merge_time_ms =
            self.merge_stats.total_merge_time_ms / self.merge_stats.total_merges as f64;
        self.merge_stats.total_vectors_merged += total_vectors;

        if original_memory > merged_memory {
            self.merge_stats.memory_savings_bytes += original_memory - merged_memory;
        }
    }
}
