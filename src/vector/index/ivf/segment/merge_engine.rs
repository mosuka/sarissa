//! Merge engine for vector index segments.
//!
//! This module handles the actual merging of segments.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::core::vector::Vector;

use super::manager::ManagedSegmentInfo;

/// Configuration for merge operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// Maximum number of segments to merge at once.
    pub max_merge_segments: u32,

    /// Target segment size after merge (in vectors).
    pub target_segment_size: u64,

    /// Whether to use parallel merging.
    pub parallel_merge: bool,

    /// Number of threads to use for parallel merging.
    pub num_threads: usize,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            max_merge_segments: 10,
            target_segment_size: 1000000,
            parallel_merge: true,
            num_threads: 4,
        }
    }
}

/// Statistics about a merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeStats {
    /// Number of segments merged.
    pub segments_merged: u32,

    /// Number of vectors in merged segment.
    pub vectors_merged: u64,

    /// Number of deleted vectors removed.
    pub deletions_removed: u64,

    /// Time taken for merge (in milliseconds).
    pub merge_time_ms: u64,

    /// Size of merged segment (in bytes).
    pub merged_size_bytes: u64,
}

impl MergeStats {
    /// Create new merge stats.
    pub fn new() -> Self {
        Self {
            segments_merged: 0,
            vectors_merged: 0,
            deletions_removed: 0,
            merge_time_ms: 0,
            merged_size_bytes: 0,
        }
    }

    /// Calculate compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.vectors_merged == 0 {
            return 1.0;
        }
        1.0 - (self.deletions_removed as f64 / self.vectors_merged as f64)
    }
}

impl Default for MergeStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Information about the new merged segment.
    pub merged_segment: ManagedSegmentInfo,

    /// Statistics about the merge.
    pub stats: MergeStats,

    /// IDs of segments that were merged.
    pub merged_segment_ids: Vec<String>,
}

/// Engine for merging vector index segments.
pub struct MergeEngine {
    config: MergeConfig,
    storage: Arc<dyn Storage>,
}

impl MergeEngine {
    /// Create a new merge engine.
    pub fn new(config: MergeConfig, storage: Arc<dyn Storage>) -> Self {
        Self { config, storage }
    }

    /// Merge multiple segments into a single segment.
    ///
    /// This is a simplified implementation. In a real system, this would:
    /// - Read vectors from all source segments
    /// - Filter out deleted vectors
    /// - Write merged vectors to new segment
    /// - Update segment metadata
    pub fn merge_segments(
        &self,
        segments: Vec<ManagedSegmentInfo>,
        new_segment_id: String,
    ) -> Result<MergeResult> {
        let start_time = std::time::Instant::now();

        // Calculate statistics
        let segments_merged = segments.len() as u32;
        let vectors_merged: u64 = segments.iter().map(|s| s.vector_count).sum();
        let total_size: u64 = segments.iter().map(|s| s.size_bytes).sum();

        // In a real implementation, we would:
        // 1. Read all vectors from source segments
        // 2. Filter out deleted vectors
        // 3. Merge and sort if needed
        // 4. Write to new segment
        // 5. Update metadata

        // For now, just create a merged segment info
        let merged_segment = ManagedSegmentInfo {
            segment_id: new_segment_id,
            vector_count: vectors_merged,
            vector_offset: 0,
            generation: segments.iter().map(|s| s.generation).max().unwrap_or(0) + 1,
            has_deletions: false,
            size_bytes: total_size,
        };

        let merge_time_ms = start_time.elapsed().as_millis() as u64;

        let stats = MergeStats {
            segments_merged,
            vectors_merged,
            deletions_removed: 0,
            merge_time_ms,
            merged_size_bytes: total_size,
        };

        Ok(MergeResult {
            merged_segment,
            stats,
            merged_segment_ids: segments.iter().map(|s| s.segment_id.clone()).collect(),
        })
    }

    /// Merge vectors from multiple sources.
    ///
    /// This helper method would be used during actual merge operations.
    #[allow(dead_code)]
    fn merge_vectors(
        &self,
        vectors: Vec<Vec<(u64, Vector)>>,
        deleted_ids: &[u64],
    ) -> Vec<(u64, Vector)> {
        // Flatten all vectors
        let mut all_vectors: Vec<(u64, Vector)> = vectors.into_iter().flatten().collect();

        // Filter out deleted vectors
        all_vectors.retain(|(id, _)| !deleted_ids.contains(id));

        // Sort by ID
        all_vectors.sort_by_key(|(id, _)| *id);

        // Deduplicate (keep latest)
        all_vectors.dedup_by_key(|(id, _)| *id);

        all_vectors
    }

    /// Get storage reference.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// Get configuration.
    pub fn config(&self) -> &MergeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    #[test]
    fn test_merge_engine_basic() {
        let config = MergeConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = MergeEngine::new(config, storage);

        let segments = vec![
            ManagedSegmentInfo {
                segment_id: "seg1".to_string(),
                vector_count: 1000,
                vector_offset: 0,
                generation: 0,
                has_deletions: false,
                size_bytes: 128000,
            },
            ManagedSegmentInfo {
                segment_id: "seg2".to_string(),
                vector_count: 2000,
                vector_offset: 1000,
                generation: 1,
                has_deletions: false,
                size_bytes: 256000,
            },
        ];

        let result = engine
            .merge_segments(segments, "merged_seg".to_string())
            .unwrap();

        assert_eq!(result.stats.segments_merged, 2);
        assert_eq!(result.stats.vectors_merged, 3000);
        assert_eq!(result.merged_segment.vector_count, 3000);
        assert_eq!(result.merged_segment.generation, 2);
    }

    #[test]
    fn test_merge_stats() {
        let stats = MergeStats {
            segments_merged: 3,
            vectors_merged: 1000,
            deletions_removed: 200,
            merge_time_ms: 100,
            merged_size_bytes: 102400,
        };

        assert_eq!(stats.compression_ratio(), 0.8);
    }
}
