//! Merge policies for vector index segments.
//!
//! This module defines policies for when and how to merge segments.

use serde::{Deserialize, Serialize};

use super::manager::{ManagedSegmentInfo, SegmentManagerConfig};

/// Merge policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergePolicyConfig {
    /// Maximum number of segments before triggering merge.
    pub max_segments: u32,

    /// Minimum segment size (in vectors) to consider for merging.
    pub min_segment_size: u64,

    /// Maximum segment size (in vectors) after merging.
    pub max_segment_size: u64,

    /// Deletion ratio threshold for triggering compaction.
    pub deletion_ratio_threshold: f64,

    /// Merge factor (number of segments to merge at once).
    pub merge_factor: u32,
}

impl Default for MergePolicyConfig {
    fn default() -> Self {
        Self {
            max_segments: 100,
            min_segment_size: 10000,
            max_segment_size: 1000000,
            deletion_ratio_threshold: 0.3,
            merge_factor: 10,
        }
    }
}

/// Merge policy trait for determining merge behavior.
pub trait MergePolicy: Send + Sync {
    /// Check if segments should be merged based on current state.
    fn should_merge(&self, segments: &[ManagedSegmentInfo], config: &SegmentManagerConfig) -> bool;

    /// Select segments to merge.
    fn select_merge_candidates(
        &self,
        segments: &[ManagedSegmentInfo],
        config: &SegmentManagerConfig,
    ) -> Vec<String>;
}

/// Tiered merge policy (similar to Lucene's TieredMergePolicy).
///
/// This policy merges segments of similar sizes together.
pub struct TieredMergePolicy {
    config: MergePolicyConfig,
}

impl TieredMergePolicy {
    /// Create a new tiered merge policy.
    pub fn new(config: MergePolicyConfig) -> Self {
        Self { config }
    }
}

impl MergePolicy for TieredMergePolicy {
    fn should_merge(
        &self,
        segments: &[ManagedSegmentInfo],
        _config: &SegmentManagerConfig,
    ) -> bool {
        // Merge if we have too many segments
        if segments.len() > self.config.max_segments as usize {
            return true;
        }

        // Merge if any segment has high deletion ratio
        segments
            .iter()
            .any(|s| s.has_deletions && s.vector_count < self.config.min_segment_size)
    }

    fn select_merge_candidates(
        &self,
        segments: &[ManagedSegmentInfo],
        _config: &SegmentManagerConfig,
    ) -> Vec<String> {
        let mut segments = segments.to_vec();

        // Sort by size (ascending)
        segments.sort_by_key(|s| s.vector_count);

        // Select smallest segments up to merge_factor
        let count = self.config.merge_factor.min(segments.len() as u32) as usize;
        segments
            .iter()
            .take(count)
            .map(|s| s.segment_id.clone())
            .collect()
    }
}

/// Log-structured merge policy.
///
/// This policy merges segments in levels, similar to LSM trees.
pub struct LogStructuredMergePolicy {
    config: MergePolicyConfig,
}

impl LogStructuredMergePolicy {
    /// Create a new log-structured merge policy.
    pub fn new(config: MergePolicyConfig) -> Self {
        Self { config }
    }
}

impl MergePolicy for LogStructuredMergePolicy {
    fn should_merge(
        &self,
        segments: &[ManagedSegmentInfo],
        _config: &SegmentManagerConfig,
    ) -> bool {
        segments.len() > self.config.merge_factor as usize
    }

    fn select_merge_candidates(
        &self,
        segments: &[ManagedSegmentInfo],
        _config: &SegmentManagerConfig,
    ) -> Vec<String> {
        let mut segments = segments.to_vec();

        // Sort by generation (oldest first)
        segments.sort_by_key(|s| s.generation);

        // Select oldest segments up to merge_factor
        let count = self.config.merge_factor.min(segments.len() as u32) as usize;
        segments
            .iter()
            .take(count)
            .map(|s| s.segment_id.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_segment(id: &str, vector_count: u64, generation: u64) -> ManagedSegmentInfo {
        ManagedSegmentInfo {
            segment_id: id.to_string(),
            vector_count,
            vector_offset: 0,
            generation,
            has_deletions: false,
            size_bytes: vector_count * 128,
        }
    }

    #[test]
    fn test_tiered_merge_policy() {
        let config = MergePolicyConfig::default();
        let policy = TieredMergePolicy::new(config);
        let manager_config = SegmentManagerConfig::default();

        let segments = vec![
            create_test_segment("seg1", 1000, 0),
            create_test_segment("seg2", 2000, 1),
            create_test_segment("seg3", 3000, 2),
        ];

        assert!(!policy.should_merge(&segments, &manager_config));

        let candidates = policy.select_merge_candidates(&segments, &manager_config);
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0], "seg1"); // Smallest first
    }

    #[test]
    fn test_log_structured_merge_policy() {
        let config = MergePolicyConfig {
            merge_factor: 2,
            ..Default::default()
        };
        let policy = LogStructuredMergePolicy::new(config);
        let manager_config = SegmentManagerConfig::default();

        let segments = vec![
            create_test_segment("seg1", 1000, 0),
            create_test_segment("seg2", 2000, 1),
            create_test_segment("seg3", 3000, 2),
        ];

        assert!(policy.should_merge(&segments, &manager_config));

        let candidates = policy.select_merge_candidates(&segments, &manager_config);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0], "seg1"); // Oldest first
    }
}
