//! Merge policies for segment management.
//!
//! This module provides different strategies for determining when and how
//! to merge segments to maintain optimal search performance.

use std::collections::HashMap;

use crate::lexical::index::segment_manager::{ManagedSegmentInfo, MergeCandidate, MergeStrategy};

/// Trait for defining merge policies.
pub trait MergePolicy: Send + Sync + std::fmt::Debug {
    /// Select segments for merging based on policy criteria.
    fn select_merges(&self, segments: &[ManagedSegmentInfo]) -> Vec<MergeCandidate>;

    /// Calculate merge priority for a candidate (higher = more urgent).
    fn merge_priority(&self, candidate: &MergeCandidate) -> f64;

    /// Check if a merge operation should be triggered.
    fn should_merge(&self, segments: &[ManagedSegmentInfo]) -> bool;

    /// Get policy configuration as key-value pairs for debugging.
    fn get_config(&self) -> HashMap<String, String>;
}

/// Tiered merge policy inspired by LSM-trees.
/// Organizes segments into tiers and merges within each tier.
#[derive(Debug, Clone)]
pub struct TieredMergePolicy {
    /// Maximum number of segments per tier.
    pub max_segments_per_tier: usize,

    /// Number of segments to merge at once.
    pub segments_per_merge: usize,

    /// Maximum size for merged segment (in bytes).
    pub max_merged_segment_mb: u64,

    /// Size ratio threshold for triggering merges.
    pub size_ratio: f64,

    /// Minimum segment count to trigger merge.
    pub min_merge_segments: usize,

    /// Deletion ratio threshold for priority merging.
    pub deletion_threshold: f64,
}

impl Default for TieredMergePolicy {
    fn default() -> Self {
        TieredMergePolicy {
            max_segments_per_tier: 4,
            segments_per_merge: 3,
            max_merged_segment_mb: 100,
            size_ratio: 2.0,
            min_merge_segments: 2,
            deletion_threshold: 0.2,
        }
    }
}

impl TieredMergePolicy {
    /// Create a new tiered merge policy with custom configuration.
    pub fn new(
        max_segments_per_tier: usize,
        segments_per_merge: usize,
        max_merged_segment_mb: u64,
    ) -> Self {
        TieredMergePolicy {
            max_segments_per_tier,
            segments_per_merge,
            max_merged_segment_mb,
            ..Default::default()
        }
    }

    /// Group segments by tier.
    fn group_by_tier(
        &self,
        segments: &[ManagedSegmentInfo],
    ) -> HashMap<u8, Vec<ManagedSegmentInfo>> {
        let mut tiers: HashMap<u8, Vec<ManagedSegmentInfo>> = HashMap::new();

        for segment in segments {
            if !segment.is_merging {
                tiers.entry(segment.tier).or_default().push(segment.clone());
            }
        }

        // Sort segments within each tier by size (smallest first)
        for tier_segments in tiers.values_mut() {
            tier_segments.sort_by_key(|s| s.size_bytes);
        }

        tiers
    }

    /// Select candidates within a tier.
    fn select_tier_candidates(
        &self,
        _tier: u8,
        segments: &[ManagedSegmentInfo],
    ) -> Vec<MergeCandidate> {
        let mut candidates = Vec::new();

        if segments.len() < self.min_merge_segments {
            return candidates;
        }

        // Strategy 1: Too many segments in tier
        if segments.len() > self.max_segments_per_tier {
            let segments_to_merge = segments
                .iter()
                .take(self.segments_per_merge)
                .map(|s| s.segment_info.segment_id.clone())
                .collect();

            let estimated_size = segments
                .iter()
                .take(self.segments_per_merge)
                .map(|s| s.size_bytes)
                .sum();

            let priority = 10.0 + (segments.len() as f64 - self.max_segments_per_tier as f64);

            candidates.push(MergeCandidate {
                segments: segments_to_merge,
                priority,
                estimated_size,
                strategy: MergeStrategy::SizeBased,
            });
        }

        // Strategy 2: High deletion ratio segments
        let high_deletion_segments: Vec<_> = segments
            .iter()
            .filter(|s| s.deletion_ratio() > self.deletion_threshold)
            .take(self.segments_per_merge)
            .collect();

        if high_deletion_segments.len() >= self.min_merge_segments {
            let segments_to_merge = high_deletion_segments
                .iter()
                .map(|s| s.segment_info.segment_id.clone())
                .collect();

            let estimated_size = high_deletion_segments
                .iter()
                .map(|s| (s.size_bytes as f64 * (1.0 - s.deletion_ratio())) as u64)
                .sum();

            let avg_deletion_ratio = high_deletion_segments
                .iter()
                .map(|s| s.deletion_ratio())
                .sum::<f64>()
                / high_deletion_segments.len() as f64;

            let priority = 5.0 + (avg_deletion_ratio * 10.0);

            candidates.push(MergeCandidate {
                segments: segments_to_merge,
                priority,
                estimated_size,
                strategy: MergeStrategy::DeletionBased,
            });
        }

        // Strategy 3: Size-based merging for small segments
        if segments.len() >= self.segments_per_merge {
            let small_segments: Vec<_> = segments
                .iter()
                .filter(|s| s.size_bytes < (self.max_merged_segment_mb * 1024 * 1024) / 4)
                .take(self.segments_per_merge)
                .collect();

            if small_segments.len() >= self.min_merge_segments {
                let segments_to_merge = small_segments
                    .iter()
                    .map(|s| s.segment_info.segment_id.clone())
                    .collect();

                let estimated_size = small_segments.iter().map(|s| s.size_bytes).sum();

                // Check if merged size would exceed limit
                if estimated_size <= self.max_merged_segment_mb * 1024 * 1024 {
                    let priority =
                        3.0 + (self.segments_per_merge as f64 - small_segments.len() as f64);

                    candidates.push(MergeCandidate {
                        segments: segments_to_merge,
                        priority,
                        estimated_size,
                        strategy: MergeStrategy::SizeBased,
                    });
                }
            }
        }

        candidates
    }
}

impl MergePolicy for TieredMergePolicy {
    fn select_merges(&self, segments: &[ManagedSegmentInfo]) -> Vec<MergeCandidate> {
        let tiers = self.group_by_tier(segments);
        let mut all_candidates = Vec::new();

        // Process each tier
        for (tier, tier_segments) in tiers {
            let tier_candidates = self.select_tier_candidates(tier, &tier_segments);
            all_candidates.extend(tier_candidates);
        }

        // Sort by priority (highest first)
        all_candidates.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        all_candidates
    }

    fn merge_priority(&self, candidate: &MergeCandidate) -> f64 {
        candidate.priority
    }

    fn should_merge(&self, segments: &[ManagedSegmentInfo]) -> bool {
        let tiers = self.group_by_tier(segments);

        // Check if any tier has too many segments
        for tier_segments in tiers.values() {
            if tier_segments.len() > self.max_segments_per_tier {
                return true;
            }
        }

        // Check for high deletion ratios
        let high_deletion_count = segments
            .iter()
            .filter(|s| s.deletion_ratio() > self.deletion_threshold)
            .count();

        high_deletion_count >= self.min_merge_segments
    }

    fn get_config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("policy_type".to_string(), "tiered".to_string());
        config.insert(
            "max_segments_per_tier".to_string(),
            self.max_segments_per_tier.to_string(),
        );
        config.insert(
            "segments_per_merge".to_string(),
            self.segments_per_merge.to_string(),
        );
        config.insert(
            "max_merged_segment_mb".to_string(),
            self.max_merged_segment_mb.to_string(),
        );
        config.insert("size_ratio".to_string(), self.size_ratio.to_string());
        config.insert(
            "deletion_threshold".to_string(),
            self.deletion_threshold.to_string(),
        );
        config
    }
}

/// Log-structured merge policy for write-heavy workloads.
#[derive(Debug, Clone)]
pub struct LogStructuredMergePolicy {
    /// Size ratio between levels.
    pub level_size_ratio: f64,

    /// Maximum number of levels.
    pub max_levels: u8,

    /// Files per level before merge.
    pub files_per_level: usize,

    /// Bloom filter false positive rate.
    pub bloom_filter_fp_rate: f64,
}

impl Default for LogStructuredMergePolicy {
    fn default() -> Self {
        LogStructuredMergePolicy {
            level_size_ratio: 10.0,
            max_levels: 7,
            files_per_level: 10,
            bloom_filter_fp_rate: 0.01,
        }
    }
}

impl MergePolicy for LogStructuredMergePolicy {
    fn select_merges(&self, segments: &[ManagedSegmentInfo]) -> Vec<MergeCandidate> {
        let mut candidates = Vec::new();

        // Group by level (using tier as level)
        let mut levels: HashMap<u8, Vec<_>> = HashMap::new();
        for segment in segments {
            if !segment.is_merging {
                levels.entry(segment.tier).or_default().push(segment);
            }
        }

        // Check each level for merge candidates
        for (&level, level_segments) in &levels {
            if level_segments.len() >= self.files_per_level {
                // Select oldest segments for merge
                let mut sorted_segments = level_segments.clone();
                sorted_segments.sort_by_key(|s| s.created_at);

                let segments_to_merge: Vec<String> = sorted_segments
                    .iter()
                    .take(self.files_per_level / 2)
                    .map(|s| s.segment_info.segment_id.clone())
                    .collect();

                let estimated_size = sorted_segments
                    .iter()
                    .take(self.files_per_level / 2)
                    .map(|s| s.size_bytes)
                    .sum();

                let priority = 5.0
                    + (level as f64)
                    + (level_segments.len() as f64 - self.files_per_level as f64);

                candidates.push(MergeCandidate {
                    segments: segments_to_merge,
                    priority,
                    estimated_size,
                    strategy: MergeStrategy::TimeBased,
                });
            }
        }

        candidates.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        candidates
    }

    fn merge_priority(&self, candidate: &MergeCandidate) -> f64 {
        candidate.priority
    }

    fn should_merge(&self, segments: &[ManagedSegmentInfo]) -> bool {
        let mut levels: HashMap<u8, usize> = HashMap::new();
        for segment in segments {
            if !segment.is_merging {
                *levels.entry(segment.tier).or_default() += 1;
            }
        }

        levels.values().any(|&count| count >= self.files_per_level)
    }

    fn get_config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("policy_type".to_string(), "log_structured".to_string());
        config.insert(
            "level_size_ratio".to_string(),
            self.level_size_ratio.to_string(),
        );
        config.insert("max_levels".to_string(), self.max_levels.to_string());
        config.insert(
            "files_per_level".to_string(),
            self.files_per_level.to_string(),
        );
        config.insert(
            "bloom_filter_fp_rate".to_string(),
            self.bloom_filter_fp_rate.to_string(),
        );
        config
    }
}

/// No-merge policy for testing or read-only scenarios.
#[derive(Debug, Clone, Default)]
pub struct NoMergePolicy;

impl MergePolicy for NoMergePolicy {
    fn select_merges(&self, _segments: &[ManagedSegmentInfo]) -> Vec<MergeCandidate> {
        Vec::new()
    }

    fn merge_priority(&self, _candidate: &MergeCandidate) -> f64 {
        0.0
    }

    fn should_merge(&self, _segments: &[ManagedSegmentInfo]) -> bool {
        false
    }

    fn get_config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("policy_type".to_string(), "no_merge".to_string());
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::segment_manager::ManagedSegmentInfo;
    use crate::lexical::inverted_index::SegmentInfo;

    #[allow(dead_code)]
    fn create_test_segment(
        id: &str,
        doc_count: u64,
        size_bytes: u64,
        tier: u8,
    ) -> ManagedSegmentInfo {
        let segment_info = SegmentInfo {
            segment_id: id.to_string(),
            doc_count,
            doc_offset: 0,
            generation: 1,
            has_deletions: false,
        };

        let mut managed_info = ManagedSegmentInfo::new(segment_info);
        managed_info.size_bytes = size_bytes;
        managed_info.tier = tier;
        managed_info
    }

    #[test]
    fn test_tiered_merge_policy_too_many_segments() {
        let policy = TieredMergePolicy {
            max_segments_per_tier: 3,
            segments_per_merge: 2,
            ..Default::default()
        };

        let segments = vec![
            create_test_segment("seg1", 1000, 1024, 0),
            create_test_segment("seg2", 1000, 2048, 0),
            create_test_segment("seg3", 1000, 1536, 0),
            create_test_segment("seg4", 1000, 1024, 0), // Triggers merge
        ];

        let candidates = policy.select_merges(&segments);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].segments.len(), 2);
        assert!(candidates[0].priority > 10.0);
    }

    #[test]
    fn test_tiered_merge_policy_high_deletion() {
        let policy = TieredMergePolicy {
            deletion_threshold: 0.2,
            min_merge_segments: 2,
            ..Default::default()
        };

        let mut seg1 = create_test_segment("seg1", 1000, 1024, 0);
        let mut seg2 = create_test_segment("seg2", 1000, 1024, 0);

        // Set high deletion ratios
        seg1.deleted_count = 300; // 30%
        seg2.deleted_count = 250; // 25%

        let segments = vec![seg1, seg2];

        let candidates = policy.select_merges(&segments);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].strategy, MergeStrategy::DeletionBased);
        assert!(candidates[0].priority >= 5.0);
    }

    #[test]
    fn test_log_structured_merge_policy() {
        let policy = LogStructuredMergePolicy {
            files_per_level: 3,
            ..Default::default()
        };

        let segments = vec![
            create_test_segment("seg1", 1000, 1024, 0),
            create_test_segment("seg2", 1000, 1024, 0),
            create_test_segment("seg3", 1000, 1024, 0),
            create_test_segment("seg4", 1000, 1024, 0), // Triggers merge
        ];

        let candidates = policy.select_merges(&segments);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].strategy, MergeStrategy::TimeBased);
    }

    #[test]
    fn test_no_merge_policy() {
        let policy = NoMergePolicy;
        let segments = vec![
            create_test_segment("seg1", 1000, 1024, 0),
            create_test_segment("seg2", 1000, 1024, 0),
        ];

        let candidates = policy.select_merges(&segments);
        assert!(candidates.is_empty());
        assert!(!policy.should_merge(&segments));
    }

    #[test]
    fn test_merge_priority_ordering() {
        let policy = TieredMergePolicy::default();

        let high_priority = MergeCandidate {
            segments: vec!["seg1".to_string()],
            priority: 15.0,
            estimated_size: 1024,
            strategy: MergeStrategy::SizeBased,
        };

        let low_priority = MergeCandidate {
            segments: vec!["seg2".to_string()],
            priority: 5.0,
            estimated_size: 1024,
            strategy: MergeStrategy::DeletionBased,
        };

        assert!(policy.merge_priority(&high_priority) > policy.merge_priority(&low_priority));
    }
}
