//! Merge policy for vector index segments.
//!
//! This module defines the strategy for selecting which segments to merge.

use std::fmt::Debug;

use crate::vector::index::hnsw::segment::manager::{ManagedSegmentInfo, SegmentManagerConfig};

/// Trait for merge policies.
pub trait MergePolicy: Debug + Send + Sync {
    /// Select segments to merge.
    ///
    /// Returns a list of segment IDs to merge, or `None` if no merge is needed.
    fn candidates(
        &self,
        segments: &[ManagedSegmentInfo],
        config: &SegmentManagerConfig,
    ) -> Option<Vec<String>>;
}

/// Simple merge policy based on segment count and size.
#[derive(Debug, Default)]
pub struct SimpleMergePolicy;

impl SimpleMergePolicy {
    /// Create a new simple merge policy.
    pub fn new() -> Self {
        Self
    }
}

impl MergePolicy for SimpleMergePolicy {
    fn candidates(
        &self,
        segments: &[ManagedSegmentInfo],
        config: &SegmentManagerConfig,
    ) -> Option<Vec<String>> {
        // If we don't have enough segments, don't merge.
        // We trigger merge only when we exceed max_segments or soft limit?
        // Let's say we trigger if we have more than max_segments / 2?
        // Or strictly strictly max_segments?
        // Currently config.max_segments is the trigger threshold.

        if segments.len() < config.max_segments as usize {
            return None;
        }

        // Strategy:
        // 1. Sort segments by generation (oldest first)? Or just use existing order (usually chronological)?
        //    ManagedSegmentInfo doesn't imply order, but SegmentManager keeps them in Vec.
        // 2. Look for sequence of segments that are candidates.
        // 3. Simple approach: Pick the smallest segments to merge.

        // Let's try finding the "best" window of segments to merge.
        // We want to merge `merge_factor` segments.

        let merge_factor = config.merge_factor as usize;
        if segments.len() < merge_factor {
            return None;
        }

        // Find a window of `merge_factor` segments with the smallest total size?
        // Or smallest vector count.

        // We only look at contiguous segments to preserve some locality/generation order if implicitly there.
        // Merging non-adjacent segments might change logical order if that matters (it might not for vectors).
        // But assuming we want to keep older/newer distinct if possible.
        // For simplicity, we assume vectors are unordered collection, so any segments can merge.
        // BUT, usually we want to merge small -> medium -> large.

        // Let's pick N smallest segments regardless of position?
        // If we pick arbitrary segments, we create a new segment.

        // Let's stick to "Smallest First" regardless of adjacency for now,
        // as vector Search is global across all segments.

        let mut sorted_segments: Vec<(usize, &ManagedSegmentInfo)> =
            segments.iter().enumerate().collect();
        sorted_segments.sort_by_key(|(_, s)| s.vector_count);

        // Take top `merge_factor` smallest segments
        let candidates: Vec<String> = sorted_segments
            .iter()
            .take(merge_factor)
            .map(|(_, s)| s.segment_id.clone())
            .collect();

        if candidates.is_empty() {
            None
        } else {
            Some(candidates)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::index::hnsw::segment::manager::ManagedSegmentInfo;

    fn create_info(id: &str, count: u64) -> ManagedSegmentInfo {
        ManagedSegmentInfo {
            segment_id: id.to_string(),
            vector_count: count,
            vector_offset: 0,
            generation: 1,
            has_deletions: false,
            size_bytes: count * 100,
        }
    }

    #[test]
    fn test_simple_merge_policy_candidates() {
        let policy = SimpleMergePolicy::new();
        let mut config = SegmentManagerConfig::default();
        config.max_segments = 5;
        config.merge_factor = 3;

        // Case 1: Not enough segments
        let segments = vec![create_info("1", 100), create_info("2", 100)];
        assert!(policy.candidates(&segments, &config).is_none());

        // Case 2: Enough segments, trigger merge
        let segments = vec![
            create_info("1", 1000), // Large
            create_info("2", 100),  // Small
            create_info("3", 100),  // Small
            create_info("4", 100),  // Small
            create_info("5", 1000), // Large
            create_info("6", 1000), // Large
        ]; // Total 6 > max 5

        let candidates = policy.candidates(&segments, &config).unwrap();
        assert_eq!(candidates.len(), 3);
        // Should pick smallest: 2, 3, 4
        assert!(candidates.contains(&"2".to_string()));
        assert!(candidates.contains(&"3".to_string()));
        assert!(candidates.contains(&"4".to_string()));
    }
}
