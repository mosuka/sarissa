//! Segment manager for vector indexes.
//!
//! This module manages vector index segments, including segment metadata,
//! merging strategies, and segment lifecycle.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::Result;

/// Configuration for segment manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentManagerConfig {
    /// Maximum number of vectors per segment.
    pub max_vectors_per_segment: u64,

    /// Minimum number of vectors per segment before merging.
    pub min_vectors_per_segment: u64,

    /// Maximum number of segments before triggering merge.
    pub max_segments: u32,

    /// Merge factor (how many segments to merge at once).
    pub merge_factor: u32,
}

impl Default for SegmentManagerConfig {
    fn default() -> Self {
        Self {
            max_vectors_per_segment: 1000000,
            min_vectors_per_segment: 10000,
            max_segments: 100,
            merge_factor: 10,
        }
    }
}

/// Information about a managed segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedSegmentInfo {
    /// Segment identifier.
    pub segment_id: String,

    /// Number of vectors in this segment.
    pub vector_count: u64,

    /// Vector offset for this segment.
    pub vector_offset: u64,

    /// Generation number of this segment.
    pub generation: u64,

    /// Whether this segment has deletions.
    pub has_deletions: bool,

    /// Size of the segment in bytes.
    pub size_bytes: u64,
}

impl ManagedSegmentInfo {
    /// Create a new managed segment info.
    pub fn new(segment_id: String, vector_count: u64, vector_offset: u64, generation: u64) -> Self {
        Self {
            segment_id,
            vector_count,
            vector_offset,
            generation,
            has_deletions: false,
            size_bytes: 0,
        }
    }

    /// Check if this segment should be merged based on config.
    pub fn should_merge(&self, config: &SegmentManagerConfig) -> bool {
        self.vector_count < config.min_vectors_per_segment
    }
}

/// Candidate segments for merging.
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// Segment IDs to merge.
    pub segment_ids: Vec<String>,

    /// Total vector count.
    pub total_vectors: u64,

    /// Total size in bytes.
    pub total_size: u64,
}

/// Strategy for selecting segments to merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Merge smallest segments first.
    Smallest,

    /// Merge segments with most deletions first.
    MostDeletions,

    /// Merge adjacent segments.
    Adjacent,
}

/// Urgency level for merge operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MergeUrgency {
    /// No urgent need to merge.
    Low,

    /// Should merge soon.
    Medium,

    /// Should merge immediately.
    High,
}

/// Plan for merging segments.
#[derive(Debug, Clone)]
pub struct MergePlan {
    /// Merge candidates.
    pub candidates: Vec<MergeCandidate>,

    /// Strategy used.
    pub strategy: MergeStrategy,

    /// Urgency level.
    pub urgency: MergeUrgency,
}

/// Statistics about segment manager.
#[derive(Debug, Clone)]
pub struct SegmentManagerStats {
    /// Total number of segments.
    pub segment_count: u32,

    /// Total number of vectors across all segments.
    pub total_vectors: u64,

    /// Total size of all segments in bytes.
    pub total_size: u64,

    /// Number of segments with deletions.
    pub segments_with_deletions: u32,

    /// Average vectors per segment.
    pub avg_vectors_per_segment: f64,
}

/// Manages segments for vector indexes.
pub struct SegmentManager {
    config: SegmentManagerConfig,
    segments: Arc<RwLock<HashMap<String, ManagedSegmentInfo>>>,
    next_segment_id: Arc<RwLock<u64>>,
}

impl SegmentManager {
    /// Create a new segment manager with the given configuration.
    pub fn new(config: SegmentManagerConfig) -> Self {
        Self {
            config,
            segments: Arc::new(RwLock::new(HashMap::new())),
            next_segment_id: Arc::new(RwLock::new(0)),
        }
    }

    /// Add a new segment.
    pub fn add_segment(&self, info: ManagedSegmentInfo) -> Result<()> {
        let mut segments = self.segments.write().unwrap();
        segments.insert(info.segment_id.clone(), info);
        Ok(())
    }

    /// Remove a segment.
    pub fn remove_segment(&self, segment_id: &str) -> Result<()> {
        let mut segments = self.segments.write().unwrap();
        segments.remove(segment_id);
        Ok(())
    }

    /// Get segment information.
    pub fn get_segment(&self, segment_id: &str) -> Option<ManagedSegmentInfo> {
        let segments = self.segments.read().unwrap();
        segments.get(segment_id).cloned()
    }

    /// List all segments.
    pub fn list_segments(&self) -> Vec<ManagedSegmentInfo> {
        let segments = self.segments.read().unwrap();
        segments.values().cloned().collect()
    }

    /// Generate a new segment ID.
    pub fn generate_segment_id(&self) -> String {
        let mut next_id = self.next_segment_id.write().unwrap();
        let id = *next_id;
        *next_id += 1;
        format!("segment_{:06}", id)
    }

    /// Check if merging is needed.
    pub fn needs_merge(&self) -> bool {
        let segments = self.segments.read().unwrap();
        segments.len() as u32 > self.config.max_segments
    }

    /// Create a merge plan.
    pub fn create_merge_plan(&self, strategy: MergeStrategy) -> Option<MergePlan> {
        let segments = self.segments.read().unwrap();

        if segments.len() <= 1 {
            return None;
        }

        let mut segment_list: Vec<_> = segments.values().cloned().collect();

        // Sort based on strategy
        match strategy {
            MergeStrategy::Smallest => {
                segment_list.sort_by_key(|s| s.vector_count);
            }
            MergeStrategy::MostDeletions => {
                segment_list.sort_by(|a, b| b.has_deletions.cmp(&a.has_deletions));
            }
            MergeStrategy::Adjacent => {
                segment_list.sort_by_key(|s| s.vector_offset);
            }
        }

        // Select segments to merge
        let merge_count = self.config.merge_factor.min(segment_list.len() as u32) as usize;
        let to_merge = &segment_list[..merge_count];

        let candidate = MergeCandidate {
            segment_ids: to_merge.iter().map(|s| s.segment_id.clone()).collect(),
            total_vectors: to_merge.iter().map(|s| s.vector_count).sum(),
            total_size: to_merge.iter().map(|s| s.size_bytes).sum(),
        };

        // Determine urgency
        let urgency = if segments.len() as u32 > self.config.max_segments * 2 {
            MergeUrgency::High
        } else if segments.len() as u32 > self.config.max_segments {
            MergeUrgency::Medium
        } else {
            MergeUrgency::Low
        };

        Some(MergePlan {
            candidates: vec![candidate],
            strategy,
            urgency,
        })
    }

    /// Get statistics.
    pub fn stats(&self) -> SegmentManagerStats {
        let segments = self.segments.read().unwrap();
        let segment_count = segments.len() as u32;
        let total_vectors: u64 = segments.values().map(|s| s.vector_count).sum();
        let total_size: u64 = segments.values().map(|s| s.size_bytes).sum();
        let segments_with_deletions = segments.values().filter(|s| s.has_deletions).count() as u32;
        let avg_vectors_per_segment = if segment_count > 0 {
            total_vectors as f64 / segment_count as f64
        } else {
            0.0
        };

        SegmentManagerStats {
            segment_count,
            total_vectors,
            total_size,
            segments_with_deletions,
            avg_vectors_per_segment,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_manager_basic() {
        let config = SegmentManagerConfig::default();
        let manager = SegmentManager::new(config);

        let segment_id = manager.generate_segment_id();
        assert_eq!(segment_id, "segment_000000");

        let info = ManagedSegmentInfo::new(segment_id.clone(), 1000, 0, 0);
        manager.add_segment(info.clone()).unwrap();

        let retrieved = manager.get_segment(&segment_id).unwrap();
        assert_eq!(retrieved.vector_count, 1000);
    }

    #[test]
    fn test_merge_plan_creation() {
        let config = SegmentManagerConfig {
            max_segments: 5,
            merge_factor: 3,
            ..Default::default()
        };
        let manager = SegmentManager::new(config);

        // Add multiple segments
        for i in 0..10 {
            let segment_id = manager.generate_segment_id();
            let info = ManagedSegmentInfo::new(segment_id, 1000 * (i + 1), i * 1000, 0);
            manager.add_segment(info).unwrap();
        }

        assert!(manager.needs_merge());

        let plan = manager.create_merge_plan(MergeStrategy::Smallest);
        assert!(plan.is_some());

        let plan = plan.unwrap();
        assert_eq!(plan.candidates.len(), 1);
        assert_eq!(plan.candidates[0].segment_ids.len(), 3);
    }
}
