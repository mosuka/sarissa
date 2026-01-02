//! Segment manager for vector indexes.
//!
//! This module manages vector index segments, including segment metadata,
//! merging strategies, and segment lifecycle.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::{Arc, RwLock};

use crate::error::Result;
use crate::storage::Storage;

use super::merge_policy::MergePolicy;

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
    /// Segments to merge.
    pub segments: Vec<ManagedSegmentInfo>,

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
#[derive(Debug)]
pub struct SegmentManager {
    config: SegmentManagerConfig,
    storage: Arc<dyn Storage>,
    segments: Arc<RwLock<Vec<ManagedSegmentInfo>>>,
    next_segment_id: Arc<RwLock<u64>>,
}

impl SegmentManager {
    /// Create a new segment manager with the given configuration.
    pub fn new(config: SegmentManagerConfig, storage: Arc<dyn Storage>) -> Result<Self> {
        let manager = Self {
            config,
            storage,
            segments: Arc::new(RwLock::new(Vec::new())),
            next_segment_id: Arc::new(RwLock::new(0)),
        };

        let _ = manager.load_state();

        Ok(manager)
    }

    fn load_state(&self) -> Result<()> {
        let mut reader = match self.storage.open_input("segments.json") {
            Ok(r) => r,
            Err(_) => return Ok(()),
        };

        let mut content = Vec::new();
        reader.read_to_end(&mut content)?;
        // If empty file, ignore
        if content.is_empty() {
            return Ok(());
        }

        let segments_info: Vec<ManagedSegmentInfo> = serde_json::from_slice(&content)?;

        let mut segments = self.segments.write().unwrap();
        *segments = segments_info;

        let max_id = segments
            .iter()
            .filter_map(|s| s.segment_id.strip_prefix("segment_"))
            .filter_map(|s| s.parse::<u64>().ok())
            .max()
            .unwrap_or(0);

        *self.next_segment_id.write().unwrap() = max_id + 1;

        Ok(())
    }

    pub fn save_state(&self) -> Result<()> {
        let segments = self.segments.read().unwrap();
        let content = serde_json::to_vec_pretty(&*segments)?;

        let mut writer = self.storage.create_output("segments.json")?;
        writer.write_all(&content)?;
        writer.flush()?;

        Ok(())
    }

    /// Add a new segment.
    pub fn add_segment(&self, info: ManagedSegmentInfo) -> Result<()> {
        let mut segments = self.segments.write().unwrap();
        segments.push(info);
        drop(segments);
        self.save_state()
    }

    /// Remove a segment.
    pub fn remove_segment(&self, segment_id: &str) -> Result<()> {
        let mut segments = self.segments.write().unwrap();
        if let Some(pos) = segments.iter().position(|s| s.segment_id == segment_id) {
            segments.remove(pos);
            drop(segments);
            self.save_state()
        } else {
            Ok(())
        }
    }

    /// Delete physical files associated with a segment.
    pub fn delete_segment_files(&self, segment_id: &str) -> Result<()> {
        // HNSW index writer uses segment_id as the main file name
        self.storage.delete_file(segment_id)?;
        Ok(())
    }

    /// Update a segment info.
    pub fn update_segment(&self, info: ManagedSegmentInfo) -> Result<()> {
        let mut segments = self.segments.write().unwrap();
        if let Some(idx) = segments
            .iter()
            .position(|s| s.segment_id == info.segment_id)
        {
            segments[idx] = info;
        }
        drop(segments);
        self.save_state()
    }

    /// Get segment information.
    pub fn get_segment(&self, segment_id: &str) -> Option<ManagedSegmentInfo> {
        let segments = self.segments.read().unwrap();
        segments
            .iter()
            .find(|s| s.segment_id == segment_id)
            .cloned()
    }

    /// List all segments.
    pub fn list_segments(&self) -> Vec<ManagedSegmentInfo> {
        let segments = self.segments.read().unwrap();
        segments.clone()
    }

    /// Check if any segments need merging.
    pub fn check_merge(&self, policy: &dyn MergePolicy) -> Option<MergeCandidate> {
        let segments_lock = self.segments.read().unwrap();

        if let Some(candidate_ids) = policy.candidates(&segments_lock, &self.config) {
            let mut total_vectors = 0;
            let mut total_size = 0;
            let mut candidates = Vec::new();

            for id in &candidate_ids {
                if let Some(segment) = segments_lock.iter().find(|s| s.segment_id == *id) {
                    total_vectors += segment.vector_count;
                    total_size += segment.size_bytes;
                    candidates.push(segment.clone());
                }
            }

            return Some(MergeCandidate {
                segments: candidates,
                total_vectors,
                total_size,
            });
        }
        None
    }

    /// Apply a merge result by replacing source segments with the merged segment.
    pub fn apply_merge(
        &self,
        candidate: MergeCandidate,
        merged_segment: ManagedSegmentInfo,
    ) -> Result<()> {
        let mut segments_lock = self.segments.write().unwrap();

        // 1. Remove source segments
        let ids_to_remove: std::collections::HashSet<_> =
            candidate.segments.iter().map(|s| &s.segment_id).collect();

        segments_lock.retain(|s| !ids_to_remove.contains(&s.segment_id));

        // 2. Add new segment
        segments_lock.push(merged_segment);

        // 3. Save state
        drop(segments_lock);
        self.save_state()?;

        // 4. Cleanup physical files of source segments
        for segment in candidate.segments {
            self.delete_segment_files(&segment.segment_id)?;
        }

        Ok(())
    }
    pub fn total_vectors(&self) -> u64 {
        self.segments
            .read()
            .unwrap()
            .iter()
            .map(|s| s.vector_count)
            .sum()
    }

    pub fn total_deleted(&self) -> u64 {
        0 // TODO: Track deleted count in ManagedSegmentInfo
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

        let mut segment_list: Vec<_> = segments.iter().cloned().collect();

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
            segments: to_merge.to_vec(),
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
        let total_vectors: u64 = segments.iter().map(|s| s.vector_count).sum();
        let total_size: u64 = segments.iter().map(|s| s.size_bytes).sum();
        let segments_with_deletions = segments.iter().filter(|s| s.has_deletions).count() as u32;
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
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::index::hnsw::segment::merge_policy::SimpleMergePolicy;

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
    fn test_segment_manager_basic() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = SegmentManager::new(config, storage).unwrap();

        let segment_id = manager.generate_segment_id();
        assert_eq!(segment_id, "segment_000000");

        let info = ManagedSegmentInfo::new(segment_id.clone(), 1000, 0, 0);
        manager.add_segment(info.clone()).unwrap();

        let retrieved = manager.get_segment(&segment_id).unwrap();
        assert_eq!(retrieved.vector_count, 1000);
    }

    // Additional tests for persistence?
    #[test]
    fn test_persistence() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager = SegmentManager::new(config.clone(), storage.clone()).unwrap();
            let info = ManagedSegmentInfo::new("segment_000000".to_string(), 1000, 0, 0);
            manager.add_segment(info).unwrap();
            // Saves automatically
        }

        // Reload
        {
            let manager = SegmentManager::new(config, storage.clone()).unwrap();
            let segments = manager.list_segments();
            assert_eq!(segments.len(), 1);
            assert_eq!(segments[0].segment_id, "segment_000000");
        }
    }

    #[test]
    fn test_check_merge() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let mut config = SegmentManagerConfig::default();
        config.max_segments = 5;
        config.merge_factor = 3;

        // We use a temporary config for the manager
        let manager = SegmentManager::new(config, storage).unwrap();

        // 1. Add segments (not enough for merge)
        manager.add_segment(create_info("1", 100)).unwrap();
        manager.add_segment(create_info("2", 100)).unwrap();

        assert!(manager.check_merge(&SimpleMergePolicy::new()).is_none());

        // 2. Add more segments to trigger merge
        manager.add_segment(create_info("3", 100)).unwrap();
        manager.add_segment(create_info("4", 100)).unwrap();
        manager.add_segment(create_info("5", 100)).unwrap();
        manager.add_segment(create_info("6", 100)).unwrap(); // Total 6 > 5

        let candidate = manager.check_merge(&SimpleMergePolicy::new());
        assert!(candidate.is_some());

        let candidate = candidate.unwrap();
        assert_eq!(candidate.segments.len(), 3);
        // Expect smallest: 1, 2, 3, 4, 5, 6 are all 100?
        // Wait, simple policy sort by vector_count.
        // If all equal, it picks stable sort order? Or arbitrary.
        // SimpleMergePolicy uses `segments.iter().enumerate()` then sort_by_key.
        // `sort_by_key` is stable. So it picks first 3: 1, 2, 3.

        let ids: Vec<String> = candidate
            .segments
            .iter()
            .map(|s| s.segment_id.clone())
            .collect();
        assert!(ids.contains(&"1".to_string()));
        assert!(ids.contains(&"2".to_string()));
        assert!(ids.contains(&"3".to_string()));
    }
}
