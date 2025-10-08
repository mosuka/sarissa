//! Segment management for efficient index organization.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::storage::Storage;

/// Segment metadata and management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Unique identifier for this segment.
    pub id: String,
    /// Number of documents in this segment.
    pub doc_count: u64,
    /// Number of deleted documents in this segment.
    pub deleted_count: u64,
    /// Generation number for ordering segments.
    pub generation: u64,
    /// Timestamp when this segment was created.
    pub created_at: u64,
    /// Size of the segment in bytes.
    pub size_bytes: u64,
    /// Whether this segment is currently being written to.
    pub is_mutable: bool,
    /// Files that make up this segment.
    pub files: Vec<String>,
}

impl Segment {
    /// Create a new segment.
    pub fn new(id: String, generation: u64) -> Self {
        Segment {
            id,
            doc_count: 0,
            deleted_count: 0,
            generation,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            size_bytes: 0,
            is_mutable: true,
            files: Vec::new(),
        }
    }

    /// Get the number of live (non-deleted) documents in this segment.
    pub fn live_doc_count(&self) -> u64 {
        self.doc_count.saturating_sub(self.deleted_count)
    }

    /// Get the deletion ratio (0.0 = no deletions, 1.0 = all deleted).
    pub fn deletion_ratio(&self) -> f64 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.deleted_count as f64 / self.doc_count as f64
        }
    }

    /// Check if this segment needs merging (high deletion ratio).
    pub fn needs_merging(&self, threshold: f64) -> bool {
        self.deletion_ratio() > threshold
    }

    /// Mark this segment as immutable (read-only).
    pub fn freeze(&mut self) {
        self.is_mutable = false;
    }

    /// Add a file to this segment.
    pub fn add_file(&mut self, filename: String) {
        if !self.files.contains(&filename) {
            self.files.push(filename);
        }
    }

    /// Remove a file from this segment.
    pub fn remove_file(&mut self, filename: &str) {
        self.files.retain(|f| f != filename);
    }
}

/// Manages multiple segments in an index.
#[derive(Debug)]
pub struct SegmentManager {
    /// Storage backend for persisting segment metadata.
    storage: Arc<dyn Storage>,
    /// Currently active segments.
    segments: HashMap<String, Segment>,
    /// Next generation number to assign.
    next_generation: u64,
    /// Maximum number of segments before triggering a merge.
    max_segments: usize,
    /// Deletion ratio threshold for triggering merges.
    merge_threshold: f64,
}

impl SegmentManager {
    /// Create a new segment manager.
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        SegmentManager {
            storage,
            segments: HashMap::new(),
            next_generation: 1,
            max_segments: 10,
            merge_threshold: 0.3, // 30% deletion ratio triggers merge
        }
    }

    /// Load segment metadata from storage.
    pub fn load(&mut self) -> Result<()> {
        // Try to load segments.json
        if self.storage.file_exists("segments.json")
            && let Ok(mut input) = self.storage.open_input("segments.json")
        {
            let mut data = Vec::new();
            if input.read_to_end(&mut data).is_ok() {
                let segments_data: HashMap<String, Segment> = serde_json::from_slice(&data)
                    .map_err(|e| {
                        crate::error::SarissaError::storage(format!(
                            "Failed to parse segments metadata: {e}"
                        ))
                    })?;

                self.segments = segments_data;

                // Update next generation number
                self.next_generation = self
                    .segments
                    .values()
                    .map(|s| s.generation)
                    .max()
                    .unwrap_or(0)
                    + 1;
            }
        }

        Ok(())
    }

    /// Save segment metadata to storage.
    pub fn save(&self) -> Result<()> {
        let data = serde_json::to_vec_pretty(&self.segments).map_err(|e| {
            crate::error::SarissaError::storage(format!(
                "Failed to serialize segments metadata: {e}"
            ))
        })?;

        if let Ok(mut output) = self.storage.create_output("segments.json") {
            output.write_all(&data)?;
            output.flush()?;
        }
        Ok(())
    }

    /// Create a new segment.
    pub fn create_segment(&mut self) -> Result<Segment> {
        let id = format!("segment_{:08x}", self.next_generation);
        let segment = Segment::new(id.clone(), self.next_generation);

        self.next_generation += 1;
        self.segments.insert(id.clone(), segment.clone());

        Ok(segment)
    }

    /// Get a segment by ID.
    pub fn get_segment(&self, id: &str) -> Option<&Segment> {
        self.segments.get(id)
    }

    /// Get a mutable reference to a segment by ID.
    pub fn get_segment_mut(&mut self, id: &str) -> Option<&mut Segment> {
        self.segments.get_mut(id)
    }

    /// Get all segments.
    pub fn segments(&self) -> impl Iterator<Item = &Segment> {
        self.segments.values()
    }

    /// Get segments sorted by generation (newest first).
    pub fn segments_by_generation(&self) -> Vec<&Segment> {
        let mut segments: Vec<&Segment> = self.segments.values().collect();
        segments.sort_by(|a, b| b.generation.cmp(&a.generation));
        segments
    }

    /// Get segments that need merging.
    pub fn segments_needing_merge(&self) -> Vec<&Segment> {
        self.segments
            .values()
            .filter(|s| !s.is_mutable && s.needs_merging(self.merge_threshold))
            .collect()
    }

    /// Remove a segment.
    pub fn remove_segment(&mut self, id: &str) -> Result<()> {
        if let Some(segment) = self.segments.remove(id) {
            // Delete segment files
            for file in &segment.files {
                if let Err(e) = self.storage.delete_file(file) {
                    // Log error but don't fail the operation
                    eprintln!("Warning: Failed to delete segment file {file}: {e}");
                }
            }
        }
        Ok(())
    }

    /// Check if we need to trigger a merge operation.
    pub fn should_merge(&self) -> bool {
        // Too many segments
        if self.segments.len() > self.max_segments {
            return true;
        }

        // Any segments with high deletion ratio
        !self.segments_needing_merge().is_empty()
    }

    /// Get merge candidates (segments that should be merged together).
    pub fn get_merge_candidates(&self) -> Vec<String> {
        let mut candidates = Vec::new();

        // Prefer to merge segments with high deletion ratios
        let mut segments: Vec<&Segment> =
            self.segments.values().filter(|s| !s.is_mutable).collect();

        // Sort by deletion ratio (highest first) and then by size (smallest first)
        segments.sort_by(|a, b| {
            b.deletion_ratio()
                .partial_cmp(&a.deletion_ratio())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.size_bytes.cmp(&b.size_bytes))
        });

        // Take up to 5 segments for merging
        for segment in segments.into_iter().take(5) {
            candidates.push(segment.id.clone());
        }

        candidates
    }

    /// Mark a segment for deletion (add to deleted count).
    pub fn mark_deleted(&mut self, segment_id: &str, doc_ids: &[u64]) -> Result<()> {
        if let Some(segment) = self.segments.get_mut(segment_id) {
            segment.deleted_count += doc_ids.len() as u64;
        }
        Ok(())
    }

    /// Update segment statistics.
    pub fn update_segment_stats(
        &mut self,
        segment_id: &str,
        doc_count: u64,
        size_bytes: u64,
    ) -> Result<()> {
        if let Some(segment) = self.segments.get_mut(segment_id) {
            segment.doc_count = doc_count;
            segment.size_bytes = size_bytes;
        }
        Ok(())
    }

    /// Freeze a segment (make it immutable).
    pub fn freeze_segment(&mut self, segment_id: &str) -> Result<()> {
        if let Some(segment) = self.segments.get_mut(segment_id) {
            segment.freeze();
        }
        Ok(())
    }

    /// Get total number of documents across all segments.
    pub fn total_doc_count(&self) -> u64 {
        self.segments.values().map(|s| s.doc_count).sum()
    }

    /// Get total number of live documents across all segments.
    pub fn total_live_doc_count(&self) -> u64 {
        self.segments.values().map(|s| s.live_doc_count()).sum()
    }

    /// Get total size of all segments in bytes.
    pub fn total_size_bytes(&self) -> u64 {
        self.segments.values().map(|s| s.size_bytes).sum()
    }

    /// Set the maximum number of segments before triggering merge.
    pub fn set_max_segments(&mut self, max_segments: usize) {
        self.max_segments = max_segments;
    }

    /// Set the deletion ratio threshold for triggering merges.
    pub fn set_merge_threshold(&mut self, threshold: f64) {
        self.merge_threshold = threshold.clamp(0.0, 1.0);
    }
}

/// Segment merge policy for deciding when and how to merge segments.
#[derive(Debug, Clone)]
pub enum MergePolicy {
    /// Merge when deletion ratio exceeds threshold.
    DeletionRatio(f64),
    /// Merge when number of segments exceeds limit.
    SegmentCount(usize),
    /// Merge based on segment size.
    SizeThreshold(u64),
    /// Custom merge policy.
    Custom,
}

impl Default for MergePolicy {
    fn default() -> Self {
        MergePolicy::DeletionRatio(0.3)
    }
}

/// Segment merge operation.
#[derive(Debug)]
pub struct MergeOperation {
    /// Segments to be merged.
    pub source_segments: Vec<String>,
    /// Target segment for merge result.
    pub target_segment: String,
    /// Estimated size after merge.
    pub estimated_size: u64,
    /// Priority of this merge (higher = more urgent).
    pub priority: u32,
}

impl MergeOperation {
    /// Create a new merge operation.
    pub fn new(source_segments: Vec<String>, target_segment: String, estimated_size: u64) -> Self {
        MergeOperation {
            source_segments,
            target_segment,
            estimated_size,
            priority: 0,
        }
    }

    /// Set the priority of this merge operation.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{MemoryStorage, StorageConfig};

    #[test]
    fn test_segment_creation() {
        let segment = Segment::new("test_segment".to_string(), 1);

        assert_eq!(segment.id, "test_segment");
        assert_eq!(segment.generation, 1);
        assert_eq!(segment.doc_count, 0);
        assert_eq!(segment.deleted_count, 0);
        assert_eq!(segment.live_doc_count(), 0);
        assert_eq!(segment.deletion_ratio(), 0.0);
        assert!(segment.is_mutable);
    }

    #[test]
    fn test_segment_deletion_ratio() {
        let mut segment = Segment::new("test".to_string(), 1);
        segment.doc_count = 100;
        segment.deleted_count = 30;

        assert_eq!(segment.live_doc_count(), 70);
        assert_eq!(segment.deletion_ratio(), 0.3);
        assert!(segment.needs_merging(0.25));
        assert!(!segment.needs_merging(0.35));
    }

    #[test]
    fn test_segment_manager() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let mut manager = SegmentManager::new(storage);

        let segment1 = manager.create_segment().unwrap();
        let segment2 = manager.create_segment().unwrap();

        assert_eq!(manager.segments().count(), 2);
        assert_eq!(segment1.generation, 1);
        assert_eq!(segment2.generation, 2);
    }

    #[test]
    fn test_segment_manager_merge_candidates() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));
        let mut manager = SegmentManager::new(storage);

        // Create segments with different deletion ratios
        let mut segment1 = manager.create_segment().unwrap();
        segment1.doc_count = 100;
        segment1.deleted_count = 50; // 50% deletion ratio
        segment1.freeze();
        manager.segments.insert(segment1.id.clone(), segment1);

        let mut segment2 = manager.create_segment().unwrap();
        segment2.doc_count = 100;
        segment2.deleted_count = 10; // 10% deletion ratio
        segment2.freeze();
        manager.segments.insert(segment2.id.clone(), segment2);

        assert!(manager.should_merge());

        let candidates = manager.get_merge_candidates();
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_merge_operation() {
        let merge_op = MergeOperation::new(
            vec!["seg1".to_string(), "seg2".to_string()],
            "merged_seg".to_string(),
            1024,
        )
        .with_priority(5);

        assert_eq!(merge_op.source_segments.len(), 2);
        assert_eq!(merge_op.target_segment, "merged_seg");
        assert_eq!(merge_op.estimated_size, 1024);
        assert_eq!(merge_op.priority, 5);
    }
}
