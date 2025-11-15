//! Deletion management for vector indexes.
//!
//! This module handles soft deletions and tombstone tracking for vector indexes.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use crate::error::Result;

/// Manages deleted vectors in an index.
#[derive(Debug, Clone)]
pub struct DeletionManager {
    /// Set of deleted vector IDs.
    deleted_ids: Arc<RwLock<HashSet<u64>>>,

    /// Total number of deletions.
    deletion_count: Arc<RwLock<u64>>,
}

impl DeletionManager {
    /// Create a new deletion manager.
    pub fn new() -> Self {
        Self {
            deleted_ids: Arc::new(RwLock::new(HashSet::new())),
            deletion_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Mark a vector as deleted.
    pub fn delete(&self, vector_id: u64) -> Result<bool> {
        let mut deleted_ids = self.deleted_ids.write().unwrap();
        let inserted = deleted_ids.insert(vector_id);
        if inserted {
            let mut count = self.deletion_count.write().unwrap();
            *count += 1;
        }
        Ok(inserted)
    }

    /// Mark multiple vectors as deleted.
    pub fn delete_batch(&self, vector_ids: &[u64]) -> Result<u64> {
        let mut deleted_ids = self.deleted_ids.write().unwrap();
        let mut count = self.deletion_count.write().unwrap();
        let mut deleted_count = 0;

        for &vector_id in vector_ids {
            if deleted_ids.insert(vector_id) {
                *count += 1;
                deleted_count += 1;
            }
        }

        Ok(deleted_count)
    }

    /// Check if a vector is deleted.
    pub fn is_deleted(&self, vector_id: u64) -> bool {
        let deleted_ids = self.deleted_ids.read().unwrap();
        deleted_ids.contains(&vector_id)
    }

    /// Get total number of deletions.
    pub fn deletion_count(&self) -> u64 {
        *self.deletion_count.read().unwrap()
    }

    /// Get all deleted vector IDs.
    pub fn deleted_ids(&self) -> Vec<u64> {
        let deleted_ids = self.deleted_ids.read().unwrap();
        deleted_ids.iter().copied().collect()
    }

    /// Clear all deletions.
    pub fn clear(&self) -> Result<()> {
        let mut deleted_ids = self.deleted_ids.write().unwrap();
        let mut count = self.deletion_count.write().unwrap();
        deleted_ids.clear();
        *count = 0;
        Ok(())
    }

    /// Remove a vector ID from the deletion set (e.g., after compaction).
    pub fn undelete(&self, vector_id: u64) -> Result<bool> {
        let mut deleted_ids = self.deleted_ids.write().unwrap();
        let removed = deleted_ids.remove(&vector_id);
        if removed {
            let mut count = self.deletion_count.write().unwrap();
            *count = count.saturating_sub(1);
        }
        Ok(removed)
    }
}

impl Default for DeletionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about deletions in a segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentDeletionInfo {
    /// Segment ID.
    pub segment_id: String,

    /// Number of deleted vectors in this segment.
    pub deleted_count: u64,

    /// Total vectors in this segment.
    pub total_count: u64,

    /// Deletion ratio (deleted / total).
    pub deletion_ratio: f64,
}

impl SegmentDeletionInfo {
    /// Create a new segment deletion info.
    pub fn new(segment_id: String, deleted_count: u64, total_count: u64) -> Self {
        let deletion_ratio = if total_count > 0 {
            deleted_count as f64 / total_count as f64
        } else {
            0.0
        };

        Self {
            segment_id,
            deleted_count,
            total_count,
            deletion_ratio,
        }
    }

    /// Check if this segment needs compaction based on deletion ratio.
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.deletion_ratio >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deletion_manager_basic() {
        let manager = DeletionManager::new();

        assert!(!manager.is_deleted(1));
        assert_eq!(manager.deletion_count(), 0);

        manager.delete(1).unwrap();
        assert!(manager.is_deleted(1));
        assert_eq!(manager.deletion_count(), 1);

        // Deleting again should not increase count
        manager.delete(1).unwrap();
        assert_eq!(manager.deletion_count(), 1);
    }

    #[test]
    fn test_deletion_batch() {
        let manager = DeletionManager::new();

        let ids = vec![1, 2, 3, 4, 5];
        let deleted = manager.delete_batch(&ids).unwrap();
        assert_eq!(deleted, 5);
        assert_eq!(manager.deletion_count(), 5);

        // Deleting again should return 0
        let deleted = manager.delete_batch(&ids).unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(manager.deletion_count(), 5);
    }

    #[test]
    fn test_segment_deletion_info() {
        let info = SegmentDeletionInfo::new("seg1".to_string(), 50, 100);
        assert_eq!(info.deletion_ratio, 0.5);
        assert!(info.needs_compaction(0.3));
        assert!(!info.needs_compaction(0.6));
    }
}
