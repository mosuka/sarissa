//! Segment management for inverted indexes.
//!
//! This module handles segment operations for inverted indexes:
//! - Segment manager for coordinating segments
//! - Merge engine for combining segments
//! - Merge policy for determining when to merge

use serde::{Deserialize, Serialize};

/// Information about a segment in the inverted index.
///
/// This structure contains metadata about an individual segment,
/// including document counts, offsets, and deletion status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// Segment identifier.
    pub segment_id: String,

    /// Number of documents in this segment.
    pub doc_count: u64,

    /// Minimum document ID in this segment.
    pub min_doc_id: u64,

    /// Maximum document ID in this segment.
    pub max_doc_id: u64,

    /// Generation number of this segment.
    pub generation: u64,

    /// Whether this segment has deletions.
    pub has_deletions: bool,

    /// Shard ID for this segment.
    pub shard_id: u16,
}

pub mod merge_engine;

// ---------------------------------------------------------------------------
// Merge data types (#1024).
//
// Extracted from the deleted `segment/manager.rs`: that file's
// `SegmentManager` and its binary `segments.manifest` ("SEGS") machinery
// were a complete parallel segment-management architecture that production
// never adopted — discovery and publication run on `segments.json` (#1021).
// These three types are the parts the live merge path genuinely uses.
// ---------------------------------------------------------------------------

/// Extended segment information with management metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManagedSegmentInfo {
    /// Core segment information.
    pub segment_info: SegmentInfo,

    /// Size of the segment in bytes.
    pub size_bytes: u64,

    /// Number of deleted documents in this segment.
    pub deleted_count: u64,

    /// Timestamp when segment was created.
    pub created_at: u64,

    /// Timestamp when segment was last modified.
    pub last_modified: u64,

    /// Merge tier (for tiered merge policy).
    pub tier: u8,

    /// Whether this segment is currently being merged.
    pub is_merging: bool,

    /// Segment file paths for cleanup.
    pub file_paths: Vec<String>,
}

impl ManagedSegmentInfo {
    /// Create new managed segment info.
    pub fn new(segment_info: SegmentInfo) -> Self {
        let now = crate::util::time::now_secs();

        ManagedSegmentInfo {
            segment_info,
            size_bytes: 0,
            deleted_count: 0,
            created_at: now,
            last_modified: now,
            tier: 0,
            is_merging: false,
            file_paths: Vec::new(),
        }
    }

    /// Get deletion ratio (deleted docs / total docs).
    pub fn deletion_ratio(&self) -> f64 {
        if self.segment_info.doc_count == 0 {
            0.0
        } else {
            self.deleted_count as f64 / self.segment_info.doc_count as f64
        }
    }

    /// Get effective document count (total - deleted).
    pub fn effective_doc_count(&self) -> u64 {
        self.segment_info
            .doc_count
            .saturating_sub(self.deleted_count)
    }

    /// Check if segment needs compaction.
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.deletion_ratio() > threshold
    }
}

/// Merge candidate representing segments to be merged.
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// Segments to merge.
    pub segments: Vec<String>,

    /// Priority score (higher = more urgent).
    pub priority: f64,

    /// Expected size after merge.
    pub estimated_size: u64,

    /// Merge strategy to use.
    pub strategy: MergeStrategy,
}

/// Merge strategy options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeStrategy {
    /// Size-based merging (small segments first).
    SizeBased,

    /// Deletion-based merging (high deletion ratio first).
    DeletionBased,

    /// Time-based merging (oldest segments first).
    TimeBased,

    /// Balanced approach considering multiple factors.
    Balanced,
}
