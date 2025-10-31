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

    /// Document ID offset for this segment.
    pub doc_offset: u64,

    /// Generation number of this segment.
    pub generation: u64,

    /// Whether this segment has deletions.
    pub has_deletions: bool,
}

pub mod manager;
pub mod merge_engine;
pub mod merge_policy;
