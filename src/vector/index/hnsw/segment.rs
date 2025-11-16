//! Segment management for HNSW vector indexes.
//!
//! This module handles segment operations for HNSW indexes:
//! - Segment manager for coordinating segments
//! - Merge engine for combining segments
//! - Merge policy for determining when to merge

use serde::{Deserialize, Serialize};

/// Information about a segment in the HNSW vector index.
///
/// This structure contains metadata about an individual segment,
/// including vector counts, offsets, and deletion status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentInfo {
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
}

pub mod manager;
pub mod merge_engine;
pub mod merge_policy;
