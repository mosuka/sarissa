//! Type definitions for segment management.

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
