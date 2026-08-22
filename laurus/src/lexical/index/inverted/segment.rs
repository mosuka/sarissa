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

    /// Whether a commit has published this segment (#1017).
    ///
    /// The writer flushes a segment as soon as its buffer fills, long
    /// before the commit that makes those documents durable, so a segment
    /// can exist on storage while its contents are not yet part of the
    /// index a reader should see. Segment discovery skips anything that is
    /// not published, which is what makes the documented contract —
    /// documents become searchable only after `commit()` — hold regardless
    /// of whether a searcher happened to be built in between.
    ///
    /// Defaults to `true` when absent so that segments written before this
    /// field existed keep reading back as visible. That is correct: they
    /// were published the instant they were written, under the old rules.
    #[serde(default = "committed_default")]
    pub committed: bool,
}

/// Serde default for [`SegmentInfo::committed`] (#1017).
///
/// # Returns
///
/// `true`, so a `.meta` written before this field existed reads back as a
/// published segment.
fn committed_default() -> bool {
    true
}

pub mod manager;
pub mod merge_engine;
pub mod merge_policy;
