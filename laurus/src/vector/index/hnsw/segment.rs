//! HNSW-specific segment file layout and merge engine.
//!
//! The generic segment manifest/manager, merge policies, and reader cache
//! shared across index types live in [`crate::vector::index::segment`]
//! (Issue #889); this module keeps only what is HNSW-specific: the merge
//! engine (its I/O is HNSW-graph-typed) and this index type's on-disk file
//! layout descriptor.

use crate::vector::index::segment::manager::SegmentFileLayout;

pub mod merge_engine;

/// On-disk file-suffix layout for HNSW segments: a primary `.hnsw` file plus
/// its `.hnsw.f32` rerank sidecar, staged through a `.hnsw.tmp` temp file.
pub const LAYOUT: SegmentFileLayout = SegmentFileLayout {
    primary: ".hnsw",
    sidecars: &[".hnsw.f32"],
    tmp: ".hnsw.tmp",
};
